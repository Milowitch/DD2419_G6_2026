#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import os
import csv
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler


class ObjectGlobalTransformer(Node):

    def __init__(self):
        super().__init__('object_global_transformer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Workspace polygon (meters)
        self.workspace_polygon = [
            (0.00, 0.00),
            (5.22, 0.00),
            (8.00, 2.02),
            (10.01, 2.04),
            (10.00, 4.22),
            (8.60, 4.23),
            (8.59, 2.67),
            (0.00, 2.70)
        ]

        # Cube register
        self.global_cubes = {}        # confirmed cubes {id: {'pos': np.array([x,y,z]), 'color':[r,g,b]}}
        self.candidate_cubes = {}     # temporary candidates {id: {'pos', 'observations', 'color'}}
        self.next_cube_id = 3

        self.position_threshold = 0.40   # 5 cm XY threshold for candidate merging
        self.z_max = 0.05                 # max Z height for a valid cube
        self.confirmations_needed = 3 # must see 7 times
        
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(
            MarkerArray,
            '/perception/markers',
            self.marker_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/perception/markersT',
            qos
        )
        
        # Optional storage from CSV
        self.cubes_active = {}
        self.boxes = {}
        self.initial_pose = None

        self.box_pub = self.create_publisher(
            MarkerArray,
            '/perception/box',
            qos
        )

        # Load static map
        self.load_from_csv()
        self.get_logger().info("Cube registry transformer started.")

    def load_from_csv(self):
        path = os.path.expanduser("~/dd2419_ws/task/map.csv")

        if not os.path.exists(path):
            self.get_logger().error(f"CSV not found: {path}")
            return

        cube_markers = MarkerArray()
        box_markers = MarkerArray()
        marker_id = 0
        marker_ida = 0
        
        with open(path, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                t = row["Type"].strip()
                x = float(row["x"]) / 100.0
                y = float(row["y"]) / 100.0
                angle = float(row["angle"])

                # --- INITIAL POSE ---
                if t == "S":
                    pose = PoseStamped()
                    pose.header.frame_id = "map"

                    pose.pose.position.x = x
                    pose.pose.position.y = y

                    q = quaternion_from_euler(0, 0, np.deg2rad(angle))
                    pose.pose.orientation.x = q[0]
                    pose.pose.orientation.y = q[1]
                    pose.pose.orientation.z = q[2]
                    pose.pose.orientation.w = q[3]

                    self.initial_pose = pose
                    continue

                if t == "O":  # cube
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.ns = "csv_cubes"         # 【修改1】增加命名空间防止RViz中闪烁覆盖
                    marker.id = marker_ida
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.color.a = 0.95
                    marker.type = Marker.CUBE
                    marker.scale.x = marker.scale.y = marker.scale.z = 0.05
                    marker.color.r = 1.0
                    cube_markers.markers.append(marker)

                    self.cubes_active[marker_ida] = (x, y)
                    marker_ida+=1
                elif t == "B":  # box
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.ns = "csv_boxes"         # 【修改1】增加命名空间防止RViz中闪烁覆盖
                    marker.id = marker_id
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.color.a = 0.95
                    marker.type = Marker.CUBE
                    marker.scale.x = 0.24
                    marker.scale.y = 0.16
                    marker.scale.z = 0.05
                    marker.color.g = 1.0
                    box_markers.markers.append(marker)

                    self.boxes[marker_id] = (x, y)
                    marker_id += 1

        # publish static map objects once
        self.marker_pub.publish(cube_markers)
        self.box_pub.publish(box_markers)
        self.get_logger().info("✅ CSV perception + initial pose loaded")
    def marker_callback(self, msg: MarkerArray):
        updated = False

        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue

            point_cam = PointStamped()
            point_cam.header = marker.header
            point_cam.point.x = marker.pose.position.x
            point_cam.point.y = marker.pose.position.y
            point_cam.point.z = marker.pose.position.z

            color = [marker.color.r, marker.color.g, marker.color.b]

            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    marker.header.frame_id,
                    rclpy.time.Time()
                )
                point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)
                new_pos = np.array([point_map.point.x, point_map.point.y, point_map.point.z])

                # Workspace & height filter
                if not self.is_inside_workspace(new_pos[0], new_pos[1]) or new_pos[2] > self.z_max:
                    continue

                # Skip if already a confirmed cube nearby (use slightly larger threshold)
                if self.is_existing_cube(new_pos, threshold=self.position_threshold * 1.5):
                    continue

                # --- Candidate merging ---
                matched_candidate = None
                for candidate in self.candidate_cubes.values():
                    dist = np.linalg.norm(new_pos[:2] - candidate['pos'][:2])
                    if dist < self.position_threshold:
                        matched_candidate = candidate
                        break

                if matched_candidate:
                    matched_candidate['observations'].append(new_pos)
                    matched_candidate['pos'] = np.mean(matched_candidate['observations'], axis=0)
                else:
                    # Add new candidate
                    candidate_id = f"cand_{len(self.candidate_cubes)}"
                    self.candidate_cubes[candidate_id] = {
                        'id': candidate_id,
                        'pos': new_pos,
                        'observations': [new_pos],
                        'color': color
                    }
                    matched_candidate = self.candidate_cubes[candidate_id]

                # Confirm cube if enough observations
                if len(matched_candidate['observations']) >= self.confirmations_needed:
                    
                    # 【修改2】最终查重：如果均值收敛后发现其实就是已知Cube（稍微放宽一点阈值，如 0.25米吸收累积噪点）
                    if self.is_existing_cube(matched_candidate['pos'], threshold=0.25):
                        # 删除候选池中的该项目，不发布为新Cube
                        del self.candidate_cubes[matched_candidate['id']]
                        continue

                    cube_id = self.next_cube_id+5
                    self.global_cubes[cube_id] = {
                        'pos': matched_candidate['pos'],
                        'color': matched_candidate['color']
                    }
                    self.next_cube_id += 1
                    updated = True
                    self.get_logger().info(
                        f"Confirmed cube → ID {cube_id} | "
                        f"X:{matched_candidate['pos'][0]:.3f} Y:{matched_candidate['pos'][1]:.3f} Z:{matched_candidate['pos'][2]:.3f} | "
                        f"Color: R{matched_candidate['color'][0]:.1f} G{matched_candidate['color'][1]:.1f} B{matched_candidate['color'][2]:.1f}"
                    )
                    # Remove candidate
                    del self.candidate_cubes[matched_candidate['id']]

            except TransformException:
                continue

        # Publish all confirmed cubes as MarkerArray if updated
        if updated:
            self.publish_full_marker_array()

    def is_existing_cube(self, new_pos, threshold=None):
        if threshold is None:
            threshold = self.position_threshold * 1.5

        # Check confirmed perception cubes
        for cube in self.global_cubes.values():
            if np.linalg.norm(new_pos[:2] - cube['pos'][:2]) < threshold:
                return True

        # ALSO check CSV cubes
        for (x, y) in self.cubes_active.values():
            if np.linalg.norm(new_pos[:2] - np.array([x, y])) < threshold:
                return True
        for (x, y) in self.boxes.values():
            if np.linalg.norm(new_pos[:2] - np.array([x, y])) < threshold:
                return True

        return False

    def publish_full_marker_array(self):
        marker_array = MarkerArray()

        for cube_id, cube in self.global_cubes.items():
            pos = cube['pos']
            color = cube['color']

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "live_cubes"           # 【修改3】分配独立命名空间
            marker.id = cube_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = 0.05      # FIXED HEIGHT

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = 0.95              

            marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()

            marker_array.markers.append(marker)

            # Broadcast TF with fixed Z
            self.broadcast_tf(cube_id, pos)

        self.marker_pub.publish(marker_array)

    def broadcast_tf(self, cube_id, pos):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = f"cube_{cube_id}"

        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = 0.05      # FIXED HEIGHT

        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    def is_inside_workspace(self, x, y):
        num = len(self.workspace_polygon)
        j = num - 1
        inside = False
        for i in range(num):
            xi, yi = self.workspace_polygon[i]
            xj, yj = self.workspace_polygon[j]
            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside


def main():
    rclpy.init()
    node = ObjectGlobalTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()