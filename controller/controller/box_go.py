#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, ColorRGBA, Bool
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import open3d as o3d
import math
import cv2

import tf2_ros
from tf2_geometry_msgs import do_transform_point

class NearBoxCatcher(Node):
    def __init__(self):
        super().__init__('nearbox_catcher')

        # ---------------- State machine & control parameters ----------------
        self.active = False
        self.finished = False
        self.collecting_box = False
        self.target_locked = False
        
        self.robot_pose = None  # (x, y, yaw)
        self.target_odom_x = None
        self.target_odom_y = None
        self.target_box_z = 0.0 # Used only for RViz height visualization

        self.box_buffer = []
        self.buffer_size = 3  # Sample 3 times and take the best to avoid single-frame noise

        # Motion control parameters
        self.v_max = 0.25
        self.v_min = 0.05
        self.w_max = 0.6
        self.rotate_threshold = 0.1  # radians (~8.5 degrees)
        self.stop_distance = 0.1     # tolerance distance to target

        # ---------------- Vision algorithm parameters ----------------
        self.max_distance = 1.8
        self.height_min, self.height_max = -0.05, 0.5
        self.voxel_size = 0.02
        self.distance_threshold = 0.02
        self.eps, self.min_points = 0.08, 30
        self.lower_hsv = np.array([43, 0, 56])
        self.upper_hsv = np.array([119, 61, 94])
        self.color_ratio_thresh = 0.3 
        self.stop_offset = 0.4 # Stop 0.4 meters in front of the box

        self.workspace_polygon = [
            (0.00, 0.00), (5.22, 0.00), (8.00, 2.02), (10.01, 2.04),
            (10.00, 4.22), (8.60, 4.23), (8.59, 2.67), (0.00, 2.70)
        ]

        # ---------------- TF & communication setup ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(PointCloud2, '/realsense/depth/color/points', self.pc_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Bool, '/box/go', self.go_callback, 10) # Trigger command

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.approach_pub = self.create_publisher(Bool, '/box/approach', 10) # Completion signal
        self.near_box_marker_pub = self.create_publisher(Marker, '/camera/depth/detected_near_box', 10)
        self.viz_target_marker_pub = self.create_publisher(MarkerArray, '/camera/depth/target_marker', 10)

        self.get_logger().info("📦 NearBox Catcher started and ready! Waiting for /box/go command...")

    # ---------------- Trigger command callback ----------------
    def go_callback(self, msg: Bool):
        if msg.data:
            self.active = True
            self.finished = False
            self.target_locked = False
            self.box_buffer = []
            self.collecting_box = True
            self.get_logger().info("🟢 Command received! Starting vision to find nearest green box...")
        else:
            self.active = False
            self.cmd_pub.publish(Twist()) # Emergency stop

    # ---------------- Odometry & motion control ----------------
    def odom_callback(self, msg: Odometry):
        # 1. Update robot pose
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

        # 2. If active and target locked, pursue target
        if self.active and self.target_locked and not self.finished:
            self.drive_to_target()

    def drive_to_target(self):
        if not self.robot_pose: return

        x, y, yaw = self.robot_pose
        dx = self.target_odom_x - x
        dy = self.target_odom_y - y
        dist = math.hypot(dx, dy)
        
        # Compute angle difference and normalize to [-pi, pi]
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        cmd = Twist()

        if dist < self.stop_distance:
            self.cmd_pub.publish(Twist()) # Send zero velocity to stop
            self.approach_pub.publish(Bool(data=True)) # Notify system arrival
            self.active = False
            self.finished = True
            self.get_logger().info("🏁 Reached box stopping position!")
            return

        # Prioritize orientation alignment
        if abs(alpha) > self.rotate_threshold:
            cmd.angular.z = math.copysign(self.w_max, alpha)
            cmd.linear.x = 0.0
        else:
            # Move forward while making small adjustments
            cmd.linear.x = max(self.v_min, min(self.v_max, 1.0 * dist)) # Simple P control
            cmd.angular.z = math.copysign(min(self.w_max, 6.0 * abs(alpha)), alpha)

        self.cmd_pub.publish(cmd)
 
    # ---------------- Vision processing core ----------------
    def pc_callback(self, msg: PointCloud2):
        # [Core optimization] If not collecting data, exit immediately to save CPU
        if not self.collecting_box or self.robot_pose is None:
            return

        points = pc2.read_points_numpy(msg, field_names=("x","y","z","rgb"), skip_nans=True)
        if len(points) < self.min_points: return

        xyz = points[:, :3]
        rgb_uint32 = points[:, 3].astype(np.float32).view(np.uint32)
        rgb_norm = np.stack([((rgb_uint32 >> 16) & 255) / 255.0, 
                             ((rgb_uint32 >> 8) & 255) / 255.0, 
                             (rgb_uint32 & 255) / 255.0], axis=1)

        mask = np.linalg.norm(xyz, axis=1) < self.max_distance
        xyz, rgb_norm = xyz[mask], rgb_norm[mask]
        if len(xyz) < self.min_points: return

        # Open3D processing
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_norm)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        _, inliers = pcd.segment_plane(self.distance_threshold, 3, 250)
        pcd_no_ground = pcd.select_by_index(inliers, invert=True)
        
        pts = np.asarray(pcd_no_ground.points)
        clr = np.asarray(pcd_no_ground.colors)
        h_mask = (pts[:, 1] > self.height_min) & (pts[:, 1] < self.height_max)
        pts, clr = pts[h_mask], clr[h_mask]

        if len(pts) < self.min_points: return

        labels = np.array(pcd_no_ground.select_by_index(np.where(h_mask)[0]).cluster_dbscan(self.eps, self.min_points))
        if labels.max() < 0: return

        try:
            trans = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
            return

        clr_u8 = (clr * 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv_all = cv2.cvtColor(clr_u8, cv2.COLOR_RGB2HSV)
        sorted_lbls = sorted(np.arange(labels.max() + 1), key=lambda i: np.sum(labels == i), reverse=True)

        nearest_box_odom = None
        min_distance = float('inf')
        nearest_pts = None

        for lbl in sorted_lbls:
            idx = (labels == lbl)
            c_pts = pts[idx]
            
            centroid = np.mean(c_pts, axis=0)
            ps = PointStamped()
            ps.header = msg.header
            ps.point.x, ps.point.y, ps.point.z = float(centroid[0]), float(centroid[1]), float(centroid[2])
            p_odom = do_transform_point(ps, trans)

            if not self.is_in_polygon(p_odom.point.x, p_odom.point.y): continue

            mask_hsv = cv2.inRange(hsv_all[idx], self.lower_hsv, self.upper_hsv)
            if (np.count_nonzero(mask_hsv) / len(c_pts)) > self.color_ratio_thresh:
                xr, yr, _ = self.robot_pose
                dist_to_robot = math.hypot(p_odom.point.x - xr, p_odom.point.y - yr)
                
                if dist_to_robot < min_distance:
                    min_distance = dist_to_robot
                    nearest_box_odom = p_odom
                    nearest_pts = c_pts

        # If a valid box is found, store it in buffer
        if nearest_box_odom is not None:
            self.box_buffer.append((nearest_box_odom.point.x, nearest_box_odom.point.y, nearest_box_odom.point.z, min_distance))
            self.get_logger().info(f"🔎 Found box sample {len(self.box_buffer)}/{self.buffer_size}")
            self.publish_near_box_marker(nearest_pts, msg.header)

            # Once enough samples are collected, compute final stop point
            if len(self.box_buffer) >= self.buffer_size:
                # Choose the closest sample as ground truth
                best_sample = min(self.box_buffer, key=lambda x: x[3])
                bx, by, bz, _ = best_sample
                
                # Compute direction vector and stopping point 0.4m before box
                xr, yr, _ = self.robot_pose
                dx, dy = bx - xr, by - yr
                dist = math.hypot(dx, dy)
                
                if dist > 0:
                    ux, uy = dx/dist, dy/dist 
                    self.target_odom_x = bx - self.stop_offset * ux
                    self.target_odom_y = by - self.stop_offset * uy
                    self.target_box_z = bz
                    
                    self.target_locked = True
                    self.collecting_box = False # ✅ Key: stop processing point cloud after locking
                    
                    self.publish_target_marker()
                    self.get_logger().info(f"🎯 Target locked! Moving to: X={self.target_odom_x:.2f}, Y={self.target_odom_y:.2f}")

    # ---------------- Helper: polygon boundary check ----------------
    def is_in_polygon(self, x, y):
        inside = False
        n = len(self.workspace_polygon)
        p1x, p1y = self.workspace_polygon[0]
        for i in range(n + 1):
            p2x, p2y = self.workspace_polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y):
                if x <= max(p1x, p2x) and p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # ---------------- Helper: RViz visualization ----------------
    def publish_near_box_marker(self, c_pts, header):
        if c_pts is None: return
        centroid = np.mean(c_pts, axis=0)
        min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
        size = max_p - min_p

        m = Marker()
        m.header = header
        m.ns = "near_box_shape"
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(centroid[0]), float(centroid[1]), float(centroid[2])
        m.scale.x, m.scale.y, m.scale.z = float(max(size[0], 0.05)), float(max(size[1], 0.05)), float(max(size[2], 0.05))
        m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
        self.near_box_marker_pub.publish(m)

    def publish_target_marker(self):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.id = 99
        m.pose.position.x = self.target_odom_x
        m.pose.position.y = self.target_odom_y
        m.pose.position.z = self.target_box_z
        m.scale.x = m.scale.y = m.scale.z = 0.08
        m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # Red target sphere
        ma.markers.append(m)
        self.viz_target_marker_pub.publish(ma)


def main():
    rclpy.init()
    node = NearBoxCatcher()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()