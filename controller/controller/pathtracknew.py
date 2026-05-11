#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Bool, Float64MultiArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs

import math
import numpy as np


class MapToOdomPathController(Node):

    def __init__(self):
        super().__init__('map_to_odom_path_controller')

        # -----------------------------
        # Parameters (better for real robot)
        # -----------------------------
        self.v_max = 0.3
        self.w_max = 1.0
        self.v_th=0.3
        self.v_min = 0.1
        self.k1 = 5.0
        self.k2 = 3.51
        self.k3 = 6.1
        self.lookahead_dist = 0.2  # bigger for real robot
        self.goal_tolerance = 0.05
        self.rotate_threshold = math.radians(2)
        self.front_pher_min=0.05
        # DEBUG FLAGS
        self.ignore_lidar = False   # 🔥 set True to test motion without obstacles

        # -----------------------------
        # State
        # -----------------------------
        self.path = []
        self.current_pose = None
        self.goal = None
        self.goal_reached_sent = False
        self.v_min = 0.2

        self.laser_ranges = []
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.path_safety_dist = 0.35

        # -----------------------------
        # TF
        # -----------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.create_subscription(Path, '/planned_path', self.map_path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/lidar/scan', self.scan_cb, 10)
        self.create_subscription(Float64MultiArray, '/controller_tuning', self.tuning_cb, 10)
        self.create_subscription(OccupancyGrid, '/pheromone_map', self.pheromone_cb, 10)

        # Pheromone map
        self.pher_map = None
        self.pher_resolution = None
        self.pher_width = None
        self.pher_height = None
        self.pher_origin = None
        # -----------------------------
        # Publishers
        # -----------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.goal_failed_pub = self.create_publisher(Bool, '/goal_failed', 10)
        self.marker_pub = self.create_publisher(Marker, '/pheromone_markers', 10)
        self.info_pub = self.create_publisher(Float64MultiArray, '/path_track/info', 10)

        # -----------------------------
        # Timer
        # -----------------------------
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(" Controller Ready")

    # -----------------------------
    # Callbacks
    # -----------------------------
    def tuning_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn("Use: [k1, k2, k3, lookahead, (optional v_min)]")
            return
        self.k1 = msg.data[0]
        self.k2 = msg.data[1]
        self.k3 = msg.data[2]
        self.lookahead_dist = msg.data[3]
        if len(msg.data) >= 5:
            self.v_min = msg.data[4]
        self.get_logger().info(
            f"Updated → k1={self.k1:.2f}, k2={self.k2:.2f}, k3={self.k3:.2f}, "
            f"lookahead={self.lookahead_dist:.2f}, v_min={self.v_min:.2f}"
        )

    def map_path_cb(self, msg: Path):
        path_odom = []

        for pose_map in msg.poses:
            point_map = PointStamped()
            point_map.header = pose_map.header
            point_map.point = pose_map.pose.position

            try:
                tf_map_to_odom = self.tf_buffer.lookup_transform(
                    'odom',
                    point_map.header.frame_id,
                    rclpy.time.Time()
                )

                point_odom = tf2_geometry_msgs.do_transform_point(point_map, tf_map_to_odom)
                path_odom.append((point_odom.point.x, point_odom.point.y))

            except TransformException as e:
                self.get_logger().warn(f"TF failed: {str(e)}")
                return

        if path_odom:
            self.path = path_odom
            self.goal = path_odom[-1]
            self.goal_reached_sent = False
            self.get_logger().info(f"✅ Path received: {len(self.path)} points")

    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose = (x, y, yaw)

    def scan_cb(self, msg: LaserScan):
        self.laser_ranges = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

        # DEBUG
        self.get_logger().info(f" Scan size: {len(self.laser_ranges)}")
    def pheromone_cb(self, msg: OccupancyGrid):
        self.pher_map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.pher_resolution = msg.info.resolution
        self.pher_width = msg.info.width
        self.pher_height = msg.info.height
        self.pher_origin = msg.info.origin
    def publish_pheromone_markers(self, points, values):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pheromone_debug"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0

        marker.points = []
        marker.colors = []
        for (px, py), val in zip(points, values):
            from geometry_msgs.msg import Point
            p = Point()
            p.x = px
            p.y = py
            p.z = 0.0
            marker.points.append(p)

            color = ColorRGBA()
            color.r = max(0.0, 1.0 - val)
            color.g = min(1.0, val)
            color.b = 0.0
            color.a = 1.0
            marker.colors.append(color)

        self.marker_pub.publish(marker)

    # Control loop
    def control_loop(self):

        self.get_logger().info(
            f"STATE → path:{len(self.path)} | pose:{self.current_pose is not None} | scan:{len(self.laser_ranges)}"
        )

        if not self.path or self.current_pose is None:
            return

        if not self.ignore_lidar and not self.laser_ranges:
            self.get_logger().warn(" No LaserScan yet")
            return

        x, y, yaw = self.current_pose
        goal_x, goal_y = self.goal
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # Goal reached
        if dist_to_goal < self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info("🧸 Goal reached")
            self.goal_reached_pub.publish(Bool(data=True))
            self.path = []
            return

        # Obstacle check
        path_blocked = False

        if not self.ignore_lidar:
            for px, py in self.path:
                dx = px - x
                dy = py - y
                dist = math.hypot(dx, dy)

                if dist < self.lookahead_dist+0.9:
                    angle = math.atan2(dy, dx) - yaw
                    angle = self.normalize_angle(angle)

                    if -math.pi/6 <= angle <= math.pi/6:
                        idx = int((angle - self.angle_min) / self.angle_increment)

                        if 0 <= idx < len(self.laser_ranges):
                            r = self.laser_ranges[idx]

                            if np.isfinite(r) and r < self.path_safety_dist:
                                self.get_logger().warn(f"🚨 Obstacle at {r:.2f} m")
                                path_blocked = True
                                break

        if path_blocked:
            self.stop_robot()
            self.goal_failed_pub.publish(Bool(data=True))
            self.path = []
            return

        # Pure pursuit
        closest_index = min(
            range(len(self.path)),
            key=lambda i: math.hypot(self.path[i][0]-x, self.path[i][1]-y)
        )

        target = self.goal
        for i in range(closest_index, len(self.path)):
            px, py = self.path[i]
            if math.hypot(px - x, py - y) >= self.lookahead_dist:
                target = (px, py)
                break

        tx, ty = target
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy)

        target_theta = math.atan2(dy, dx)
        delta_theta = self.normalize_angle(target_theta - yaw)
        # pheromone computation
        avg_pher = 0.0
        lookahead_points = []
        pher_values = []

        if self.pher_map is not None:
            pher_vals = []
            weights = []
            lookahead_max = 0.4
            lookahead_mid = lookahead_max / 2.0

            for i in range(closest_index, len(self.path)):
                px, py = self.path[i]
                pdist = math.hypot(px - x, py - y)
                if pdist < 0.05:
                    continue
                if pdist > lookahead_max:
                    break
                angle = math.atan2(py - y, px - x)
                dtheta = self.normalize_angle(angle - yaw)
                if abs(dtheta) > np.pi / 2:
                    continue  # ignore points behind

                # Map indices
                gx = int((px - self.pher_origin.position.x) / self.pher_resolution)
                gy = int((py - self.pher_origin.position.y) / self.pher_resolution)
                if 0 <= gx < self.pher_width and 0 <= gy < self.pher_height:
                    # Average over 3x3 neighborhood
                    neighborhood = self.pher_map[max(0, gy-1):min(self.pher_height, gy+2),
                                                 max(0, gx-1):min(self.pher_width, gx+2)]
                    val = np.clip(np.mean(neighborhood), 0, 100) / 100.0
                else:
                    val = 0.0

                pher_vals.append(val)
                weights.append(1.0)
                lookahead_points.append((px, py))
                pher_values.append(val)

            if weights:
                # Weighted average along lookahead
                avg_pher = sum([p*w for p,w in zip(pher_vals, weights)]) / sum(weights)

            # Debug: log avg pheromone
            self.get_logger().info(f"Avg pheromone ahead: {avg_pher:.3f}")

            # Publish RViz markers
            if lookahead_points:
                self.publish_pheromone_markers(lookahead_points, pher_values)

        # -----------------------------
        # Velocity mapping
        # -----------------------------
        if avg_pher >= 0.7:
            va = self.v_max
        elif avg_pher <= 0.3:
            va = self.v_min
        else:
            scale = (avg_pher - 0.3) / (0.7 - 0.3)
            va = self.v_min + scale * (self.v_max - self.v_min)
        if va>0.4:
                    self.k1 = 5.0
                    self.k2 = 2.0
                    self.k3 = 4.0
        else :
                    self.k1 = 5.0
                    self.k2 = 2.5
                    self.k3 = 5.1
        # Safety: do not go full speed if front pheromone too low
        if avg_pher < self.front_pher_min:
            va = min(va, self.v_min + 0.1)
        v = max(self.v_th, min(va, self.k1 * d))* math.exp(-self.k3 * delta_theta**2)
        w = math.copysign(min(self.w_max, self.k2 * abs(delta_theta)), delta_theta)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)

        self.get_logger().info(f"CMD → v:{v:.2f}, w:{w:.2f}")

        self.cmd_pub.publish(cmd)
        # Publish info
        info = Float64MultiArray()
        info.data = [
            x, y, yaw,
            tx, ty, target_theta,
            dx, dy, delta_theta,
            d, v, w,
            closest_index, va, avg_pher
        ]
        self.info_pub.publish(info)
    # -----------------------------
    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def quaternion_to_yaw(self, q):
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)

    node = MapToOdomPathController()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()