#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Bool, Float64MultiArray

from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs

import math


class MapToOdomPathController(Node):

    def __init__(self):
        super().__init__('map_to_odom_path_controller')

        # Controller Parameters

        self.v_max = 0.35
        self.w_max = 0.55

        self.k1 = 0.6
        self.k2 = 2.25
        self.k3 = 0.87
        self.lookahead_dist = 0.25
        self.v_min = 0.00  # anti-stuck
        self.goal_tolerance = 0.05
        self.rotate_threshold = math.radians(45)  # rotate-first threshold


        self.path = []
        self.current_pose = None
        self.goal = None
        self.goal_reached_sent = False
        self.path_active = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)


        self.create_subscription(Path, '/planned_path', self.map_path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Live tuning
        self.create_subscription(
            Float64MultiArray,
            '/controller_tuning',
            self.tuning_cb,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)

        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("🔥 Map→Odom Controller + Rotate-first + Live Tuning Ready")


    def tuning_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn(
                "Use: [k1, k2, k3, lookahead, (optional v_min)]"
            )
            return

        self.k1 = msg.data[0]
        self.k2 = msg.data[1]
        self.k3 = msg.data[2]
        self.lookahead_dist = msg.data[3]

        if len(msg.data) >= 5:
            self.v_min = msg.data[4]

        self.get_logger().info(
            f"Updated → k1={self.k1:.2f}, k2={self.k2:.2f}, "
            f"k3={self.k3:.2f}, lookahead={self.lookahead_dist:.2f}, "
            f"v_min={self.v_min:.2f}"
        )


    def map_path_cb(self, msg: Path):
        path_odom = []

        for pose_map in msg.poses:
            point_map = PointStamped()
            point_map.header = pose_map.header
            point_map.point.x = pose_map.pose.position.x
            point_map.point.y = pose_map.pose.position.y
            point_map.point.z = pose_map.pose.position.z

            try:
                tf_map_to_odom = self.tf_buffer.lookup_transform(
                    'odom',
                    point_map.header.frame_id,
                    point_map.header.stamp
                )

                point_odom = tf2_geometry_msgs.do_transform_point(
                    point_map, tf_map_to_odom
                )

            except TransformException as ex:
                self.get_logger().warn(f"TF failed: {ex}")
                continue

            path_odom.append((point_odom.point.x, point_odom.point.y))

        if path_odom:
            self.path = path_odom
            self.goal = path_odom[-1]
            self.goal_reached_sent = False
            self.get_logger().info(f"Path received ({len(self.path)} pts)")

            # 🚀 Resume movement when a new path is received
            self.path_active = True


    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose = (x, y, yaw)

    def control_loop(self):
        if not self.path or self.current_pose is None:
            return

        x, y, yaw = self.current_pose
        goal_x, goal_y = self.goal
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # ----------------------------
        # Goal reached
        # ----------------------------
        if dist_to_goal < self.goal_tolerance:
            self.stop_robot()
            if not self.goal_reached_sent:
                self.get_logger().info("Goal reached")
                self.goal_reached_pub.publish(Bool(data=True))
                self.goal_reached_sent = True
                self.path_active = True
            return
        else:
            self.goal_reached_sent = False

        # ----------------------------
        # Closest point
        # ----------------------------
        closest_index = min(
            range(len(self.path)),
            key=lambda i: math.hypot(self.path[i][0]-x, self.path[i][1]-y)
        )

        # ----------------------------
        # Lookahead target
        # ----------------------------
        target = None
        for i in range(closest_index, len(self.path)):
            px, py = self.path[i]
            if math.hypot(px - x, py - y) >= self.lookahead_dist:
                target = (px, py)
                break

        if target is None:
            target = self.goal

        tx, ty = target

        # ----------------------------
        # Errors
        # ----------------------------
        dx = tx - x
        dy = ty - y
        d = math.hypot(dx, dy)
        target_theta = math.atan2(dy, dx)
        delta_theta = self.normalize_angle(target_theta - yaw)

        # =============================
        # 🚀 Rotate-first + Nonlinear Controller
        # =============================
        if abs(delta_theta) > self.rotate_threshold:
            v = 0.0
            w = math.copysign(self.w_max, delta_theta)
        else:
            v = max(self.v_min, min(self.v_max, self.k1*d) * math.exp(-self.k3*delta_theta**2))
            w = math.copysign(min(self.w_max, self.k2*abs(delta_theta)), delta_theta)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

    # =====================================================
    # Helpers
    # =====================================================
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

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

