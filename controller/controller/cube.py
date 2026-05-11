#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool, Float64MultiArray
import math
import numpy as np


class NearCubeCatcher(Node):

    def __init__(self):
        super().__init__('nearcube_catcher')

        # ---------------- PARAMETERS ----------------
        self.target_z = 0.22
        # self.k_lin = 5.0
        # self.max_lin = 0.25
        # self.max_ang = 1.0
        self.k_lin = 5.0
        self.max_lin = 0.25
        self.max_ang = 0.8
        self.angle_threshold_deg = 10.0

        # ---------------- ROBOT STATE ----------------
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.initialized = False

        # ---------------- CUBE TARGET ----------------
        self.target_odom_x = None
        self.target_odom_y = None
        self.target_set = False

        # Collect closest buffer
        self.cube_buffer = []
        self.buffer_size = 5
        self.collecting_cube = False

        # ---------------- MOTION STATE ----------------
        self.active = False
        self.finished = False

        # Camera offsets (from base_link)
        self.realsense_offset_x = 0.0175
        self.realsense_offset_z = 0.10456

        # ---------------- SUBSCRIBERS ----------------
        self.create_subscription(Marker, '/perception/nearcube', self.cube_callback, 10)
        self.create_subscription(Bool, '/cube/go', self.go_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 50)
        self.create_subscription(Float64MultiArray, '/nearcube_controller_params', self.update_params_callback, 10)

        # ---------------- PUBLISHERS ----------------
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.approach_pub = self.create_publisher(Bool, '/cube/approach', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/cube/target_pose', 10)  # NEW: publish goal pose

        self.get_logger().info("🎯 NearCube Catcher (Closest Position Mode)")

    # ---------------- GO CALLBACK ----------------
    def go_callback(self, msg: Bool):
        if msg.data:
            self.active = True
            self.finished = False
            self.target_set = False

            self.cube_buffer = []
            self.collecting_cube = True

            self.initial_x = self.odom_x
            self.initial_y = self.odom_y
            self.initial_yaw = self.odom_yaw

            self.get_logger().info(" Collecting cube samples for closest selection...")
        else:
            self.active = False

    # ---------------- ODOM CALLBACK ----------------
    def odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        if not self.initialized:
            self.initialized = True
            self.initial_x = self.odom_x
            self.initial_y = self.odom_y
            self.initial_yaw = self.odom_yaw

        if self.active and self.target_set and not self.finished:
            self.drive_to_target()

    # ---------------- CUBE CALLBACK ----------------
    def cube_callback(self, msg: Marker):

        if not self.collecting_cube or not self.initialized:
            return
        self.initial_x = self.odom_x
        self.initial_y = self.odom_y
        self.initial_yaw = self.odom_yaw
        # Camera frame coordinates

        cube_x  = msg.pose.position.z
        cube_y  = -msg.pose.position.x
        cube_x_cam = cube_x * math.cos(self.initial_yaw) - cube_y  * math.sin(self.initial_yaw)
        cube_y_cam = cube_x * math.sin(self.initial_yaw) + cube_y  * math.cos(self.initial_yaw)
        dist = math.sqrt(cube_x**2 + cube_y**2)
        self.cube_buffer.append((cube_x, cube_y, dist))

        self.get_logger().info(
            f"📏 Sample {len(self.cube_buffer)}/{self.buffer_size} | "
            f"x={cube_x_cam:.3f}, y={cube_y_cam:.3f}, d={dist:.3f}"
        )

        # When enough samples collected
        if len(self.cube_buffer) >= self.buffer_size:

            # Select closest sample
            closest = min(self.cube_buffer, key=lambda x: x[2])
            best_x, best_y, best_d = closest

            self.get_logger().info(
                f"🏆 Closest sample selected | x={best_x:.3f}, y={best_y:.3f}, d={best_d:.3f}"
            )
            cube_base_x = best_x + self.realsense_offset_z  # forward offset
            cube_base_y = best_y + self.realsense_offset_x  # lateral offset
            cube_x_rot = cube_base_x * math.cos(self.initial_yaw) -cube_base_y * math.sin(self.initial_yaw)
            cube_y_rot = cube_base_x* math.sin(self.initial_yaw) + cube_base_y * math.cos(self.initial_yaw)

            
            self.target_odom_x = self.initial_x + cube_x_rot 
            self.target_odom_y = self.initial_y + cube_y_rot 
            self.target_set = True
            self.collecting_cube = False

            self.get_logger().info(
                f"📍 Target locked at odom: x={self.target_odom_x:.3f}, y={self.target_odom_y:.3f}"
            )

            # --- Publish as PoseStamped for visualization ---
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'odom'
            pose_msg.pose.position.x = self.target_odom_x
            pose_msg.pose.position.y = self.target_odom_y
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0  # Neutral orientation
            self.pose_pub.publish(pose_msg)

    # ---------------- DRIVE CONTROL ----------------
    def drive_to_target(self):
        dx = self.target_odom_x - self.odom_x
        dy = self.target_odom_y - self.odom_y
        dist = math.sqrt(dx * dx + dy * dy)

        theta = math.atan2(dy, dx) - self.odom_yaw
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        cmd = Twist()

        if dist <= self.target_z:
            self.pub.publish(Twist())
            self.approach_pub.publish(Bool(data=True))
            self.active = False
            self.finished = True
            self.get_logger().info("🏁 Cube reached!")
            return

        if abs(np.degrees(theta)) > self.angle_threshold_deg:
            cmd.angular.z = math.copysign(self.max_ang, theta)
            cmd.linear.x = 0.0
        else:
            cmd.angular.z = 0.0
            cmd.linear.x = np.clip(self.k_lin * (dist - self.target_z),
                                   -self.max_lin, self.max_lin)

        self.pub.publish(cmd)

    # ---------------- HELPERS ----------------
    def quaternion_to_yaw(self, q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def update_params_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 7:
            return
        self.target_z = msg.data[0]
        self.k_lin = msg.data[2]
        self.max_lin = msg.data[4]
        self.max_ang = msg.data[5]
        self.angle_threshold_deg = msg.data[6]


def main(args=None):
    rclpy.init(args=args)
    node = NearCubeCatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()