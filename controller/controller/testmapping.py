#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import PoseStamped

import numpy as np
import math
import open3d
import os
import csv

from open3d.pipelines.registration import (
    registration_icp,
    TransformationEstimationPointToPoint,
    ICPConvergenceCriteria
)

from rclpy.qos import QoSProfile


class ICP_SLAM_Node(Node):
    def __init__(self):
        super().__init__('icp_slam_node')

        # ---------------- Parameters ----------------
        self.declare_parameter("fitness_threshold", 0.2)
        self.declare_parameter("rmse_threshold", 0.2)

        self.fitness_threshold = self.get_parameter("fitness_threshold").value
        self.rmse_threshold = self.get_parameter("rmse_threshold").value

        # ---------------- Pose ----------------
        self.pose = np.eye(4)

        if not self.load_initial_pose_from_csv():
            self.get_logger().error("Failed to initialize pose from CSV. Shutting down.")
            raise RuntimeError("CSV initialization failed")

        # ---------------- ICP ----------------
        self.global_map = open3d.geometry.PointCloud()
        self.first_scan = True

        # ---------------- Scan stability ----------------
        self.scan_count = 0
        self.init_after_n_scans = 5
        self.min_valid_points = 50

        # ---------------- Grid ----------------
        self.resolution = 0.15
        self.min_x, self.max_x = -5.0, 10.0
        self.min_y, self.max_y = -5.0, 5.0

        self.size_x = int((self.max_x - self.min_x) / self.resolution)
        self.size_y = int((self.max_y - self.min_y) / self.resolution)

        self.origin_x = self.min_x
        self.origin_y = self.min_y

        self.grid = np.full((self.size_y, self.size_x), -1, dtype=np.int8)

        # ---------------- Cached data ----------------
        self.latest_cloud = None
        self.latest_ranges = None
        self.latest_angles = None
        self.latest_stamp = None

        # ---------------- ROS ----------------
        qos = QoSProfile(depth=1)

        self.scan_sub = self.create_subscription(
            LaserScan, '/lidar/scan', self.scan_callback, 1)

        self.update_sub = self.create_subscription(
            Bool, '/update_map', self.update_callback, 1)

        self.odom_pub = self.create_publisher(Odometry, '/odomA', qos)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map_icp', qos)

        # NEW: external pose reset topic
        self.set_pose_pub = self.create_publisher(PoseStamped, '/set_pose', 1)

        self.get_logger().info("ICP SLAM Started (CSV REQUIRED)")

    # ---------------- CSV pose ----------------
    def load_initial_pose_from_csv(self):
        path = os.path.expanduser("~/dd2419_ws/task/map.csv")

        if not os.path.exists(path):
            self.get_logger().error(f"CSV not found: {path}")
            return False

        with open(path, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row["Type"].strip() == "S":
                    x = float(row["x"]) / 100.0
                    y = float(row["y"]) / 100.0
                    yaw = math.radians(float(row["angle"]))

                    self.pose = np.eye(4)
                    c, s = math.cos(yaw), math.sin(yaw)

                    self.pose[0:2, 0:2] = [[c, -s], [s, c]]
                    self.pose[0, 3] = x
                    self.pose[1, 3] = y

                    self.get_logger().info(
                        f"Initialized pose from CSV: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
                    )
                    return True

        self.get_logger().error("No 'S' entry found in CSV")
        return False

    # ---------------- Scan → cloud ----------------
    def scan_to_cloud(self, scan):
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

        mask = np.isfinite(ranges)
        ranges = ranges[mask]
        angles = angles[mask]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)

        pts = np.vstack((x, y, z)).T
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(pts)

        return cloud, ranges, angles

    # ---------------- ICP ----------------
    def do_icp(self, source, target):
        return registration_icp(
            source, target,
            0.3,
            np.identity(4),
            TransformationEstimationPointToPoint(),
            ICPConvergenceCriteria(max_iteration=30)
        )

    # ---------------- Odometry ----------------
    def publish_odometry(self, stamp):
        x, y = self.pose[0, 3], self.pose[1, 3]
        yaw = math.atan2(self.pose[1, 0], self.pose[0, 0])

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw / 2)
        odom.pose.pose.orientation.w = math.cos(yaw / 2)

        self.odom_pub.publish(odom)

    # ---------------- NEW: external pose reset ----------------
    def publish_set_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"

        x, y = self.pose[0, 3], self.pose[1, 3]
        yaw = math.atan2(self.pose[1, 0], self.pose[0, 0])

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = yaw


        self.set_pose_pub.publish(msg)

    # ---------------- Grid ----------------
    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def update_grid(self, ranges, angles):
        rx, ry = self.pose[0, 3], self.pose[1, 3]
        yaw = math.atan2(self.pose[1, 0], self.pose[0, 0])

        for r, a in zip(ranges, angles):
            wx = rx + r * math.cos(a + yaw)
            wy = ry + r * math.sin(a + yaw)

            gx, gy = self.world_to_grid(wx, wy)
            if 0 <= gx < self.size_x and 0 <= gy < self.size_y:
                self.grid[gy, gx] = 100

    def publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.size_x
        msg.info.height = self.size_y
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = self.grid.flatten(order='C').tolist()
        self.map_pub.publish(msg)

    # ---------------- Map update ----------------
    def process_map_update(self):
        if self.latest_cloud is None:
            return

        self.update_grid(self.latest_ranges, self.latest_angles)

        cloud_map = open3d.geometry.PointCloud(self.latest_cloud)
        cloud_map.transform(self.pose)

        self.global_map += cloud_map
        self.global_map = self.global_map.voxel_down_sample(0.15)

        stamp = self.latest_stamp

        self.publish_map(stamp)
        self.publish_odometry(stamp)

        # NEW: send pose reset signal to external odometry node
        #self.publish_set_pose(stamp)

        self.get_logger().info("Map + odometry + set_pose published")

    # ---------------- Scan callback ----------------
    def scan_callback(self, scan):
        cloud, ranges, angles = self.scan_to_cloud(scan)

        self.scan_count += 1

        if len(ranges) < self.min_valid_points:
            return

        self.latest_cloud = cloud
        self.latest_ranges = ranges
        self.latest_angles = angles
        self.latest_stamp = scan.header.stamp

        # -------- initialization --------
        if self.first_scan:
            if self.scan_count < self.init_after_n_scans:
                return

            init_cloud = open3d.geometry.PointCloud(cloud)
            init_cloud.transform(self.pose)
            self.global_map = init_cloud

            self.update_grid(ranges, angles)
            self.publish_map(scan.header.stamp)
            self.publish_odometry(scan.header.stamp)

            self.first_scan = False
            return

        # -------- ICP --------
        guess = open3d.geometry.PointCloud(cloud)
        guess.transform(self.pose)

        result = self.do_icp(guess, self.global_map)

        if result.fitness < self.fitness_threshold or result.inlier_rmse > self.rmse_threshold:
            self.get_logger().warn("ICP rejected")
            return

        self.pose = result.transformation @ self.pose
        self.publish_odometry(scan.header.stamp)

    # ---------------- Trigger ----------------
    def update_callback(self, msg):
        if msg.data:
            self.get_logger().info("Map update triggered")
            self.process_map_update()


def main():
    rclpy.init()
    node = ICP_SLAM_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
