#!/usr/bin/env python3
import csv
import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped


class DummyPerceptionNode(Node):
    def __init__(self):
        super().__init__('dummy_perception')

        qos_pub = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/perception/markersT', qos_pub)
        self.box_pub = self.create_publisher(MarkerArray, '/perception/box', qos_pub)
        self.pose_pub = self.create_publisher(PoseStamped, '/set_pose', 10)

        self.marker_id = 0
        self.cube_size = 0.05
        self.box_L = 0.24
        self.box_W = 0.16

        self.cube_queue = []

        self.load_from_csv()

    def publish_initial_pose(self, x, y, yaw_deg):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x / 100.0
        pose.pose.position.y = y / 100.0

        yaw = np.deg2rad(yaw_deg)
        pose.pose.orientation.z =  yaw
        pose.pose.orientation.w =  0.0

        self.pose_pub.publish(pose)
        self.get_logger().info(f"Initial pose set: {x},{y},{yaw_deg}")

    def publish_marker(self, obj):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = obj["id"]
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = obj["x"]
        marker.pose.position.y = obj["y"]
        marker.pose.position.z = 0.05

        if obj["type"] == "cube":
            marker.type = Marker.CUBE
            marker.scale.x = marker.scale.y = marker.scale.z = self.cube_size
            pub = self.marker_pub
        else:
            marker.type = Marker.CUBE
            marker.scale.x = self.box_L
            marker.scale.y = self.box_W
            marker.scale.z = 0.05
            pub = self.box_pub

        r, g, b = obj["color"]
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.95

        ma = MarkerArray()
        ma.markers.append(marker)
        pub.publish(ma)

    def load_from_csv(self):
        path = os.path.expanduser("~/dd2419_ws/task/map.csv")

        if not os.path.exists(path):
            self.get_logger().error(f"CSV not found: {path}")
            return

        with open(path, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                t = row["Type"].strip()
                x = float(row["x"])
                y = float(row["y"])
                angle = float(row["angle"])

                if t == "S":
                    self.publish_initial_pose(x, y, angle)
                    continue

                if t == "O":
                    color = (1.0, 0.0, 0.0)
                    obj_type = "cube"
                elif t == "B":
                    color = (0.8, 0.8, 0.2)
                    obj_type = "box"
                else:
                    continue

                obj = {
                    "id": self.marker_id,
                    "type": obj_type,
                    "x": x / 100.0,
                    "y": y / 100.0,
                    "color": color
                }

                self.marker_id += 1

                self.publish_marker(obj)

                if obj_type == "cube":
                    self.cube_queue.append(obj)

        self.get_logger().info("CSV loaded successfully")


def main():
    rclpy.init()

    node = DummyPerceptionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()