#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math

class CircleAroundObjectSLAM(Node):
    def __init__(self):
        super().__init__('circle_object_slam')

        # ===== Map setup =====
        self.resolution = 0.01
        self.width = 1001
        self.height = 423
        self.map = np.zeros((self.height, self.width), dtype=np.int8)

        self.pub = self.create_publisher(OccupancyGrid, '/slam_map', 10)

        # ===== Object to circle around =====
        self.obj_x = 49   # example object coordinates (from your workspace)
        self.obj_y = 50
        self.radius = 20  # cells away from object
        self.theta_circ = 0.0  # angle around object

        # Robot initial position
        self.robot_x = int(self.obj_x + self.radius * math.cos(self.theta_circ))
        self.robot_y = int(self.obj_y + self.radius * math.sin(self.theta_circ))
        self.robot_theta = self.theta_circ + math.pi/2  # tangent to circle

        self.timer = self.create_timer(0.2, self.update_map)

    def simulate_sensor(self):
        max_range = 50
        for d in range(1, max_range):
            x = int(self.robot_x + d * math.cos(self.robot_theta))
            y = int(self.robot_y + d * math.sin(self.robot_theta))
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.map[y, x] == 100:  # obstacle
                    return d
        return max_range

    def update_map(self):
        # Move robot around object
        self.theta_circ += 0.1  # rad per step
        self.robot_x = int(self.obj_x + self.radius * math.cos(self.theta_circ))
        self.robot_y = int(self.obj_y + self.radius * math.sin(self.theta_circ))
        self.robot_theta = self.theta_circ + math.pi/2  # tangent heading

        # Simulate sensor
        distance = self.simulate_sensor()
        for d in range(distance):
            x = int(self.robot_x + d * math.cos(self.robot_theta))
            y = int(self.robot_y + d * math.sin(self.robot_theta))
            if 0 <= x < self.width and 0 <= y < self.height:
                self.map[y, x] = -1  # free

        # Mark obstacle
        ox = int(self.robot_x + distance * math.cos(self.robot_theta))
        oy = int(self.robot_y + distance * math.sin(self.robot_theta))
        if 0 <= ox < self.width and 0 <= oy < self.height:
            self.map[oy, ox] = 100

        # Publish map
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = 0.0
        grid.data = self.map.flatten().tolist()
        self.pub.publish(grid)
        self.get_logger().info(f"Robot circling object at ({self.robot_x},{self.robot_y})")


def main():
    rclpy.init()
    node = CircleAroundObjectSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

