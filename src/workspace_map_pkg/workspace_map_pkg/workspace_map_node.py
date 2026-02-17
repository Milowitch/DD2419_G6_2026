#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class WorkspaceMap(Node):

    def __init__(self):
        super().__init__('workspace_map')

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(OccupancyGrid, '/map', qos)

        # Map settings
        self.resolution = 0.01
        self.width = 1001
        self.height = 423

        # Inflation
        self.inflation_radius = 0.22  # meters

        # IMPORTANT: coordinates clipped to valid range
        self.workspace = np.array([
            [0, 0],
            [522, 0],
            [800, 202],
            [1000, 204],
            [1000, 422],
            [860, 422],
            [859, 267],
            [0, 270]
        ])
        self.workspace_path = Path(self.workspace)

        # Objects
        self.cube_size = 5
        self.box_L = 24
        self.box_W = 16

        self.objects = [
            ("O", 133, 222, 0),
            ("B", 138, 16, 0),
            ("O", 320, 146, 0),
        ]

        self.timer = self.create_timer(1.0, self.publish_map)

    def publish_map(self):

        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()

        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = 0.0

        # =========================
        # 1) Workspace mask
        # =========================
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        points = np.vstack((xx.ravel(), yy.ravel())).T

        # KEY FIX: include boundary pixels
        inside = self.workspace_path.contains_points(points, radius=-1e-9)
        inside = inside.reshape((self.height, self.width))

        # Outside = obstacle
        binary = np.ones((self.height, self.width), dtype=np.uint8)
        binary[inside] = 0

        # =========================
        # 2) Add objects
        # =========================
        for typ, x, y, ang in self.objects:

            if typ == "O":
                r = self.cube_size // 2
                binary[y-r:y+r, x-r:x+r] = 1

            if typ == "B":
                binary[
                    y-self.box_W//2:y+self.box_W//2,
                    x-self.box_L//2:x+self.box_L//2
                ] = 1

        # =========================
        # 3) Inflation
        # =========================
        inflation_cells = int(self.inflation_radius / self.resolution)

        dist = distance_transform_edt(binary == 0)

        inflated = np.zeros_like(binary)

        # Hard obstacles
        inflated[binary == 1] = 100

        # Inflation band
        inflated[(dist <= inflation_cells) & (binary == 0)] = 50

        grid.data = inflated.flatten().tolist()

        self.pub.publish(grid)
        self.get_logger().info("Workspace map with full border inflation published")


def main():
    rclpy.init()
    node = WorkspaceMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
