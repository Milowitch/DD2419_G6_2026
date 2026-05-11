#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DepthToOccupancyGrid(Node):
    def __init__(self):
        super().__init__('depth_to_occupancy_grid')

        # QoS for point clouds
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscriber
        self.create_subscription(
            PointCloud2,
            '/realsense/depth/color/points',
            self.cloud_callback,
            qos_profile
        )

        # Publisher
        self._grid_pub = self.create_publisher(OccupancyGrid, '/perception/occupancy_grid', 10)

        # Grid parameters
        self.resolution = 0.05  # meters per cell
        self.width = 200        # number of cells (10m / 0.05m)
        self.height = 200
        self.origin_x = -5.0    # grid origin in world frame
        self.origin_y = -5.0

        # Filtering parameters
        self.min_z = 0.15       # meters
        self.max_z = 2.5
        self.max_distance = 5.0 # XY distance from sensor

        self.get_logger().info("Depth to OccupancyGrid node started!")

    def cloud_callback(self, msg: PointCloud2):
        # Parse PointCloud2
        pt_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pt_data), dtype=np.float32)
        if points.size == 0:
            return

        # Apply filtering
        distances = np.linalg.norm(points[:, :2], axis=1)
        mask = (points[:, 2] >= self.min_z) & (points[:, 2] <= self.max_z) & (distances <= self.max_distance)
        filtered_points = points[mask]

        if filtered_points.size == 0:
            return

        # Create and publish OccupancyGrid
        self.create_occupancy_grid(filtered_points, msg.header)

    def create_occupancy_grid(self, coords, header):
        grid = np.full((self.height, self.width), -1, dtype=np.int8)  # unknown

        for point in coords:
            x, y, z = point

            grid_x = int((x - self.origin_x) / self.resolution)
            grid_y = int((y - self.origin_y) / self.resolution)

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                grid[grid_y, grid_x] = 100  # occupied

        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()

        self._grid_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToOccupancyGrid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()