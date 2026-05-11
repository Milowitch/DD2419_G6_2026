#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
import numpy as np
from scipy.ndimage import distance_transform_edt
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

def points_in_polygon(x, y, poly):
    """Ray-casting algorithm to check if points (x,y) are inside polygon poly"""
    n = len(poly)
    inside = np.zeros_like(x, dtype=bool)
    px = poly[:, 0]
    py = poly[:, 1]
    for i in range(n):
        j = (i - 1) % n
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > y) != (yj > y)) & \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        inside ^= intersect
    return inside

def rasterize_line(x0, y0, x1, y1):
    """Bresenham-like integer line rasterization"""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((y, x))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((y, x))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((y1, x1))
    return zip(*points)

class WorkspaceMap(Node):
    def __init__(self):
        super().__init__('workspace_map')

        # --- QoS for markers and map ---
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.create_subscription(MarkerArray, '/perception/marker', self.marker_callback, map_qos)
        self.create_subscription(MarkerArray, '/perception/box', self.marker_callback, map_qos)

        # Map parameters
        self.resolution = 0.01  # meters per cell
        self.width = 1000       # columns
        self.height = 423       # rows
        self.inflation_radius = 0.22  # meters

        # Workspace polygon in cm
        self.workspace = np.array([
            [0, 0],
            [522, 0],
            [800, 202],
            [999, 204],
            [999, 422],
            [860, 422],
            [859, 267],
            [0, 270]
        ], dtype=int)

        # Objects storage: key=id, value=dict(type,x_cm,y_cm)
        self.objects = {}

        # Preload objects (id, type, x_cm, y_cm)
        initial_objects = [
            (0, "B", 138, 16)
            # (0, "O", 133, 222),
            # (1, "B", 138, 16),
            # (2, "O", 320, 146)
        ]
        for mid, typ, x, y in initial_objects:
            self.objects[mid] = {"type": typ, "x": x, "y": y}

        # Object sizes in cm
        self.cube_size = 5
        self.box_L = 24
        self.box_W = 16

        # Timer to publish map at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_map)

    def marker_callback(self, msg: MarkerArray):
        """Add or delete objects based on MarkerArray"""
        for m in msg.markers:
            if m.action == m.DELETE:
                if m.id in self.objects:
                    del self.objects[m.id]
                    self.get_logger().info(f"Deleted object {m.id}")
            else:
                typ = "O" if m.scale.x <= 0.1 else "B"
                x_cm = int(m.pose.position.x * 100)
                y_cm = int(m.pose.position.y * 100)
                self.objects[m.id] = {"type": typ, "x": x_cm, "y": y_cm}
                self.get_logger().info(f"Added/Updated object {m.id}: {typ} at {x_cm},{y_cm}")

    def publish_map(self):
        """Generate occupancy grid"""
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = 0.0

        # Base occupancy: workspace polygon
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        inside = points_in_polygon(xx, yy, self.workspace)
        binary = np.ones((self.height, self.width), dtype=np.uint8)  # 1=occupied
        binary[inside] = 0  # free space

        # Rasterize workspace edges
        for i in range(len(self.workspace)):
            start = self.workspace[i]
            end = self.workspace[(i + 1) % len(self.workspace)]
            rr, cc = rasterize_line(start[0], start[1], end[0], end[1])
            rr = np.clip(rr, 0, self.height - 1)
            cc = np.clip(cc, 0, self.width - 1)
            binary[rr, cc] = 1

        # Add objects
        for obj in self.objects.values():
            col = np.clip(obj["x"], 0, self.width - 1)
            row = np.clip(obj["y"], 0, self.height - 1)
            if obj["type"] == "O":
                r = self.cube_size // 2
                rmin = max(0, row - r)
                rmax = min(self.height, row + r + 1)
                cmin = max(0, col - r)
                cmax = min(self.width, col + r + 1)
                binary[rmin:rmax, cmin:cmax] = 1
            elif obj["type"] == "B":
                rhalf = self.box_W // 2
                chalf = self.box_L // 2
                rmin = max(0, row - rhalf)
                rmax = min(self.height, row + rhalf + 1)
                cmin = max(0, col - chalf)
                cmax = min(self.width, col + chalf + 1)
                binary[rmin:rmax, cmin:cmax] = 1

        # Inflate obstacles (stable)
        obstacles = (binary == 1)
        dist = distance_transform_edt(~obstacles)
        inflation_cells = int(self.inflation_radius / self.resolution)

        inflated = np.zeros_like(binary, dtype=np.int8)
        inflated[obstacles] = 100

        mask = (dist > 0) & (dist <= inflation_cells)
        normalized = dist[mask] / float(inflation_cells)
        inflated[mask] = (99 * np.exp(-2.0 * normalized)).astype(np.int8)

        grid.data = inflated.flatten().tolist()
        self.pub.publish(grid)
        self.get_logger().info(f"Map published with {len(self.objects)} objects")

def main():
    rclpy.init()
    node = WorkspaceMap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()