#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import math

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


class AStarPlanner(Node):

    def __init__(self):
        super().__init__('a_star_planner')

        # Subscribers
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.start_callback,
            10)

        self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10)

        # Publisher
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        self.map = None
        self.resolution = None
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.start = None
        self.goal = None

        self.get_logger().info("Inflation-aware A* Planner Ready")

    # ===============================
    # Map callback
    # ===============================
    def map_callback(self, msg):
        w = msg.info.width
        h = msg.info.height

        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.map = np.array(msg.data).reshape((h, w))
        self.get_logger().info("Map received")

    # ===============================
    # Start pose from RViz
    # ===============================
    def start_callback(self, msg):
        self.start = self.world_to_grid(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        self.get_logger().info(f"Start set: {self.start}")
        self.try_plan()

    # ===============================
    # Goal pose from RViz
    # ===============================
    def goal_callback(self, msg):
        self.goal = self.world_to_grid(
            msg.pose.position.x,
            msg.pose.position.y
        )
        self.get_logger().info(f"Goal set: {self.goal}")
        self.try_plan()

    # ===============================
    # Coordinate transforms
    # ===============================
    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin_x
        y = gy * self.resolution + self.origin_y
        return x, y

    # ===============================
    def try_plan(self):
        if self.map is None:
            return
        if self.start is None or self.goal is None:
            return

        path = self.a_star(self.start, self.goal)

        if path is None:
            self.get_logger().warn("No path found")
            return

        self.publish_path(path)

    # ===============================
    # A* WITH INFLATION BLOCKING
    # ===============================
    def a_star(self, start, goal):

        h, w = self.map.shape

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        # 8-connected movement
        moves = [
            (1,0,1), (-1,0,1), (0,1,1), (0,-1,1),
            (1,1,math.sqrt(2)),
            (1,-1,math.sqrt(2)),
            (-1,1,math.sqrt(2)),
            (-1,-1,math.sqrt(2))
        ]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy, cost in moves:
                nx = current[0] + dx
                ny = current[1] + dy

                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue

                # BLOCK inflation + obstacles
                if self.map[ny, nx] >= 50:
                    continue

                # Prevent diagonal cutting through inflation
                if dx != 0 and dy != 0:
                    if self.map[current[1], nx] >= 50:
                        continue
                    if self.map[ny, current[0]] >= 50:
                        continue

                tentative = g_score[current] + cost

                if (nx, ny) not in g_score or tentative < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None

    # ===============================
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ===============================
    def publish_path(self, grid_path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for gx, gy in grid_path:
            x, y = self.grid_to_world(gx, gy)

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(grid_path)} points")


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
