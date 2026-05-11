#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import math

from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener, TransformException


class AStarPlanner(Node):

    def __init__(self):
        super().__init__('a_star_planner')

        # ===============================
        # Subscribers
        # ===============================
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10)

        self.create_subscription(
            Bool,
            '/goal_reached',
            self.goal_reached_cb,
            10)

        # ===============================
        # Publisher
        # ===============================
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # ===============================
        # TF Setup
        # ===============================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ===============================
        # Internal variables
        # ===============================
        self.map = None
        self.resolution = None
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.start = None
        self.goal = None
        self.last_goal_world = None

        # Inflation parameters
        self.inflation_radius = 0.22
        self.inflation_scale = 3.0

        # Timer to continuously update start pose
        self.create_timer(0.1, self.update_start_from_tf)

        self.get_logger().info("A* Planner Ready")

    # ===============================
    # Map callback
    # ===============================
    def map_callback(self, msg):
        w = msg.info.width
        h = msg.info.height

        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.map = np.array(msg.data).reshape((h, w)).astype(np.int8)

        self.apply_inflation()

    # ===============================
    # Goal callback
    # ===============================
    def goal_callback(self, msg):
        self.goal = self.world_to_grid(
            msg.pose.position.x,
            msg.pose.position.y
        )
        self.get_logger().info(f"Goal set: {self.goal}")
        self.try_plan()

    # ===============================
    # Goal reached callback
    # ===============================
    def goal_reached_cb(self, msg):
        if msg.data and self.goal is not None:
            self.get_logger().info(" Goal reached")

            self.last_goal_world = self.grid_to_world(*self.goal)
            self.start = self.goal

    # ===============================
    # TF-based start pose
    # ===============================
    def update_start_from_tf(self):
        if self.map is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            #print(str(x)+","+str(y))
            self.start = self.world_to_grid(x, y)

        except TransformException as e:
            self.get_logger().warn(f"TF failed: {str(e)}")

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
    # Inflation
    # ===============================
    def apply_inflation(self):
        if self.map is None:
            return

        h, w = self.map.shape
        inflation_cells = int(self.inflation_radius / self.resolution)

        dist = distance_transform_edt(self.map < 100)

        mask = (self.map < 100) & (dist <= inflation_cells)
        normalized = dist[mask] / float(inflation_cells)

        max_cost = 99
        cost_scaling = 2.0
        soft_cost = max_cost * np.exp(-cost_scaling * normalized)

        self.map[mask] = soft_cost.astype(np.int8)

    # ===============================
    # Try planning
    # ===============================
    def try_plan(self):
        if self.map is None or self.start is None or self.goal is None:
            return

        path = self.a_star(self.start, self.goal)

        if path is None:
            self.get_logger().warn(" No path found")
            return

        # Optional smoothing
        # path = self.smooth_path_bspline(path, smoothing=1.0)

        self.publish_path(path)

    # ===============================
    # A*
    # ===============================
    
    def a_star(self, start, goal):
        h, w = self.map.shape

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        moves = [
            (1,0,1), (-1,0,1), (0,1,1), (0,-1,1),
            (1,1,math.sqrt(2)), (1,-1,math.sqrt(2)),
            (-1,1,math.sqrt(2)), (-1,-1,math.sqrt(2))
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

                if self.map[ny, nx] >= 100:
                    continue

                if dx != 0 and dy != 0:
                    if self.map[current[1], nx] >= 100 or self.map[ny, current[0]] >= 100:
                        continue

                infl_cost = self.map[ny, nx] / 100.0
                tentative = g_score[current] + cost + self.inflation_scale * infl_cost

                if (nx, ny) not in g_score or tentative < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        return None

    # ===============================
    # Path reconstruction
    # ===============================
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ===============================
    # Optional smoothing
    # ===============================
    def smooth_path_bspline(self, path, smoothing=0):
        if len(path) < 3:
            return path

        path_np = np.array(path, dtype=np.float32)
        x = path_np[:,0]
        y = path_np[:,1]

        tck, _ = splprep([x, y], s=smoothing, k=3)
        u = np.linspace(0, 1, max(100, len(path)*10))
        x_s, y_s = splev(u, tck)

        return [(float(x_), float(y_)) for x_, y_ in zip(x_s, y_s)]

    # ===============================
    # Publish path
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
        self.get_logger().info(f" Path published: {len(grid_path)} points")


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
