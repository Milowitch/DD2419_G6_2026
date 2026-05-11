#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import math

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener, TransformException


class DifferentialHybridPlanner(Node):

    def __init__(self):
        super().__init__('differential_hybrid_planner')

        # Subscribers
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(Bool, '/goal_reached', self.goal_reached_cb, 10)

        # Publisher
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Map
        self.map = None
        self.resolution = None
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Robot state
        self.start = None
        self.goal = None

        # Planner parameters
        self.step_size = 0.1    # distance per motion step
        self.angle_step = math.pi / 8  # discretized rotations
        self.inflation_weight = 3.0

        self.create_timer(0.1, self.update_start_from_tf)

        self.get_logger().info("Differential Hybrid Planner Ready")

    # ===============================
    # Map
    # ===============================
    def map_callback(self, msg):
        w = msg.info.width
        h = msg.info.height

        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.map = np.array(msg.data).reshape((h, w)).astype(np.int16)

    # ===============================
    # Goal
    # ===============================
    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x,
                     msg.pose.position.y)

        self.get_logger().info(f"Goal received: {self.goal}")
        self.try_plan()

    def goal_reached_cb(self, msg):
        if msg.data:
            self.get_logger().info("Goal reached")

    # ===============================
    # TF → Start pose
    # ===============================
    def update_start_from_tf(self):
        if self.map is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            theta = self.yaw_from_quaternion(trans.transform.rotation)

            self.start = (x, y, theta)

        except TransformException:
            pass

    # ===============================
    # Quaternion → yaw
    # ===============================
    def yaw_from_quaternion(self, q):
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    # ===============================
    # World → Grid
    # ===============================
    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    # ===============================
    # Plan trigger
    # ===============================
    def try_plan(self):
        if self.map is None or self.start is None or self.goal is None:
            return

        path = self.hybrid_a_star(self.start, self.goal)

        if path is None:
            self.get_logger().warn("No path found")
            return

        self.publish_path(path)

    # ===============================
    # Hybrid A* for differential drive
    # ===============================
    def hybrid_a_star(self, start, goal):

        def heuristic(x, y):
            return math.hypot(x - goal[0], y - goal[1])

        def theta_to_bin(theta):
            return round((theta + 2 * math.pi) % (2 * math.pi), 2)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {}
        start_key = (round(start[0], 2), round(start[1], 2), theta_to_bin(start[2]))
        g_score[start_key] = 0
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            x, y, theta = current

            if heuristic(x, y) < 0.2:
                return self.reconstruct_path(came_from, current)

            key = (round(x, 2), round(y, 2), theta_to_bin(theta))
            if key in visited:
                continue
            visited.add(key)

            # 3 possible moves: forward, rotate left, rotate right
            motions = [
                (self.step_size, 0.0),          # forward
                (0.0, self.angle_step),         # rotate left
                (0.0, -self.angle_step)         # rotate right
            ]

            for dx, dtheta in motions:
                nx = x + dx * math.cos(theta)
                ny = y + dx * math.sin(theta)
                ntheta = theta + dtheta

                if not self.is_motion_valid(nx, ny):
                    continue

                new_key = (round(nx, 2), round(ny, 2), theta_to_bin(ntheta))
                gx, gy = self.world_to_grid(nx, ny)
                infl_cost = self.map[gy, gx] / 100.0

                tentative = g_score[key] + dx + self.inflation_weight * infl_cost

                if new_key not in g_score or tentative < g_score[new_key]:
                    g_score[new_key] = tentative
                    f = tentative + heuristic(nx, ny)
                    heapq.heappush(open_set, (f, (nx, ny, ntheta)))
                    came_from[(nx, ny, ntheta)] = current

        return None

    # ===============================
    # Collision check
    # ===============================
    def is_motion_valid(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if gx < 0 or gy < 0 or gx >= self.map.shape[1] or gy >= self.map.shape[0]:
            return False

        cell = self.map[gy, gx]
        if cell >= 100 or cell > 70:   # hard obstacle + high inflation
            return False
        return True

    # ===============================
    # Reconstruct path
    # ===============================
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ===============================
    # Publish path
    # ===============================
    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y, theta in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Differential Hybrid path published: {len(path)} points")


def main(args=None):
    rclpy.init(args=args)
    node = DifferentialHybridPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()