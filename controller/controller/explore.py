#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from scipy.ndimage import gaussian_filter

UNKNOWN = -1
FREE = 0
OBSTACLE = 100
VISITED = 50

SAFE_LIMIT = 10

class PheromoneExplorer(Node):

    def __init__(self):
        super().__init__("pheromone_explorer")

        # Subscribers
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.create_subscription(Bool, "/goal_reached", self.goal_reached_callback, 10)
        self.create_subscription(Bool, "/goal_failed", self.goal_failed_callback, 10)
        self.create_subscription(Bool, "/exploration", self.exploration_callback, 10)  

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.explore_pub = self.create_publisher(OccupancyGrid, "/exploration_map", 10)
        self.pher_pub = self.create_publisher(OccupancyGrid, "/pheromone_map", 10)

        self.timer = self.create_timer(0.8, self.update)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Maps
        self.workspace_map = None
        self.exploration_map = None
        self.pheromone = None

        # Map info
        self.resolution = None
        self.width = None
        self.height = None
        self.origin = None

        # Robot pose
        self.robot_x = 0
        self.robot_y = 0
        self.robot_yaw = 0

        # Sensor
        self.sensor_range = 1.0

        self.deposit = 100000.0
        self.evap = 0.999

        # Frontier weights
        self.w_info = 2.1
        self.w_pher = 5.1
        self.w_dist = 0.05
        self.w_heading = 2.5

        self.waiting_goal = False
        self.goal_grid = None

        # ✅ Exploration control
        self.exploration_enabled = False

    # ---------------- CONTROL ----------------
    def exploration_callback(self, msg):
        self.exploration_enabled = msg.data
        self.get_logger().info(f"Exploration: {'ON' if msg.data else 'OFF'}")

        # 🚫 Stop current goal immediately
        if not msg.data:
            self.waiting_goal = False

    # ---------------- GOAL STATUS ----------------
    def goal_failed_callback(self, msg):
        if msg.data:
            self.waiting_goal = False

    def goal_reached_callback(self, msg):
        if msg.data:
            self.waiting_goal = False
            if self.goal_grid is not None:
                gx, gy = self.goal_grid
                self.exploration_map[gy, gx] = VISITED

    # ---------------- MAP ----------------
    def map_callback(self, msg):
        self.resolution = msg.info.resolution
        self.width = msg.info.width
        self.height = msg.info.height
        self.origin = msg.info.origin

        self.workspace_map = np.array(msg.data).reshape(self.height, self.width)

        if self.exploration_map is None:
            self.exploration_map = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)
            self.pheromone = np.zeros((self.height, self.width))
            self.get_logger().info("Explorer initialized")

    # ---------------- POSE ----------------
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                             1 - 2*(q.y*q.y + q.z*q.z))
            return x, y, yaw
        except:
            return None

    def world_to_grid(self, x, y):
        gx = int((x - self.origin.position.x) / self.resolution)
        gy = int((y - self.origin.position.y) / self.resolution)

        gx = max(0, min(self.width - 1, gx))
        gy = max(0, min(self.height - 1, gy))
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.origin.position.x
        y = gy * self.resolution + self.origin.position.y
        return x, y

    # ---------------- SENSING ----------------
    def sense(self, gx, gy, yaw):
        max_cells = int(self.sensor_range / self.resolution)
        fov = math.radians(90)

        for dx in range(-max_cells, max_cells + 1):
            for dy in range(-max_cells, max_cells + 1):

                wx = gx + dx
                wy = gy + dy

                if not (0 <= wx < self.width and 0 <= wy < self.height):
                    continue

                dist = math.hypot(dx, dy)
                if dist == 0 or dist > max_cells:
                    continue

                angle = math.atan2(dy, dx)
                dtheta = abs(math.atan2(math.sin(angle - yaw), math.cos(angle - yaw)))

                if dtheta <= fov / 2:
                    val = self.workspace_map[wy, wx]
                    if val >= SAFE_LIMIT:
                        self.exploration_map[wy, wx] = OBSTACLE
                    elif self.exploration_map[wy, wx] != VISITED:
                        self.exploration_map[wy, wx] = FREE

    # ---------------- FRONTIERS ----------------
    def detect_frontiers(self):
        frontiers = []
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.exploration_map[y, x] != FREE:
                    continue
                if self.workspace_map[y, x] >= SAFE_LIMIT:
                    continue
                if np.any(self.exploration_map[y-1:y+2, x-1:x+2] == UNKNOWN):
                    frontiers.append((x, y))
        return frontiers

    def information_gain(self, x, y):
        gain = 0
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.exploration_map[ny, nx] == UNKNOWN:
                        gain += 1
        return gain

    # ---------------- PHEROMONE ----------------
    def update_pheromone(self, gx, gy):

        self.pheromone *= self.evap

        gx = max(0, min(self.width - 1, gx))
        gy = max(0, min(self.height - 1, gy))

        self.pheromone[gy, gx] += self.deposit

        self.pheromone[:] = gaussian_filter(self.pheromone, sigma=2.0)
        self.pheromone[self.pheromone > 80] = 80

    # ---------------- PLANNING ----------------
    def choose_frontier(self, gx, gy):
        frontiers = self.detect_frontiers()
        if not frontiers:
            return None

        best = None
        best_score = -1e9
        max_pher = np.max(self.pheromone) + 1e-6

        for fx, fy in frontiers:

            if self.exploration_map[fy, fx] == VISITED:
                continue

            gain = self.information_gain(fx, fy)
            pher = self.pheromone[fy, fx] / max_pher
            dist = math.hypot(fx - gx, fy - gy)

            goal_dir = math.atan2(fy - gy, fx - gx)
            dtheta = abs(math.atan2(math.sin(goal_dir - self.robot_yaw),
                                    math.cos(goal_dir - self.robot_yaw)))

            score = self.w_info*gain - self.w_pher*pher - self.w_dist*dist - self.w_heading*dtheta

            if score > best_score:
                best_score = score
                best = (fx, fy)

        return best

    def publish_goal(self, gx, gy):
        x, y = self.grid_to_world(gx, gy)

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.x = 1.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        self.waiting_goal = True
        self.goal_grid = (gx, gy)

        self.get_logger().info(f"Goal {x:.2f} {y:.2f}")

    # ---------------- OUTPUT ----------------
    def publish_maps(self):

        m = OccupancyGrid()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = self.resolution
        m.info.width = self.width
        m.info.height = self.height
        m.info.origin = self.origin
        m.data = self.exploration_map.flatten().tolist()
        self.explore_pub.publish(m)

        p = OccupancyGrid()
        p.header.frame_id = "map"
        p.header.stamp = self.get_clock().now().to_msg()
        p.info.resolution = self.resolution
        p.info.width = self.width
        p.info.height = self.height
        p.info.origin = self.origin

        scaled = np.clip(self.pheromone, 0, 100)
        p.data = scaled.astype(np.int8).flatten().tolist()
        self.pher_pub.publish(p)

    # ---------------- MAIN LOOP ----------------
    def update(self):



        if self.workspace_map is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose
        self.robot_x, self.robot_y, self.robot_yaw = x, y, yaw

        gx, gy = self.world_to_grid(x, y)

        self.sense(gx, gy, yaw)
        self.update_pheromone(gx, gy)
        self.publish_maps()
        if not self.exploration_enabled:
            return
        if self.waiting_goal:
            return

        goal = self.choose_frontier(gx, gy)
        if goal:
            self.publish_goal(goal[0], goal[1])


def main():
    rclpy.init()
    node = PheromoneExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()