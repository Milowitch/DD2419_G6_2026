#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import math
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener, TransformException


class SimpleAStarPlanner(Node):

    def __init__(self):
        super().__init__('simple_a_star_planner')

        # ---------------- Parameters ----------------
        self.declare_parameter("inflation_radius", 0.22)
        self.declare_parameter("inflation_scale", 3.0)
        self.declare_parameter("b_spline_smooth", 0.02)
        self.declare_parameter("goal_approach_points", 1)
        self.declare_parameter("goal_approach_step", 0.001)

        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.inflation_scale = self.get_parameter("inflation_scale").value
        self.b_spline_smooth = self.get_parameter("b_spline_smooth").value
        self.goal_approach_points = self.get_parameter("goal_approach_points").value
        self.goal_approach_step = self.get_parameter("goal_approach_step").value

        # ---------------- Subscribers ----------------
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(PoseArray, '/waypoints', self.waypoints_callback, 10)
        self.create_subscription(Bool, '/goal_reached', self.goal_reached_cb, 10)

        # ---------------- Publisher ----------------
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # ---------------- TF ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- State ----------------
        self.map = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.waypoints = []  # list of goals in grid coordinates

        self.get_logger().info(" TF-based A* Planner with multi-waypoint support Ready")

    # ---------------- Utilities ----------------
    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw*0.5)
        qw = math.cos(yaw*0.5)
        return 0.0, 0.0, qz, qw

    def world_to_grid(self, x, y):
        gx = int((x - self.origin_x)/self.resolution)
        gy = int((y - self.origin_y)/self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx*self.resolution + self.origin_x
        y = gy*self.resolution + self.origin_y
        return x, y

    # ---------------- Map ----------------
    def map_callback(self, msg):
        w = msg.info.width
        h = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.map = np.array(msg.data, dtype=np.int16).reshape((h, w))
        self.apply_inflation()
        self.get_logger().info(" Map received")

    def apply_inflation(self):
        inflation_cells = int(self.inflation_radius/self.resolution)
        dist = distance_transform_edt(self.map < 100)

        mask = (self.map < 100) & (dist <= inflation_cells)

        if inflation_cells > 0:
            normalized = dist[mask]/float(inflation_cells)
            soft_cost = 99*np.exp(-2.0*normalized)
            self.map[mask] = soft_cost.astype(np.int16)

    # ---------------- Goal / Waypoints ----------------
    def goal_callback(self, msg):
        # wrap single goal_pose as a one-element waypoint list
        self.waypoints = [self.world_to_grid(msg.pose.position.x, msg.pose.position.y)]
        self.plan_through_waypoints()

    def waypoints_callback(self, msg):
        if self.map is None:
            self.get_logger().warn("Map not ready yet")
            return

        # convert PoseArray to grid coordinates
        self.waypoints = [self.world_to_grid(p.position.x, p.position.y) for p in msg.poses]
        self.plan_through_waypoints()

    def goal_reached_cb(self, msg):
        if not msg.data:
            return

        self.get_logger().info("Goal reached — clearing path")
        self.waypoints = []

        empty_path = Path()
        empty_path.header.frame_id = "map"
        empty_path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(empty_path)

    # ---------------- Planner ----------------
    def plan_through_waypoints(self):
        if self.map is None or not self.waypoints:
            return

        # get robot current position
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )
            sx = trans.transform.translation.x
            sy = trans.transform.translation.y
            start = self.world_to_grid(sx, sy)
        except TransformException as ex:
            self.get_logger().warn(f"TF not ready: {ex}")
            return

        full_path = []
        current_start = start

        for goal in self.waypoints:
            segment = self.a_star(current_start, goal)
            if segment is None:
                self.get_logger().warn(f"No path found to {goal}")
                return

            if full_path and segment[0] == full_path[-1]:
                segment = segment[1:]  # avoid duplicate points

            full_path += segment
            current_start = goal

        full_path = self.smooth_path_bspline(full_path)
        self.publish_path(full_path)

    # ---------------- A* ----------------
    def a_star(self, start, goal):
        h, w = self.map.shape

        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            D = 1.0        # cost for straight
            D2 = math.sqrt(3)  # cost for diagonal
            return D * (dx + dy) + (D2 - 2*D) * min(dx, dy)

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        moves = [(1,0),(0,1),(-1,0),(0,-1),
                 (1,1),(1,-1),(-1,1),(-1,-1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            cx, cy = current

            for dx, dy in moves:
                nx, ny = cx+dx, cy+dy

                if nx<0 or ny<0 or nx>=w or ny>=h:
                    continue

                if self.map[ny, nx] >= 100:
                    continue

                if dx!=0 and dy!=0:
                    if self.map[cy, nx] >=100 or self.map[ny, cx]>=100:
                        continue

                move_cost = math.hypot(dx, dy)
                cell_cost = self.map[ny, nx]/100.0
                tentative_g = g_score[current] + move_cost*(1+self.inflation_scale*cell_cost)

                neighbor = (nx, ny)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ---------------- B-spline ----------------
    def smooth_path_bspline(self, path):
        xs, ys = zip(*path)
        xs_w, ys_w = zip(*[self.grid_to_world(x, y) for x, y in zip(xs, ys)])

        try:
            tck, u = splprep([xs_w, ys_w], s=self.b_spline_smooth)
            u_new = np.linspace(0, 1, len(xs_w)*3)
            xs_smooth, ys_smooth = splev(u_new, tck)

            thetas = []
            for i in range(len(xs_smooth)-1):
                theta = math.atan2(
                    ys_smooth[i+1]-ys_smooth[i],
                    xs_smooth[i+1]-xs_smooth[i]
                )
                thetas.append(theta)
            thetas.append(thetas[-1])

            return list(zip(xs_smooth, ys_smooth, thetas))

        except:
            thetas = []
            for i in range(len(xs_w)-1):
                theta = math.atan2(
                    ys_w[i+1]-ys_w[i],
                    xs_w[i+1]-xs_w[i]
                )
                thetas.append(theta)
            thetas.append(thetas[-1])

            return list(zip(xs_w, ys_w, thetas))

    # ---------------- Publish ----------------
    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y, yaw in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y

            qx,qy,qz,qw = self.yaw_to_quaternion(yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f" Path published: {len(path)} points")


# ---------------- Main ----------------
def main(args=None):
    rclpy.init(args=args)
    node = SimpleAStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()