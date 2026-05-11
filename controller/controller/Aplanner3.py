


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


class SimpleAStarPlanner(Node):

    def __init__(self):
        super().__init__('simple_a_star_planner')

        # ---------------- Parameters ----------------
        self.declare_parameter("inflation_radius", 0.22)
        self.declare_parameter("inflation_scale", 3.0)
        self.declare_parameter("b_spline_smooth", 0.001)
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
        self.goal = None
        self.goal_world = None

        self.get_logger().info(" TF-based A* Planner with B-spline smoothing Ready")

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

    # ---------------- Goal ----------------
    def goal_callback(self, msg):

        if self.map is None:
            self.get_logger().warn("Map not ready yet")
            return

        # 1️⃣ Get robot start pose from TF
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

        # 2️⃣ Get goal
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        yaw = self.quaternion_to_yaw(msg.pose.orientation)

        self.goal_world = (gx, gy, yaw)
        self.goal = self.world_to_grid(gx, gy)

        self.get_logger().info(
            f"Planning from ({sx:.2f},{sy:.2f}) "
            f"to ({gx:.2f},{gy:.2f})"
        )

        # 3️⃣ Plan
        self.try_plan(start)

    def goal_reached_cb(self, msg):

        if not msg.data:
            return

        self.get_logger().info("Goal reached — clearing path")

        # 1️⃣ Clear internal state
        self.goal = None
        self.goal_world = None

        # 2️⃣ Publish empty path to clear RViz & controller
        empty_path = Path()
        empty_path.header.frame_id = "map"
        empty_path.header.stamp = self.get_clock().now().to_msg()

        self.path_pub.publish(empty_path)
    # ---------------- Planning ----------------
    def try_plan(self, start):

        if self.map is None or self.goal is None:
            return

        path = self.a_star(start, self.goal)

        if path is None:
            self.get_logger().warn(" No path found")
            return

        path = self.smooth_path_bspline(path)
        self.publish_path(path)

    # ---------------- A* ----------------
    def a_star(self, start, goal):

        h, w = self.map.shape

        def heuristic(a, b):
            dx = abs(a[0]-b[0])
            dy = abs(a[1]-b[1])
            D = 1
            D2 = math.sqrt(3)
            return D*(dx+dy) + (D2-2*D)*min(dx, dy)

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




#  #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# import numpy as np
# import heapq
# import math
# from scipy.ndimage import distance_transform_edt
# from scipy.interpolate import splprep, splev

# from nav_msgs.msg import OccupancyGrid, Path
# from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Bool

# from tf2_ros import Buffer, TransformListener, TransformException

# class SimpleAStarPlanner(Node):
#     def __init__(self):
#         super().__init__('simple_a_star_planner')

#         # ---------------- Parameters ----------------
#         self.declare_parameter("inflation_radius", 0.22)
#         self.declare_parameter("inflation_scale", 3.0)
#         self.declare_parameter("b_spline_smooth", 0.01)
#         self.declare_parameter("goal_approach_points", 1)
#         self.declare_parameter("goal_approach_step", 0.001)

#         self.inflation_radius = self.get_parameter("inflation_radius").get_parameter_value().double_value
#         self.inflation_scale = self.get_parameter("inflation_scale").get_parameter_value().double_value
#         self.b_spline_smooth = self.get_parameter("b_spline_smooth").get_parameter_value().double_value
#         self.goal_approach_points = self.get_parameter("goal_approach_points").get_parameter_value().integer_value
#         self.goal_approach_step = self.get_parameter("goal_approach_step").get_parameter_value().double_value

#         # ---------------- Subscribers ----------------
#         self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
#         self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
#         self.create_subscription(Bool, '/goal_reached', self.goal_reached_cb, 10)

#         # ---------------- Publisher ----------------
#         self.path_pub = self.create_publisher(Path, '/planned_path', 10)

#         # ---------------- TF ----------------
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # ---------------- State ----------------
#         self.map = None
#         self.resolution = None
#         self.origin_x = None
#         self.origin_y = None
#         self.start = None
#         self.start_world = None
#         self.goal = None
#         self.goal_world = None

#         self.create_timer(0.1, self.update_start_from_tf)
#         self.get_logger().info(" A* Planner with B-spline smoothing")

#     # ---------------- Utilities ----------------
#     def quaternion_to_yaw(self, q):
#         siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
#         cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
#         return math.atan2(siny_cosp, cosy_cosp)

#     def yaw_to_quaternion(self, yaw):
#         qz = math.sin(yaw*0.5)
#         qw = math.cos(yaw*0.5)
#         return 0.0, 0.0, qz, qw

#     def world_to_grid(self, x, y):
#         gx = int((x - self.origin_x)/self.resolution)
#         gy = int((y - self.origin_y)/self.resolution)
#         return gx, gy

#     def grid_to_world(self, gx, gy):
#         x = gx*self.resolution + self.origin_x
#         y = gy*self.resolution + self.origin_y
#         return x, y

#     # ---------------- Map ----------------
#     def map_callback(self, msg):
#         w = msg.info.width
#         h = msg.info.height
#         self.resolution = msg.info.resolution
#         self.origin_x = msg.info.origin.position.x
#         self.origin_y = msg.info.origin.position.y
#         self.map = np.array(msg.data, dtype=np.int16).reshape((h,w))
#         self.apply_inflation()
#         self.get_logger().info("🗺 Map received")

#     def apply_inflation(self):
#         inflation_cells = int(self.inflation_radius/self.resolution)
#         dist = distance_transform_edt(self.map < 100)
#         mask = (self.map < 100) & (dist <= inflation_cells)
#         if inflation_cells > 0:
#             normalized = dist[mask]/float(inflation_cells)
#             soft_cost = 99*np.exp(-2.0*normalized)
#             self.map[mask] = soft_cost.astype(np.int16)

#     # ---------------- TF Start ----------------
#     def update_start_from_tf(self):
#         if self.map is None or (self.goal is None and self.start is not None):
#             return
#         try:
#             trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
#             x = trans.transform.translation.x
#             y = trans.transform.translation.y
#             self.start_world = (x, y)
#             self.start = self.world_to_grid(x, y)
#         except TransformException:
#             return

#     # ---------------- Goal ----------------
#     def goal_callback(self, msg):
#         if self.map is None:
#             self.get_logger().warn("Map not ready yet")
#             return
#         x = msg.pose.position.x
#         y = msg.pose.position.y
#         yaw = self.quaternion_to_yaw(msg.pose.orientation)
#         self.goal_world = (x, y, yaw)
#         self.goal = self.world_to_grid(x, y)
#         self.get_logger().info(f"Goal set: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
#         self.try_plan()

#     def goal_reached_cb(self, msg):
#         if not msg.data or self.goal is None:
#             return
#         self.get_logger().info("Goal reached")
#         self.start = self.goal
#         self.start_world = self.goal_world
#         self.goal = None
#         self.goal_world = None

#     # ---------------- Planning ----------------
#     def try_plan(self):
#         if self.map is None or self.start is None or self.goal is None:
#             return

#         approach_segment = self.create_goal_approach()
#         if not approach_segment:
#             return

#         first_approach = approach_segment[0]

#         approach_start_grid = self.world_to_grid(first_approach[0], first_approach[1])

#         path = self.a_star(self.start,self.goal)# approach_start_grid)
#         if path is None:
#             self.get_logger().warn(" No path found")
#             return

#         path = self.smooth_path_bspline(path)
#         full_path = path #+ approach_segment
#         self.publish_path(full_path)

#     # ---------------- A* ----------------
#     def a_star(self, start, goal):
#         h, w = self.map.shape

#         def heuristic(a, b):
#             dx = abs(a[0]-b[0])
#             dy = abs(a[1]-b[1])
#             D = 1
#             D2 = math.sqrt(3)
#             return D*(dx+dy) + (D2-2*D)*min(dx, dy)

#         open_set = []
#         heapq.heappush(open_set, (0, start))
#         came_from = {}
#         g_score = {start:0}
#         moves = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

#         while open_set:
#             _, current = heapq.heappop(open_set)
#             if current == goal:
#                 return self.reconstruct_path(came_from, current)

#             cx, cy = current
#             for dx, dy in moves:
#                 nx, ny = cx+dx, cy+dy
#                 if nx<0 or ny<0 or nx>=w or ny>=h:
#                     continue
#                 if self.map[ny, nx]>=100:
#                     continue
#                 if dx!=0 and dy!=0:
#                     if self.map[cy,nx]>=100 or self.map[ny,cx]>=100:
#                         continue
#                 move_cost = math.hypot(dx, dy)
#                 cell_cost = self.map[ny, nx]/100.0
#                 tentative_g = g_score[current] + move_cost*(1+self.inflation_scale*cell_cost)

#                 neighbor = (nx, ny)
#                 if neighbor not in g_score or tentative_g<g_score[neighbor]:
#                     g_score[neighbor] = tentative_g
#                     f = tentative_g + heuristic(neighbor, goal)
#                     heapq.heappush(open_set, (f, neighbor))
#                     came_from[neighbor] = current
#         return None

#     def reconstruct_path(self, came_from, current):
#         path = [current]
#         while current in came_from:
#             current = came_from[current]
#             path.append(current)
#         path.reverse()
#         return path

#     # ---------------- B-spline ----------------
#     def smooth_path_bspline(self, path):
#         if len(path) < 2:
#             # Not enough points to smooth
#             return [(x, y, 0.0) for x, y in path]

#         xs, ys = zip(*path)
#         xs_w, ys_w = zip(*[self.grid_to_world(x, y) for x, y in zip(xs, ys)])

#         # Only fit B-spline if enough points
#         if len(xs_w) >= 4:
#             try:
#                 tck, u = splprep([xs_w, ys_w], s=self.b_spline_smooth)
#                 u_new = np.linspace(0, 1, len(xs_w)*3)
#                 xs_smooth, ys_smooth = splev(u_new, tck)
#                 thetas = [math.atan2(ys_smooth[i+1]-ys_smooth[i], xs_smooth[i+1]-xs_smooth[i])
#                         for i in range(len(xs_smooth)-1)]
#                 thetas.append(thetas[-1])
#                 return list(zip(xs_smooth, ys_smooth, thetas))
#             except Exception as e:
#                 self.get_logger().warn(f"B-spline smoothing failed: {e}")

#         # Fallback: no smoothing
#         thetas = [math.atan2(ys_w[i+1]-ys_w[i], xs_w[i+1]-xs_w[i]) for i in range(len(xs_w)-1)]
#         thetas.append(thetas[-1])
#         return list(zip(xs_w, ys_w, thetas))

#     # ---------------- Goal Approach ----------------
#     def create_goal_approach(self):
#         if self.goal_world is None:
#             return []
#         gx, gy, goal_yaw = self.goal_world
#         segment = []
#         for i in reversed(range(self.goal_approach_points)):
#             alpha = (i+1)/self.goal_approach_points
#             x = gx - alpha*self.goal_approach_step*math.cos(goal_yaw)
#             y = gy - alpha*self.goal_approach_step*math.sin(goal_yaw)
#             segment.append((x, y, goal_yaw))
#         return segment

#     # ---------------- Publish ----------------
#     def publish_path(self, path):
#         path_msg = Path()
#         path_msg.header.frame_id = "map"
#         path_msg.header.stamp = self.get_clock().now().to_msg()
#         for x, y, yaw in path:
#             pose = PoseStamped()
#             pose.header.frame_id = "map"
#             pose.pose.position.x = x
#             pose.pose.position.y = y
#             qx,qy,qz,qw = self.yaw_to_quaternion(yaw)
#             pose.pose.orientation.x = qx
#             pose.pose.orientation.y = qy
#             pose.pose.orientation.z = qz
#             pose.pose.orientation.w = qw
#             path_msg.poses.append(pose)
#         self.path_pub.publish(path_msg)
#         self.get_logger().info(f" Path published: {len(path)} points")

# # ---------------- Main ----------------
# def main(args=None):
#     rclpy.init(args=args)
#     node = SimpleAStarPlanner()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()