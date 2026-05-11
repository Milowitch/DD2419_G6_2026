#!/usr/bin/env python3
#ros2 topic pub /cube/approach  std_msgs/msg/Bool "{data: True}"

#!/usr/bin/env python3
#ros2 topic pub /cube/approach  std_msgs/msg/Bool "{data: True}"

#!/usr/bin/env python3
#ros2 topic pub /cube/approach  std_msgs/msg/Bool "{data: True}"
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Twist
from std_msgs.msg import Bool
from collections import deque
import math
from std_msgs.msg import Float64MultiArray
import csv
import os
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
import time
# ===========================
# --- EXPLORATION PARAMETERS ---
# ===========================
workspace2 = [
    [50, 50],
    [522, 50],
    [800, 202],
    [980, 204],
    [980, 422],
    [880, 422],
    [880, 267],
    [50, 230]
]


class RobotTaskManager(Node):

    def __init__(self):
        super().__init__('robot_task_manager')

        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Subscribers
        self.cube_ring = {}   
        self.box_ring = {}   
        self.csv_load_time = None
        self.wait_after_csv_sec = 5.0   # adjust (1–3 sec typical)
        self.create_subscription(Float64MultiArray, '/ring_pos/cube', self.cube_ring_cb, 10)
        self.create_subscription(Float64MultiArray, '/ring_pos/box', self.box_ring_cb, 10)
        self.create_subscription(MarkerArray, '/perception/markersT', self.cube_cb, qos)
        self.create_subscription(MarkerArray, '/perception/box', self.box_cb, qos)
        self.create_subscription(Bool, '/goal_reached', self.goal_reached_cb, 10)
        self.create_subscription(Bool, '/grip/finished', self.grip_finished_cb, 10)
        self.create_subscription(Bool, '/cube/approach', self.cube_approach_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.current_pose = None
        self.goal_abort = False
        self.create_subscription(Bool, '/goal_abort', self.goal_abort_cb, 10)

        # Publishers
        # --- CSV Perception Publishers ---
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/markersT', qos)
        self.marker_pub_per = self.create_publisher(MarkerArray, '/perception/markers/all', qos)
        self.box_pub = self.create_publisher(MarkerArray, '/perception/box', qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/set_pose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.grip_ready_pub = self.create_publisher(Bool, '/grip/ready', 10)
        self.grip_grasp_pub = self.create_publisher(Bool, '/grip/grasp', 10)
        self.grip_release_pub = self.create_publisher(Bool, '/grip/release', 10)
        self.cube_go_pub = self.create_publisher(Bool, '/cube/go', 10)
        self.exploration_pub = self.create_publisher(Bool, '/exploration', 10)
        self.waypoint_pub = self.create_publisher(PoseArray, '/waypoints', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Robot State
        self.cubes_active = {}      # cubes detected but not collected
        self.collected_cubes = set()
        self.boxes = {}             # box positions
        self.home_pose = (5.22, 2.21, 0.0)
        self.last_robot_pose = self.home_pose

        self.task_queue = deque()
        self.current_task = None
        self.current_cube_id = None

        self.goal_reached = False
        self.grip_done = False
        self.goal_sent = False
        self.cube_approached = False
        self.delivery_in_progress = False
        self.from_exploration = False
        # Backward motion
        self.backward_active = False
        self.backward_end_time = 0.0
        self.backward_msg = Twist()

        # Exploration
        self.exploration_mode = False
        self.first_cube=False
        # Timer
        self.create_timer(0.1, self.task_loop)
        self.create_timer(0.5, self.publish_exploration)
        self.active_cube_targets = {}
        self.active_box_targets = {}
        self.csv_loaded = False
    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.current_pose = (x, y, yaw)
    def goal_abort_cb(self, msg):
        self.goal_abort = msg.data
    def load_from_csv(self):
        path = os.path.expanduser("~/dd2419_ws/task/map.csv")

        if not os.path.exists(path):
            self.get_logger().error(f"CSV not found: {path}")
            return

        cube_markers = MarkerArray()
        box_markers = MarkerArray()

        marker_id = 0

        with open(path, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                t = row["Type"].strip()
                x = float(row["x"]) / 100.0
                y = float(row["y"]) / 100.0
                angle = float(row["angle"])

                # --- INITIAL POSE ---
                if t == "S":
                    pose = PoseStamped()
                    pose.header.frame_id = "map"

                    pose.pose.position.x = x
                    pose.pose.position.y = y

                    yaw = np.deg2rad(angle)
                    pose.pose.orientation.z = yaw

                    self.pose_pub.publish(pose)
                    continue

                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = marker_id
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.orientation.w = 1.0
                marker.color.a = 0.95

                if t == "O":  # cube
                    marker.type = Marker.CUBE
                    marker.scale.x = marker.scale.y = marker.scale.z = 0.05
                    marker.color.r = 1.0
                    cube_markers.markers.append(marker)

                    # 🔥 OPTIONAL (strongly recommended)
                    self.cubes_active[marker_id] = (x, y)

                elif t == "B":  # box
                    marker.type = Marker.CUBE
                    marker.scale.x = 0.24
                    marker.scale.y = 0.16
                    marker.scale.z = 0.05
                    marker.color.g = 1.0
                    box_markers.markers.append(marker)

                    # 🔥 OPTIONAL (strongly recommended)
                    self.boxes[marker_id] = (x, y)

                marker_id += 1

        # publish markers
        self.marker_pub.publish(cube_markers)
        self.marker_pub_per.publish(cube_markers)
        self.box_pub.publish(box_markers)
        self.get_logger().info("✅ CSV perception loaded")
        self.csv_load_time = self.get_clock().now().nanoseconds / 1e9
        if self.csv_load_time is not None:
            now = self.get_clock().now().nanoseconds / 1e9
            elapsed = now - self.csv_load_time

            if elapsed < self.wait_after_csv_sec:
                self.get_logger().warn(
                    f"⏳ Waiting {self.wait_after_csv_sec - elapsed:.2f}s for ring data..."
                )
                return


    def publish_exploration(self):
        msg = Bool()
        msg.data = self.exploration_mode
        self.exploration_pub.publish(msg)
    # =======================
    # --- Callbacks ---
    # =======================

    def box_cb(self, msg):
        for m in msg.markers:
            self.boxes[m.id] = (m.pose.position.x, m.pose.position.y)

    def goal_reached_cb(self, msg):
        self.goal_reached = msg.data
        if msg.data and self.current_task and self.current_task[0] in ["move", "waypoint"]:
            self.current_task = None

    def grip_finished_cb(self, msg):
        self.grip_done = msg.data

    def cube_approach_cb(self, msg):
        if msg.data:
            self.cube_approached = True

    def cube_cb(self, msg):
        self.first_cube=True
        new_cube = False
        for m in msg.markers:
            cube_pose = (m.pose.position.x, m.pose.position.y)
            if m.id in self.collected_cubes:
                continue
            if m.id not in self.cubes_active:
                self.cubes_active[m.id] = cube_pose
                new_cube = True

        if new_cube:
            self.get_logger().info("🆕 New cube detected")

            # interrupt ONLY if idle OR exploring
            if not self.task_queue and not self.current_task:
                self.get_logger().info("🛑 Switching to cube task")

                self.exploration_mode = False

                msg_bool = Bool()
                msg_bool.data = False
                self.exploration_pub.publish(msg_bool)

                self.replan(self.last_robot_pose)
    # =======================
    # --- Helpers ---
    # =======================
    def distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def send_goal(self, pose):
        # pose = (x, y, yaw) OR (x, y)
        if len(pose) == 3:
            x, y, yaw = pose
        else:
            x, y = pose
            yaw = 0.0

        self.last_robot_pose = (x, y, yaw)

        msg = PoseStamped()
        msg.header.frame_id = "map"

        msg.pose.position.x = x
        msg.pose.position.y = y

        # yaw → quaternion
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f"➡ MOVE → ({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
        )

        self.goal_pub.publish(msg)
        # self.last_robot_pose = pose
        # msg = PoseStamped()
        # msg.header.frame_id = "map"
        # msg.pose.position.x = pose[0]
        # msg.pose.position.y = pose[1]
        # msg.pose.orientation.w = 1.0
        # self.get_logger().info(f"➡ MOVE → ({pose[0]:.2f}, {pose[1]:.2f})")
        # self.goal_pub.publish(msg)

    def send_waypoints(self, points):
        msg = PoseArray()
        msg.header.frame_id = "map"
        for p in points:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.get_logger().info(f"➡ WAYPOINT PATH → {len(points)} points")
        self.waypoint_pub.publish(msg)

    def send_bool(self, pub, name):
        msg = Bool()
        msg.data = True
        self.get_logger().info(f"⚙ ACTION → {name}")
        pub.publish(msg)

    # =======================
    # --- Cube Collection Planner ---
    # =======================
    def cube_ring_cb(self, msg):
        data = msg.data

        new_dict = {}
        if len(data) % 4 != 0:
            self.get_logger().warn("Invalid cube ring data (expect 4 values)")
            return

        for i in range(0, len(data), 4):
            obj_id = int(data[i])
            x = data[i + 1]
            y = data[i + 2]
            yaw = data[i + 3]

            new_dict[obj_id] = (x, y, yaw)
        self.cube_ring = new_dict

        # 🔥 update active map in real-time too
        self.active_cube_targets = dict(new_dict)
    def box_ring_cb(self, msg):
        data = msg.data
        new_dict = {}
        if len(data) % 4 != 0:
            self.get_logger().warn("Invalid box ring data (expect 4 values)")
            return

        for i in range(0, len(data), 4):
            obj_id = int(data[i])
            x = data[i + 1]
            y = data[i + 2]
            yaw = data[i + 3]

            new_dict[obj_id] = (x, y, yaw)

        self.box_ring = new_dict

        # 🔥 always keep latest box target snapshot
        self.active_box_targets = dict(new_dict)
    def replan(self, start_pose=None):
        if not self.boxes or not self.cubes_active:
            return
        
        if not self.cube_ring:
                    self.get_logger().warn("⏳ Waiting for box ring data...")
                    return
        self.get_logger().info("♻ Replanning cube collection...")
        self.task_queue.clear()
        self.current_task = None
        self.delivery_in_progress = True
        # BEFORE adding cube tasks
        if self.from_exploration:
            self.task_queue.append(("wait_abort_or_timeout", 2.0))
            self.from_exploration=False
        if self.current_pose is not None:
            robot_pos = self.current_pose
        elif start_pose:
            robot_pos = start_pose
        else:
            robot_pos = self.home_pose
        remaining = dict(self.cubes_active)
        bx, by = list(self.boxes.values())[0]

        while remaining:
            cube_id, cube = min(
                remaining.items(),
                key=lambda item: self.distance(robot_pos, item[1])
            )

            # ===========================
            # USE RING APPROACH (NEW)
            # ===========================
            approach = self.cube_ring.get(cube_id)

            if approach is None:
                self.get_logger().warn(f"No ring pose for cube {cube_id}, fallback used")

                approach = self.cubes_active.get(cube_id)
                if approach is None:
                    self.get_logger().warn(f"Cube {cube_id} missing completely → skipping")
                    remaining.pop(cube_id)
                    continue
            box_id = None
            if self.box_ring:
                box_id = next(iter(self.box_ring.keys()))


            approach_box = None

            if self.box_ring:
                box_id, approach_box = min(
                    self.box_ring.items(),
                    key=lambda item: self.distance(robot_pos, item[1])
                )
            self.task_queue.extend([
                ("move_cube", cube_id),
                ("ready", None),
                ("cubego", None),
                ("grasp", cube_id),
                ("backward", 1.0),
                ("move_box", None),
                ("release", cube_id),
                ("backward", 1.0)
            ])
            robot_pos = approach_box
            remaining.pop(cube_id)

    # =======================
    # --- Main Task Loop ---
    # =======================
    def task_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if not self.first_cube:
                return 
        if not self.csv_loaded:

            #self.load_from_csv()
            self.csv_loaded = True
            time.sleep(3)
            if self.cubes_active and self.boxes:
                self.get_logger().info("🚀 Initial cubes detected → planning")

                self.exploration_mode = False
                msg_bool = Bool()
                msg_bool.data = False
                self.exploration_pub.publish(msg_bool)

                self.replan(self.home_pose)
        # --- handle backward motion ---
        if self.backward_active:
            if now < self.backward_end_time:
                self.cmd_vel_pub.publish(self.backward_msg)
                return
            else:
                self.cmd_vel_pub.publish(Twist())
                self.backward_active = False
                self.get_logger().info("↩ BACKWARD DONE")
                self.current_task = None
                self.goal_sent = False

        # --- cube tasks have priority ---
        if self.task_queue or self.current_task:
            if self.current_task is None and self.task_queue:
                self.current_task = self.task_queue.popleft()
                self.goal_reached = False
                self.grip_done = False
                self.goal_sent = False

            action, data = self.current_task

            if action == "move":
                if not self.goal_sent:
                    self.send_goal(data)
                    self.goal_sent = True
                elif self.goal_reached:
                    self.current_task = None
            elif action == "wait_abort_or_timeout":
                timeout = data

                if not self.goal_sent:
                    self.wait_start_time = now
                    self.goal_abort = False
                    self.goal_sent = True
                    self.get_logger().info(f"⏳ Waiting for /goal_abort or {timeout}s")

                elif self.goal_abort:
                    self.get_logger().info("🛑 Goal aborted → continuing")
                    self.current_task = None

                elif now - self.wait_start_time >= timeout:
                    self.get_logger().info("⏱ Timeout reached → continuing")
                    self.current_task = None
            elif action == "waypoint":
                if not self.goal_sent:
                    self.send_waypoints(data)
                    self.goal_sent = True
                elif self.goal_reached:
                    self.current_task = None

            elif action == "ready":
                self.send_bool(self.grip_ready_pub, "GRIP READY")
                self.current_task = None
            elif action == "move_cube":
                cube_id = data

                pose = self.active_cube_targets.get(cube_id)

                if pose is None:
                    self.get_logger().warn(f"Cube {cube_id} disappeared → skipping move")
                    self.current_task = None
                    return

                if not self.goal_sent:
                    self.send_goal(pose)
                    self.goal_sent = True
                elif self.goal_reached:
                    self.current_task = None
            elif action == "move_box":
                if not self.active_box_targets:
                    self.get_logger().warn("No box available")
                    self.current_task = None
                    return

                box_id, pose = min(
                    self.active_box_targets.items(),
                    key=lambda item: self.distance(self.last_robot_pose, item[1])
                )

                if not self.goal_sent:
                    self.send_goal(pose)
                    self.goal_sent = True

                elif self.goal_reached:
                    self.current_task = None
                    self.goal_sent = False
            elif action == "cubego":
                if not self.goal_sent:
                    self.cube_approached = False
                    self.send_bool(self.cube_go_pub, "CUBE APPROACH")
                    self.goal_sent = True
                elif self.cube_approached:
                    self.current_task = None

            elif action == "grasp":
                if not self.goal_sent:
                    self.current_cube_id = data
                    self.send_bool(self.grip_grasp_pub, "GRASP")
                    self.goal_sent = True

                elif self.grip_done:
                    cube_id = self.current_cube_id   # ✅ define it once

                    if cube_id in self.cubes_active:
                        del self.cubes_active[cube_id]

                    self.collected_cubes.add(cube_id)

                    # OPTIONAL: remove from targets
                    if cube_id in self.active_cube_targets:
                        del self.active_cube_targets[cube_id]

                    marker = Marker()
                    marker.id = cube_id
                    marker.action = Marker.DELETE
                    marker.header.frame_id = "map"

                    ma = MarkerArray()
                    ma.markers.append(marker)
                    #
                    # 
                    # 
                    # self.marker_pub.publish(ma)
                    #self.marker_pub_per.publish(ma)
                    self.current_task = None
            elif action == "release":
                if not self.goal_sent:
                    self.send_bool(self.grip_release_pub, "RELEASE")
                    self.goal_sent = True
                elif self.grip_done:
                    self.current_task = None

            elif action == "backward":
                if not self.backward_active:
                    duration = data
                    self.backward_msg = Twist()
                    self.backward_msg.linear.x = -0.1
                    self.backward_active = True
                    self.backward_end_time = now + duration
                    self.get_logger().info(f"↩ START BACKWARD for {duration}s")

            return  # skip exploration if cube tasks exist
        if not self.task_queue and self.current_task is None:
            if self.delivery_in_progress:
                self.get_logger().info("✅ Delivery cycle finished")
            self.delivery_in_progress = False
        # ✅ if cubes exist but no active plan → replan
        if self.cubes_active and not self.task_queue and not self.current_task:
            self.get_logger().info("🔁 New cubes available → replanning")
            self.exploration_mode = False

            msg_bool = Bool()
            msg_bool.data = False
            self.exploration_pub.publish(msg_bool)

            self.replan(self.last_robot_pose)
            return
        # --- if no cube tasks, enable exploration ---
        if not self.exploration_mode:
            self.get_logger().info("🧭 Exploration ON")
            self.exploration_mode = True
            self.from_exploration = True

            msg_bool = Bool()
            msg_bool.data = True
            self.exploration_pub.publish(msg_bool)


# ===========================
# Main
# ===========================
def main(args=None):
    rclpy.init(args=args)
    node = RobotTaskManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()