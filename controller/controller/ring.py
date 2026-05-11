#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import math


class RingApproachPlanner(Node):
    def __init__(self):
        super().__init__('ring_approach_planner')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.cube_objects = {}
        self.box_objects = {}

        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.create_subscription(MarkerArray, '/perception/markersT', self.cube_cb, qos)
        self.create_subscription(MarkerArray, '/perception/box', self.box_cb, qos)

        self.pose_pub = self.create_publisher(PoseArray, '/approach_poses', 10)
        self.ring_pub = self.create_publisher(MarkerArray, '/ring_debug', 10)

        self.cube_pub = self.create_publisher(Float64MultiArray, '/ring_pos/cube', 10)
        self.box_pub = self.create_publisher(Float64MultiArray, '/ring_pos/box', 10)

        self.map = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.timer = self.create_timer(0.1, self.compute)

        self.get_logger().info("Ring Approach Planner READY")

    # =========================
    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )

    # =========================
    def cube_cb(self, msg):
        for m in msg.markers:
            obj_id = f"cube_{m.id}"

            if m.action == Marker.DELETE:
                self.cube_objects.pop(obj_id, None)
                continue

            self.cube_objects[obj_id] = {
                "x": m.pose.position.x,
                "y": m.pose.position.y
            }

    def box_cb(self, msg):
        for m in msg.markers:
            obj_id = f"box_{m.id}"

            if m.action == Marker.DELETE:
                self.box_objects.pop(obj_id, None)
                continue

            self.box_objects[obj_id] = {
                "x": m.pose.position.x,
                "y": m.pose.position.y
            }

    # =========================
    def map_cb(self, msg):
        self.map = msg

    # =========================
    def compute(self):

        if self.map is None:
            return

        all_objects = {}

        for k, v in self.cube_objects.items():
            all_objects[k] = {"x": v["x"], "y": v["y"], "type": "cube"}

        for k, v in self.box_objects.items():
            all_objects[k] = {"x": v["x"], "y": v["y"], "type": "box"}

        if len(all_objects) == 0:
            return

        width = self.map.info.width
        height = self.map.info.height
        res = self.map.info.resolution

        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        data = np.array(self.map.data).reshape((height, width))

        pose_array = PoseArray()
        pose_array.header = self.map.header

        ring_msg = MarkerArray()

        cube_data = []
        box_data = []

        marker_id = 0

        for obj_id, obj in all_objects.items():

            x_o = obj["x"]
            y_o = obj["y"]
            obj_type = obj["type"]

            r = 0.45 if obj_type == "cube" else 0.34
            angles = np.linspace(0, 2*np.pi, 15)

            best_pose = None
            best_yaw = None
            best_score = float('inf')

            obj_index = int(obj_id.split('_')[1])

            # ✅ CHECK IF ROBOT IS NEAR RING
            dist_to_object = math.sqrt(
                (self.robot_x - x_o)**2 +
                (self.robot_y - y_o)**2
            )

            threshold = 0.1
            near_ring = abs(dist_to_object - r) < threshold

            for a in angles:

                x = x_o + r * math.cos(a)
                y = y_o + r * math.sin(a)

                gx = int((x - origin_x) / res)
                gy = int((y - origin_y) / res)

                valid = (0 <= gx < width and 0 <= gy < height)

                marker = Marker()
                marker.header = self.map.header
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.color.a = 1.0

                if valid and data[gy, gx] == 0:
                    marker.color.g = 1.0
                else:
                    marker.color.r = 1.0

                ring_msg.markers.append(marker)
                marker_id += 1

                if not valid or data[gy, gx] != 0:
                    continue

                # ✅ ORIENTATION TOWARD OBJECT
                theta = math.atan2(y_o - y, x_o - x)

                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.orientation.z = math.sin(theta / 2.0)
                pose.orientation.w = math.cos(theta / 2.0)

                if near_ring:
                    # 🔥 publish ALL ring poses
                    pose_array.poses.append(pose)

                    if obj_type == "cube":
                        cube_data += [float(obj_index), x, y, theta]
                    else:
                        box_data += [float(obj_index), x, y, theta]

                else:
                    # 🔁 best pose selection
                    dx = x - self.robot_x
                    dy = y - self.robot_y
                    dist = math.sqrt(dx*dx + dy*dy)

                    heading = abs(self.normalize_angle(theta - self.robot_yaw))
                    score = 10.0 * dist + 1.2 * heading

                    if score < best_score:
                        best_score = score
                        best_pose = pose
                        best_yaw = theta

            # ✅ ONLY add best if NOT near ring
            if not near_ring and best_pose:
                pose_array.poses.append(best_pose)

                if obj_type == "cube":
                    cube_data += [float(obj_index),
                                  best_pose.position.x,
                                  best_pose.position.y,
                                  best_yaw]
                else:
                    box_data += [float(obj_index),
                                 best_pose.position.x,
                                 best_pose.position.y,
                                 best_yaw]

        # =========================
        self.pose_pub.publish(pose_array)
        self.ring_pub.publish(ring_msg)

        cube_msg = Float64MultiArray()
        box_msg = Float64MultiArray()

        cube_msg.data = cube_data
        box_msg.data = box_data

        self.cube_pub.publish(cube_msg)
        self.box_pub.publish(box_msg)

    # =========================
    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))


def main():
    rclpy.init()
    node = RingApproachPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node

# from nav_msgs.msg import OccupancyGrid, Odometry
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Pose, PoseArray
# from std_msgs.msg import Float64MultiArray
# from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# import numpy as np
# import math


# class RingApproachPlanner(Node):
#     def __init__(self):
#         super().__init__('ring_approach_planner')

#         qos = QoSProfile(
#             depth=10,
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL
#         )

#         # =========================
#         # OBJECT STORAGE (persistent)
#         # =========================
#         self.cube_objects = {}
#         self.box_objects = {}

#         # =========================
#         # SUBSCRIBERS
#         # =========================
#         self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
#         self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

#         self.create_subscription(MarkerArray, '/perception/markersT', self.cube_cb, qos)
#         self.create_subscription(MarkerArray, '/perception/box', self.box_cb, qos)

#         # =========================
#         # PUBLISHERS
#         # =========================
#         self.pose_pub = self.create_publisher(PoseArray, '/approach_poses', 10)
#         self.ring_pub = self.create_publisher(MarkerArray, '/ring_debug', 10)

#         self.cube_pub = self.create_publisher(Float64MultiArray, '/ring_pos/cube', 10)
#         self.box_pub = self.create_publisher(Float64MultiArray, '/ring_pos/box', 10)

#         # =========================
#         # STATE
#         # =========================
#         self.map = None

#         self.robot_x = 0.0
#         self.robot_y = 0.0
#         self.robot_yaw = 0.0

#         self.timer = self.create_timer(0.1, self.compute)

#         self.get_logger().info("Ring Approach Planner READY")

#     # =========================================================
#     # ODOM
#     # =========================================================
#     def odom_cb(self, msg):
#         self.robot_x = msg.pose.pose.position.x
#         self.robot_y = msg.pose.pose.position.y

#         q = msg.pose.pose.orientation
#         self.robot_yaw = math.atan2(
#             2.0 * (q.w*q.z + q.x*q.y),
#             1.0 - 2.0 * (q.y*q.y + q.z*q.z)
#         )

#     # =========================================================
#     # OBJECT CALLBACKS (FIXED)
#     # =========================================================
#     def cube_cb(self, msg):
#         for m in msg.markers:
#             obj_id = f"cube_{m.id}"

#             if m.action == Marker.DELETE:
#                 if obj_id in self.cube_objects:
#                     del self.cube_objects[obj_id]
#                 continue

#             self.cube_objects[obj_id] = {
#                 "x": m.pose.position.x,
#                 "y": m.pose.position.y
#             }

#     def box_cb(self, msg):
#         for m in msg.markers:
#             obj_id = f"box_{m.id}"

#             if m.action == Marker.DELETE:
#                 if obj_id in self.box_objects:
#                     del self.box_objects[obj_id]
#                 continue

#             self.box_objects[obj_id] = {
#                 "x": m.pose.position.x,
#                 "y": m.pose.position.y
#             }

#     # =========================================================
#     # MAP
#     # =========================================================
#     def map_cb(self, msg):
#         self.map = msg

#     # =========================================================
#     # CORE
#     # =========================================================
#     def compute(self):

#         if self.map is None:
#             return

#         # =========================
#         # MERGE OBJECTS
#         # =========================
#         all_objects = {}

#         for k, v in self.cube_objects.items():
#             all_objects[k] = {"x": v["x"], "y": v["y"], "type": "cube"}

#         for k, v in self.box_objects.items():
#             all_objects[k] = {"x": v["x"], "y": v["y"], "type": "box"}

#         if len(all_objects) == 0:
#             return

#         # DEBUG
#         self.get_logger().info(
#             f"Cubes: {len(self.cube_objects)} | Boxes: {len(self.box_objects)}"
#         )

#         # =========================
#         # MAP DATA
#         # =========================
#         width = self.map.info.width
#         height = self.map.info.height
#         res = self.map.info.resolution

#         origin_x = self.map.info.origin.position.x
#         origin_y = self.map.info.origin.position.y

#         data = np.array(self.map.data).reshape((height, width))

#         pose_array = PoseArray()
#         pose_array.header = self.map.header

#         ring_msg = MarkerArray()

#         cube_data = []
#         box_data = []

#         marker_id = 0

#         # =====================================================
#         for obj_id, obj in all_objects.items():

#             x_o = obj["x"]
#             y_o = obj["y"]
#             obj_type = obj["type"]


#             r = 0.45
#             if obj_type == "box":
#                 r=0.34
#             angles = np.linspace(0, 2*np.pi,15)

#             best_pose = None
#             best_yaw  = None
#             best_score = float('inf')

#             obj_index = int(obj_id.split('_')[1])

#             for a in angles:

#                 x = x_o + r * math.cos(a)
#                 y = y_o + r * math.sin(a)

#                 gx = int((x - origin_x) / res)
#                 gy = int((y - origin_y) / res)

#                 marker = Marker()
#                 marker.header = self.map.header
#                 marker.id = marker_id
#                 marker.type = Marker.SPHERE
#                 marker.scale.x = 0.05
#                 marker.scale.y = 0.05
#                 marker.scale.z = 0.05
#                 marker.pose.position.x = x
#                 marker.pose.position.y = y
#                 marker.color.a = 1.0

#                 valid = (0 <= gx < width and 0 <= gy < height)

#                 if valid and data[gy, gx] == 0:
#                     marker.color.g = 1.0
#                 else:
#                     marker.color.r = 1.0

#                 ring_msg.markers.append(marker)
#                 marker_id += 1

#                 if not valid or data[gy, gx] != 0:
#                     continue

#                 dx = x - self.robot_x
#                 dy = y - self.robot_y
#                 dist = math.sqrt(dx*dx + dy*dy)

#                 theta = math.atan2(y_o - y, x_o - x)
#                 heading = abs(self.normalize_angle(theta - self.robot_yaw))

#                 score = 10.0 * dist + 1.2 * heading

#                 if score < best_score:
#                     best_score = score

#                     best_pose = Pose()
#                     best_pose.position.x = x
#                     best_pose.position.y = y
#                     best_pose.orientation.z = math.sin(theta / 2.0)
#                     best_pose.orientation.w = math.cos(theta / 2.0)
#                     best_yaw = theta

#             if best_pose:
#                 pose_array.poses.append(best_pose)

#                 if obj_type == "cube":
#                     cube_data += [float(obj_index), best_pose.position.x, best_pose.position.y,best_yaw]
#                 else:
#                     box_data += [float(obj_index), best_pose.position.x, best_pose.position.y,best_yaw]

#         # =====================================================
#         # PUBLISH
#         # =====================================================
#         self.pose_pub.publish(pose_array)
#         self.ring_pub.publish(ring_msg)

#         cube_msg = Float64MultiArray()
#         box_msg = Float64MultiArray()

#         cube_msg.data = cube_data
#         box_msg.data = box_data

#         self.cube_pub.publish(cube_msg)
#         self.box_pub.publish(box_msg)

#     # =========================================================
#     def normalize_angle(self, a):
#         return math.atan2(math.sin(a), math.cos(a))


# def main():
#     rclpy.init()
#     node = RingApproachPlanner()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()