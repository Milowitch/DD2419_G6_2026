#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class BoxFollower(Node):
    def __init__(self):
        super().__init__('box_follower')

        # Parameters
        self.max_distance = 0.4      # max distance to consider
        self.min_height = -0.1        # min height (camera frame)
        self.max_height = 0.02       # max height (10 cm)
        self.target_distance = 0.20  # desired distance to box
        self.Kp_linear = 5.0
        self.Kp_angular = 5.0
        self.max_linear = 0.25
        self.max_angular = 0.8

        # Subscribers & Publishers
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/realsense/depth/color/points',
            self.pc_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/box_center_marker', 10)
        self.filtered_pc_pub = self.create_publisher(PointCloud2, '/filtered_box_points', 10)

        # Marker setup
        self.marker = Marker()
        self.marker.header.frame_id = "realsense_camera_link"  # camera frame
        self.marker.ns = "box_tracker"
        self.marker.id = 0
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.05
        self.marker.scale.y = 0.05
        self.marker.scale.z = 0.05
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.pose.orientation.w = 1.0

    def pc_callback(self, msg):
        # Convert PointCloud2 to safe Python floats
        points = []
        for p in pc2.read_points(msg, skip_nans=True):
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                points.append((x, y, z))
            except:
                continue
        if len(points) == 0:
            return

        points = np.array(points)

        # Filter by distance and height
        dists = np.linalg.norm(points, axis=1)
        mask = (dists < self.max_distance) & (points[:,1] >= self.min_height) & (points[:,1] <= self.max_height)
        filtered_points = points[mask]
        if len(filtered_points) == 0:
            return

        # Publish filtered points for visualization
        filtered_msg = pc2.create_cloud_xyz32(msg.header, filtered_points.tolist())
        self.filtered_pc_pub.publish(filtered_msg)

        # Weighted center (mean)
        box_center_cam = np.mean(filtered_points, axis=0)

        # Convert RealSense camera frame (X_left,Y_up,Z_forward) -> robot frame
        forward = float(box_center_cam[2])  # Z_forward
        left = float(box_center_cam[0])     # X_left

        # Update marker in RViz (camera frame)
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.pose.position.x = float(box_center_cam[0])
        self.marker.pose.position.y = float(box_center_cam[1])
        self.marker.pose.position.z = float(box_center_cam[2])
        self.marker_pub.publish(self.marker)

        # Compute simple proportional control
        dx = forward - self.target_distance
        dy = left
        linear_vel = float(np.clip(self.Kp_linear * dx, -self.max_linear, self.max_linear))
        angular_vel = float(np.clip(self.Kp_angular * -dy, -self.max_angular, self.max_angular))

        # Publish cmd_vel
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_pub.publish(twist)

        self.get_logger().info(f"Box center (cam): {box_center_cam}, linear: {linear_vel:.2f}, angular: {angular_vel:.2f}, points used: {len(filtered_points)}")


def main(args=None):
    rclpy.init(args=args)
    node = BoxFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from visualization_msgs.msg import Marker
# from std_msgs.msg import Float64MultiArray, Bool
# import numpy as np
# import math

# class NearCubeCatcher(Node):

#     def __init__(self):
#         super().__init__('nearcube_catcher')

#         # Desired final distance
#         self.target_z = 0.18 # 22 cm

#         # Controller gains
#         self.k_ang_rotate = 0.05
#         self.k_lin = 8.0
#         self.k_ang_center = 2.0

#         # Limits
#         self.max_lin = 0.25
#         self.max_ang = 0.8

#         # Thresholds
#         self.angle_threshold_deg = 5.0
#         self.distance_threshold = 0.01

#         self.stage = 1  # 1 = rotate, 2 = approach
#         self.active = False
#         self.finished = False

#         # Subscribe to Marker
#         self.sub = self.create_subscription(
#             Marker,
#             '/perception/nearcube',
#             self.cube_callback,
#             10)

#         # Subscribe to parameter updates
#         self.param_sub = self.create_subscription(
#             Float64MultiArray,
#             '/nearcube_controller_params',
#             self.update_params_callback,
#             10)

#         # Subscribe to activation
#         self.go_sub = self.create_subscription(
#             Bool,
#             '/cube/go',
#             self.go_callback,
#             10)

#         # Publish velocity
#         self.pub = self.create_publisher(
#             Twist,
#             '/cmd_vel',
#             10)

#         # Publish approach success / failure
#         self.approach_pub = self.create_publisher(Bool, '/cube/approach', 10)
#         self.fail_pub = self.create_publisher(Bool, '/cube/fail', 10)

#         self.get_logger().info("🎯 NearCube Catcher (Marker Version) Started")
#     def go_callback(self, msg: Bool):

#         if msg.data:  # Activation request
#             self.active = True
#             self.finished = False
#             self.stage = 1

#             # Reset single-publish flags
#             self._approach_published = False
#             self._fail_published = False
#             self._stop_published = False

#             self.get_logger().info("🚀 Cube approach activated")

#         else:
#             self.active = False
#             self.get_logger().info("⏹ Cube approach deactivated")

#     def cube_callback(self, msg):

#         if not self.active or self.finished:
#             return

#         # Ignore delete markers
#         if msg.action == Marker.DELETE:
#             return

#         # Extract cube position from Marker pose
#         x = msg.pose.position.x   # + right
#         z = msg.pose.position.z   # forward

#         cmd = Twist()

#         # Neglect if cube too far
#         if z > 0.4:
#             self.publish_fail_once()
#             self.stop_robot_once()
#             return

#         # Safety: ignore invalid depth
#         if z <= 0.0:
#             self.stop_robot_once()
#             return

#         # Compute angle error
#         theta = np.arctan2(x, z)
#         theta_deg = np.degrees(theta)

#         # =================================
#         # STAGE 1: ROTATE ONLY
#         # =================================
#         if self.stage == 1:

#             if abs(theta_deg) > self.angle_threshold_deg:

#                 ang =( -theta_deg/29)# self.k_ang_rotate * theta
#                 ang = np.clip(ang, -self.max_ang, self.max_ang)

#                 cmd.angular.z =ang
#                 cmd.linear.x = 0.0
#                 self.get_logger().info(
#                     f"x={x:.3f}, z={z:.3f}, theta_deg={theta_deg:.2f}, ang_cmd={ang:.2f}"
#                 )
#             else:
#                 self.stage = 2
#                 self.get_logger().info("✅ Rotation aligned. Switching to approach stage.")

#         # =================================
#         # STAGE 2: APPROACH
#         # =================================
#         if self.stage == 2:

#             dz = z - self.target_z

#             if abs(dz) < self.distance_threshold:

#                 cmd.linear.x = 0.0
#                 cmd.angular.z = 0.0
#                 self.get_logger().info("🏁 Cube reached at target distance")
#                 self.publish_approach_once()
#                 self.stop_robot_once()

#                 self.active = False
#                 self.finished = True
#                 self.get_logger().info("🛑 Waiting for next /cube/go")
#                 return
#             else:
#                 # Forward speed
#                 lin = self.k_lin * dz
#                 lin = np.clip(lin, -self.max_lin, self.max_lin)

#                 # Small angular correction while moving
#                 ang = ( -theta_deg/22)
#                 ang = np.clip(ang, -self.max_ang, self.max_ang)

#                 cmd.linear.x = lin
#                 cmd.angular.z = ang

#         self.pub.publish(cmd)

#     # ------------------ Single-publish helpers ------------------
#     def publish_approach_once(self):
#         if not hasattr(self, '_approach_published') or not self._approach_published:
#             msg = Bool()
#             msg.data = True
#             self.approach_pub.publish(msg)
#             self._approach_published = True
#             self.finished = True

#     def publish_fail_once(self):
#         if not hasattr(self, '_fail_published') or not self._fail_published:
#             msg = Bool()
#             msg.data = True
#             self.fail_pub.publish(msg)
#             self._fail_published = True
#             self.finished = True

#     def stop_robot_once(self):
#         if not self._stop_published:
#             cmd = Twist()
#             self.pub.publish(cmd)
#             self._stop_published = True

#     # ------------------ Parameter updates ------------------
#     def update_params_callback(self, msg: Float64MultiArray):
#         """
#         Expected order of msg.data:
#         [target_z, k_ang_rotate, k_lin, k_ang_center, max_lin, max_ang, angle_threshold_deg, distance_threshold]
#         """
#         data = msg.data
#         if len(data) != 8:
#             self.get_logger().warn("Invalid parameter array length, expected 8 values.")
#             return

#         self.target_z = data[0]
#         self.k_ang_rotate = data[1]
#         self.k_lin = data[2]
#         self.k_ang_center = data[3]
#         self.max_lin = data[4]
#         self.max_ang = data[5]
#         self.angle_threshold_deg = data[6]
#         self.distance_threshold = data[7]

#         self.get_logger().info(f"🎛 Updated parameters: {data}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = NearCubeCatcher()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()