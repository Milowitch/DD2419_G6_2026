# !/usr/bin/env python3
import sys
import threading
import time

import numpy as np
import rclpy
import cv2
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

from robp_interfaces.msg import ArmControl
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer

class DummyGripperNode(Node):
    def __init__(self):
        super().__init__('dummy_gripper')

        # Subscribers
        self.create_subscription(Bool, '/grip/ready', self.grip_ready_cb, 10)
        self.create_subscription(Bool, '/grip/grasp', self.grip_grasp_cb, 10)
        self.create_subscription(Pose, '/grip/coords', self.grip_coords_cb, 10)
        self.create_subscription(Bool, '/grip/release', self.grip_release_cb, 10)
        self.create_subscription(Image, '/arm/camera/image_raw', self.image_callback, 10)

        # Publishers
        self.grip_finished_pub = self.create_publisher(Bool, '/grip/finished', 10)
        self.detection_ready_pub = self.create_publisher(Bool, '/grip/detection_ready', 10)
        self.move_arm_pub = self.create_publisher(ArmControl, '/arm/control', 10)
        self.grip_success_pub = self.create_publisher(Bool, '/grip/success', 10)

        # Joints
        self.last_positions = [0]*6
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Track status for GUI
        self.topic_status = {
            'ready': False,
            'release': False,
            'detection_ready': False,
            'coords': False
        }
        self.frame = None
        self.success = False
        self.lock = threading.Lock()

        self.get_logger().info("Gripper node initialized")

    # ------------------ Callbacks ------------------
    def grip_ready_cb(self, msg: Bool):
        self.topic_status['ready'] = msg.data
        self.get_logger().info(f"Pickup ready received: {msg.data}")
        threading.Thread(
            target=self.move_arm_to_wp,
            daemon=True
        ).start()

    def grip_grasp_cb(self, msg: Bool):
        self.topic_status['ready'] = msg.data
        self.get_logger().info(f"Grasping received: {msg.data}")
        threading.Thread(
            target=self.start_grasp,
            daemon=True
        ).start()

    def grip_coords_cb(self, msg: Pose):
        if msg.position.z < 0:
            self.get_logger().warn("No object detected")
            self.success = False
            self.send_finished()
            self.move_arm([120, 120, 30, 220, 180, 120], time_ms=2000)
            time.sleep(2)
            return
        
        # get x_world, y_world from message
        x_world = msg.position.x
        y_world = msg.position.y
        yaw = msg.orientation.w
        self.get_logger().info(f"Received grasp target: x={x_world}, y={y_world}, yaw={yaw}")
        threading.Thread(
            target=self.move_arm_grasp,
            args=(x_world, y_world, yaw),
            daemon=True
        ).start()

    def grip_release_cb(self, msg: Bool):
        self.topic_status['release'] = msg.data
        self.get_logger().info(f"Grip Release received: {msg.data}")
        threading.Thread(
            target=self.move_arm_release,
            daemon=True
        ).start()

    def image_callback(self, msg):

        np_arr = np.frombuffer(msg.data, dtype=np.uint8)

        # self.get_logger().warn(f"{msg.width}x{msg.height} {msg.encoding} len={len(msg.data)}")

        if msg.encoding == "rgb8":
                    frame = np_arr.reshape((msg.height, msg.width, 3))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        elif msg.encoding == "bgr8":
            frame = np_arr.reshape((msg.height, msg.width, 3))

        elif msg.encoding in ["yuyv", "yuv422_yuy2"]:
            frame = np_arr.reshape((msg.height, msg.width, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

        else:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
            return

        with self.lock:
            self.frame = frame.copy()


    def joint_state_callback(self, msg):
        self.last_positions = list(msg.position[:6])

    # ------------------ Arm move commands ------------------
    def move_arm(self, positions, time_ms=3000):
        """
        positions: list of 6 joint angles (degrees)
        time_ms: int or list of 6 ints (milliseconds)
        """

        if len(positions) != 6:
            self.get_logger().error("Expected 6 joint positions")
            return

        msg = ArmControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = positions
        self.target_positions = positions

        # Time
        if isinstance(time_ms, int):
            msg.time = [time_ms] * 6
        else:
            msg.time = time_ms

        self.move_arm_pub.publish(msg)
        self.get_logger().info(f"Sent arm command: {positions}")

    def move_arm_ik(self, x_world, y_world, yaw, j5, j1, time_ms = 3000):
        theta = np.degrees(np.atan2(x_world,y_world))
        if theta > 90:
            theta = theta - 180
        r_x = np.sqrt(x_world**2 + y_world**2) - 0.12
        r_z = 0.180
        l_1 = 0.095
        l_2 = 0.165
        D = (r_x**2 + r_z**2 - l_1**2 - l_2**2) / (2*l_1*l_2)
        D = np.clip(D, -1.0, 1.0)
        phi_2 = np.acos(D)
        phi_1 = np.atan2(r_z, r_x) - np.atan2(l_2* np.sin(phi_2), l_1+l_2 * np.cos(phi_2))
        phi_1 = np.degrees(phi_1)
        phi_2 = np.degrees(phi_2)

        self.get_logger().info(f"Sent arm IK command: {yaw, theta, phi_1, phi_2}")

        if (-45 < theta < 45) and ((abs(theta)*2 + phi_2) < 180) and (phi_2 < 110) and (r_x < 0.18):
            self.move_arm((j5, yaw+120, 120-phi_2, 120+phi_1, j1, 120 - theta))
        else:
            self.get_logger().warn(f"Position out of reach!")
            


    def move_arm_to_start(self):
        self.move_arm([40, 120, 30, 165, 80, 120], time_ms=2000)
        time.sleep(2)
        self.move_arm([40, 120, 30, 220, 180, 120], time_ms=2000)
        time.sleep(4)


    def move_arm_to_wp(self):
        self.move_arm([40, 120, 27, 165, 80, 120], time_ms=2000)
        time.sleep(4)
        self.topic_status['ready'] = self.has_reached_goal()

    def start_grasp(self):
        time.sleep(3)
        self.detection_ready()


    def move_arm_grasp(self, x_world, y_world, yaw):
        self.success = False
        self.move_arm_ik(x_world, y_world, yaw, 40, 40, time_ms=2000)
        time.sleep(4)
        self.move_arm_ik(x_world, y_world, yaw, 120, 40, time_ms=2000)
        time.sleep(4)
        self.move_arm([120, 120, 30, 220, 180, 120], time_ms=2000)
        time.sleep(6)

        self.success = self.pickup_success()
        self.send_success()
        self.send_finished()


    def move_arm_release(self):
        self.move_arm([120, 120, 80, 165, 80, 120], time_ms=2000)
        time.sleep(4)
        self.move_arm([40, 120, 80, 165, 80, 120], time_ms=2000)
        time.sleep(2)
        self.move_arm([40, 120, 80, 165, 180, 120], time_ms=2000)
        time.sleep(2)
        self.move_arm([40, 120, 30, 220, 180, 120], time_ms=2000)
        time.sleep(2)
        self.topic_status['release'] = self.has_reached_goal()
        self.send_finished()


    # ------------------ Position comparison ------------------
    def has_reached_goal(self, tolerance=2.0):

        if not hasattr(self, 'last_positions') or not hasattr(self, 'target_positions'):
            return False

        for actual, target in zip(self.last_positions, self.target_positions):
            if abs(actual - target) > tolerance:
                return False

        return True
    
    # ------------------ Waiting ------------------
    def wait_until_reached(self, timeout=3.0):
        start_time = time.time()


        while not self.has_reached_goal():

            if time.time() - start_time > timeout:
                self.get_logger().warn("Timeout: goal not reached")
                return False

        self.get_logger().info("Goal reached ")
        return True
    
    # ------------------ Pickup success detection ------------------
    def pickup_success(self):  
        with self.lock:
            if self.frame is None:
                self.get_logger().warn("No frame available")
                return False
            frame = self.frame.copy()

        # ROI
        x1, y1 = 280, 375
        x2, y2 = 370, 390
        roi = frame[y1:y2, x1:x2]

        mean_color = np.mean(roi, axis=(0, 1))
        b,g,r = mean_color

        # detection:
        if (r > 220 and g < 160 and b < 200) or (r > 180 and g < 140 and b < 150) or (r < 80 and g < b and b > 70) or (r < 140 and g < b and b > 200) or (r < 20 and g > 40 and b < 40) or (r < 50 and g > b*1.5 and g > 50):
            return True
        else:
            self.get_logger().warn(f"R:{r:.1f} G:{g:.1f} B:{b:.1f}")
            return False  


    # ------------------ Button trigger ------------------
    def send_finished(self):
        msg = Bool()
        # msg.data = self.success
        msg.data = True
        self.grip_finished_pub.publish(msg)
        self.get_logger().info("Grip Finished sent from gripper node.")
        self.topic_status['success'] = self.success

    # ------------------ Success feedback ------------------
    def send_success(self):
        msg = Bool()
        msg.data = self.success
        self.grip_success_pub.publish(msg)
        self.get_logger().info("Grasp success sent from gripper node.")
        self.topic_status['success'] = self.success

    # ----------------- Detection ready -----------------
    def detection_ready(self):
        msg = Bool()
        msg.data = True
        self.detection_ready_pub.publish(msg)
        self.get_logger().info("Detection ready from gripper node")
        self.topic_status['detection_ready'] = True

# ------------------ PyQt GUI ------------------
# class GripperGUI(QWidget):
#     def __init__(self, gripper_node):
#         super().__init__()
#         self.gripper_node = gripper_node
#         self.setWindowTitle("Dummy Gripper Control")
#         self.setGeometry(100, 100, 250, 200)

#         layout = QVBoxLayout()
#         self.setLayout(layout)

#         # Topic LEDs
#         self.leds = {}
#         for topic in ['ready', 'release', 'detection_ready', 'coords']:
#             row = QHBoxLayout()
#             label = QLabel(topic)
#             label.setAlignment(Qt.AlignCenter)
#             led = QLabel()
#             led.setFixedSize(30, 30)
#             led.setStyleSheet("background-color: red; border-radius: 15px;")
#             row.addWidget(label)
#             row.addWidget(led)
#             layout.addLayout(row)
#             self.leds[topic] = led

#         # Button to send grip/finished
#         self.button = QPushButton("Send Finished")
#         self.button.clicked.connect(self.send_finished)
#         layout.addWidget(self.button)

#         # Timer to update LEDs every 200ms
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_leds)
#         self.timer.start(200)

#     def update_leds(self):
#         for topic, led in self.leds.items():
#             if self.gripper_node.topic_status[topic]:
#                 led.setStyleSheet("background-color: green; border-radius: 15px;")
#             else:
#                 led.setStyleSheet("background-color: red; border-radius: 15px;")

#     def send_finished(self):
        
#         self.gripper_node.send_finished()


# ------------------ ROS Spin Thread ------------------
def ros_spin(node):
    rclpy.spin(node)


# ------------------ Main ------------------
def main():
    rclpy.init()
    gripper_node = DummyGripperNode()

    rclpy.spin(gripper_node)   # blocks forever

    gripper_node.destroy_node()
    rclpy.shutdown()
    # rclpy.init()
    # gripper_node = DummyGripperNode()

    # Start ROS2 spinning in a separate thread
    # ros_thread = threading.Thread(target=ros_spin, args=(gripper_node,), daemon=True)
    # ros_thread.start()

    # Start Qt application
    # app = QApplication(sys.argv)
    # gui = GripperGUI(gripper_node)
    # gui.show()

    # sys.exit(app.exec_())


if __name__ == "__main__":
    main()