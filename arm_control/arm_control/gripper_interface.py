#!/usr/bin/env python3
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from robp_interfaces.msg import ArmControl
from sensor_msgs.msg import JointState

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer

class GripperInterface(Node):
    def __init__(self):
        super().__init__('gripper_interface')

                # Publishers
        self.ready_pub = self.create_publisher(Bool, '/grip/ready', 10)
        self.grasp_pub = self.create_publisher(Bool, '/grip/grasp', 10)
        self.release_pub = self.create_publisher(Bool, '/grip/release', 10)

        # Subscriber
        self.create_subscription(
            Bool,
            '/grip/success',
            self.finished_callback,
            10
        )

        self.finished_state = False

        self.get_logger().info("Gripper Interface Node Started")

    # ------------------ Functions ------------------
    def publish_ready(self):
        msg = Bool()
        msg.data = True
        self.ready_pub.publish(msg)
        self.get_logger().info("Published READY")

    def publish_release(self):
        msg = Bool()
        msg.data = True
        self.release_pub.publish(msg)
        self.get_logger().info("Published RELEASE")

    def finished_callback(self, msg):
        self.finished_state = msg.data
        self.get_logger().info(f"Received SUCCESS: {msg.data}")


# ------------------ PyQt GUI ------------------
class GripperGUI(QWidget):

    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node

        self.setWindowTitle("Gripper Control")
        self.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout()

        # Buttons
        self.ready_button = QPushButton("Send Ready")
        self.ready_button.clicked.connect(self.node.publish_ready)
        layout.addWidget(self.ready_button)

        self.release_button = QPushButton("Send Release")
        self.release_button.clicked.connect(self.node.publish_release)
        layout.addWidget(self.release_button)

        # LED indicator
        self.status_label = QLabel("SUCCESS: FALSE")
        self.status_label.setStyleSheet("background-color: red; color: white;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(100)

    def update_status(self):
        if self.node.finished_state:
            self.status_label.setText("SUCCESS: TRUE")
            self.status_label.setStyleSheet("background-color: green; color: white;")
        else:
            self.status_label.setText("SUCCESS: FALSE")
            self.status_label.setStyleSheet("background-color: red; color: white;")


# ------------------ ROS Spin Thread ------------------
def ros_spin(node):
    rclpy.spin(node)


# ------------------ Main ------------------
def main():
    rclpy.init()
    gripper_node = GripperInterface()

    # Start ROS2 spinning in a separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(gripper_node,), daemon=True)
    ros_thread.start()

    # Start Qt application
    app = QApplication(sys.argv)
    gui = GripperGUI(gripper_node)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()