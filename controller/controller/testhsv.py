#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import threading


class HSVTuner(Node):

    def __init__(self):
        super().__init__('hsv_realsense_threaded')

        self.subscription = self.create_subscription(
            Image,
            '/realsense/color/image_raw',
            self.image_callback,
            10)

        self.lock = threading.Lock()
        self.frame = None

    def image_callback(self, msg):

        np_arr = np.frombuffer(msg.data, dtype=np.uint8)

        frame = np_arr.reshape((msg.height, msg.width, 3))

        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with self.lock:
            self.frame = frame.copy()


def ros_spin(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = HSVTuner()

    # Start ROS spinning in separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    # Create GUI in main thread
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("H Min", "Trackbars", 0, 179, lambda x: None)
    cv2.createTrackbar("H Max", "Trackbars", 179, 179, lambda x: None)
    cv2.createTrackbar("S Min", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("S Max", "Trackbars", 255, 255, lambda x: None)
    cv2.createTrackbar("V Min", "Trackbars", 0, 255, lambda x: None)
    cv2.createTrackbar("V Max", "Trackbars", 255, 255, lambda x: None)

    while rclpy.ok():

        frame = None
        with node.lock:
            if node.frame is not None:
                frame = node.frame.copy()

        if frame is not None:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            h_min = cv2.getTrackbarPos("H Min", "Trackbars")
            h_max = cv2.getTrackbarPos("H Max", "Trackbars")
            s_min = cv2.getTrackbarPos("S Min", "Trackbars")
            s_max = cv2.getTrackbarPos("S Max", "Trackbars")
            v_min = cv2.getTrackbarPos("V Min", "Trackbars")
            v_max = cv2.getTrackbarPos("V Max", "Trackbars")

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        