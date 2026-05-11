import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import threading

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
import math


class HSVTuner(Node):

    def __init__(self):
        super().__init__('hsv_realsense_threaded')

        self.camsub = self.create_subscription(
            Image,
            '/arm/camera/image_raw',
            self.image_callback,
            10)
                
        self.armsub = self.create_subscription(
            Float32MultiArray,
            '/arm/feedback',
            self.arm_position_callback,
            10
        )

        self.lock = threading.Lock()
        self.frame = None

    def image_callback(self, msg):

        np_arr = np.frombuffer(msg.data, dtype=np.uint8)


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

    def arm_position_callback(self, msg):
        arm_pos = msg.data
        self.arm_pos = arm_pos



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

    # Tracker: 0 179 0 255 179 255
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

            #  # ROI
            # x1, y1 = 275, 375
            # x2, y2 = 375, 405
            # roi = hsv[y1:y2, x1:x2]

            # mean_color = np.mean(roi, axis=(0, 1))
            # h,s,v = mean_color

            # node.get_logger().info(f"Hue value: {h}")

            # undistortion
            K = np.array([[914.47640228, 0., 292.27721482],
                        [0., 919.65564668, 251.73568227],
                        [0., 0., 1.]])
            distCoeffs = np.array([[-1.43355018, 1.79907038, -0.01499228, 0.01978075, -0.81396968]])

            h, w = frame.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w,h), 1, (w,h))
            undistorted_frame = cv2.undistort(frame, K, distCoeffs, None, new_K)
            result = undistorted_frame

            h_min = cv2.getTrackbarPos("H Min", "Trackbars")
            h_max = cv2.getTrackbarPos("H Max", "Trackbars")
            s_min = cv2.getTrackbarPos("S Min", "Trackbars")
            s_max = cv2.getTrackbarPos("S Max", "Trackbars")
            v_min = cv2.getTrackbarPos("V Min", "Trackbars")
            v_max = cv2.getTrackbarPos("V Max", "Trackbars")

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            # lower = np.array([0, 0, 210])
            # upper = np.array([255, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)
            undistorted = cv2.undistort(mask, K, distCoeffs, None, new_K)
            # result = cv2.bitwise_and(undistorted_frame, undistorted_frame, mask=mask)

            cv2.imshow("Undistorted", undistorted)
            # cv2.imshow("Original", frame)
            # cv2.imshow("Mask", mask)
            # cv2.imshow("Result", result)

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # # Blur to reduce noise
            # blur_val = cv2.getTrackbarPos("Blur", "Canny Trackbars")
            # if blur_val % 2 == 0:
            #     blur_val += 1
            # if blur_val < 1:
            #     blur_val = 1
            # gray = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)

            # # Canny thresholds
            # threshold1 = cv2.getTrackbarPos("Threshold1", "Canny Trackbars")
            # threshold2 = cv2.getTrackbarPos("Threshold2", "Canny Trackbars")
            # aperture = cv2.getTrackbarPos("Aperture Size", "Canny Trackbars")
            # if aperture % 2 == 0:
            #     aperture += 1
            # if aperture < 3:
            #     aperture = 3
            # elif aperture > 7:
            #     aperture = 7

            # edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture)

            # Contour detection
            contours, _ = cv2.findContours(undistorted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area < 800:  # filter small noise
                        continue

                    rect = cv2.minAreaRect(approx)
                    x_center, y_center = rect[0]
                    w, h = rect[1]
                    yaw = rect[2]
                    aspect_ratio = w / float(h)
                    shape_type = "Square" if 0.4 < aspect_ratio < 1.6 else "Rectangle"
                    if(shape_type == "Square"):

                        # calculate world coordinates
                        # image: 635 x 477
                        x_world = (x_center - 319) * 0.541 / 1000
                        y_world = 0.21 - (y_center - 239) * 0.541 / 1000                    

                        # node.get_logger().info(f"x world coord: {x_world}")
                        # node.get_logger().info(f"y world coord: {y_world}")

                        # pack data
                        p = Pose()
                        p.position.x = float(x_world)
                        p.position.y = float(y_world)
                        p.position.z = 0.165

                        # orientation
                        if yaw <= -45:
                            yaw += 90
                        if yaw >= 45:
                            yaw -= 90

                        node.get_logger().info(f"yaw: {yaw}")
                        p.orientation.w = 1.0

                        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
                        cv2.circle(result, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                        cv2.putText(result, f"{shape_type} {yaw:.1f}", (int(x_center), int(y_center)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Annotated Result", result)   

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()






