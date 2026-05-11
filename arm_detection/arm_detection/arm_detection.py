# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import numpy as np
# import cv2
# import threading

# from std_msgs.msg import Float32MultiArray
# from geometry_msgs.msg import Pose


# class ShapeDetector(Node):

#     def _init_(self):
#         super()._init_('shape_detector_threaded')

#         self.subscription = self.create_subscription(
#             Image,
#             '/arm/camera/image_raw',
#             self.image_callback,
#             10)
        
#         self.subscription = self.create_subscription(
#             Float32MultiArray,
#             '/arm/feedback',
#             self.arm_position_callback,
#             10
#         )

#         self.lock = threading.Lock()
#         self.frame = None

#     def image_callback(self, msg):
#         np_arr = np.frombuffer(msg.data, dtype=np.uint8)
#         frame = np_arr.reshape((msg.height, msg.width, 3))

#         if msg.encoding == "rgb8":
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         with self.lock:
#             self.frame = frame.copy()

#     def arm_position_callback(self, msg):
#         arm_pos = msg.data
#         self.arm_pos = arm_pos



# def ros_spin(node):
#     rclpy.spin(node)


# def main():
#     rclpy.init()
#     node = ShapeDetector()

#     # Start ROS spinning in separate thread
#     ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
#     ros_thread.start()

#     # Create GUI in main thread
#     cv2.namedWindow("Canny Trackbars")
#     cv2.createTrackbar("Threshold1", "Canny Trackbars", 50, 500, lambda x: None)
#     cv2.createTrackbar("Threshold2", "Canny Trackbars", 150, 500, lambda x: None)
#     cv2.createTrackbar("Aperture Size", "Canny Trackbars", 3, 7, lambda x: None)
#     cv2.createTrackbar("Blur", "Canny Trackbars", 5, 20, lambda x: None)  # kernel size for Gaussian blur

#     while rclpy.ok():
#         frame = None
#         with node.lock:
#             if node.frame is not None:
#                 frame = node.frame.copy()

#         if frame is not None:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Blur to reduce noise
#             blur_val = cv2.getTrackbarPos("Blur", "Canny Trackbars")
#             if blur_val % 2 == 0:
#                 blur_val += 1
#             if blur_val < 1:
#                 blur_val = 1
#             gray = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)

#             # Canny thresholds
#             threshold1 = cv2.getTrackbarPos("Threshold1", "Canny Trackbars")
#             threshold2 = cv2.getTrackbarPos("Threshold2", "Canny Trackbars")
#             aperture = cv2.getTrackbarPos("Aperture Size", "Canny Trackbars")
#             if aperture % 2 == 0:
#                 aperture += 1
#             if aperture < 3:
#                 aperture = 3
#             elif aperture > 7:
#                 aperture = 7

#             edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture)

#             # Contour detection
#             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             for cnt in contours:
#                 epsilon = 0.02 * cv2.arcLength(cnt, True)
#                 approx = cv2.approxPolyDP(cnt, epsilon, True)

#                 if len(approx) == 4:
#                     area = cv2.contourArea(approx)
#                     if area > 1000:  # filter small noise
#                         rect = cv2.minAreaRect(approx)
#                         x_center, y_center = rect[0]
#                         w, h = rect[1]
#                         yaw = rect[2]
#                         aspect_ratio = w / float(h)
#                         shape_type = "Square" if 0.8 < aspect_ratio < 1.2 else "Rectangle"
#                         # if(shape_type == "Square"):

#                         # calculate world coordinates
#                         x_world = x_center
#                         y_world = y_center
#                         z_world = 0.15

#                         # # Sending coordinates                       
#                         # p = Pose()
#                         # p.position.x = x_world
#                         # p.position.y = y_world
#                         # p.position.z = z_world
#                         # # optionally, set orientation if you know it
#                         # p.orientation.x = 0
#                         # p.orientation.y = 0
#                         # p.orientation.z = 0
#                         # p.orientation.w = 1

#                         cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
#                         cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
#                         cv2.putText(frame, f"{shape_type} {yaw:.1f}", (int(x_center), int(y_center)-10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#             cv2.imshow("Original with Shapes", frame)
#             cv2.imshow("Edges", edges)          

#         if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
#             break

#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()


# if _name_ == '_main_':
#     main()


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import threading
import time

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
import math
from std_msgs.msg import Bool

from geometry_msgs.msg import Pose

class HSVTuner(Node):

    def __init__(self):
        super().__init__('hsv_realsense_threaded')
        # self.arm_pos = None

        # Subscribers
        self.img_sub = self.create_subscription(Image, '/arm/camera/image_raw', self.image_callback, 10)     
        self.detection_ready_sub = self.create_subscription(Bool, '/grip/detection_ready', self.detection_callback, 10)
        # self.arm_sub = self.create_subscription(Float32MultiArray, '/arm/feedback', self.arm_position_callback, 10)

        # Publishers
        self.grip_grasp_pub = self.create_publisher(Pose, '/grip/coords', 10)

        self.lock = threading.Lock()
        self.debug_annotated = None
        self.debug_mask = None
        self.debug_undistorted = None
        self.detecting = False
        self.frame = None
        self.frame_id = 0

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
            self.frame_id += 1

    def arm_position_callback(self, msg):
        arm_pos = msg.data
        self.arm_pos = arm_pos
        self.get_logger().info(f"x inverse coord: {arm_pos}")

    def detection_callback(self, msg: Bool):
        if not msg.data:
            return
        
        if self.detecting:
            self.get_logger().warn("Detection already running, skipping trigger")
            return

        self.detecting = True
    
        self.get_logger().info(f"Detection ready received: {msg.data}")
        threading.Thread(
            target=self.detect_w_timeout, # detect with timeout
            daemon=True
        ).start()

    # Detection
    def detect(self, frame):
        p = None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # undistortion
        K = np.array([[914.47640228, 0., 292.27721482],
                    [0., 919.65564668, 251.73568227],
                    [0., 0., 1.]])
        distCoeffs = np.array([[-1.43355018, 1.79907038, -0.01499228, 0.01978075, -0.81396968]])

        h, w = frame.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w,h), 1, (w,h))
        undistorted_frame = cv2.undistort(frame, K, distCoeffs, None, new_K)
        result = undistorted_frame

        lower = np.array([68, 30, 110])
        upper = np.array([171, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        undistorted = cv2.undistort(mask, K, distCoeffs, None, new_K)
        # result = cv2.bitwise_and(undistorted_frame, undistorted_frame, mask=mask)

        # cv2.imshow("Undistorted", undistorted)
        # cv2.imshow("Original", frame)
        # cv2.imshow("Mask", mask)
    
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

                    self.get_logger().info(f"x world coord: {x_world}")
                    self.get_logger().info(f"y world coord: {y_world}")

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

                    self.get_logger().info(f"yaw: {yaw}")
                    p.orientation.w = yaw

                    # cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
                    # cv2.circle(result, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                    # cv2.putText(result, f"{shape_type} {yaw:.1f}", (int(x_center), int(y_center)-10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    # cv2.imshow("Annotated Result", result)
        with self.lock:
            self.debug_annotated = result.copy()
            self.debug_mask = undistorted.copy()
            self.debug_undistorted = undistorted.copy()
        return p
    
    def detect_w_timeout(self):
        start_time = time.time()
        timeout = 1.0
        last_frame_id = -1

        while time.time() - start_time < timeout:
            frame = None
            with self.lock:
                if self.frame is not None:
                    frame = self.frame.copy()
                    frame_id = self.frame_id

            if frame is None:
                self.get_logger().warn("No frame available")
                time.sleep(0.001)
                continue

            if frame_id == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = frame_id

            pose = self.detect(frame)

            if pose is not None:
                self.grip_grasp_pub.publish(pose)
                self.get_logger().info(f"Detection successful")
                self.detecting = False
                return  # cube found

            time.sleep(0.01)

        self.get_logger().warn("No object detected after 1 seconds")
        p = Pose()
        p.position.z = -1.0   # no cube detected
        self.grip_grasp_pub.publish(p)
        self.detecting = False
        return

        # frame = None

        # with self.lock:
        #     if self.frame is not None:
        #         frame = self.frame.copy()

        # if frame is None:
        #     self.get_logger().warn("No frame available")
        #     return

        # pose = self.detect(frame)

        # if pose is not None:
        #     self.grip_grasp_pub.publish(pose)
        #     self.get_logger().info("Published grasp pose")
        # else:
        #     self.get_logger().warn("No object detected")



def ros_spin(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = HSVTuner()

    # Start ROS spinning in separate thread
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    while rclpy.ok():
        time.sleep(0.01)
        # annotated = None
        # mask = None
        # undist = None

        # with node.lock:
        #     if node.debug_annotated is not None:
        #         annotated = node.debug_annotated.copy()
        #     if node.debug_mask is not None:
        #         mask = node.debug_mask.copy()
        #     if node.debug_undistorted is not None:
        #         undist = node.debug_undistorted.copy()

        # if annotated is not None:
        #     cv2.imshow("Annotated Result", annotated)
        # if mask is not None:
        #     cv2.imshow("Mask", mask)
        # if undist is not None:
        #     cv2.imshow("Undistorted", undist)

if __name__ == "__main__":
    main()