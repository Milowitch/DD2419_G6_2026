#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
import time


class MaskVisualizer(Node):

    def __init__(self):
        super().__init__('mask_visualizer')

        self.depth_sub = self.create_subscription(
            Image,
            '/realsense/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/realsense/aligned_depth_to_color/camera_info',
            self.info_callback,
            10)

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_depth = None
        self.mask_to_show = None
        self.lock = threading.Lock()

        # Processing thread
        self.processing_thread = threading.Thread(
            target=self.compute_mask_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info("Mask Visualizer with Floor Rejection Started")

    # ---------------------------------------------------------
    def info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # ---------------------------------------------------------
    def depth_callback(self, msg):
        if self.fx is None:
            return

        depth = np.frombuffer(msg.data, dtype=np.uint16)
        depth = depth.astype(np.float32) / 1000.0
        depth = depth.reshape((msg.height, msg.width))

        with self.lock:
            self.latest_depth = depth

    # ---------------------------------------------------------
    def compute_mask_loop(self):

        while rclpy.ok():

            if self.latest_depth is None:
                time.sleep(0.01)
                continue

            with self.lock:
                depth = self.latest_depth.copy()

            h, w = depth.shape

            # -------------------------------------------------
            # 1️⃣ Estimate floor depth from bottom of image
            # -------------------------------------------------
            floor_region = depth[h-40:h, :]
            floor_valid = floor_region[floor_region > 0.1]

            if floor_valid.size > 50:
                floor_depth = np.median(floor_valid)
            else:
                floor_depth = None

            # -------------------------------------------------
            # 2️⃣ Distance mask
            # -------------------------------------------------
            depth_mask = (depth > 0.12) & (depth < 0.4)

            # Back-project to compute height (Y)
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            X = (u - self.cx) * depth / self.fx
            Y = (v - self.cy) * depth / self.fy

            # -------------------------------------------------
            # 3️⃣ Height mask
            # -------------------------------------------------
            height_mask = (Y > 0.03) & (Y < 0.10)

            mask = depth_mask & height_mask
            mask_img = (mask.astype(np.uint8) * 255)

            # -------------------------------------------------
            # 4️⃣ Morphological cleanup
            # -------------------------------------------------
            kernel = np.ones((5, 5), np.uint8)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

            # -------------------------------------------------
            # 5️⃣ Contour detection
            # -------------------------------------------------
            contours, _ = cv2.findContours(
                mask_img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            contour_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

            if contours:
                largest = max(contours, key=cv2.contourArea)

                if cv2.contourArea(largest) > 1000:

                    x, y, w_box, h_box = cv2.boundingRect(largest)

                    # Use median depth inside bounding box (robust)
                    roi = depth[y:y+h_box, x:x+w_box]
                    roi_valid = roi[roi > 0]

                    if roi_valid.size > 50:
                        center_depth = np.median(roi_valid)
                    else:
                        continue

                    # -------------------------------------------------
                    # 6️⃣ Reject if too close to floor
                    # -------------------------------------------------
                    if floor_depth is not None:
                        depth_difference = abs(center_depth - floor_depth)

                        if depth_difference < 0.01:
                            # Too close to floor → ignore
                            continue

                    # Draw contour + bounding box
                    cv2.drawContours(contour_img, [largest], -1,
                                     (0, 255, 0), 2)

                    cv2.rectangle(contour_img,
                                  (x, y),
                                  (x + w_box, y + h_box),
                                  (0, 0, 255), 2)

                    # -------------------------------------------------
                    # 7️⃣ Compute 3D center
                    # -------------------------------------------------
                    center_u = x + w_box // 2
                    center_v = y + h_box // 2

                    Xc = (center_u - self.cx) * center_depth / self.fx
                    Yc = (center_v - self.cy) * center_depth / self.fy
                    Zc = center_depth

                    self.get_logger().info(
                        f"Box center 3D -> "
                        f"X: {Xc:.3f} m, "
                        f"Y: {Yc:.3f} m, "
                        f"Z: {Zc:.3f} m "
                        f"| Height above floor: "
                        f"{abs(center_depth - floor_depth):.3f} m"
                        if floor_depth is not None else "")

            display_img = cv2.resize(contour_img, (640, 360))

            with self.lock:
                self.mask_to_show = display_img

            time.sleep(0.02)

    # ---------------------------------------------------------
    def show_mask(self):
        while rclpy.ok():
            img = None
            with self.lock:
                if self.mask_to_show is not None:
                    img = self.mask_to_show.copy()

            if img is not None:
                cv2.imshow("Mask + Floor Rejection", img)

            key = cv2.waitKey(1)
            if key == 27:
                break

            time.sleep(0.01)

        cv2.destroyAllWindows()


# ---------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    node = MaskVisualizer()

    display_thread = threading.Thread(target=node.show_mask)
    display_thread.start()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()