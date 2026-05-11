#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
import sensor_msgs_py.point_cloud2 as pc2


class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_final')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=5
        )

        # --- Publishers ---
        self._marker_pub = self.create_publisher(
            MarkerArray,
            '/perception/markers',
            qos
        )

        # NEW: separate nearcube topic
        self._near_cube_pub = self.create_publisher(
            Marker,
            '/perception/nearcube',
            qos
        )

        self._red_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/red_points', qos)
        self._blue_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/blue_points', qos)
        self._wood_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/wood_points', qos)
        self._green_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/green_points', qos)

        # --- State storage ---
        self.last_positions = {"Red": None, "Blue": None, "Green": None, "Wood": None}

        # --- Stability parameters ---
        self.alpha = 0.98 #0.8
        self.min_point_count = 100

        # --- Subscriber ---
        self.create_subscription(
            PointCloud2,
            '/realsense/depth/color/points',
            self.cloud_callback,
            qos
        )

        self.get_logger().info("Color Detection Node started (Height filter: Y-axis 0.1cm to 10cm).")

    def cloud_callback(self, msg: PointCloud2):

        pt_data = pc2.read_points(
            msg,
            field_names=("x", "y", "z", "rgb"),
            skip_nans=True
        )

        data = np.array(list(pt_data),
                        dtype=[('x', 'f4'),
                               ('y', 'f4'),
                               ('z', 'f4'),
                               ('rgb', 'f4')])

        if data.size == 0:
            return

        coords = np.stack([data['x'], data['y'], data['z']], axis=-1)

        rgb_int = data['rgb'].view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_int & 0xFF).astype(np.uint8)
        colors_rgb = np.stack([r, g, b], axis=-1)

        # --- 修改部分：高度滤波器 (Y轴) ---
        # 0.1cm = 0.001m, 10cm = 0.1m
        # 假设相机 Y 轴指向下方，或根据你的坐标系定义控制高低范围
        y_min = 0.001
        y_max = 0.1
        
        # 过滤 Y 轴坐标在 0.1cm 到 10cm 之间的点
        spatial_mask = (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        
        # 保留一个合理的深度 Z 范围（例如 1米内），避免远距离噪声干扰
        spatial_mask &= (coords[:, 2] < 4.0) & (coords[:, 2] > 0.05)

        roi_coords = coords[spatial_mask]
        roi_colors = colors_rgb[spatial_mask]

        if roi_coords.shape[0] == 0:
            return

        roi_hsv = cv2.cvtColor(
            roi_colors.reshape(-1, 1, 3),
            cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)

        # --- Color masks ---
        red_mask = ((roi_hsv[:, 0] <= 15) | (roi_hsv[:, 0] >= 165)) & \
                    (roi_hsv[:, 1] > 150) & (roi_hsv[:, 2] > 50)

        blue_mask = (roi_hsv[:, 0] >= 98) & (roi_hsv[:, 0] <= 113) & \
                     (roi_hsv[:, 1] >= 104) & (roi_hsv[:, 2] >= 87)

        green_mask = (roi_hsv[:, 0] >= 67) & (roi_hsv[:, 0] <= 96) & \
                      (roi_hsv[:, 1] >= 101) & (roi_hsv[:, 2] >= 54)

        # wood_mask = (roi_hsv[:, 0] >= 10) & (roi_hsv[:, 0] <= 28) & \
        #              (roi_hsv[:, 1] >= 80) & (roi_hsv[:, 2] >= 50)

        # --- Debug clouds ---
        self.publish_debug_cloud(roi_coords[red_mask], msg.header, self._red_debug_pub)
        self.publish_debug_cloud(roi_coords[blue_mask], msg.header, self._blue_debug_pub)
        self.publish_debug_cloud(roi_coords[green_mask], msg.header, self._green_debug_pub)
        # self.publish_debug_cloud(roi_coords[wood_mask], msg.header, self._wood_debug_pub)

        marker_array = MarkerArray()

        nearest_pos = None
        min_dist = float('inf')

        tasks = [
            (red_mask, "Red", [1, 0, 0], 0),
            (blue_mask, "Blue", [0, 0, 1], 10),
            (green_mask, "Green", [0, 1, 0], 20),
            # (wood_mask, "Wood", [0.6, 0.4, 0.2], 30)
        ]

        tasks = [
            (red_mask, "Red", [1, 0, 0], 0),
            (blue_mask, "Blue", [0, 0, 1], 10),
            (green_mask, "Green", [0, 1, 0], 20),
        ]
        for mask, label, color, m_id in tasks:

            points = roi_coords[mask]

            if len(points) > self.min_point_count:

                current_pos = np.median(points, axis=0)

                if self.last_positions[label] is None:
                    self.last_positions[label] = current_pos
                else:
                    self.last_positions[label] = \
                        (1 - self.alpha) * self.last_positions[label] + \
                        self.alpha * current_pos

                smoothed_pos = self.last_positions[label]

                # Track nearest cube
                dist = np.linalg.norm(smoothed_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = smoothed_pos

                marker_array.markers.append(
                    self.create_marker(smoothed_pos,
                                       msg.header.frame_id,
                                       color,
                                       m_id)
                )
            else:
                self.last_positions[label] = None

        # Publish all cubes
        self._marker_pub.publish(marker_array)

        # Publish nearest cube separately
        if nearest_pos is not None:
            near_marker = self.create_marker(
                nearest_pos,
                msg.header.frame_id,
                [1.0, 1.0, 0.0],  # yellow
                999
            )
            self._near_cube_pub.publish(near_marker)

    def publish_debug_cloud(self, points, header, pub):
        if len(points) == 0:
            return

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, points)
        pub.publish(cloud_msg)

    def create_marker(self, pos, frame_id, color, m_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0

        marker.lifetime = rclpy.duration.Duration(seconds=0.15).to_msg()

        return marker


def main():
    rclpy.init()
    node = ColorDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '_main_':
    main()