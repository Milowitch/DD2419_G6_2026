#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import sensor_msgs_py.point_cloud2 as pc2

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_improved')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # --- 核心发布者 ---
        self._marker_pub = self.create_publisher(MarkerArray, '/perception/markers', 10)
        
        # --- 调试发布者 ---
        self._red_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/red_points', qos_profile)
        self._blue_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/blue_points', qos_profile)
        self._green_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/green_points', qos_profile)
        self._wood_debug_pub = self.create_publisher(PointCloud2, '/perception/debug/wood_points', qos_profile)
        
        # --- 订阅者 ---
        self.create_subscription(PointCloud2, '/realsense/depth/color/points', self.cloud_callback, qos_profile)

        # --- 平滑滤波参数 ---
        # 存储每个类别的上一次位置: {label: np.array([x, y, z])}
        self.last_positions = {}
        self.alpha = 0.8  # 平滑系数 (0.0 到 1.0)。越小越平滑，但延迟越高；越大越灵敏，但抖动多。

        self.get_logger().info("检测节点已启动！已添加绿色检测与位置平滑。")

    def cloud_callback(self, msg: PointCloud2):
        # 1. 解析点云
        pt_data = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        data = np.array(list(pt_data), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        if data.size == 0: return

        coords = np.stack([data['x'], data['y'], data['z']], axis=-1)
        rgb_int = data['rgb'].view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_int & 0xFF).astype(np.uint8)
        colors_rgb = np.stack([r, g, b], axis=-1)

        # 2. 空间过滤 (只保留前方 0.1m 到 1.5m 的点)
        spatial_mask = (coords[:, 2] < 1.5) & (coords[:, 2] > 0.1)
        roi_coords = coords[spatial_mask]
        roi_colors = colors_rgb[spatial_mask]
        if roi_coords.shape[0] < 50: return # 点太少直接跳过

        # 3. 颜色转换 (HSV)
        roi_hsv = cv2.cvtColor(roi_colors.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

        # --- 颜色掩码定义 ---
        masks = {
            "Red":   ((roi_hsv[:, 0] < 10) | (roi_hsv[:, 0] > 170)) & (roi_hsv[:, 1] > 120),
            "Blue":  (roi_hsv[:, 0] >= 100) & (roi_hsv[:, 0] <= 130) & (roi_hsv[:, 1] > 100),
            "Green": (roi_hsv[:, 0] >= 35) & (roi_hsv[:, 0] <= 85) & (roi_hsv[:, 1] > 60), # 新增绿色
            "Wood":  (roi_hsv[:, 0] >= 10) & (roi_hsv[:, 0] <= 25) & (roi_hsv[:, 1] >= 40) & (roi_hsv[:, 2] > 40)
        }

        # 4. 发布调试点云
        self.publish_debug_cloud(roi_coords[masks["Red"]], msg.header, self._red_debug_pub)
        self.publish_debug_cloud(roi_coords[masks["Blue"]], msg.header, self._blue_debug_pub)
        self.publish_debug_cloud(roi_coords[masks["Green"]], msg.header, self._green_debug_pub)
        self.publish_debug_cloud(roi_coords[masks["Wood"]], msg.header, self._wood_debug_pub)

        # 5. 生成并平滑 Marker
        marker_array = MarkerArray()
        # 配置: (label, color_rgb_norm, m_id)
        configs = [
            ("Red", [1.0, 0.0, 0.0], 0),
            ("Blue", [0.0, 0.0, 1.0], 10),
            ("Green", [0.0, 1.0, 0.0], 30),
            ("Wood", [0.6, 0.4, 0.2], 20)
        ]

        for label, color, m_id in configs:
            points = roi_coords[masks[label]]
            # 提高点数阈值，减少误报引起的闪烁
            if len(points) > 100: 
                # 计算当前观测位置
                current_pos = np.median(points, axis=0)

                # --- 核心平滑逻辑 ---
                if label in self.last_positions:
                    # 指数加权移动平均滤波 (Low-pass filter)
                    smoothed_pos = self.alpha * current_pos + (1 - self.alpha) * self.last_positions[label]
                else:
                    smoothed_pos = current_pos
                
                self.last_positions[label] = smoothed_pos
                marker_array.markers.append(self.create_marker(smoothed_pos, msg.header.frame_id, color, m_id, label))
            
        self._marker_pub.publish(marker_array)

    def publish_debug_cloud(self, points, header, pub):
        if len(points) == 0: return
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg = pc2.create_cloud(header, fields, points)
        pub.publish(msg)

    def create_marker(self, pos, frame_id, color, m_id, label):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, pos)
        marker.scale.x = marker.scale.y = marker.scale.z = 0.06 # 稍微加大一点方便观察
        marker.color.r, marker.color.g, marker.color.b = map(float, color)
        marker.color.a = 0.9
        # 增加生存时间，防止因丢帧导致的消失闪烁
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        return marker

def main():
    rclpy.init()
    rclpy.spin(ColorDetectionNode())
    rclpy.shutdown()

if __name__ == '_main_':
    main()