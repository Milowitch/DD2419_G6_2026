#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

# TF2 相关
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

# 消息类型
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, TransformStamped

class BoxGlobalTransformer(Node):

    def __init__(self):
        super().__init__('box_global_transformer')

        # ===============================
        # TF2 设置
        # ===============================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ===============================
        # 工作区多边形 (米)
        # ===============================
        self.workspace_polygon = [
            (0.00, 0.00), (5.22, 0.00), (8.00, 2.02), (10.01, 2.04),
            (10.00, 4.22), (8.60, 4.23), (8.59, 2.67), (0.00, 2.70)
        ]

        # ===============================
        # 订阅者 (监听你之前代码发布的灰色长方体 Marker)
        # ===============================
        self.subscription = self.create_subscription(
            Marker,
            '/camera/depth/detected_box_marker',
            self.marker_callback,
            10
        )

        # ===============================
        # 发布者 (发布转换后的全局坐标点)
        # ===============================
        self.global_point_pub = self.create_publisher(
            PointStamped,
            '/perception/global_box_point',
            10
        )

        self.get_logger().info("Box Global Transformer 启动：正在将检测到的盒子转换至 map 系...")

    # ============================================================
    # Marker 回调函数
    # ============================================================
    def marker_callback(self, marker: Marker):
        # 仅处理添加操作的 Marker
        if marker.action != Marker.ADD:
            return

        # 准备待转换的点 (来自相机或 odom 系的局部坐标)
        point_local = PointStamped()
        point_local.header = marker.header
        point_local.point.x = marker.pose.position.x
        point_local.point.y = marker.pose.position.y
        point_local.point.z = marker.pose.position.z

        try:
            # 查找从局部系到全局 map 系的转换
            # 使用 rclpy.time.Time() 获取最新的可用变换
            transform = self.tf_buffer.lookup_transform(
                'map',
                marker.header.frame_id,
                rclpy.time.Time()
            )

            # 执行坐标转换
            point_global = tf2_geometry_msgs.do_transform_point(
                point_local,
                transform
            )

            gx = point_global.point.x
            gy = point_global.point.y

            # ===============================
            # 工作区过滤
            # ===============================
            if self.is_inside_workspace(gx, gy):
                # 发布全局 PointStamped
                point_global.header.frame_id = 'map'
                point_global.header.stamp = self.get_clock().now().to_msg()
                self.global_point_pub.publish(point_global)

                # 广播 TF 坐标轴 (显示为名为 'detected_box' 的坐标系)
                self.broadcast_box_tf(point_global)

                self.get_logger().info(f"[区域内] 盒子位置: X={gx:.2f}, Y={gy:.2f}")
            else:
                self.get_logger().debug(f"[区域外] 忽略盒子: X={gx:.2f}, Y={gy:.2f}")

        except (TransformException) as e:
            self.get_logger().warn(f"无法转换坐标: {str(e)}")

    # ============================================================
    # 多边形检测 (射线法)
    # ============================================================
    def is_inside_workspace(self, x, y):
        num = len(self.workspace_polygon)
        j = num - 1
        inside = False
        for i in range(num):
            xi, yi = self.workspace_polygon[i]
            xj, yj = self.workspace_polygon[j]

            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    # ============================================================
    # 广播 TF 坐标系
    # ============================================================
    def broadcast_box_tf(self, point_stamped):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'detected_box_frame'

        t.transform.translation.x = point_stamped.point.x
        t.transform.translation.y = point_stamped.point.y
        t.transform.translation.z = point_stamped.point.z

        # 盒子默认不带旋转，设为单位四元数
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

# ============================================================
# 主函数
# ============================================================
def main():
    rclpy.init()
    node = BoxGlobalTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '_main_':
    main()