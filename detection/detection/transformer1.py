#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# TF2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs

# Messages
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PointStamped, TransformStamped


class ObjectGlobalTransformer(Node):

    def __init__(self):
        super().__init__('object_global_transformer')

        # ===============================
        # TF2 Setup
        # ===============================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ===============================
        # Workspace Polygon (meters)
        # Converted from cm → meters
        # ===============================
        self.workspace_polygon = [
            (0.00, 0.00),
            (5.22, 0.00),
            (8.00, 2.02),
            (10.01, 2.04),
            (10.00, 4.22),
            (8.60, 4.23),
            (8.59, 2.67),
            (0.00, 2.70)
        ]

        # ===============================
        # Subscriber
        # ===============================
        self.subscription = self.create_subscription(
            MarkerArray,
            '/perception/markers',
            self.marker_callback,
            10
        )

        # ===============================
        # Publisher
        # ===============================
        self.point_pub = self.create_publisher(
            PointStamped,
            '/perception/global_points',
            10
        )

        # ID mapping (must match detection node IDs)
        self.id_to_name = {
            0: "Red",
            1: "Blue",
            2: "Green",
            3: "Wood"
        }

        self.get_logger().info("Workspace-filtered global transformer started.")

    # ============================================================
    # Marker Callback
    # ============================================================

    def marker_callback(self, msg: MarkerArray):

        for marker in msg.markers:

            if marker.action != marker.ADD:
                continue

            color_name = self.id_to_name.get(marker.id, "Unknown")

            point_in_camera = PointStamped()
            point_in_camera.header = marker.header
            point_in_camera.point.x = marker.pose.position.x
            point_in_camera.point.y = marker.pose.position.y
            point_in_camera.point.z = marker.pose.position.z

            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    marker.header.frame_id,
                    rclpy.time.Time()
                )

                point_in_map = tf2_geometry_msgs.do_transform_point(
                    point_in_camera,
                    transform
                )

                x = point_in_map.point.x
                y = point_in_map.point.y

                # ===============================
                # WORKSPACE FILTER
                # ===============================
                if self.is_inside_workspace(x, y):

                    # Publish global point
                    point_in_map.header.frame_id = 'map'
                    self.point_pub.publish(point_in_map)

                    # Broadcast TF frame
                    self.broadcast_object_tf(point_in_map, color_name)

                    self.get_logger().info(
                        f"[INSIDE] {color_name:5} | "
                        f"X: {x:.3f} Y: {y:.3f}"
                    )

                else:
                    self.get_logger().info(
                        f"[OUTSIDE - IGNORED] {color_name:5} | "
                        f"X: {x:.3f} Y: {y:.3f}"
                    )

            except TransformException:
                continue

    # ============================================================
    # Point In Polygon Test (Ray Casting)
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
    # Broadcast TF Frame
    # ============================================================

    def broadcast_object_tf(self, point_stamped, color_name):

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = f'obj_{color_name.lower()}'

        t.transform.translation.x = point_stamped.point.x
        t.transform.translation.y = point_stamped.point.y
        t.transform.translation.z = point_stamped.point.z

        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)


# ============================================================
# Main
# ============================================================

def main():
    rclpy.init()
    node = ObjectGlobalTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()