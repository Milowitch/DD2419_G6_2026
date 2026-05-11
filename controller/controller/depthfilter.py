#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import struct
import random

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class GroundDBSCAN(Node):

    def __init__(self):
        super().__init__('ground_dbscan')

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            '/realsense/depth/color/points',
            self.cloud_callback,
            10
        )

        # Publishers
        self.cluster_pub = self.create_publisher(
            PointCloud2,
            '/camera/depth/clusters',
            10
        )
        self.centroid_pub = self.create_publisher(
            PointStamped,
            '/camera/depth/cluster_centroid',
            10
        )
        self.target_pub = self.create_publisher(
            PointStamped,
            '/camera/depth/target_20cm_ahead',
            10
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Parameters (debug-friendly)
        self.min_distance = 0.0
        self.max_distance = 0.6
        self.height_min = -0.05
        self.height_max = 0.5
        self.voxel_size = 0.02

        self.distance_threshold = 0.02  # RANSAC plane
        self.eps = 0.08                # DBSCAN
        self.min_points = 20            # DBSCAN

        self.get_logger().info("Ground + DBSCAN + 20cm Offset Node Started")

    # -----------------------------
    def cloud_callback(self, msg: PointCloud2):

        # Convert ROS → NumPy
        points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)

        if points.shape[0] < 50:
            return

        xyz = points[:, :3]

        # Distance filter
        dist = np.linalg.norm(xyz, axis=1)
        mask = (dist > self.min_distance) & (dist < self.max_distance)
        xyz = xyz[mask]

        if xyz.shape[0] < 50:
            return

        # Open3D cloud + voxel downsample
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud = cloud.voxel_down_sample(self.voxel_size)

        # RANSAC ground removal
        plane_model, inliers = cloud.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=4,
            num_iterations=1000
        )
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)

        # Only remove horizontal plane
        if abs(normal[1]) > 0.9:
            non_ground_cloud = cloud.select_by_index(inliers, invert=True)
        else:
            non_ground_cloud = cloud

        filtered_xyz = np.asarray(non_ground_cloud.points)
        if filtered_xyz.shape[0] < 20:
            return

        # Height filter
        height_mask = (filtered_xyz[:, 1] > self.height_min) & (filtered_xyz[:, 1] < self.height_max)
        filtered_xyz = filtered_xyz[height_mask]
        if filtered_xyz.shape[0] < 20:
            return

        # -----------------------------
        # DBSCAN clustering
        # -----------------------------
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(filtered_xyz)

        labels = np.array(cluster_cloud.cluster_dbscan(
            eps=self.eps,
            min_points=self.min_points,
            print_progress=False
        ))
        if labels.max() < 0:
            # No clusters detected
            return

        # -----------------------------
        # Largest cluster (single box assumption)
        # -----------------------------
        largest_cluster = 0
        largest_size = 0
        for i in range(labels.max() + 1):
            size = np.sum(labels == i)
            if size > largest_size:
                largest_size = size
                largest_cluster = i

        cluster_pts = filtered_xyz[labels == largest_cluster]
        if cluster_pts.shape[0] < 5:
            return

        centroid = np.mean(cluster_pts, axis=0)
        distance = np.linalg.norm(centroid)

        # -----------------------------
        # Transform to robot frame (base_link) if available
        # -----------------------------
        p = PointStamped()
        p.header.frame_id = msg.header.frame_id
        p.header.stamp = msg.header.stamp
        p.point.x = float(centroid[0])
        p.point.y = float(centroid[1])
        p.point.z = float(centroid[2])

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            p_base = do_transform_point(p, transform)
            x_r = p_base.point.x
            y_r = p_base.point.y
            z_r = p_base.point.z
            distance = np.linalg.norm([x_r, y_r, z_r])

            self.get_logger().info(
                f"BOX POSITION (robot frame): X={x_r:.3f} Y={y_r:.3f} Z={z_r:.3f} Dist={distance:.3f}"
            )
            self.centroid_pub.publish(p_base)

            # -----------------------------
            # 20 cm ahead target point (along +X robot frame)
            # -----------------------------
            target = PointStamped()
            target.header.frame_id = "base_link"
            target.header.stamp = msg.header.stamp
            target.point.x = x_r + 0.2  # 20 cm ahead
            target.point.y = y_r
            target.point.z = z_r
            self.target_pub.publish(target)

        except Exception as e:
            # Fallback: print in camera frame
            self.get_logger().info(
                f"[Camera frame] BOX centroid: X={centroid[0]:.3f} "
                f"Y={centroid[1]:.3f} Z={centroid[2]:.3f} Dist={distance:.3f} "
                f"(TF not ready)"
            )

            # Publish 20 cm ahead in camera frame
            target = PointStamped()
            target.header.frame_id = msg.header.frame_id
            target.header.stamp = msg.header.stamp
            target.point.x = centroid[0] + 0.2
            target.point.y = centroid[1]
            target.point.z = centroid[2]
            self.target_pub.publish(target)

        # -----------------------------
        # Publish colored cluster cloud
        # -----------------------------
        colors = np.zeros((filtered_xyz.shape[0], 3))
        colors[labels == largest_cluster] = [0.0, 1.0, 0.0]  # green for box
        colors[labels != largest_cluster] = [1.0, 0.0, 0.0]  # red for others/noise

        cluster_msg = self.numpy_to_pointcloud2_rgb(filtered_xyz, colors, msg.header.stamp, msg.header.frame_id)
        self.cluster_pub.publish(cluster_msg)

    # -----------------------------
    def numpy_to_pointcloud2_rgb(self, points, colors, stamp, frame_id):

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        cloud_data = []
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r = int(colors[i][0] * 255)
            g = int(colors[i][1] * 255)
            b = int(colors[i][2] * 255)
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            cloud_data.append([x, y, z, rgb])

        return pc2.create_cloud(header, fields, cloud_data)


def main():
    rclpy.init()
    node = GroundDBSCAN()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()