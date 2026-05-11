#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import cv2
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2

class FloorCannyNode(Node):
    def __init__(self):
        super().__init__('floor_canny_node')

        # -----------------------------
        # Tunable Parameters
        # -----------------------------
        self.declare_parameter("max_distance", 0.6)   
        self.declare_parameter("height_min", -0.1)    
        self.declare_parameter("height_max", 0.1)     
        self.declare_parameter("voxel_size", 0.002)    
        self.declare_parameter("projection_scale", 100) 
        self.declare_parameter("image_size", 500)       
        self.declare_parameter("ransac_distance_threshold", 0.02)  
        self.declare_parameter("ransac_iterations", 100)  
        self.declare_parameter("canny_threshold1", 5)  # lower threshold
        self.declare_parameter("canny_threshold2", 250) # upper threshold

        self.max_distance = self.get_parameter("max_distance").value
        self.height_min = self.get_parameter("height_min").value
        self.height_max = self.get_parameter("height_max").value
        self.voxel_size = self.get_parameter("voxel_size").value
        self.scale = self.get_parameter("projection_scale").value
        self.img_size = self.get_parameter("image_size").value
        self.ransac_distance_threshold = self.get_parameter("ransac_distance_threshold").value
        self.ransac_iterations = self.get_parameter("ransac_iterations").value
        self.canny_threshold1 = self.get_parameter("canny_threshold1").value
        self.canny_threshold2 = self.get_parameter("canny_threshold2").value

        # Subscriber / Publisher
        self.pc_sub = self.create_subscription(
            PointCloud2, '/realsense/depth/color/points', self.pc_callback, 10
        )
        self.edge_pub = self.create_publisher(
            PointCloud2, '/camera/depth/canny_edges', 10
        )

        self.get_logger().info("FloorCannyNode started")

    def pc_callback(self, msg: PointCloud2):
        points = pc2.read_points_numpy(msg, field_names=("x","y","z"), skip_nans=True)
        if points.shape[0] < 50:
            return

        xyz = points[:, :3]

        # Distance filter
        dist = np.linalg.norm(xyz, axis=1)
        mask_dist = dist < self.max_distance
        xyz = xyz[mask_dist]
        if xyz.shape[0] < 10:
            return

        # Ground removal using RANSAC
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        try:
            plane_model, inliers = cloud.segment_plane(
                distance_threshold=self.ransac_distance_threshold,
                ransac_n=3,
                num_iterations=self.ransac_iterations
            )
            a,b,c,d = plane_model
            normal = np.array([a,b,c])
            normal /= np.linalg.norm(normal)
            if abs(normal[1]) > 0.9:
                non_ground = cloud.select_by_index(inliers, invert=True)
                xyz = np.asarray(non_ground.points)
        except Exception as e:
            self.get_logger().warn(f"RANSAC failed: {e}")
            xyz = np.asarray(cloud.points)

        # Height filter
        mask_height = (xyz[:,1] > self.height_min) & (xyz[:,1] < self.height_max)
        xyz = xyz[mask_height]
        if xyz.shape[0] < 10:
            return

        # Top-down projection x,z -> 2D image
        pts_2d = (xyz[:, [0,2]] * self.scale).astype(np.int32)
        pts_2d[:,0] += self.img_size // 2
        pts_2d[:,1] += self.img_size // 2
        pts_2d = np.clip(pts_2d, 0, self.img_size-1)

        img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img[pts_2d[:,1], pts_2d[:,0]] = 255

        # Apply Canny edge detection
        edges = cv2.Canny(img, self.canny_threshold1, self.canny_threshold2)

        # Extract points corresponding to edges
        edge_indices = np.argwhere(edges > 0)
        if edge_indices.shape[0] == 0:
            return

        edge_points = []
        for pix_y, pix_x in edge_indices:
            idx = np.argmin(np.linalg.norm(pts_2d - [pix_x, pix_y], axis=1))
            edge_points.append(xyz[idx])

        # Downsample
        cloud_edge = o3d.geometry.PointCloud()
        cloud_edge.points = o3d.utility.Vector3dVector(np.array(edge_points))
        cloud_edge = cloud_edge.voxel_down_sample(self.voxel_size)
        down_pts = np.asarray(cloud_edge.points)

        # Publish edge point cloud
        edge_msg = self.numpy_to_pointcloud2(down_pts, msg.header.stamp, msg.header.frame_id)
        self.edge_pub.publish(edge_msg)

    def numpy_to_pointcloud2(self, points, stamp, frame_id):
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, points.tolist())


def main():
    rclpy.init()
    node = FloorCannyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()