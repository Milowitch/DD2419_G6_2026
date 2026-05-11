#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import struct
import math
import cv2

import tf2_ros
from tf2_geometry_msgs import do_transform_point

class BoxOnlyVisualizerNode(Node):
    def __init__(self):
        super().__init__('box_only_visualizer_node')

        # --- 基础订阅与发布 ---
        self.pc_sub = self.create_subscription(PointCloud2, '/realsense/depth/color/points', self.pc_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.target_pub = self.create_publisher(PointStamped, '/camera/depth/target_point_odom', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- RViz 只显示盒子的发布者 ---
        # 仅发布识别到的“盒子”点云
        self.viz_box_pub = self.create_publisher(PointCloud2, '/camera/depth/box_only', 10)
        # 目标停靠点标记
        self.viz_target_marker_pub = self.create_publisher(MarkerArray, '/camera/depth/target_marker', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- 参数设置 ---
        self.max_distance = 1.8
        self.height_min, self.height_max = -0.05, 0.5
        self.voxel_size = 0.01
        self.distance_threshold = 0.02
        self.eps, self.min_points = 0.08, 30

        # HSV 颜色范围 (绿色盒子示例)
        self.lower_hsv = np.array([43, 0, 56])
        self.upper_hsv = np.array([119, 61, 94])
        self.color_ratio_thresh = 0.3 

        self.robot_pose = None        
        self.target_point_odom = None 
        self.target_published = False 
        
        # 工作区多边形 (odom 系)
        self.workspace_polygon = [
            (0.00, 0.00), (5.22, 0.00), (8.00, 2.02), (10.01, 2.04),
            (10.00, 4.22), (8.60, 4.23), (8.59, 2.67), (0.00, 2.70)
        ]

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("只显示盒子模式已启动")

    def is_in_polygon(self, x, y):
        inside = False
        n = len(self.workspace_polygon)
        p1x, p1y = self.workspace_polygon[0]
        for i in range(n + 1):
            p2x, p2y = self.workspace_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def pc_callback(self, msg: PointCloud2):
        if self.target_published or self.robot_pose is None:
            return

        # 1. 读取并预处理
        points = pc2.read_points_numpy(msg, field_names=("x","y","z","rgb"), skip_nans=True)
        if points.shape[0] < self.min_points: return

        xyz = points[:, :3]
        rgb_uint32 = points[:, 3].astype(np.float32).view(np.uint32)
        rgb_norm = np.stack([((rgb_uint32 >> 16) & 255) / 255.0, 
                             ((rgb_uint32 >> 8) & 255) / 255.0, 
                             (rgb_uint32 & 255) / 255.0], axis=1)

        mask = np.linalg.norm(xyz, axis=1) < self.max_distance
        xyz, rgb_norm = xyz[mask], rgb_norm[mask]

        # 2. Open3D 过滤
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_norm)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        _, inliers = pcd.segment_plane(self.distance_threshold, 3, 500)
        pcd_no_ground = pcd.select_by_index(inliers, invert=True)
        
        pts = np.asarray(pcd_no_ground.points)
        clr = np.asarray(pcd_no_ground.colors)
        h_mask = (pts[:, 1] > self.height_min) & (pts[:, 1] < self.height_max)
        pts, clr = pts[h_mask], clr[h_mask]

        if len(pts) < self.min_points: return

        # 3. 聚类并排序
        labels = np.array(pcd_no_ground.select_by_index(np.where(h_mask)[0]).cluster_dbscan(self.eps, self.min_points))
        if labels.max() < 0: return

        sorted_lbls = sorted(np.arange(labels.max() + 1), key=lambda i: np.sum(labels == i), reverse=True)

        # 4. 筛选盒子
        try:
            trans = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, msg.header.stamp, 
                                                  timeout=rclpy.duration.Duration(seconds=0.1))
        except: return

        clr_u8 = (clr * 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv_all = cv2.cvtColor(clr_u8, cv2.COLOR_RGB2HSV)

        for lbl in sorted_lbls:
            idx = (labels == lbl)
            c_pts = pts[idx]
            
            # 位置检查
            centroid = np.mean(c_pts, axis=0)
            ps = PointStamped(header=msg.header)
            ps.point.x, ps.point.y, ps.point.z = float(centroid[0]), float(centroid[1]), float(centroid[2])
            p_odom = do_transform_point(ps, trans)
            if not self.is_in_polygon(p_odom.point.x, p_odom.point.y): continue

            # 颜色检查
            mask_hsv = cv2.inRange(hsv_all[idx], self.lower_hsv, self.upper_hsv)
            if (np.count_nonzero(mask_hsv) / len(c_pts)) > self.color_ratio_thresh:
                # --- 核心：只发布这个盒子的点云到 RViz ---
                self.publish_box_only(c_pts, msg.header)
                
                self.calculate_target_and_publish_marker(p_odom)
                self.target_published = True
                break

    def publish_box_only(self, box_pts, header):
        """将识别到的盒子点云发布为纯绿色，方便 RViz 查看"""
        # 绿色编码
        green_val = struct.unpack('f', struct.pack('i', (0 << 16) | (255 << 8) | 0))[0]
        rgb_column = np.full((box_pts.shape[0], 1), green_val, dtype=np.float32)
        cloud_data = np.hstack((box_pts.astype(np.float32), rgb_column)).tobytes()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        box_msg = PointCloud2(header=header, height=1, width=box_pts.shape[0], is_bigendian=False,
                              point_step=16, row_step=16 * box_pts.shape[0], fields=fields,
                              is_dense=True, data=cloud_data)
        self.viz_box_pub.publish(box_msg)

    def calculate_target_and_publish_marker(self, p_odom):
        """计算目标点并发布标记"""
        xr, yr, _ = self.robot_pose
        dx, dy = p_odom.point.x - xr, p_odom.point.y - yr
        dist = math.hypot(dx, dy)
        ux, uy = dx/dist, dy/dist 
        tx, ty = p_odom.point.x - 0.35 * ux, p_odom.point.y - 0.35 * uy

        self.target_point_odom = PointStamped(header=Header(frame_id='odom', stamp=self.get_clock().now().to_msg()))
        self.target_point_odom.point.x, self.target_point_odom.point.y = tx, ty
        self.target_pub.publish(self.target_point_odom)

        # RViz 视觉反馈
        ma = MarkerArray()
        m = Marker(header=self.target_point_odom.header, type=Marker.SPHERE, action=Marker.ADD, id=0)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = tx, ty, p_odom.point.z
        m.scale.x = m.scale.y = m.scale.z = 0.1
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        ma.markers.append(m)
        self.viz_target_marker_pub.publish(ma)

    def control_loop(self):
        if not self.target_point_odom or not self.robot_pose: return
        x, y, yaw = self.robot_pose
        tx, ty = self.target_point_odom.point.x, self.target_point_odom.point.y
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        alpha = self.normalize_angle(math.atan2(dy, dx) - yaw)
        cmd = Twist()
        if dist < 0.05:
            self.target_point_odom = None 
        elif abs(alpha) > self.rotate_threshold:
            cmd.angular.z = math.copysign(self.w_max, alpha)
        else:
            cmd.linear.x = max(self.v_min, min(self.v_max, 0.7 * dist) * math.exp(-5.0 * alpha**2))
            cmd.angular.z = math.copysign(min(self.w_max, 9.0 * abs(alpha)), alpha)
        self.cmd_pub.publish(cmd)

    @staticmethod
    def normalize_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

def main():
    rclpy.init()
    node = BoxOnlyVisualizerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '_main_':
    main()
