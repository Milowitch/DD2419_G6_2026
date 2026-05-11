#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
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
        
        # 1. 识别到的盒子：显示为灰色长方体 (Marker)
        self.box_marker_pub = self.create_publisher(Marker, '/camera/depth/detected_box_marker', 10)
        # 1b. 最近的盒子：显示为蓝色长方体 (Marker)
        self.near_box_marker_pub = self.create_publisher(Marker, '/camera/depth/detected_near_box_marker', 10)
        # 2. 目标点：逻辑位置与视觉球体
        self.target_pub = self.create_publisher(PointStamped, '/camera/depth/target_point_odom', 10)
        self.viz_target_marker_pub = self.create_publisher(MarkerArray, '/camera/depth/target_marker', 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_velA', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- 算法参数 ---
        self.max_distance = 2.0
        self.height_min, self.height_max = 0.05, 0.5
        self.voxel_size = 0.02
        self.distance_threshold = 0.02
        self.eps, self.min_points = 0.08, 30

        # HSV 颜色范围 (绿色盒子)
        self.lower_hsv = np.array([43, 0, 56])
        self.upper_hsv = np.array([119, 61, 94])
        self.color_ratio_thresh = 0.3 

        # --- 控制参数 (修复 AttributeError 关键点) ---
        self.v_max = 0.2              # 最大线速度
        self.v_min = 0.05             # 最小线速度
        self.w_max = 0.6              # 最大角速度
        self.rotate_threshold = 0.15  # 转向阈值 (弧度)

        self.robot_pose = None        
        self.target_point_odom = None 
        
        # --- 防止重复发布 ---
        self.last_detected_box_id = None  # 记录上一帧检测到的盒子ID
        self.detection_stable_count = 0   # 稳定帧数计数
        self.stable_threshold = 3         # 需要连续检测N帧才认为是同一个盒子
        
        # 工作区多边形
        self.workspace_polygon = [
            (0.00, 0.00), (5.22, 0.00), (8.00, 2.02), (10.01, 2.04),
            (10.00, 4.22), (8.60, 4.23), (8.59, 2.67), (0.00, 2.70)
        ]

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("只显示灰色盒子模式已启动")
        self.get_logger().info(f"绿色Box HSV范围 -> H: [{self.lower_hsv[0]}-{self.upper_hsv[0]}], "
                               f"S: [{self.lower_hsv[1]}-{self.upper_hsv[1]}], "
                               f"V: [{self.lower_hsv[2]}-{self.upper_hsv[2]}]")

    def is_in_polygon(self, x, y):
        inside = False
        n = len(self.workspace_polygon)
        p1x, p1y = self.workspace_polygon[0]
        for i in range(n + 1):
            p2x, p2y = self.workspace_polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y):
                if x <= max(p1x, p2x) and p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def publish_box_marker(self, c_pts, header):
        """将点云簇渲染为灰色长方体"""
        centroid = np.mean(c_pts, axis=0)
        min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
        size = max_p - min_p

        m = Marker()
        m.header = header
        m.ns = "box_shape"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(centroid[0])
        m.pose.position.y = float(centroid[1])
        m.pose.position.z = float(centroid[2])
        m.scale.x = float(max(size[0], 0.05))
        m.scale.y = float(max(size[1], 0.05))
        m.scale.z = float(max(size[2], 0.05))
        # 灰色，稍微带点透明度 A=0.8
        m.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
        self.box_marker_pub.publish(m)

    def publish_near_box_marker(self, c_pts, header):
        """将最近的点云簇渲染为蓝色长方体"""
        centroid = np.mean(c_pts, axis=0)
        min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
        size = max_p - min_p

        m = Marker()
        m.header = header
        m.ns = "near_box_shape"
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(centroid[0])
        m.pose.position.y = float(centroid[1])
        m.pose.position.z = float(centroid[2])
        m.scale.x = float(max(size[0], 0.05))
        m.scale.y = float(max(size[1], 0.05))
        m.scale.z = float(max(size[2], 0.05))
        # 蓝色，强调这是最近的box
        m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
        self.near_box_marker_pub.publish(m)

    def pc_callback(self, msg: PointCloud2):
        if self.robot_pose is None: return

        # 1. 读取数据
        points = pc2.read_points_numpy(msg, field_names=("x","y","z","rgb"), skip_nans=True)
        if len(points) < self.min_points: return

        xyz = points[:, :3]
        rgb_uint32 = points[:, 3].astype(np.float32).view(np.uint32)
        rgb_norm = np.stack([((rgb_uint32 >> 16) & 255) / 255.0, 
                             ((rgb_uint32 >> 8) & 255) / 255.0, 
                             (rgb_uint32 & 255) / 255.0], axis=1)

        mask = np.linalg.norm(xyz, axis=1) < self.max_distance
        xyz, rgb_norm = xyz[mask], rgb_norm[mask]
        if len(xyz) < self.min_points: return

        # 2. Open3D 滤波与平面分割
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb_norm)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        _, inliers = pcd.segment_plane(self.distance_threshold, 3, 250)
        pcd_no_ground = pcd.select_by_index(inliers, invert=True)
        
        pts = np.asarray(pcd_no_ground.points)
        clr = np.asarray(pcd_no_ground.colors)
        h_mask = (pts[:, 1] > self.height_min) & (pts[:, 1] < self.height_max)
        pts, clr = pts[h_mask], clr[h_mask]

        if len(pts) < self.min_points: return

        # 3. 聚类
        labels = np.array(pcd_no_ground.select_by_index(np.where(h_mask)[0]).cluster_dbscan(self.eps, self.min_points))
        if labels.max() < 0: return

        # 4. 筛选盒子
        try:
            trans = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
        except: return

        clr_u8 = (clr * 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv_all = cv2.cvtColor(clr_u8, cv2.COLOR_RGB2HSV)
        sorted_lbls = sorted(np.arange(labels.max() + 1), key=lambda i: np.sum(labels == i), reverse=True)

        # 追踪最近的box
        nearest_box_pts = None
        nearest_box_odom = None
        min_distance = float('inf')
        current_box_id = None

        for lbl in sorted_lbls:
            idx = (labels == lbl)
            c_pts = pts[idx]
            
            centroid = np.mean(c_pts, axis=0)
            ps = PointStamped(header=msg.header)
            ps.point.x, ps.point.y, ps.point.z = float(centroid[0]), float(centroid[1]), float(centroid[2])
            p_odom = do_transform_point(ps, trans)

            if not self.is_in_polygon(p_odom.point.x, p_odom.point.y): continue

            mask_hsv = cv2.inRange(hsv_all[idx], self.lower_hsv, self.upper_hsv)
            if (np.count_nonzero(mask_hsv) / len(c_pts)) > self.color_ratio_thresh:
                # 计算到机器人的距离
                xr, yr, _ = self.robot_pose
                dist_to_robot = math.hypot(p_odom.point.x - xr, p_odom.point.y - yr)
                
                # 记录最近的box
                if dist_to_robot < min_distance:
                    min_distance = dist_to_robot
                    nearest_box_pts = c_pts
                    nearest_box_odom = p_odom
                    current_box_id = lbl
                
                # 只在稳定后才发布（防止重复发布同一个盒子）
                if current_box_id == self.last_detected_box_id:
                    self.detection_stable_count += 1
                else:
                    self.detection_stable_count = 1
                
                if self.detection_stable_count >= self.stable_threshold:
                    # 视觉反馈：发布灰色长方体
                    self.publish_box_marker(c_pts, msg.header)
                    # 计算并发布目标停靠点
                    self.calculate_target_and_publish_marker(p_odom)
                
                self.last_detected_box_id = current_box_id
                break
        
        # 发布最近的box（蓝色）
        if nearest_box_pts is not None:
            self.publish_near_box_marker(nearest_box_pts, msg.header)

    def calculate_target_and_publish_marker(self, p_odom):
        xr, yr, _ = self.robot_pose
        dx, dy = p_odom.point.x - xr, p_odom.point.y - yr
        dist = math.hypot(dx, dy)
        if dist < 0.01: return
        
        ux, uy = dx/dist, dy/dist 
        # 停在盒子前 0.4 米处
        tx, ty = p_odom.point.x - 0.4 * ux, p_odom.point.y - 0.4 * uy

        self.target_point_odom = PointStamped(header=Header(frame_id='odom', stamp=self.get_clock().now().to_msg()))
        self.target_point_odom.point.x, self.target_point_odom.point.y = tx, ty
        self.target_pub.publish(self.target_point_odom)

        # RViz 目标球体
        ma = MarkerArray()
        m = Marker(header=self.target_point_odom.header, type=Marker.SPHERE, action=Marker.ADD, id=99)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = tx, ty, p_odom.point.z
        m.scale.x = m.scale.y = m.scale.z = 0.08
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # 绿色目标球
        ma.markers.append(m)
        self.viz_target_marker_pub.publish(ma)

    def control_loop(self):
        if not self.target_point_odom or not self.robot_pose: return
        
        x, y, yaw = self.robot_pose
        tx, ty = self.target_point_odom.point.x, self.target_point_odom.point.y
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        alpha = self.normalize_angle(math.atan2(dy, dx) - yaw)
        
        # cmd = Twist()
        # if dist < 0.08:
        #     self.get_logger().info("已到达停靠位置")
        #     self.target_point_odom = None # 停止控制
        # elif abs(alpha) > self.rotate_threshold:
        #     # 先原地转向
        #     cmd.angular.z = math.copysign(self.w_max, alpha)
        # else:
        #     # 边走边修正
        #     cmd.linear.x = max(self.v_min, min(self.v_max, 0.5 * dist))
        #     cmd.angular.z = math.copysign(min(self.w_max, 2.0 * abs(alpha)), alpha)
            
        # self.cmd_pub.publish(cmd)

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

if __name__ == '__main__':
    main()
# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PointStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Header, ColorRGBA
# from visualization_msgs.msg import Marker, MarkerArray
# import sensor_msgs_py.point_cloud2 as pc2
# import numpy as np
# import open3d as o3d
# import math
# import cv2

# import tf2_ros
# from tf2_geometry_msgs import do_transform_point

# class BoxOnlyVisualizerNode(Node):
#     def __init__(self):
#         super().__init__('box_only_visualizer_node')

#         # --- 基础订阅与发布 ---
#         self.pc_sub = self.create_subscription(PointCloud2, '/realsense/depth/color/points', self.pc_callback, 10)
#         self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
#         # 1. 识别到的盒子：显示为灰色长方体 (Marker)
#         self.box_marker_pub = self.create_publisher(Marker, '/camera/depth/detected_box_marker', 10)
#         # 1b. 最近的盒子：显示为蓝色长方体 (Marker)
#         self.near_box_marker_pub = self.create_publisher(Marker, '/camera/depth/detected_near_box_marker', 10)
#         # 2. 目标点：逻辑位置与视觉球体
#         self.target_pub = self.create_publisher(PointStamped, '/camera/depth/target_point_odom', 10)
#         self.viz_target_marker_pub = self.create_publisher(MarkerArray, '/camera/depth/target_marker', 10)
        
#         self.cmd_pub = self.create_publisher(Twist, '/cmd_velA', 10)

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#         # --- 算法参数 ---
#         self.max_distance = 1.8
#         self.height_min, self.height_max = -0.05, 0.5
#         self.voxel_size = 0.02
#         self.distance_threshold = 0.02
#         self.eps, self.min_points = 0.08, 30

#         # HSV 颜色范围 (绿色盒子)
#         self.lower_hsv = np.array([43, 0, 56])
#         self.upper_hsv = np.array([119, 61, 94])
#         self.color_ratio_thresh = 0.3 

#         # --- 控制参数 (修复 AttributeError 关键点) ---
#         self.v_max = 0.2              # 最大线速度
#         self.v_min = 0.05             # 最小线速度
#         self.w_max = 0.6              # 最大角速度
#         self.rotate_threshold = 0.15  # 转向阈值 (弧度)

#         self.robot_pose = None        
#         self.target_point_odom = None 
        
#         # 工作区多边形
#         self.workspace_polygon = [
#             (0.00, 0.00), (5.22, 0.00), (8.00, 2.02), (10.01, 2.04),
#             (10.00, 4.22), (8.60, 4.23), (8.59, 2.67), (0.00, 2.70)
#         ]

#         self.control_timer = self.create_timer(0.05, self.control_loop)
#         self.get_logger().info("只显示灰色盒子模式已启动")
#         self.get_logger().info(f"绿色Box HSV范围 -> H: [{self.lower_hsv[0]}-{self.upper_hsv[0]}], "
#                                f"S: [{self.lower_hsv[1]}-{self.upper_hsv[1]}], "
#                                f"V: [{self.lower_hsv[2]}-{self.upper_hsv[2]}]")

#     def is_in_polygon(self, x, y):
#         inside = False
#         n = len(self.workspace_polygon)
#         p1x, p1y = self.workspace_polygon[0]
#         for i in range(n + 1):
#             p2x, p2y = self.workspace_polygon[i % n]
#             if y > min(p1y, p2y) and y <= max(p1y, p2y):
#                 if x <= max(p1x, p2x) and p1y != p2y:
#                     xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                     if p1x == p2x or x <= xints:
#                         inside = not inside
#             p1x, p1y = p2x, p2y
#         return inside

#     def odom_callback(self, msg):
#         q = msg.pose.pose.orientation
#         yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
#         self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

#     def publish_box_marker(self, c_pts, header):
#         """将点云簇渲染为灰色长方体"""
#         centroid = np.mean(c_pts, axis=0)
#         min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
#         size = max_p - min_p

#         m = Marker()
#         m.header = header
#         m.ns = "box_shape"
#         m.id = 0
#         m.type = Marker.CUBE
#         m.action = Marker.ADD
#         m.pose.position.x = float(centroid[0])
#         m.pose.position.y = float(centroid[1])
#         m.pose.position.z = float(centroid[2])
#         m.scale.x = float(max(size[0], 0.05))
#         m.scale.y = float(max(size[1], 0.05))
#         m.scale.z = float(max(size[2], 0.05))
#         # 灰色，稍微带点透明度 A=0.8
#         m.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
#         self.box_marker_pub.publish(m)

#     def publish_near_box_marker(self, c_pts, header):
#         """将最近的点云簇渲染为蓝色长方体"""
#         centroid = np.mean(c_pts, axis=0)
#         min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
#         size = max_p - min_p

#         m = Marker()
#         m.header = header
#         m.ns = "near_box_shape"
#         m.id = 1
#         m.type = Marker.CUBE
#         m.action = Marker.ADD
#         m.pose.position.x = float(centroid[0])
#         m.pose.position.y = float(centroid[1])
#         m.pose.position.z = float(centroid[2])
#         m.scale.x = float(max(size[0], 0.05))
#         m.scale.y = float(max(size[1], 0.05))
#         m.scale.z = float(max(size[2], 0.05))
#         # 蓝色，强调这是最近的box
#         m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)
#         self.near_box_marker_pub.publish(m)

#     def pc_callback(self, msg: PointCloud2):
#         if self.robot_pose is None: return

#         # 1. 读取数据
#         points = pc2.read_points_numpy(msg, field_names=("x","y","z","rgb"), skip_nans=True)
#         if len(points) < self.min_points: return

#         xyz = points[:, :3]
#         rgb_uint32 = points[:, 3].astype(np.float32).view(np.uint32)
#         rgb_norm = np.stack([((rgb_uint32 >> 16) & 255) / 255.0, 
#                              ((rgb_uint32 >> 8) & 255) / 255.0, 
#                              (rgb_uint32 & 255) / 255.0], axis=1)

#         mask = np.linalg.norm(xyz, axis=1) < self.max_distance
#         xyz, rgb_norm = xyz[mask], rgb_norm[mask]
#         if len(xyz) < self.min_points: return

#         # 2. Open3D 滤波与平面分割
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(rgb_norm)
#         pcd = pcd.voxel_down_sample(self.voxel_size)
        
#         _, inliers = pcd.segment_plane(self.distance_threshold, 3, 250)
#         pcd_no_ground = pcd.select_by_index(inliers, invert=True)
        
#         pts = np.asarray(pcd_no_ground.points)
#         clr = np.asarray(pcd_no_ground.colors)
#         h_mask = (pts[:, 1] > self.height_min) & (pts[:, 1] < self.height_max)
#         pts, clr = pts[h_mask], clr[h_mask]

#         if len(pts) < self.min_points: return

#         # 3. 聚类
#         labels = np.array(pcd_no_ground.select_by_index(np.where(h_mask)[0]).cluster_dbscan(self.eps, self.min_points))
#         if labels.max() < 0: return

#         # 4. 筛选盒子
#         try:
#             trans = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
#         except: return

#         clr_u8 = (clr * 255).astype(np.uint8).reshape(-1, 1, 3)
#         hsv_all = cv2.cvtColor(clr_u8, cv2.COLOR_RGB2HSV)
#         sorted_lbls = sorted(np.arange(labels.max() + 1), key=lambda i: np.sum(labels == i), reverse=True)

#         # 追踪最近的box
#         nearest_box_pts = None
#         nearest_box_odom = None
#         min_distance = float('inf')

#         for lbl in sorted_lbls:
#             idx = (labels == lbl)
#             c_pts = pts[idx]
            
#             centroid = np.mean(c_pts, axis=0)
#             ps = PointStamped(header=msg.header)
#             ps.point.x, ps.point.y, ps.point.z = float(centroid[0]), float(centroid[1]), float(centroid[2])
#             p_odom = do_transform_point(ps, trans)

#             if not self.is_in_polygon(p_odom.point.x, p_odom.point.y): continue

#             mask_hsv = cv2.inRange(hsv_all[idx], self.lower_hsv, self.upper_hsv)
#             if (np.count_nonzero(mask_hsv) / len(c_pts)) > self.color_ratio_thresh:
#                 # 计算到机器人的距离
#                 xr, yr, _ = self.robot_pose
#                 dist_to_robot = math.hypot(p_odom.point.x - xr, p_odom.point.y - yr)
                
#                 # 记录最近的box
#                 if dist_to_robot < min_distance:
#                     min_distance = dist_to_robot
#                     nearest_box_pts = c_pts
#                     nearest_box_odom = p_odom
                
#                 # 视觉反馈：发布灰色长方体
#                 self.publish_box_marker(c_pts, msg.header)
#                 # 计算并发布目标停靠点
#                 self.calculate_target_and_publish_marker(p_odom)
#                 break
        
#         # 发布最近的box（蓝色）
#         if nearest_box_pts is not None:
#             self.publish_near_box_marker(nearest_box_pts, msg.header)

#     def calculate_target_and_publish_marker(self, p_odom):
#         xr, yr, _ = self.robot_pose
#         dx, dy = p_odom.point.x - xr, p_odom.point.y - yr
#         dist = math.hypot(dx, dy)
#         if dist < 0.01: return
        
#         ux, uy = dx/dist, dy/dist 
#         # 停在盒子前 0.4 米处
#         tx, ty = p_odom.point.x - 0.4 * ux, p_odom.point.y - 0.4 * uy

#         self.target_point_odom = PointStamped(header=Header(frame_id='odom', stamp=self.get_clock().now().to_msg()))
#         self.target_point_odom.point.x, self.target_point_odom.point.y = tx, ty
#         self.target_pub.publish(self.target_point_odom)

#         # RViz 目标球体
#         ma = MarkerArray()
#         m = Marker(header=self.target_point_odom.header, type=Marker.SPHERE, action=Marker.ADD, id=99)
#         m.pose.position.x, m.pose.position.y, m.pose.position.z = tx, ty, p_odom.point.z
#         m.scale.x = m.scale.y = m.scale.z = 0.08
#         m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # 绿色目标球
#         ma.markers.append(m)
#         self.viz_target_marker_pub.publish(ma)

#     def control_loop(self):
#         if not self.target_point_odom or not self.robot_pose: return
        
#         x, y, yaw = self.robot_pose
#         tx, ty = self.target_point_odom.point.x, self.target_point_odom.point.y
#         dx, dy = tx - x, ty - y
#         dist = math.hypot(dx, dy)
#         alpha = self.normalize_angle(math.atan2(dy, dx) - yaw)
        
#         # cmd = Twist()
#         # if dist < 0.08:
#         #     self.get_logger().info("已到达停靠位置")
#         #     self.target_point_odom = None # 停止控制
#         # elif abs(alpha) > self.rotate_threshold:
#         #     # 先原地转向
#         #     cmd.angular.z = math.copysign(self.w_max, alpha)
#         # else:
#         #     # 边走边修正
#         #     cmd.linear.x = max(self.v_min, min(self.v_max, 0.5 * dist))
#         #     cmd.angular.z = math.copysign(min(self.w_max, 2.0 * abs(alpha)), alpha)
            
#         # self.cmd_pub.publish(cmd)

#     @staticmethod
#     def normalize_angle(a):
#         return math.atan2(math.sin(a), math.cos(a))

# def main():
#     rclpy.init()
#     node = BoxOnlyVisualizerNode()
#     try: rclpy.spin(node)
#     except KeyboardInterrupt: pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PointStamped
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Header, ColorRGBA
# from visualization_msgs.msg import Marker, MarkerArray
# import sensor_msgs_py.point_cloud2 as pc2
# import numpy as np
# import open3d as o3d
# import math
# import cv2

# import tf2_ros
# from tf2_geometry_msgs import do_transform_point

# class BoxOnlyVisualizerNode(Node):
#     def __init__(self):
#         super().__init__('box_only_visualizer_node')

#         # --- 基础订阅与发布 ---
#         self.pc_sub = self.create_subscription(PointCloud2, '/realsense/depth/color/points', self.pc_callback, 10)
#         self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
#         # 1. 识别到的盒子：显示为灰色长方体 (Marker)
#         self.box_marker_pub = self.create_publisher(Marker, '/camera/depth/detected_box_marker', 10)
#         # 2. 目标点：逻辑位置与视觉球体
#         self.target_pub = self.create_publisher(PointStamped, '/camera/depth/target_point_odom', 10)
#         self.viz_target_marker_pub = self.create_publisher(MarkerArray, '/camera/depth/target_marker', 10)
        
#         self.cmd_pub = self.create_publisher(Twist, '/cmd_velA', 10)

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#         # --- 算法参数 ---
#         self.max_distance = 1.8
#         self.height_min, self.height_max = -0.05, 0.5
#         self.voxel_size = 0.02
#         self.distance_threshold = 0.02
#         self.eps, self.min_points = 0.08, 30

#         # HSV 颜色范围 (绿色盒子)
#         self.lower_hsv = np.array([43, 0, 56])
#         self.upper_hsv = np.array([119, 61, 94])
#         self.color_ratio_thresh = 0.3 

#         # --- 控制参数 (修复 AttributeError 关键点) ---
#         self.v_max = 0.2              # 最大线速度
#         self.v_min = 0.05             # 最小线速度
#         self.w_max = 0.6              # 最大角速度
#         self.rotate_threshold = 0.15  # 转向阈值 (弧度)

#         self.robot_pose = None        
#         self.target_point_odom = None 
        
#         # 工作区多边形
#         self.workspace_polygon = [
#             (0.00, 0.00), (5.22, 0.00), (8.00, 2.02), (10.01, 2.04),
#             (10.00, 4.22), (8.60, 4.23), (8.59, 2.67), (0.00, 2.70)
#         ]

#         self.control_timer = self.create_timer(0.05, self.control_loop)
#         self.get_logger().info("只显示灰色盒子模式已启动")

#     def is_in_polygon(self, x, y):
#         inside = False
#         n = len(self.workspace_polygon)
#         p1x, p1y = self.workspace_polygon[0]
#         for i in range(n + 1):
#             p2x, p2y = self.workspace_polygon[i % n]
#             if y > min(p1y, p2y) and y <= max(p1y, p2y):
#                 if x <= max(p1x, p2x) and p1y != p2y:
#                     xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                     if p1x == p2x or x <= xints:
#                         inside = not inside
#             p1x, p1y = p2x, p2y
#         return inside

#     def odom_callback(self, msg):
#         q = msg.pose.pose.orientation
#         yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
#         self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

#     def publish_box_marker(self, c_pts, header):
#         """将点云簇渲染为灰色长方体"""
#         centroid = np.mean(c_pts, axis=0)
#         min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
#         size = max_p - min_p

#         m = Marker()
#         m.header = header
#         m.ns = "box_shape"
#         m.id = 0
#         m.type = Marker.CUBE
#         m.action = Marker.ADD
#         m.pose.position.x = float(centroid[0])
#         m.pose.position.y = float(centroid[1])
#         m.pose.position.z = float(centroid[2])
#         m.scale.x = float(max(size[0], 0.05))
#         m.scale.y = float(max(size[1], 0.05))
#         m.scale.z = float(max(size[2], 0.05))
#         # 灰色，稍微带点透明度 A=0.8
#         m.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
#         self.box_marker_pub.publish(m)

#     def pc_callback(self, msg: PointCloud2):
#         if self.robot_pose is None: return

#         # 1. 读取数据
#         points = pc2.read_points_numpy(msg, field_names=("x","y","z","rgb"), skip_nans=True)
#         if len(points) < self.min_points: return

#         xyz = points[:, :3]
#         rgb_uint32 = points[:, 3].astype(np.float32).view(np.uint32)
#         rgb_norm = np.stack([((rgb_uint32 >> 16) & 255) / 255.0, 
#                              ((rgb_uint32 >> 8) & 255) / 255.0, 
#                              (rgb_uint32 & 255) / 255.0], axis=1)

#         mask = np.linalg.norm(xyz, axis=1) < self.max_distance
#         xyz, rgb_norm = xyz[mask], rgb_norm[mask]
#         if len(xyz) < self.min_points: return

#         # 2. Open3D 滤波与平面分割
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(rgb_norm)
#         pcd = pcd.voxel_down_sample(self.voxel_size)
        
#         _, inliers = pcd.segment_plane(self.distance_threshold, 3, 250)
#         pcd_no_ground = pcd.select_by_index(inliers, invert=True)
        
#         pts = np.asarray(pcd_no_ground.points)
#         clr = np.asarray(pcd_no_ground.colors)
#         h_mask = (pts[:, 1] > self.height_min) & (pts[:, 1] < self.height_max)
#         pts, clr = pts[h_mask], clr[h_mask]

#         if len(pts) < self.min_points: return

#         # 3. 聚类
#         labels = np.array(pcd_no_ground.select_by_index(np.where(h_mask)[0]).cluster_dbscan(self.eps, self.min_points))
#         if labels.max() < 0: return

#         # 4. 筛选盒子
#         try:
#             trans = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
#         except: return

#         clr_u8 = (clr * 255).astype(np.uint8).reshape(-1, 1, 3)
#         hsv_all = cv2.cvtColor(clr_u8, cv2.COLOR_RGB2HSV)
#         sorted_lbls = sorted(np.arange(labels.max() + 1), key=lambda i: np.sum(labels == i), reverse=True)

#         for lbl in sorted_lbls:
#             idx = (labels == lbl)
#             c_pts = pts[idx]
            
#             centroid = np.mean(c_pts, axis=0)
#             ps = PointStamped(header=msg.header)
#             ps.point.x, ps.point.y, ps.point.z = float(centroid[0]), float(centroid[1]), float(centroid[2])
#             p_odom = do_transform_point(ps, trans)

#             if not self.is_in_polygon(p_odom.point.x, p_odom.point.y): continue

#             mask_hsv = cv2.inRange(hsv_all[idx], self.lower_hsv, self.upper_hsv)
#             if (np.count_nonzero(mask_hsv) / len(c_pts)) > self.color_ratio_thresh:
#                 # 视觉反馈：发布灰色长方体
#                 self.publish_box_marker(c_pts, msg.header)
#                 # 计算并发布目标停靠点
#                 self.calculate_target_and_publish_marker(p_odom)
#                 break

#     def calculate_target_and_publish_marker(self, p_odom):
#         xr, yr, _ = self.robot_pose
#         dx, dy = p_odom.point.x - xr, p_odom.point.y - yr
#         dist = math.hypot(dx, dy)
#         if dist < 0.01: return
        
#         ux, uy = dx/dist, dy/dist 
#         # 停在盒子前 0.4 米处
#         tx, ty = p_odom.point.x - 0.4 * ux, p_odom.point.y - 0.4 * uy

#         self.target_point_odom = PointStamped(header=Header(frame_id='odom', stamp=self.get_clock().now().to_msg()))
#         self.target_point_odom.point.x, self.target_point_odom.point.y = tx, ty
#         self.target_pub.publish(self.target_point_odom)

#         # RViz 目标球体
#         ma = MarkerArray()
#         m = Marker(header=self.target_point_odom.header, type=Marker.SPHERE, action=Marker.ADD, id=99)
#         m.pose.position.x, m.pose.position.y, m.pose.position.z = tx, ty, p_odom.point.z
#         m.scale.x = m.scale.y = m.scale.z = 0.08
#         m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # 绿色目标球
#         ma.markers.append(m)
#         self.viz_target_marker_pub.publish(ma)

#     def control_loop(self):
#         if not self.target_point_odom or not self.robot_pose: return
        
#         x, y, yaw = self.robot_pose
#         tx, ty = self.target_point_odom.point.x, self.target_point_odom.point.y
#         dx, dy = tx - x, ty - y
#         dist = math.hypot(dx, dy)
#         alpha = self.normalize_angle(math.atan2(dy, dx) - yaw)
        
#         # cmd = Twist()
#         # if dist < 0.08:
#         #     self.get_logger().info("已到达停靠位置")
#         #     self.target_point_odom = None # 停止控制
#         # elif abs(alpha) > self.rotate_threshold:
#         #     # 先原地转向
#         #     cmd.angular.z = math.copysign(self.w_max, alpha)
#         # else:
#         #     # 边走边修正
#         #     cmd.linear.x = max(self.v_min, min(self.v_max, 0.5 * dist))
#         #     cmd.angular.z = math.copysign(min(self.w_max, 2.0 * abs(alpha)), alpha)
            
#         # self.cmd_pub.publish(cmd)

#     @staticmethod
#     def normalize_angle(a):
#         return math.atan2(math.sin(a), math.cos(a))

# def main():
#     rclpy.init()
#     node = BoxOnlyVisualizerNode()
#     try: rclpy.spin(node)
#     except KeyboardInterrupt: pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '_main_':
#     main()