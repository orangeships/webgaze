import cv2
import mediapipe as mp
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # 35个稳定的人脸关键点索引
        self.pnp_indices = [10,151,9,168,6,1,152,103,332,54,284,21,251,127,356,133,362,33,263,234,454,93,323,58,288,197,5,2,164,172,397,176,400,148,377]
        
        # 对应的标准 3D 模型点 (基于精确的人脸测量学数据)
        self.model_points = np.array([
            [0.0, 112.7, 36.8],     # 10: forehead top
            [0.0, 77.5, 59.8],      # 151: glabella (above nose)
            [0.0, 55.2, 63.8],      # 9: between eyebrows
            [0.0, 41.5, 65.4],      # 168: nose bridge high
            [0.0, 21.0, 70.8],      # 6: nose bridge mid
            [0.0, -12.4, 93.9],     # 1: nose tip
            [0.0, -103.5, 23.9],    # 152: chin
            [-44.3, 75.3, 40.8],    # 103: left eyebrow outer
            [44.3, 75.3, 40.8],     # 332: right eyebrow outer
            [-23.6, 68.3, 56.4],    # 54: left eyebrow inner
            [23.6, 68.3, 56.4],     # 284: right eyebrow inner
            [-45.5, 45.4, 21.9],    # 21: left eye upper bone
            [45.5, 45.4, 21.9],     # 251: right eye upper bone
            [-75.4, 18.2, -14.6],   # 127: left temple
            [75.4, 18.2, -14.6],    # 356: right temple
            [-15.1, 15.5, 44.4],    # 133: left eye inner corner
            [15.1, 15.5, 44.4],     # 362: right eye inner corner
            [-46.1, 14.8, 30.6],    # 33: left eye outer corner
            [46.1, 14.8, 30.6],     # 263: right eye outer corner
            [-82.9, -15.1, -12.1],  # 234: left cheek edge
            [82.9, -15.1, -12.1],   # 454: right cheek edge
            [-43.9, -35.2, 28.5],   # 93: left jaw mid
            [43.9, -35.2, 28.5],    # 323: right jaw mid
            [-47.9, -46.7, 32.3],   # 58: left mouth corner area
            [47.9, -46.7, 32.3],    # 288: right mouth corner area
            [0.0, 11.2, 75.8],      # 197: philtrum top
            [0.0, -1.9, 81.3],      # 5: philtrum mid
            [0.0, -32.8, 77.0],     # 2: upper lip
            [0.0, -51.3, 62.4],     # 164: lower lip
            [-18.7, -76.8, 48.0],   # 172: jaw left curve
            [18.7, -76.8, 48.0],    # 397: jaw right curve
            [-13.7, -88.1, 40.0],   # 176: jaw left bottom
            [13.7, -88.1, 40.0],    # 400: jaw right bottom
            [-34.2, -87.8, 20.3],   # 148: chin left side
            [34.2, -87.8, 20.3]     # 377: chin right side
        ], dtype=np.float64)
        
        # 相机内参
        self.cam_matrix = np.array([
            [1408.572643, 0.000000, 918.986646],
            [0.000000, 1378.049178, 553.273009],
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float64)
        self.dist_coeffs = np.array([
            [-0.355882, 0.159152, 0.002311, -0.001401, -0.055078]
        ], dtype=np.float64)
        
        # 原始标称分辨率（相机内参是基于此分辨率标定的）
        self.nominal_width = 1920
        self.nominal_height = 1080
        
        # 实际分辨率
        self.actual_width = None
        self.actual_height = None
        
        # 缩放后的相机内参
        self.scaled_cam_matrix = None
        self.scaled_dist_coeffs = None
        
        # 初始化旋转和平移向量
        self.rvec = None
        self.tvec = None
    
    def update_camera_params(self, frame_width, frame_height):
        """根据实际分辨率更新相机内参
        
        Args:
            frame_width: 实际帧宽度
            frame_height: 实际帧高度
        """
        if frame_width == self.nominal_width and frame_height == self.nominal_height:
            # 如果分辨率没有变化，直接使用原始参数
            self.scaled_cam_matrix = self.cam_matrix.copy()
            self.scaled_dist_coeffs = self.dist_coeffs.copy()
        else:
            # 计算缩放比例
            scale_x = frame_width / self.nominal_width
            scale_y = frame_height / self.nominal_height
            
            # 缩放相机内参
            self.scaled_cam_matrix = self.cam_matrix.copy()
            self.scaled_cam_matrix[0, 0] *= scale_x  # fx
            self.scaled_cam_matrix[1, 1] *= scale_y  # fy
            self.scaled_cam_matrix[0, 2] *= scale_x  # cx
            self.scaled_cam_matrix[1, 2] *= scale_y  # cy
            
            # 畸变系数通常不需要缩放
            self.scaled_dist_coeffs = self.dist_coeffs.copy()
        
        # 保存实际分辨率
        self.actual_width = frame_width
        self.actual_height = frame_height
        
        print(f"更新相机内参: {frame_width}x{frame_height} "
              f"(原始: {self.nominal_width}x{self.nominal_height})")
        print(f"缩放比例: {frame_width/self.nominal_width:.3f}x{frame_height/self.nominal_height:.3f}")
    
    def get_pose(self, frame):
        h, w, _ = frame.shape
        
        # 检查是否需要更新相机参数
        if self.actual_width != w or self.actual_height != h:
            self.update_camera_params(w, h)
        
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # 提取PnP关键点的2D坐标
            face_landmarks = np.array([
                [landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                for idx in self.pnp_indices
            ], dtype=np.float32)

            # 使用缩放后的相机内参
            # 使用solvePnPRansac进行初始拟合
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points, 
                face_landmarks, 
                self.scaled_cam_matrix, 
                self.scaled_dist_coeffs, 
                rvec=self.rvec, 
                tvec=self.tvec, 
                useExtrinsicGuess=True, 
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success:
                # 使用solvePnP进行多次迭代以提高精度
                for _ in range(10):
                    success, rvec, tvec = cv2.solvePnP(
                        self.model_points, 
                        face_landmarks, 
                        self.scaled_cam_matrix, 
                        self.scaled_dist_coeffs, 
                        rvec=rvec, 
                        tvec=tvec, 
                        useExtrinsicGuess=True, 
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                
                # 保存当前结果作为下一帧的初始猜测
                self.rvec, self.tvec = rvec, tvec
                
                # 计算欧拉角
                rmat, _ = cv2.Rodrigues(rvec)
                p_mtx = np.hstack((rmat, tvec))
                euler_angles = cv2.decomposeProjectionMatrix(p_mtx)[6]
                
                # 计算距离（使用tvec的Z分量）
                # 使用tvec[2, 0]的Z分量值，并转换为厘米
                distance_cm = tvec[2, 0] / 10.0  # 从毫米转换为厘米
                return {
                    "rvec": rvec, 
                    "tvec": tvec, 
                    "euler": euler_angles, 
                    "rmat": rmat,
                    "distance": distance_cm,
                    "landmarks": landmarks
                }
        
        return None

    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0), thickness=2):
        """绘制35个关键点到图像上"""
        h, w, _ = frame.shape
        
        for i, idx in enumerate(self.pnp_indices):
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                
                # 使用不同颜色显示不同区域的关键点
                if idx in [1, 6, 152]:  # 鼻子和下巴区域 - 蓝色
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), thickness)
                elif idx in [133, 362, 33, 263]:  # 眼睛区域 - 绿色
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), thickness)
                elif idx in [58, 288, 2, 164]:  # 嘴巴区域 - 红色
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), thickness)
                elif idx in [10, 151, 9, 168]:  # 额头和眉毛区域 - 黄色
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), thickness)
                else:  # 其他关键点 - 白色
                    cv2.circle(frame, (x, y), 3, color, thickness)
                
                # 只对重要的几个关键点添加标签
                if idx in [1, 152, 33, 263]:  # 鼻尖、下巴、眼角
                    cv2.putText(frame, str(idx), (x+5, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def draw_pose(self, frame, pose_data):
        if not pose_data:
            return frame
            
        h, w, _ = frame.shape
        rvec = pose_data["rvec"]
        tvec = pose_data["tvec"]
        euler = pose_data["euler"]
        distance = pose_data["distance"]  # 距离已经在get_pose中计算为厘米
        landmarks = pose_data["landmarks"]
        
        # 首先绘制35个关键点
        frame = self.draw_landmarks(frame, landmarks)
        
        # 绘制3D坐标轴（位于鼻尖位置）
        axis_length = 100  # 缩短坐标轴长度以便更好显示
        
        # 定义3D坐标轴点（相对于鼻尖）
        axis_3d = np.float32([
            [axis_length, 0, 0],      # X轴终点
            [0, -axis_length, 0],     # Y轴终点
            [0, 0, axis_length],      # Z轴终点
            [0, 0, 0]                 # 原点（鼻尖）
        ])
        
        # 投影到2D图像平面
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, self.scaled_cam_matrix, self.scaled_dist_coeffs)
        imgpts = imgpts.astype(int)
        
        # 获取原点和各轴终点
        origin = tuple(imgpts[3].ravel())
        x_end = tuple(imgpts[0].ravel())
        y_end = tuple(imgpts[1].ravel())
        z_end = tuple(imgpts[2].ravel())
        
        # 绘制坐标轴
        cv2.line(frame, origin, x_end, (0, 0, 255), 3)  # X轴 - 红色
        cv2.line(frame, origin, y_end, (0, 255, 0), 3)  # Y轴 - 绿色
        cv2.line(frame, origin, z_end, (255, 0, 0), 3)  # Z轴 - 蓝色
        
        # 绘制坐标轴标签
        cv2.putText(frame, 'X', (x_end[0]+5, x_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', (y_end[0], y_end[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, 'Z', (z_end[0], z_end[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 显示距离信息（厘米）
        cv2.putText(frame, f"Distance: {distance:.1f} cm", (30, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 显示平移向量（增强版）
        tx, ty, tz = tvec[0, 0], tvec[1, 0], tvec[2, 0]
        cv2.putText(frame, f"T: ({tx:.1f}, {ty:.1f}, {tz:.1f})", (30, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 实时输出tvec详细信息

        
        # 显示平移向量的详细分量信息
        cv2.putText(frame, f"Tx: {tx:.1f} Ty: {ty:.1f} Tz: {tz:.1f}", (30, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        

        # 添加图例说明
        legend_y = 220
        cv2.putText(frame, "Keypoints:", (30, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Blue: Nose/Chin", (30, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, "Green: Eyes", (30, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Red: Mouth", (30, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "Yellow: Forehead", (30, legend_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, f"Total: {len(self.pnp_indices)} points", (30, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

if __name__ == "__main__":
    estimator = HeadPoseEstimator()
    # 使用默认摄像头（0）。如需其它摄像头，请改为 1、2 或者摄像头设备索引/路径
    cap = cv2.VideoCapture(0)

    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        exit(1)
    
    print("开始测试实时tvec输出功能...")
    print("按ESC键退出程序")
    
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("视频播放完毕或读取失败")
            break
        
        
        # 获取头部姿态
        pose_data = estimator.get_pose(frame)
        
        # 绘制姿态信息
        frame = estimator.draw_pose(frame, pose_data)
        
        # 显示视频窗口（实时摄像头预览）
        cv2.imshow('Head Pose Estimation - 实时tvec输出测试', frame)
        
        # 按ESC或 q 键退出（waitKey 延迟设为1以获得更流畅的摄像头帧）
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            print("用户按下ESC或q键，退出程序")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n测试完成，共处理 {frame_count} 帧")