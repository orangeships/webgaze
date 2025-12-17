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

        # PnP关键点索引
        self.pnp_indices = [1, 152, 33, 263, 61, 291, 234, 454, 10, 168, 197, 5, 4, 19]
        
        # 对应的标准 3D 模型点 (基于经典人脸比例)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 1: Nose tip
            (0.0, -330.0, -65.0),        # 152: Chin
            (-225.0, 170.0, -135.0),     # 33: Left eye corner
            (225.0, 170.0, -135.0),      # 263: Right eye corner
            (-150.0, -150.0, -125.0),    # 61: Left mouth corner
            (150.0, -150.0, -125.0),     # 291: Right mouth corner
            (-480.0, 0.0, -280.0),       # 234: Left cheek
            (480.0, 0.0, -280.0),        # 454: Right cheek
            (0.0, 310.0, -80.0),         # 10: Top of forehead
            (0.0, 110.0, -110.0),        # 168: Between eyes
            (0.0, 35.0, -40.0),          # 197: Nose bridge
            (0.0, -70.0, -45.0),         # 5: Below nose
            (0.0, -45.0, -35.0),         # 4: Nose tip base
            (0.0, -180.0, -75.0)         # 19: Upper lip
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
        
        # 初始化旋转和平移向量
        self.rvec = None
        self.tvec = None

    def get_pose(self, frame):
        h, w, _ = frame.shape
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # 提取PnP关键点的2D坐标
            face_landmarks = np.array([
                [landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                for idx in self.pnp_indices
            ], dtype=np.float32)

            # 使用solvePnPRansac进行初始拟合
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points, 
                face_landmarks, 
                self.cam_matrix, 
                self.dist_coeffs, 
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
                        self.cam_matrix, 
                        self.dist_coeffs, 
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
                distance = np.linalg.norm(tvec)
                
                return {
                    "rvec": rvec, 
                    "tvec": tvec, 
                    "euler": euler_angles, 
                    "rmat": rmat,
                    "distance": distance,
                    "landmarks": landmarks
                }
        
        return None

    def draw_pose(self, frame, pose_data):
        if not pose_data:
            return frame
            
        h, w, _ = frame.shape
        rvec = pose_data["rvec"]
        tvec = pose_data["tvec"]
        euler = pose_data["euler"]
        distance = pose_data["distance"]/100 # 转换为厘米
        
        # 绘制3D坐标轴（位于鼻尖位置）
        axis_length = 500
        
        # 定义3D坐标轴点（相对于鼻尖）
        axis_3d = np.float32([
            [axis_length, 0, 0],      # X轴终点
            [0, -axis_length, 0],     # Y轴终点
            [0, 0, axis_length],      # Z轴终点
            [0, 0, 0]                 # 原点（鼻尖）
        ])
        
        # 投影到2D图像平面
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, self.cam_matrix, self.dist_coeffs)
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
        
        # 显示平移向量
        tx, ty, tz = tvec[0, 0], tvec[1, 0], tvec[2, 0]
        cv2.putText(frame, f"T: ({tx:.1f}, {ty:.1f}, {tz:.1f})", (30, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

if __name__ == "__main__":
    estimator = HeadPoseEstimator()
    cap = cv2.VideoCapture(1)  # 使用默认摄像头
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 获取头部姿态
        pose_data = estimator.get_pose(frame)
        
        # 绘制姿态信息
        frame = estimator.draw_pose(frame, pose_data)
        
        # 显示结果
        cv2.imshow('Head Pose Estimation', frame)
        
        # 按ESC退出
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()