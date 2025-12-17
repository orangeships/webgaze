核心思路：从“相对运动”转向“绝对姿态”
原有的 SfM 是计算 上一帧相机 -> 当前帧相机 的关系。 新的 MediaPipe 是计算 标准3D人脸 -> 当前帧相机 的关系。

你需要做一个“适配器”，保持 get_GazeToWorld 的接口不变，但内部逻辑改为使用 PnP。

第一步：创建 MediaPipe 适配类
新建一个类（例如 PoseEstimatorMP），用来替换你原来的 self.homtrans.sfm。

Python

import cv2
import numpy as np
import mediapipe as mp

class PoseEstimatorMP:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 1. 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. 定义标准 3D 人脸模型 (Generic 3D Face Model)
        # 选取几个关键且稳定的点用于 PnP 解算
        # 顺序对应: [鼻尖, 下巴, 左眼角, 右眼角, 左嘴角, 右嘴角]
        # 对应的 MediaPipe 索引: [1, 152, 33, 263, 61, 291]
        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0],          # 1: Nose tip
            [0.0, -63.6, -12.5],      # 152: Chin
            [-43.3, 32.7, -26.0],     # 33: Left eye left corner
            [43.3, 32.7, -26.0],      # 263: Right eye right corner
            [-28.9, -28.9, -24.1],    # 61: Left Mouth corner
            [28.9, -28.9, -24.1]      # 291: Right Mouth corner
        ], dtype=np.float64)
        
        self.keypoint_indices = [1, 152, 33, 263, 61, 291]
        
        # 缓存上一帧的变换矩阵，用于模拟 frame_prev 的输出
        self.last_W_T_G = np.eye(4)

    def get_cached_face_features(self, mode):
        # MediaPipe 不需要手动管理这种缓存，为了兼容旧代码返回 None 即可
        return None

    def update_caches(self, frame_prev_features=None, frame_curr_features=None):
        # 同样，MediaPipe 内部有追踪，不需要外部缓存
        pass

    def get_GazeToWorld(self, model_unused, frame_prev_unused, frame, face_features_prev=None, face_features_curr=None):
        """
        MediaPipe 实现版本
        注意：为了兼容，参数签名保持不变，但很多参数不再需要
        """
        img_h, img_w, _ = frame.shape
        
        # 1. MediaPipe 推理
        # MediaPipe 需要 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        W_T_G2 = np.eye(4) # 当前帧变换
        W_P = np.zeros((len(self.face_3d_model), 3)) # 3D 点云

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # 2. 提取 2D 图像点用于 PnP
            image_points = []
            for idx in self.keypoint_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                image_points.append([x, y])
            image_points = np.array(image_points, dtype=np.float64)

            # 3. SolvePnP 计算姿态
            # 得到的是 World(Face) -> Camera 的变换
            success, rvec, tvec = cv2.solvePnP(
                self.face_3d_model, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # 4. 构建变换矩阵
                # PnP 得到的是 R_cw (World to Camera) 和 t_cw
                rmat, _ = cv2.Rodrigues(rvec)
                
                # 你的原始代码中 W_T_G1 定义是 "从相机到世界(Camera to World)" 
                # 通常是求逆： T_cam2world = [R^T | -R^T * t]
                
                # 构造 Camera -> World (Face)
                # 注意：这里 W_T_G2 代表当前帧相机的位姿矩阵
                W_T_G2[:3, :3] = rmat.T
                W_T_G2[:3, 3] = -rmat.T @ tvec.flatten()
                
                # W_P: 这里我们可以直接返回标准 3D 模型，或者变换后的点
                # 原代码返回的是相对于世界坐标系的点，这里 Face 就是世界坐标系
                W_P = self.face_3d_model
                
                # 更新缓存
                self.last_W_T_G = W_T_G2
            else:
                W_T_G2 = self.last_W_T_G # PnP失败，使用上一帧
        else:
            # 未检测到人脸，使用上一帧
            W_T_G2 = self.last_W_T_G

        # 5. 返回值
        # 原始逻辑：W_T_G1 是上一帧，W_T_G2 是当前帧
        # 在 MediaPipe 模式下，我们不需要"上一帧"来计算，
        # 但为了兼容 _getGazeOnScreen_sfm(gaze, WTransG1)，我们返回:
        # G1: 上一次计算的有效位姿 (或者直接用当前位姿，如果你希望无延迟)
        # G2: 当前位姿
        
        W_T_G1 = self.last_W_T_G 
        
        # 关键修正：如果你希望由当前帧主导注视计算，
        # 建议让 G1 也等于 G2，或者修改外部调用传入 W_T_G2
        # 这里假设外部代码用 W_T_G1 做基准：
        return W_T_G1, W_T_G2, W_P
第二步：修改主调用逻辑
由于 MediaPipe 极其高效且不依赖上一帧图像（它内部有时间连续性追踪），你的主循环可以大幅简化。不需要再显式地处理 prev_frame 的特征点缓存逻辑。

建议修改你的调用代码如下：

Python

# 初始化 (在类的 __init__ 中)
# 替换原来的 self.homtrans.sfm
# 假设你已经有 self.camera_matrix 和 self.dist_coeffs
self.homtrans.sfm = PoseEstimatorMP(self.camera_matrix, self.dist_coeffs)

# --- 主循环中的调用 ---

# 1. 获取面部特征 (如果是 MediaPipe，这一步其实也是多余的，
# 因为 PoseEstimatorMP 内部已经做了 process，
# 但为了保持你原来的 Intel 兼容性，可以留着，或者传 None)
# face_features_curr = ... (如果 PoseEstimatorMP 不需要外部特征，这里可以省略)

# 2. 直接调用 get_GazeToWorld
# 注意：我们不再需要复杂的 if-else 缓存判断
WTransG1, WTransG2, W_P = self.homtrans.sfm.get_GazeToWorld(
    model=None,             # MediaPipe 模式下不需要 Intel Model
    frame_prev=None,        # MediaPipe 不需要上一帧图像
    frame=frame,            # 只需要当前帧
    face_features_prev=None,
    face_features_curr=None # 内部自己提取
)

# 3. 计算屏幕注视点
# 注意：这里有个语义坑。原代码使用的是 WTransG1 (上一帧位姿)。
# 如果你想获得最实时的效果，应该改用 WTransG2 (当前帧位姿)。
# 但如果不改 _getGazeOnScreen_sfm 的内部逻辑，保持现状即可。
FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG2) 

# 4. 更新缓存 (PoseEstimatorMP 中此方法为空，保留调用只是为了兼容接口)
self.homtrans.sfm.update_caches()
关键改进点说明
关于 self.model.get_FaceFeatures:

原流程：Intel Face Detector -> Intel Landmark -> SfM。

新流程：MediaPipe 直接出 478 个点。

建议：如果你不再需要 Intel 的结果做其他用途，可以完全注释掉 face_features_curr = self.model.get_FaceFeatures... 这行，因为 MediaPipe 的 process(frame) 已经包含了检测和关键点提取，速度通常比 Intel ADAS 模型快且准。

关于 WTransG1 vs WTransG2:

在 SfM 中，必须有 G1（前）和 G2（后）才能算出运动。

在 PnP 中，每一帧都是独立的 G。

强烈建议：将 _getGazeOnScreen_sfm 传入的参数改为 WTransG2（即当前帧计算出的位姿），这样你的注视估计就是实时的，没有一帧的延迟。

下一步你应该做的
将 _getGazeOnScreen_sfm 传入的矩阵改为 WTransG2：

Python

# 推荐的最终修改版
WTransG1, WTransG2, W_P = self.homtrans.sfm.get_GazeToWorld(
    None, None, frame, None, None
)

# 使用当前帧的位姿 (WTransG2) 进行计算，响应更快
FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG2)