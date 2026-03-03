import os
import sys
import abc
import cv2
import time

import logging as log
from turtle import width

import numpy as np
import math
from openvino.inference_engine import IENetwork, IECore

from skimage.measure import label, regionprops
import utilities.utils as util
import mediapipe as mp
# MediaPipe导入
def expand_face_box(box, scale=1.1):
    """只在y方向上扩大face_box尺寸"""
    xmin, ymin, xmax, ymax = box
    center_y = (ymin + ymax) / 2
    height = ymax - ymin
    
    # 只在y方向扩大尺寸
    new_height = height * scale
    
    new_ymin = center_y - new_height / 2 - new_height / 12
    new_ymax = center_y + new_height / 2 - new_height / 12
    
    return [int(xmin), int(new_ymin), int(xmax), int(new_ymax)]

def to_relative_coordinates(absolute_boxes, absolute_centers, face_box):
    """
    将MediaPipe的absolute坐标转换为相对于face的relative坐标
    
    Args:
        absolute_boxes: [[xmin,ymin,xmax,ymax], ...] (absolute frame coords)
        absolute_centers: [[x,y], ...] (absolute frame coords)  
        face_box: [xmin,ymin,xmax,ymax] (absolute frame coords)
    
    Returns:
        relative_boxes, relative_centers (相对于face的坐标)
    """
    face_xmin, face_ymin, face_xmax, face_ymax = face_box
    
    # 转换eye_boxes到relative坐标
    relative_eye_boxes = []
    for eye_box in absolute_boxes:
        xmin, ymin, xmax, ymax = eye_box
        relative_eye_boxes.append([
            xmin - face_xmin, ymin - face_ymin,
            xmax - face_xmin, ymax - face_ymin
        ])
    
    # 转换eye_centers到relative坐标
    relative_eye_centers = []
    for eye_center in absolute_centers:
        cx, cy = eye_center
        relative_eye_centers.append([
            cx - face_xmin, cy - face_ymin
        ])
        
    return relative_eye_boxes, relative_eye_centers

class Model(metaclass=abc.ABCMeta):
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model_name, device="AUTO", extensions=None):
        # 自动选择设备：优先GPU，如果没有则使用CPU
        if device == "AUTO":
            try:
                from openvino.inference_engine import IECore
                ie = IECore()
                available_devices = ie.available_devices
                if 'GPU' in available_devices:
                    self.device = "GPU"
                else:
                    self.device = "CPU"
            except:
                self.device = "CPU"
        else:
            self.device = device
            
        print(f"🎯 使用设备: {self.device}")
        self._init_model(model_name, self.device, extensions)
        # self._check_model(self.core, self.model, device)
        self._init_input_output(self.model)

    def _init_model(self, model_name, device, extensions):
        # 处理目录路径和文件路径
        if os.path.isdir(model_name):
            # 如果是目录，查找XML文件
            model_files = [f for f in os.listdir(model_name) if f.endswith('.xml')]
            if not model_files:
                raise ValueError(f"No model files found in directory: {model_name}")
            model_base = os.path.join(model_name, os.path.splitext(model_files[0])[0])
        else:
            # 如果是文件路径，直接使用
            model_base = model_name
            
        model_structure = model_base + '.xml'
        model_weights = model_base + '.bin'
        
        self.core = IECore()
        if extensions and "CPU" in device:
            self.core.add_extension(extensions, device)
        self.model = self.core.read_network(model=model_structure, weights=model_weights)

    def _init_input_output(self, model):
        self.input_name = next(iter(model.input_info))
        self.input_shape = model.input_info[self.input_name].input_data.shape
        self.output_name = next(iter(model.outputs))
        self.output_shape = model.outputs[self.output_name].shape

    def load_model(self):
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        except Exception as e:
            print(f"Something went wrong when loading model: {e}")
            exit()

    def get_input_shape(self, input_name=None):
        if input_name is None:
            return self.input_shape
        return self.model.input_info[input_name].input_data.shape

    def exec_net(self, request_id, inputs):
        if isinstance(inputs, dict):
            self.net.start_async(request_id=request_id, inputs=inputs)
        else:
            self.net.start_async(request_id=request_id, inputs={self.input_name: inputs})
    
    @abc.abstractmethod
    def predict(self, image):
        """ Predict Output  """
        raise NotImplementedError

    @abc.abstractmethod
    def _preprocess_output(self, outputs, image):
        """ Process Output  """
        raise NotImplementedError

    def _preprocess_input(self, image, input_name=None):
        n, c, h, w = self.get_input_shape(input_name)
        input_image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))
        # image = image.reshape(1, *input_image.shape)
        return input_image

    def get_outputs(self, request_id):
        outputs = self.net.requests[request_id].output_blobs
        return outputs

    def get_output(self, request_id):     
        output = self.net.requests[request_id].output_blobs[self.output_name]
        return output

    def wait(self, request_id):
        status = self.net.requests[request_id].wait()
        return status



class FacialLandmarkDetection35(Model):
    """
    Facial Landmark Detection Model
    https://docs.openvino.ai/2022.3/omz_models_model_facial_landmarks_35_adas_0002.html
    """
    def __init__(self, model_name, device="CPU", extensions=None):
        super().__init__(model_name, device, extensions)

    def predict(self, face_image):
        input_image = self._preprocess_input(face_image)
        self.exec_net(0, input_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            landmarks = self._preprocess_output(outputs, face_image)

        return landmarks
    
    def _preprocess_output(self, outputs, image):
        normalized_landmarks = np.squeeze(outputs.buffer).reshape((35,2))
        h, w, _ = image.shape
        landmarks = np.zeros((35,2))
        for idx, l in enumerate(normalized_landmarks):
            x, y = l
            landmarks[idx] = [int(x*w), int(y*h)]
        return landmarks

class HeadPoseEstimation(Model):
    '''
    Head Pose Estimation Model
    '''
    def __init__(self, model_name, device="CPU", extensions=None):
        super().__init__(model_name, device, extensions)

    def predict(self, image):
        input_image = self._preprocess_input(image)
        self.exec_net(0, input_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_outputs(0)
            head_pose_angles = self._preprocess_output(outputs)
            return head_pose_angles        

    def _preprocess_output(self, outputs):
        yaw = outputs['angle_y_fc'].buffer[0][0]
        pitch = outputs['angle_p_fc'].buffer[0][0]
        roll = outputs['angle_r_fc'].buffer[0][0]
        return [yaw, pitch, roll]

class GazeEstimation(Model):
    '''
    Gaze Estimation Model
    '''
    def __init__(self, model_name, device="CPU", extensions=None):
        super().__init__(model_name, device, extensions)

    def predict(self, right_eye_image, head_pose_angles, left_eye_image):
        _, _, roll = head_pose_angles
        right_eye_image, head_pose_angles, left_eye_image = self._preprocess_gaze_input(right_eye_image, head_pose_angles, left_eye_image)
        input_dict = {"left_eye_image": left_eye_image, "right_eye_image": right_eye_image, "head_pose_angles": head_pose_angles}
        self.exec_net(0, input_dict)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            gaze_vector = self._preprocess_output(outputs, roll)
            return gaze_vector

    def _preprocess_gaze_input(self, right_eye_image, head_pose_angles, left_eye_image):
        left_eye_image = self._preprocess_input(left_eye_image, "left_eye_image")
        right_eye_image = self._preprocess_input(right_eye_image, "right_eye_image")
        head_pose_angles = self._preprocess_angels(head_pose_angles)
        return right_eye_image, head_pose_angles, left_eye_image   

    def _preprocess_angels(self, head_pose_angles):
        input_shape = self.get_input_shape("head_pose_angles")
        head_pose_angles = np.reshape(head_pose_angles, input_shape)
        return head_pose_angles

    def _preprocess_output(self, outputs, roll):
        gaze_vector = outputs.buffer[0]
        gaze_vector_n = gaze_vector / np.linalg.norm(gaze_vector)
        gaze_vector_n[2] = (-1)*gaze_vector_n[2]
        # vcos = math.cos(math.radians(roll))
        # vsin = math.sin(math.radians(roll))
        # x =  gaze_vector_n[0]*vcos + gaze_vector_n[1]*vsin
        # y = -gaze_vector_n[0]*vsin + gaze_vector_n[1]*vcos
        # return [x, y], total_preprocess_time
        return gaze_vector_n

class OpenClosedEye(Model):
    """
    Fixed eye state detection - always returns open eyes
    """
    def __init__(self, model_name, device="CPU", extensions=None):
        # 不需要真正初始化模型，直接设置固定返回值
        self.out_val = ["close", "open"]
        
    def predict(self, right_eye_image, left_eye_image):
        # 固定返回两只眼睛都是睁开状态
        return ["open", "open"]
        
    def _preprocess_output(self, outputs):
        # 固定返回"open"
        return "open"


class MediaPipeFace:
    """
    MediaPipe面部检测与关键点检测模块
    替换OpenVINO的FaceDetection和FacialLandmarkDetection
    """
    
    def __init__(self):
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,          # 视频流必须 False
            max_num_faces=1,                  # 单人场景，减少误检
            refine_landmarks=True,             # 虹膜 + 眼部细化（强烈建议）
            min_detection_confidence=0.7,      # ↑ 减少假检
            min_tracking_confidence=0.75       # ↑ 抗抖动、遮挡
        )
        
        # 眼睛8点包围圈 (用于裁切)
        # 顺序：外眼角, 内眼角, 上缘(3点), 下缘(3点)
        self.l_eye_8 = [33, 133, 160, 159, 158, 144, 145, 153]
        self.r_eye_8 = [362, 263, 385, 386, 387, 373, 374, 380]
        
        # 稳定面部关键点（35点）
        self.stable_indices = [10,151,9,168,6,1,152,103,332,54,284,21,251,127,356,133,362,33,263,234,454,93,323,58,288,197,5,2,164,172,397,176,400,148,377]
        
        # PnP头部姿态估计相关初始化
        self._init_pnp_parameters()
    
    def _init_pnp_parameters(self):
        """初始化PnP头部姿态估计参数"""
        # 35个稳定的人脸关键点索引（与stable_indices保持一致）
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
        
        self.rvec = None
        self.tvec = None
        
        # 坐标系修正矩阵（可选）
        self.R_fix = np.eye(3)
        
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
        """使用PnP算法估计头部姿态
        
        返回:
            tvec: PnP绝对平移向量，失败时返回None
        """
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
                # 将 rvec 转换为旋转矩阵并返回
                R_matrix = cv2.Rodrigues(rvec)[0]
                return tvec, R_matrix
        
        return None, None

    def detect(self, frame):
        """
        检测人脸和关键点
        
        返回：
        - face_box: [xmin, ymin, xmax, ymax]
        - eye_boxes: [[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax]] (右眼, 左眼)
        - eye_centers: [[x,y], [x,y]] (右眼, 左眼)
        - landmarks_35: (len(stable_indices), 2) ndarray
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        pts = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in face_landmarks.landmark
        ])

        # ---------- face box ----------
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        face_box = [xmin, ymin, xmax, ymax]

        # ---------- 眼部中心点 ----------
        # 使用眼睛8点包围圈的中心作为眼部中心
        left_eye_center = pts[self.l_eye_8].mean(axis=0).astype(int)
        right_eye_center = pts[self.r_eye_8].mean(axis=0).astype(int)

        # ---------- 眼部包围框 ----------
        # 眼部裁切：长*2，宽*1.2
        def get_eye_box(eye_center, eye_points):
            # 计算眼部包围框
            eye_pts = pts[eye_points]
            x_min, y_min = eye_pts.min(axis=0)
            x_max, y_max = eye_pts.max(axis=0)
            
            # 扩展包围框：长*2，宽*1.2
            eye_width = max(x_max - x_min, 10)  # 最小宽度为10
            eye_height = max(y_max - y_min, 6)  # 最小高度为6
            
            center_x, center_y = eye_center

            new_width = max(int(eye_width * 1.5), 20)  # 最小宽度20
            new_height = new_width  
            
            xmin = max(0, center_x - new_width // 2)
            ymin = max(0, center_y - new_height // 2)
            xmax = min(w, center_x + new_width // 2)
            ymax = min(h, center_y + new_height // 2)
            
            
            return [int(xmin), int(ymin), int(xmax), int(ymax)]

        right_eye_box = get_eye_box(right_eye_center, self.r_eye_8)
        left_eye_box = get_eye_box(left_eye_center, self.l_eye_8)
        
        
        eye_boxes = [right_eye_box, left_eye_box]
        eye_centers = [right_eye_center.tolist(), left_eye_center.tolist()]

        # ---------- 35个稳定关键点 ----------
        if len(self.stable_indices) >= 35:
            selected_indices = self.stable_indices[:35]
        else:
            # 如果稳定点不够35个，补充其他点
            selected_indices = self.stable_indices + [i for i in range(468) if i not in self.stable_indices][:35-len(self.stable_indices)]
        
        landmarks_35 = pts[selected_indices]

        return face_box, eye_boxes, eye_centers, landmarks_35

class EyeModel:
    '''
    Computes the gaze Vector    
    '''
    def __init__(self, directory, subdir_face=os.path.join("intel","face-detection-adas-0001","FP32"),subdir_landmark=os.path.join("intel","landmarks-regression-retail-0009","FP32"),\
                                subdir_headpose=os.path.join("intel","head-pose-estimation-adas-0001","FP32"), subdir_gaze=os.path.join("intel","gaze-estimation-adas-0002","FP32"),\
                                subdir_open_close=os.path.join("intel","open-closed-eye-0001","FP32"),\
                                subdir_landmark_35 = os.path.join("intel","facial-landmarks-35-adas-0002","FP32"), device="AUTO") -> None:


        self.mp_face = MediaPipeFace()
   
        # 移除眼部检测模型初始化，只保留注视估计和头部姿态估计
        self.head_pose_estimation = HeadPoseEstimation(os.path.join(directory,subdir_headpose))
        self.gaze_estimation = GazeEstimation(os.path.join(directory,subdir_gaze))
        self.open_close_eye = OpenClosedEye(os.path.join(directory,subdir_open_close))  # 现在只是占位符

        self.head_pose_estimation.load_model()
        self.gaze_estimation.load_model()
        # 不再需要加载眼部检测模型
        
        self.right_eye_prev = np.array([])
        self.left_eye_prev = np.array([])

        self.QueueGaze = np.nan*np.zeros((3,5))

        self.dir = directory

    def get_crop_image(self, image, box):
        xmin, ymin, xmax, ymax = box
        
        # 验证输入参数
        if image is None or image.size == 0:
            print(f"❌ 输入图像为空")
            return None
            
        h, w = image.shape[:2]
        
        # 确保坐标在有效范围内
        xmin = max(0, min(xmin, w-1))
        ymin = max(0, min(ymin, h-1))
        xmax = max(xmin+1, min(xmax, w))
        ymax = max(ymin+1, min(ymax, h))
        
        if xmax <= xmin or ymax <= ymin:
            print(f"❌ 无效的裁切框: [{xmin}, {ymin}, {xmax}, {ymax}], 图像尺寸: {w}x{h}")
            return None
            
        crop_image = image[ymin:ymax, xmin:xmax]
        
        # 检查裁切结果
        if crop_image is None or crop_image.size == 0:
            print(f"❌ 裁切后图像为空，原始框: [{xmin}, {ymin}, {xmax}, {ymax}]")
            return None
            
        return crop_image

    def draw_eye_line(self, image, face_box, eye_boxes, eye_centers, gaze_x, gaze_y):
        """
        使用absolute坐标绘制可视化和注视向量
        
        Args:
            image: 原始帧 (absolute坐标系统)
            face_box: 人脸框 [xmin,ymin,xmax,ymax] (absolute坐标)
            eye_boxes: 眼部框列表 (absolute坐标)  
            eye_centers: 眼部中心点列表 (absolute坐标)
            gaze_x, gaze_y: 注视向量
        """
        xmin, ymin, xmax, ymax = face_box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
        
        # 直接使用absolute坐标绘制眼部框和中心点
        if len(eye_boxes) > 0:
            # eye_boxes已经是absolute frame坐标，直接绘制
            for i, eye_box in enumerate(eye_boxes):
                xmin2, ymin2, xmax2, ymax2 = eye_box
                cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), (255,255,255), 1)
            
            # 使用absolute坐标绘制注视向量
            for x, y in eye_centers:
                start = (x, y)
                end = (x + int(gaze_x*90), y - int(gaze_y*90))
                cv2.arrowedLine(image, start, end, (0,0,255), 2)


    def _OpticalFlow(self, eye_prev, eye_image):
        try:
            if np.size(eye_prev) > 0:
                if np.shape(eye_prev) != np.shape(eye_image):
                    eye_prev = cv2.resize(eye_prev, np.shape(eye_image)[1::-1], interpolation = cv2.INTER_AREA)
                prvs = cv2.cvtColor(eye_prev, cv2.COLOR_BGR2GRAY)
                next = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                return np.mean(flow, axis=(0,1))
            else:
                return np.array([0,0])
        except Exception as e:
            print(f"Error in Optical Flow: {e}")
            return np.array([0,0])

    def get_gaze(self, frame, imshow=False, face_boxes=None):
        """
        生成并返回当前帧的注视信息和35个人脸关键点
        1. 使用MediaPipe检测人脸和关键点（如果未提供）
        2. 提取双眼图像与中心坐标
        3. 估算头部姿态（OpenVINO模型）
        4. 使用PnP算法估算绝对头部姿态（tvec/rvec）
        5. 估算注视向量（经中值滤波平滑）
        
        返回:
            eye_info: 注视信息字典，包含：
                - gaze: 注视向量 (3,)
                - EyeRLCenterPos: 双眼中心位置 (relative坐标)
                - HeadPosAnglesYPR: 头部姿态欧拉角 (OpenVINO模型)
                - HeadPosInFrame: 人脸在帧中的位置 (absolute坐标)
                - right_eye_box, left_eye_box: 眼部包围框 (relative坐标)
                - EyeState: 睁眼状态 [right, left]
            landmarks_35: 35个稳定人脸关键点坐标 (35, 2)
            pnp_info: PnP头部姿态信息字典，包含：
                - pnp_tvec: PnP绝对平移向量 (3,1) 或 None
                - pnp_distance: PnP计算的相机距离(cm) 或 None
                - pnp_success: PnP是否成功的布尔值
        """
        open_close = {'close': 0, 'open': 1}
        eye_info = None
        landmarks_35 = None
        mp_result = self.mp_face.detect(frame)
        if mp_result is None:
            print("No face detected for get_gaze!")
            cv2.imwrite(os.path.join(self.dir, "results", "no_face.jpg"), frame)
            return None, None, None
                
        face_box, eye_boxes, eye_centers, landmarks_35 = mp_result
        landmarks_points = np.c_[landmarks_35, np.ones((landmarks_35.shape[0], 1))].T
        face = self.get_crop_image(frame, face_box)
        face_boxes = [face_box]
        
        # 处理第一张人脸（通常也是唯一的人脸）
        for face_box in face_boxes:
            # 扩大人脸框
            face_box = expand_face_box(face_box, scale=1.2)
            face = self.get_crop_image(frame, face_box)
            if face is None:
                continue

            # MediaPipe返回的是absolute坐标，需要转换为relative坐标用于crop
            # 使用独立的坐标转换函数
            relative_eye_boxes, relative_eye_centers = to_relative_coordinates(
                eye_boxes, eye_centers, face_box
            )
            
            # 获取双眼图像（使用relative坐标进行裁切）
            right_eye_image, left_eye_image = [
                self.get_crop_image(face, eye_box) for eye_box in relative_eye_boxes
            ]
            
            # 检查眼部图像是否有效
            if right_eye_image is None or left_eye_image is None:
                print("❌ 眼部图像裁切失败")
                continue

            # 头部姿态（yaw, pitch, roll）
            head_pose_angles = self.head_pose_estimation.predict(face)

            gaze_vector = self.gaze_estimation.predict(
                right_eye_image, head_pose_angles, left_eye_image
            )
            gaze_vector = util.MedianFilter(self.QueueGaze, gaze_vector)

            # 头部/眼睛在图像中的位置（使用relative坐标）
            xmin, ymin, xmax, ymax = face_box
            head_box = np.array([xmin, ymin])
            right_eye_box = np.array([relative_eye_boxes[0][0], relative_eye_boxes[0][1]])
            left_eye_box  = np.array([relative_eye_boxes[1][0], relative_eye_boxes[1][1]])

            # 睁眼/闭眼状态 - 固定返回睁开状态
            out_right, out_left = self.open_close_eye.predict(
                right_eye_image, left_eye_image
            )

            # PnP头部姿态估计（获取绝对位移tvec和旋转矩阵）
            pnp_result = self.mp_face.get_pose(frame)
            pnp_tvec = None
            pnp_R = None
            pnp_success = False
            pnp_distance = None
            
            if pnp_result is not None:
                pnp_tvec, pnp_R = pnp_result
                if pnp_tvec is not None and pnp_R is not None:
                    pnp_success = True
                    # 验证 pnp_tvec 的合理性 - 应该都在合理范围内（-1000 到 1000 mm）
                    if (np.abs(pnp_tvec) > 10000).any():
                        print(f"PnP tvec 值异常: {pnp_tvec.flatten()}")
                        pnp_tvec = None
                        pnp_R = None
                        pnp_success = False
                    else:
                        pnp_distance = pnp_tvec[2, 0] / 10.0  # 从毫米转换为厘米
            else:
                print("PnP头部姿态估计失败")

            # 组装结果字典
            eye_info = {
                'gaze': gaze_vector,
                'EyeRLCenterPos': np.array(relative_eye_centers).reshape(-1),  # 相对坐标
                'HeadPosAnglesYPR': head_pose_angles,
                'HeadPosInFrame': head_box,  # absolute坐标
                'right_eye_box': right_eye_box,  # relative坐标  
                'left_eye_box': left_eye_box,   # relative坐标
                'EyeState': [open_close[out_right], open_close[out_left]]
            }
            
            # PnP信息作为单独的返回值
            pnp_info = {
                'pnp_tvec': pnp_tvec,      # PnP绝对平移向量
                'pnp_R': pnp_R,            # PnP旋转矩阵
                'pnp_distance': pnp_distance,  # PnP计算的距离(cm)
                'pnp_success': pnp_success  # PnP是否成功的布尔值
            }

            # 可视化（使用absolute坐标绘制）
            self.draw_eye_line(
                frame, face_box, eye_boxes, eye_centers,  # 传入absolute坐标
                gaze_vector[0], gaze_vector[1]
            )
            if imshow:
                cv2.putText(
                    frame,
                    f"RightEye {out_right}; LeftEye {out_left}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
                cv2.imshow('image', frame)
                
            # 只处理第一张人脸
            break
        
        # 返回注视信息、关键点和PnP信息
        return eye_info, landmarks_points, pnp_info

    def get_FaceFeatures(self, frame, imshow=False, face_boxes=None):
        # 如果没有提供人脸框，则执行人脸检测
        mp_result = self.mp_face.detect(frame)
        face_box, _, _, landmarks_35 = mp_result
        if mp_result is None:
            print("No face detected for get_FaceFeatures!")
            return np.zeros((3, 35))
        return np.c_[landmarks_35, np.ones((landmarks_35.shape[0], 1))].T
       