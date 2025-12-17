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
        # 检查model_name是目录路径还是文件路径
        if os.path.isdir(model_name):
            # 如果是目录路径，查找目录中的模型文件
            model_files = [f for f in os.listdir(model_name) if f.endswith(('.xml', '.bin'))]
            if not model_files:
                raise ValueError(f"No model files found in directory: {model_name}")
            
            # 提取模型文件名（不带扩展名）
            base_names = set(os.path.splitext(f)[0] for f in model_files)
            if len(base_names) != 1:
                raise ValueError(f"Multiple model files found in directory: {model_name}")
            
            model_base = os.path.join(model_name, list(base_names)[0])
            model_weights = model_base + '.bin'
            model_structure = model_base + '.xml'
        else:
            # 如果是文件路径（不带扩展名）
            model_weights = model_name + '.bin'
            model_structure = model_name + '.xml'
        
        self.core = IECore()
        if extensions and "CPU" in device:
            self.core.add_extension(extensions, device)
        try:
            self.model = self.core.read_network(model=model_structure, weights=model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

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

class FaceDetection(Model):
    '''
    Face detection
    '''
    def __init__(self, model_name, device="CPU", extensions=None, threshold=0.60):
        super().__init__(model_name, device, extensions)
        self.threshold = threshold

    def predict(self, image):
        input_image = self._preprocess_input(image)
        self.exec_net(0, input_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            face_boxes = self._preprocess_output(outputs, image)
            return face_boxes

    def _preprocess_output(self, outputs, image):
        face_boxes = []
        height, width, _ = image.shape        
        for obj in outputs.buffer[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * width)
                ymin = int(obj[4] * height)
                xmax = int(obj[5] * width)
                ymax = int(obj[6] * height)
                face_boxes.append([xmin, ymin, xmax, ymax])                
        return face_boxes

class FacialLandmarkDetection(Model):
    '''
    Facial Landmark Detection Model    
    '''
    def __init__(self, model_name, device="CPU", extensions=None):
        super().__init__(model_name, device, extensions)

    def predict(self, face_image):
        input_image = self._preprocess_input(face_image)
        self.exec_net(0, input_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            eye_boxes, eye_centers = self._preprocess_output(outputs, face_image)
            return eye_boxes, eye_centers

    def _preprocess_output(self, outputs, image):
        normalized_landmarks = np.squeeze(outputs.buffer).reshape((5,2))
        h, w, _ = image.shape        
        length_offset = int(w * 0.15) 
        eye_boxes, eye_centers = [], []
        for i in range(2):
            normalized_x, normalized_y = normalized_landmarks[i]
            x = int(normalized_x*w)
            y = int(normalized_y*h)
            eye_centers.append([x, y])
            xmin, xmax = max(0, x - length_offset), min(w, x + length_offset)
            ymin, ymax = max(0, y - length_offset), min(h, y + length_offset)
            eye_boxes.append([xmin, ymin, xmax, ymax])            
        return eye_boxes, eye_centers

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
    Determine state of eye
    """
    def __init__(self, model_name, device="CPU", extensions=None):
        super().__init__(model_name, device, extensions)
        self.out_val = ["close", "open"]

    def predict(self, right_eye_image, left_eye_image):
        # right eye
        right_eye_image = self._preprocess_input(right_eye_image)
        self.exec_net(0, right_eye_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            out_right = self._preprocess_output(outputs)
        # left eye
        left_eye_image = self._preprocess_input(left_eye_image)
        self.exec_net(0, left_eye_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            out_left = self._preprocess_output(outputs)
        return [out_right, out_left]

    def _preprocess_output(self, outputs):
        return self.out_val[np.argmax(np.reshape(outputs.buffer, (1,2)))]

class Pupils(Model):
    """
    Computes the pupil diameter
    """
    def __init__(self, model_name, device="CPU", extensions=None):
        super().__init__(model_name, device, extensions)

    def _preprocess_input(self, image, input_name=None):
        start_preprocess_time = time.time()
        n, h, w, c = self.get_input_shape(input_name)
        input_image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        input_image = input_image.transpose((0, 1))  # requires channel, hight, width
        input_image = input_image.reshape((n, h, w, c))
        #input_image = input_image.reshape(1, *input_image.shape)
        total_preprocess_time = time.time() - start_preprocess_time
        return input_image, total_preprocess_time

    def predict(self, eye_image):
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)# Convert to grayscale
        eye_image = cv2.bitwise_not(eye_image)
        eye_image = eye_image.astype(np.float32) / 255.0  # convert from uint8 [0, 255] to float32 [0, 1]        
        eye_image, _ = self._preprocess_input(eye_image)
        self.exec_net(0, eye_image)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            prediction, morph, centroid = self._preprocess_output(outputs)
            return eye_image[0,:,:,0], prediction, morph, centroid

    def _preprocess_output(self, outputs):
        image = outputs.buffer[0,:,:,0]
        THRESHOLD = 0.5
        IMCLOSING = 13 # pixel radius of circular kernel for morphological closing
        # Binarize 
        binarized = image > THRESHOLD
        # Divide in regions and keep only the biggest
        label_img = label(binarized)
        regions = regionprops(label_img)
        if len(regions)==0:
            morph = np.zeros(image.shape, dtype='uint8')
            centroid = (np.nan, np.nan)
            return (image, morph, centroid)
        regions.sort(key=lambda x: x.area, reverse=True)
        centroid = regions[0].centroid # centroid coordinates of the biggest object
        if len(regions) > 1:
            for rg in regions[1:]:
                label_img[rg.coords[:,0], rg.coords[:,1]] = 0
        label_img[label_img!=0] = 1
        biggestRegion = (label_img*255).astype(np.uint8)
        # Morphological
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(IMCLOSING,IMCLOSING))
        morph = cv2.morphologyEx(biggestRegion, cv2.MORPH_CLOSE, kernel)
        return (image, morph, centroid)

    def get_iris_pupil(self, eye_image):
        # Find the iris
        image = self._do_gray_blur(eye_image, 17)
        try:
            iris = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if iris != None:
                self._draw_circle(eye_image, iris)
            # Find the pupil
            image = self._do_gray_blur(eye_image, 19)
            pupil = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if pupil != None:
                self._draw_circle(eye_image, pupil)

        except Exception as e:
            print(f"Could not find pupil: {e}")

    def _do_gray_blur(self, image, blur_factor):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # return cv2.medianBlur(image, blur_factor)
        blur = cv2.GaussianBlur(image, (5,5), 10)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(blur, kernel, iterations = 2)
        # ret3,th3 = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return erosion

    # Detect the iris or pupil
    def _draw_circle(self, image, circles):
        for c in circles[0,:]:
            cv2.circle(image,(c[0], c[1]), c[2], (0,0,255), 1)

class OGTransGt(Model):
    """
    Computes
    """
    def __init__(self, dir, model_name="intel\\CNNGTransB\\h_model", device="CPU", extensions=None):
        model_name = dir+model_name
        super().__init__(model_name, device, extensions)
        self.load_model()

    def predict(self, X_data):
        """ screen gaze vector """
        self.exec_net(0, X_data)
        status = self.wait(0)
        if status == 0:
            outputs = self.get_output(0)
            vec = self._preprocess_output(outputs)
        return vec

    def _preprocess_output(self, outputs):
        return outputs.buffer

class EyeModel:
    '''
    Computes the gaze Vector    
    '''
    def __init__(self, directory, subdir_face=os.path.join("intel","face-detection-adas-0001","FP32"),subdir_landmark=os.path.join("intel","landmarks-regression-retail-0009","FP32"),\
                                subdir_headpose=os.path.join("intel","head-pose-estimation-adas-0001","FP32"), subdir_gaze=os.path.join("intel","gaze-estimation-adas-0002","FP32"),\
                                subdir_open_close=os.path.join("intel","open-closed-eye-0001","FP32"), subdir_pupil=os.path.join("intel","PupilSegmentation","pupils_segmentation"),\
                                subdir_landmark_35 = os.path.join("intel","facial-landmarks-35-adas-0002","FP32"), device="AUTO") -> None:

        self.face_detection = FaceDetection(os.path.join(directory,subdir_face))
        self.facial_landmark_detection = FacialLandmarkDetection(os.path.join(directory,subdir_landmark))
        self.head_pose_estimation = HeadPoseEstimation(os.path.join(directory,subdir_headpose))
        self.gaze_estimation = GazeEstimation(os.path.join(directory,subdir_gaze))
        self.open_close_eye = OpenClosedEye(os.path.join(directory,subdir_open_close))
        self.pupil = Pupils(os.path.join(directory,subdir_pupil))
        self.facial_landmark_35 = FacialLandmarkDetection35(os.path.join(directory,subdir_landmark_35))

        self.face_detection.load_model()
        self.facial_landmark_detection.load_model()
        self.head_pose_estimation.load_model()
        self.gaze_estimation.load_model()
        self.open_close_eye.load_model()
        self.pupil.load_model()
        self.facial_landmark_35.load_model()
        self.right_eye_prev = np.array([])
        self.left_eye_prev = np.array([])

        self.QueueGaze = np.nan*np.zeros((3,5))

        self.dir = directory

    def get_crop_image(self, image, box):
        xmin, ymin, xmax, ymax = box
        crop_image = image[ymin:ymax, xmin:xmax]
        return crop_image

    def draw_gaze_point(self, image, gaze_x, gaze_y):
        cv2.circle(image,(int(gaze_x), int(gaze_y)), 10, (0,0,255), -1)

    def draw_eye_line(self, image, face_box, eye_boxes, eye_centers, gaze_x, gaze_y):
        xmin, ymin, xmax, ymax = face_box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
        for eye_box in eye_boxes:
            xmin2, ymin2, xmax2, ymax2 = eye_box
            cv2.rectangle(image, (xmin+xmin2, ymin+ymin2), (xmin+xmax2, ymin+ymax2), (255,255,255), 1)
        for x, y in eye_centers:
            start = (x+xmin, y+ymin)
            end = (x+xmin+int(gaze_x*90), y+ymin-int(gaze_y*90))
            cv2.arrowedLine(image, start, end, (0,0,255), 2)

    def get_faces(self, frame):
        face_boxes = self.face_detection.predict(frame)
        faces = []        
        for face_box in face_boxes:
            face_image = self.get_crop_image(frame, face_box)
            faces.append(face_image)
        return faces

    def get_eye_pairs(self, faces):
        eye_pairs = []        
        for face in faces:
            eye_boxes, eye_centers = self.facial_landmark_detection.predict(face)
            # get eye images
            right_eye_image, left_eye_image = [self.get_crop_image(face, eye_box) for eye_box in eye_boxes]
            eye_pairs.append((right_eye_image, left_eye_image, eye_centers))
        return eye_pairs

    def get_head_poses(self, faces):
        head_poses = []        
        for face in faces:
            head_pose_angles = self.head_pose_estimation.predict(face)
            head_poses.append(head_pose_angles)
        return head_poses

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
        生成并返回当前帧的注视信息
        1. 检测人脸（如果未提供）
        2. 提取双眼图像与中心坐标
        3. 估算头部姿态
        4. 估算注视向量（经中值滤波平滑）
        5. 判断睁眼/闭眼状态
        6. 组装并返回结构化结果
        """
        # 如果没有提供人脸框，则执行人脸检测
        if face_boxes is None:
            face_boxes = self.face_detection.predict(frame)
        open_close = {'close': 0, 'open': 1}
        eye_info = None

        for face_box in face_boxes:
            # 裁剪人脸区域
            face = self.get_crop_image(frame, face_box)

            # 获取双眼框与中心点
            eye_boxes, eye_centers = self.facial_landmark_detection.predict(face)
            right_eye_image, left_eye_image = [
                self.get_crop_image(face, eye_box) for eye_box in eye_boxes
            ]

            # 头部姿态（yaw, pitch, roll）
            head_pose_angles = self.head_pose_estimation.predict(face)

            # 注视向量估计 + 中值滤波平滑
            gaze_vector = self.gaze_estimation.predict(
                right_eye_image, head_pose_angles, left_eye_image
            )
            gaze_vector = util.MedianFilter(self.QueueGaze, gaze_vector)

            # 头部/眼睛在图像中的位置
            xmin, ymin, xmax, ymax = face_box
            head_box = np.array([xmin, ymin])
            right_eye_box = np.array([eye_boxes[0][0], eye_boxes[0][1]])
            left_eye_box  = np.array([eye_boxes[1][0], eye_boxes[1][1]])

            # 睁眼/闭眼状态
            out_right, out_left = self.open_close_eye.predict(
                right_eye_image, left_eye_image
            )

            # 组装结果字典
            # 返回结果字典，保持与原始代码的数据结构完全一致
            eye_info = {
                'gaze': gaze_vector,
                'EyeRLCenterPos': np.array(eye_centers).reshape(-1),  # 恢复原始格式
                'HeadPosAnglesYPR': head_pose_angles,
                'HeadPosInFrame': head_box,
                'right_eye_box': right_eye_box,
                'left_eye_box': left_eye_box,
                'EyeState': [open_close[out_right], open_close[out_left]]
            }

            # 可视化
            self.draw_eye_line(
                frame, face_box, eye_boxes, eye_centers,
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

        # 未检测到人脸
        if eye_info is None:
            print("No face detected for get_gaze!")
            cv2.imwrite(os.path.join(self.dir, "results", "no_face.jpg"), frame)

        return eye_info

    def get_FaceFeatures(self, frame, imshow=False, face_boxes=None):
        # 如果没有提供人脸框，则执行人脸检测
        if face_boxes is None:
            face_boxes = self.face_detection.predict(frame)
        if len(face_boxes) < 1:
            print("No face detected for get_FaceFeatures!")
            cv2.imwrite(os.path.join(self.dir, "results", "no_face.jpg"), frame)
            # 返回一个合适的2维数组，包含35个特征点
            return np.zeros((3, 35))

        else:
            if len(face_boxes) > 1:
                print("Warning: More than one face detected! Using first face detected!")
            face_box = face_boxes[0]
            face = self.get_crop_image(frame, face_box)
            landmarks = self.facial_landmark_35.predict(face)   # shape 35,2
            
            # 确保返回35个特征点，与sfm_module.py中期望的一致
            xmin, ymin, xmax, ymax = face_box
            points = np.zeros_like(landmarks)  # 保持为35个特征点
            
            for idx, pos in enumerate(landmarks):
                x = pos[0] + xmin
                y = pos[1] + ymin
                points[idx] = [x, y]

            return np.c_[points, np.ones((points.shape[0], 1))].T

