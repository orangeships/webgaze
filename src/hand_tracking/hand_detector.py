"""
基于MediaPipe的手部检测器

该模块实现了基于MediaPipe的手部检测功能，支持：
- 实时手部检测和关键点识别
- 单手和双手检测
- 手部关键点可视化
- 手势识别基础功能
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math


class HandDetector:
    """
    基于MediaPipe的手部检测器类
    
    支持实时手部检测和关键点识别，提供手部关键点坐标、
    手势识别和可视化功能。
    """
    
    # MediaPipe初始化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    def __init__(self, 
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.8,
                 min_tracking_confidence: float = 0.8,
                 model_complexity: int = 1,
                 static_image_mode: bool = False):
        """
        初始化手部检测器
        
        Args:
            max_num_hands: 最大检测手部数量
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            model_complexity: 模型复杂度 (0, 1, 2)
            static_image_mode: 是否为静态图像模式
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode
        
        # 初始化MediaPipe手部检测
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        # 手部关键点索引定义
        self.hand_landmarks = {
            # 手腕
            'WRIST': 0,
            # 拇指
            'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
            # 食指
            'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
            # 中指
            'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
            # 无名指
            'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
            # 小指
            'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
        }
        
        # 手指名称列表
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
    def detect_hands(self, image: np.ndarray, draw: bool = True) -> List[Dict[str, Any]]:
        """
        检测图像中的手部
        
        Args:
            image: 输入图像 (BGR格式)
            draw: 是否绘制关键点和连线
            
        Returns:
            List[Dict]: 检测到的手部信息列表
        """
        # 转换颜色空间 BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_image)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 获取手部信息
                hand_info = self._extract_hand_info(hand_landmarks, hand_idx)
                hands_data.append(hand_info)
                
                # 绘制关键点和连线
                if draw:
                    self._draw_hand_landmarks(image, hand_landmarks, hand_idx)
        
        return hands_data
    
    def _extract_hand_info(self, hand_landmarks, hand_idx: int) -> Dict[str, Any]:
        """
        提取手部关键点信息
        
        Args:
            hand_landmarks: MediaPipe手部关键点
            hand_idx: 手部索引
            
        Returns:
            Dict: 手部信息字典
        """
        hand_info = {
            'hand_id': hand_idx,
            'landmarks': [],
            'landmarks_2d': [],
            'center': None,
            'fingers_status': {},
            'gesture': None
        }
        
        # 提取关键点坐标
        landmarks_3d = []
        landmarks_2d = []
        
        for landmark in hand_landmarks.landmark:
            # 3D坐标 (归一化)
            landmarks_3d.append([landmark.x, landmark.y, landmark.z])
            
            # 2D像素坐标
            landmarks_2d.append([landmark.x, landmark.y])
        
        hand_info['landmarks'] = landmarks_3d
        hand_info['landmarks_2d'] = landmarks_2d
        
        # 计算手部中心点
        center_x = sum(point[0] for point in landmarks_2d) / len(landmarks_2d)
        center_y = sum(point[1] for point in landmarks_2d) / len(landmarks_2d)
        hand_info['center'] = [center_x, center_y]
        
        # 检测手指状态
        hand_info['fingers_status'] = self._detect_fingers_status(landmarks_2d)
        
        # 识别手势
        hand_info['gesture'] = self._recognize_gesture(hand_info['fingers_status'])
        
        return hand_info
    
    def _detect_fingers_status(self, landmarks_2d: List[List[float]]) -> Dict[str, bool]:
        """
        检测手指的弯曲状态
        
        Args:
            landmarks_2d: 2D关键点坐标列表
            
        Returns:
            Dict: 各手指的状态 (True=伸直, False=弯曲)
        """
        fingers_status = {
            'thumb': False,
            'index': False, 
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        try:
            # 拇指检测 (比较拇指尖和拇指掌指关节)
            thumb_tip = landmarks_2d[self.hand_landmarks['THUMB_TIP']]
            thumb_mcp = landmarks_2d[self.hand_landmarks['THUMB_MCP']]
            
            # 横向比较 (因为拇指是横向的)
            fingers_status['thumb'] = thumb_tip[0] < thumb_mcp[0] if thumb_tip[0] < 0.5 else thumb_tip[0] > thumb_mcp[0]
            
            # 其他手指检测 (垂直比较)
            finger_tips = [
                self.hand_landmarks['INDEX_FINGER_TIP'],
                self.hand_landmarks['MIDDLE_FINGER_TIP'], 
                self.hand_landmarks['RING_FINGER_TIP'],
                self.hand_landmarks['PINKY_TIP']
            ]
            
            finger_mcps = [
                self.hand_landmarks['INDEX_FINGER_MCP'],
                self.hand_landmarks['MIDDLE_FINGER_MCP'],
                self.hand_landmarks['RING_FINGER_MCP'], 
                self.hand_landmarks['PINKY_MCP']
            ]
            
            for i, (tip_idx, mcp_idx) in enumerate(zip(finger_tips, finger_mcps)):
                finger_name = self.finger_names[i + 1]  # 跳过拇指
                tip_y = landmarks_2d[tip_idx][1]
                mcp_y = landmarks_2d[mcp_idx][1]
                
                # 如果指尖y坐标小于掌指关节y坐标，说明手指伸直
                fingers_status[finger_name] = tip_y < mcp_y
                
        except IndexError as e:
            print(f"关键点索引错误: {e}")
            
        return fingers_status
    
    def _recognize_gesture(self, fingers_status: Dict[str, bool]) -> str:
        """
        基于手指状态识别手势
        
        Args:
            fingers_status: 手指状态字典
            
        Returns:
            str: 手势名称
        """
        extended_fingers = [name for name, status in fingers_status.items() if status]
        
        if len(extended_fingers) == 0:
            return "fist"
        elif len(extended_fingers) == 1:
            if 'thumb' in extended_fingers:
                return "thumb_up"
            else:
                return "one_finger"
        elif len(extended_fingers) == 2:
            if 'thumb' in extended_fingers and 'index' in extended_fingers:
                return "thumbs_up"
            elif 'index' in extended_fingers and 'middle' in extended_fingers:
                return "two_fingers"
            else:
                return "two_fingers"
        elif len(extended_fingers) == 3:
            if all(f in extended_fingers for f in ['index', 'middle', 'ring']):
                return "three_fingers"
            else:
                return "three_fingers"
        elif len(extended_fingers) == 4:
            return "four_fingers"
        elif len(extended_fingers) == 5:
            return "open_hand"
        else:
            return "unknown"
    
    def _draw_hand_landmarks(self, image: np.ndarray, hand_landmarks, hand_idx: int):
        """
        绘制手部关键点和连线
        
        Args:
            image: 输入图像
            hand_landmarks: MediaPipe手部关键点
            hand_idx: 手部索引
        """
        # 绘制关键点和连线
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # 绘制关键点索引
        for idx, landmark in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # 绘制关键点
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
            
            # 绘制索引号
            cv2.putText(image, str(idx), (cx + 5, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def get_hand_center(self, hands_data: List[Dict[str, Any]], hand_id: int = 0) -> Optional[Tuple[float, float]]:
        """
        获取指定手部的中心点坐标
        
        Args:
            hands_data: 手部检测结果
            hand_id: 手部ID
            
        Returns:
            Tuple[float, float]: 中心点坐标 (x, y)，如果未找到则返回None
        """
        for hand_info in hands_data:
            if hand_info['hand_id'] == hand_id and hand_info['center']:
                return hand_info['center']
        return None
    
    def get_finger_positions(self, hands_data: List[Dict[str, Any]], 
                           hand_id: int = 0) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        获取指定手部各手指的指尖位置
        
        Args:
            hands_data: 手部检测结果
            hand_id: 手部ID
            
        Returns:
            Dict[str, Tuple[float, float]]: 各手指位置坐标
        """
        for hand_info in hands_data:
            if hand_info['hand_id'] == hand_id:
                landmarks_2d = hand_info['landmarks_2d']
                return {
                    'thumb': landmarks_2d[self.hand_landmarks['THUMB_TIP']],
                    'index': landmarks_2d[self.hand_landmarks['INDEX_FINGER_TIP']],
                    'middle': landmarks_2d[self.hand_landmarks['MIDDLE_FINGER_TIP']],
                    'ring': landmarks_2d[self.hand_landmarks['RING_FINGER_TIP']],
                    'pinky': landmarks_2d[self.hand_landmarks['PINKY_TIP']]
                }
        return None
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        计算两点之间的距离
        
        Args:
            point1: 第一个点坐标 (x, y)
            point2: 第二个点坐标 (x, y)
            
        Returns:
            float: 距离值
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def draw_hand_info(self, image: np.ndarray, hands_data: List[Dict[str, Any]]):
        """
        在图像上绘制手部信息
        
        Args:
            image: 输入图像
            hands_data: 手部检测结果
        """
        for hand_info in hands_data:
            center = hand_info['center']
            gesture = hand_info['gesture']
            fingers_status = hand_info['fingers_status']
            
            if center:
                # 绘制中心点
                h, w, c = image.shape
                center_x, center_y = int(center[0] * w), int(center[1] * h)
                cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), -1)
                
                # 绘制手势信息
                cv2.putText(image, gesture, (center_x + 15, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 绘制手指状态
                y_offset = 30
                for finger, status in fingers_status.items():
                    status_text = "UP" if status else "DOWN"
                    color = (0, 255, 0) if status else (0, 0, 255)
                    cv2.putText(image, f"{finger}: {status_text}", 
                               (center_x + 15, center_y + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20