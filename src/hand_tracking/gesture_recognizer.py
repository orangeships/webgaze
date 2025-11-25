"""
手势识别器

该模块提供了基于手部关键点的高级手势识别功能，支持：
- 数字手势识别 (0-10)
- 常用手势识别 (点赞、比心、OK等)
- 手势序列识别
- 自定义手势定义
- 手势置信度评估
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional, Set
from enum import Enum
from collections import deque, Counter
from hand_detector import HandDetector


class GestureType(Enum):
    """手势类型枚举"""
    UNKNOWN = "unknown"
    FIST = "fist"
    OPEN_HAND = "open_hand"
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    INDEX_POINTING = "index_pointing"
    PEACE = "peace"
    I_LOVE_YOU = "i_love_you"
    CALL_ME = "call_me"
    OK_GESTURE = "ok_gesture"
    ROCK_N_ROLL = "rock_n_roll"
    STOP_GESTURE = "stop_gesture"


class GestureRecognizer:
    """
    高级手势识别器
    
    基于手部关键点识别各种手势，支持手势序列识别和自定义手势。
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        初始化手势识别器
        
        Args:
            confidence_threshold: 手势识别置信度阈值
        """
        self.detector = HandDetector()
        self.confidence_threshold = confidence_threshold
        
        # 手势序列缓冲区
        self.gesture_buffer = deque(maxlen=10)
        self.gesture_history = deque(maxlen=30)
        
        # 手势定义
        self.gesture_definitions = self._initialize_gesture_definitions()
        
        # MediaPipe初始化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
    def _initialize_gesture_definitions(self) -> Dict[GestureType, Dict[str, Any]]:
        """
        初始化手势定义
        
        Returns:
            Dict: 手势定义字典
        """
        definitions = {
            GestureType.FIST: {
                'description': '握拳',
                'finger_conditions': {
                    'index': False,
                    'middle': False,
                    'ring': False,
                    'pinky': False
                },
                'thumb_condition': 'any'
            },
            
            GestureType.OPEN_HAND: {
                'description': '张开手掌',
                'finger_conditions': {
                    'index': True,
                    'middle': True,
                    'ring': True,
                    'pinky': True
                },
                'thumb_condition': True
            },
            
            GestureType.THUMB_UP: {
                'description': '竖起大拇指',
                'finger_conditions': {
                    'index': False,
                    'middle': False,
                    'ring': False,
                    'pinky': False
                },
                'thumb_condition': True
            },
            
            GestureType.THUMB_DOWN: {
                'description': '倒竖大拇指',
                'finger_conditions': {
                    'index': False,
                    'middle': False,
                    'ring': False,
                    'pinky': False
                },
                'thumb_condition': False
            },
            
            GestureType.INDEX_POINTING: {
                'description': '食指指向',
                'finger_conditions': {
                    'index': True,
                    'middle': False,
                    'ring': False,
                    'pinky': False
                },
                'thumb_condition': 'any'
            },
            
            GestureType.PEACE: {
                'description': '胜利手势',
                'finger_conditions': {
                    'index': True,
                    'middle': True,
                    'ring': False,
                    'pinky': False
                },
                'thumb_condition': 'any'
            },
            
            GestureType.I_LOVE_YOU: {
                'description': '我爱你手势',
                'finger_conditions': {
                    'index': True,
                    'middle': True,
                    'ring': False,
                    'pinky': True
                },
                'thumb_condition': 'any'
            },
            
            GestureType.OK_GESTURE: {
                'description': 'OK手势',
                'finger_conditions': {
                    'index': False,
                    'middle': False,
                    'ring': False,
                    'pinky': False
                },
                'special_condition': 'ok_shape'
            },
            
            GestureType.ROCK_N_ROLL: {
                'description': '摇滚手势',
                'finger_conditions': {
                    'index': False,
                    'middle': True,
                    'ring': True,
                    'pinky': True
                },
                'thumb_condition': False
            },
            
            GestureType.STOP_GESTURE: {
                'description': '停止手势',
                'finger_conditions': {
                    'index': True,
                    'middle': True,
                    'ring': True,
                    'pinky': True
                },
                'thumb_condition': False
            }
        }
        
        return definitions
    
    def recognize_gesture(self, hand_landmarks) -> Tuple[GestureType, float]:
        """
        识别手势
        
        Args:
            hand_landmarks: MediaPipe手部关键点
            
        Returns:
            Tuple[GestureType, float]: 手势类型和置信度
        """
        # 提取关键点
        landmarks_2d = []
        for landmark in hand_landmarks.landmark:
            landmarks_2d.append([landmark.x, landmark.y])
        
        # 检测手指状态
        fingers_status = self.detector._detect_fingers_status(landmarks_2d)
        
        # 识别数字手势
        digit_gesture = self._recognize_digit_gesture(fingers_status)
        digit_confidence = self._calculate_digit_confidence(fingers_status)
        
        # 识别标准手势
        standard_gesture = self._recognize_standard_gesture(fingers_status, landmarks_2d)
        standard_confidence = self._calculate_standard_confidence(fingers_status, landmarks_2d)
        
        # 选择置信度更高的手势
        if digit_confidence > standard_confidence:
            return digit_gesture, digit_confidence
        else:
            return standard_gesture, standard_confidence
    
    def _recognize_digit_gesture(self, fingers_status: Dict[str, bool]) -> GestureType:
        """
        识别数字手势
        
        Args:
            fingers_status: 手指状态
            
        Returns:
            GestureType: 数字手势类型
        """
        extended_count = sum(fingers_status.values())
        
        # 如果拇指是水平伸出的，不算作数字手势
        if not fingers_status.get('thumb', False):
            extended_count -= 1
        
        if extended_count == 0:
            return GestureType.FIST
        elif extended_count == 1:
            return GestureType.INDEX_POINTING
        elif extended_count == 2:
            if fingers_status.get('index', False) and fingers_status.get('middle', False):
                return GestureType.PEACE
            else:
                return GestureType.INDEX_POINTING
        elif extended_count == 3:
            return GestureType.I_LOVE_YOU
        elif extended_count == 4:
            return GestureType.OPEN_HAND
        elif extended_count == 5:
            return GestureType.OPEN_HAND
        else:
            return GestureType.UNKNOWN
    
    def _recognize_standard_gesture(self, fingers_status: Dict[str, bool], 
                                  landmarks_2d: List[List[float]]) -> GestureType:
        """
        识别标准手势
        
        Args:
            fingers_status: 手指状态
            landmarks_2d: 2D关键点坐标
            
        Returns:
            GestureType: 标准手势类型
        """
        thumb_status = fingers_status.get('thumb', False)
        index_status = fingers_status.get('index', False)
        middle_status = fingers_status.get('middle', False)
        ring_status = fingers_status.get('ring', False)
        pinky_status = fingers_status.get('pinky', False)
        
        # OK手势检测 (拇指和食指形成圆圈)
        if not index_status and not middle_status and not ring_status and not pinky_status:
            if self._is_ok_shape(landmarks_2d):
                return GestureType.OK_GESTURE
        
        # 竖大拇指检测
        if not index_status and not middle_status and not ring_status and not pinky_status:
            if thumb_status and self._is_thumb_up(landmarks_2d):
                return GestureType.THUMB_UP
            elif not thumb_status and self._is_thumb_down(landmarks_2d):
                return GestureType.THUMB_DOWN
        
        # 摇滚手势 (中指、无名指、小指伸出，拇指弯曲，食指收起)
        if not index_status and middle_status and ring_status and pinky_status and not thumb_status:
            return GestureType.ROCK_N_ROLL
        
        # 停止手势 (所有手指伸出，拇指收起)
        if index_status and middle_status and ring_status and pinky_status and not thumb_status:
            return GestureType.STOP_GESTURE
        
        # 根据手指数量推断
        extended_count = sum([index_status, middle_status, ring_status, pinky_status])
        
        if extended_count == 0:
            return GestureType.FIST
        elif extended_count == 4 and thumb_status:
            return GestureType.OPEN_HAND
        
        return GestureType.UNKNOWN
    
    def _is_ok_shape(self, landmarks_2d: List[List[float]]) -> bool:
        """
        检测是否为OK手势形状
        
        Args:
            landmarks_2d: 2D关键点坐标
            
        Returns:
            bool: 是否为OK手势
        """
        try:
            # 获取拇指尖和食指尖位置
            thumb_tip = landmarks_2d[4]
            index_tip = landmarks_2d[8]
            index_pip = landmarks_2d[6]
            thumb_mcp = landmarks_2d[2]
            
            # 计算拇指尖和食指尖的距离
            distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                               (thumb_tip[1] - index_tip[1])**2)
            
            # 计算拇指掌指关节和食指掌指关节的距离作为参考
            reference_distance = math.sqrt((thumb_mcp[0] - index_pip[0])**2 + 
                                         (thumb_mcp[1] - index_pip[1])**2)
            
            # 如果两个指尖距离相对较小，则可能是OK手势
            return distance < reference_distance * 0.5
            
        except IndexError:
            return False
    
    def _is_thumb_up(self, landmarks_2d: List[List[float]]) -> bool:
        """
        检测是否为竖大拇指
        
        Args:
            landmarks_2d: 2D关键点坐标
            
        Returns:
            bool: 是否为竖大拇指
        """
        try:
            thumb_tip = landmarks_2d[4]
            thumb_mcp = landmarks_2d[2]
            
            # 拇指尖y坐标应该小于掌指关节y坐标 (向上)
            return thumb_tip[1] < thumb_mcp[1]
            
        except IndexError:
            return False
    
    def _is_thumb_down(self, landmarks_2d: List[List[float]]) -> bool:
        """
        检测是否为倒竖大拇指
        
        Args:
            landmarks_2d: 2D关键点坐标
            
        Returns:
            bool: 是否为倒竖大拇指
        """
        try:
            thumb_tip = landmarks_2d[4]
            thumb_mcp = landmarks_2d[2]
            
            # 拇指尖y坐标应该大于掌指关节y坐标 (向下)
            return thumb_tip[1] > thumb_mcp[1]
            
        except IndexError:
            return False
    
    def _calculate_digit_confidence(self, fingers_status: Dict[str, bool]) -> float:
        """
        计算数字手势置信度
        
        Args:
            fingers_status: 手指状态
            
        Returns:
            float: 置信度 (0-1)
        """
        # 基于手指状态的清晰度计算置信度
        clear_fingers = sum(fingers_status.values())
        
        if clear_fingers == 0:  # 握拳
            return 0.9
        elif clear_fingers == 1:  # 单手指
            return 0.8
        elif clear_fingers == 2:  # 双手指
            return 0.75
        elif clear_fingers >= 3:  # 多手指
            return 0.7
        else:
            return 0.3
    
    def _calculate_standard_confidence(self, fingers_status: Dict[str, bool], 
                                     landmarks_2d: List[List[float]]) -> float:
        """
        计算标准手势置信度
        
        Args:
            fingers_status: 手指状态
            landmarks_2d: 2D关键点坐标
            
        Returns:
            float: 置信度 (0-1)
        """
        # 基础置信度
        base_confidence = 0.5
        
        # 根据手势特殊条件调整置信度
        thumb_status = fingers_status.get('thumb', False)
        index_status = fingers_status.get('index', False)
        
        if not index_status and not thumb_status:
            # 可能的大拇指手势
            if self._is_thumb_up(landmarks_2d):
                return 0.85
            elif self._is_thumb_down(landmarks_2d):
                return 0.85
        
        if not any([index_status, fingers_status.get('middle', False), 
                   fingers_status.get('ring', False), fingers_status.get('pinky', False)]):
            # 所有手指收起
            if self._is_ok_shape(landmarks_2d):
                return 0.8
        
        return base_confidence
    
    def recognize_gesture_sequence(self, hands_data: List[Dict[str, Any]]) -> List[Tuple[int, GestureType, float]]:
        """
        识别手势序列
        
        Args:
            hands_data: 手部检测结果
            
        Returns:
            List[Tuple[int, GestureType, float]]: 手势序列 [(hand_id, gesture_type, confidence)]
        """
        sequence = []
        
        for hand_info in hands_data:
            hand_id = hand_info['hand_id']
            
            # 提取当前手势
            current_gesture = hand_info.get('gesture', GestureType.UNKNOWN)
            current_confidence = 0.5  # 简单处理，实际应基于更复杂的计算
            
            # 添加到手势缓冲区
            self.gesture_buffer.append((hand_id, current_gesture, current_confidence))
            self.gesture_history.append((hand_id, current_gesture, current_confidence))
            
            sequence.append((hand_id, current_gesture, current_confidence))
        
        return sequence
    
    def get_gesture_stability(self, hand_id: int, window_size: int = 5) -> float:
        """
        计算手势稳定性
        
        Args:
            hand_id: 手部ID
            window_size: 统计窗口大小
            
        Returns:
            float: 稳定性分数 (0-1)
        """
        # 获取指定手部的最近手势历史
        recent_gestures = []
        for gesture_data in reversed(list(self.gesture_history)):
            if len(recent_gestures) >= window_size:
                break
            if gesture_data[0] == hand_id:
                recent_gestures.append(gesture_data[1])
        
        if len(recent_gestures) < 2:
            return 0.0
        
        # 计算最常见的手势
        gesture_counter = Counter(recent_gestures)
        most_common_gesture = gesture_counter.most_common(1)[0]
        
        # 稳定性 = 最常见手势的比例
        stability = most_common_gesture[1] / len(recent_gestures)
        return stability
    
    def add_custom_gesture(self, name: str, gesture_definition: Dict[str, Any]):
        """
        添加自定义手势
        
        Args:
            name: 手势名称
            gesture_definition: 手势定义
        """
        # 这里可以实现自定义手势的添加逻辑
        # 需要定义手势的特定条件，如特定的手指角度、位置等
        pass
    
    def get_gesture_description(self, gesture_type: GestureType) -> str:
        """
        获取手势描述
        
        Args:
            gesture_type: 手势类型
            
        Returns:
            str: 手势描述
        """
        if gesture_type in self.gesture_definitions:
            return self.gesture_definitions[gesture_type]['description']
        else:
            return "未知手势"
    
    def draw_gesture_info(self, image: np.ndarray, hand_info: Dict[str, Any]):
        """
        绘制手势信息
        
        Args:
            image: 输入图像
            hand_info: 手部信息
        """
        center = hand_info['center']
        gesture = hand_info.get('gesture', GestureType.UNKNOWN)
        
        if center:
            h, w, c = image.shape
            center_x, center_y = int(center[0] * w), int(center[1] * h)
            
            # 绘制手势名称
            gesture_text = gesture.value.replace('_', ' ').title()
            cv2.putText(image, gesture_text, (center_x + 15, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 绘制置信度
            stability = self.get_gesture_stability(hand_info['hand_id'])
            cv2.putText(image, f"Confidence: {stability:.2f}", 
                       (center_x + 15, center_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)