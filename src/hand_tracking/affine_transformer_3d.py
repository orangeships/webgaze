"""
3D仿射变换模块

实现3D坐标映射、距离计算、姿态估计等功能，用于手部关键点的空间映射和分析。

主要功能：
- 3D仿射变换
- 3D距离计算
- 姿态估计
- 坐标归一化
- 角度计算
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import math


class AffineTransformer3D:
    """
    3D仿射变换类
    
    用于手部关键点的3D空间映射和分析，提供3D距离计算、
    姿态估计、坐标归一化等功能。
    """
    
    def __init__(self):
        """
        初始化3D仿射变换模块
        """
        self.reference_points = None
        self.target_points = None
        self.transform_matrix = None
        
    def calculate_3d_distance(self, point1: List[float], point2: List[float]) -> float:
        """
        计算3D空间中两点之间的距离
        
        Args:
            point1: 第一个点的3D坐标 [x, y, z]
            point2: 第二个点的3D坐标 [x, y, z]
            
        Returns:
            float: 两点之间的3D距离
        """
        if len(point1) < 3 or len(point2) < 3:
            raise ValueError("点坐标必须包含3D信息")
        
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 + 
            (point1[1] - point2[1]) ** 2 + 
            (point1[2] - point2[2]) ** 2
        )
    
    def calculate_2d_distance(self, point1: List[float], point2: List[float]) -> float:
        """
        计算2D空间中两点之间的距离
        
        Args:
            point1: 第一个点的2D坐标 [x, y]
            point2: 第二个点的2D坐标 [x, y]
            
        Returns:
            float: 两点之间的2D距离
        """
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 + 
            (point1[1] - point2[1]) ** 2
        )
    
    def normalize_coordinates(self, landmarks_3d: List[List[float]]) -> List[List[float]]:
        """
        对3D关键点坐标进行归一化
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            
        Returns:
            List[List[float]]: 归一化后的3D关键点坐标
        """
        if not landmarks_3d or len(landmarks_3d) < 21:
            return landmarks_3d
        
        # 以手腕点(0号点)为原点进行归一化
        wrist = np.array(landmarks_3d[0])
        normalized = []
        
        for point in landmarks_3d:
            normalized_point = (np.array(point) - wrist).tolist()
            normalized.append(normalized_point)
        
        return normalized
    
    def calculate_hand_size(self, landmarks_3d: List[List[float]]) -> float:
        """
        计算手部大小，用于相对距离判断
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            
        Returns:
            float: 手部大小（手腕到中指指尖的距离）
        """
        if not landmarks_3d or len(landmarks_3d) < 13:
            return 0.1  # 默认值
        
        # 手腕点
        wrist = landmarks_3d[0]
        # 中指指根点
        middle_tip = landmarks_3d[9]
        
        return self.calculate_3d_distance(wrist, middle_tip)
    
    def calculate_finger_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """
        计算三个点形成的角度（点2为顶点）
        
        Args:
            point1: 第一个点的3D坐标
            point2: 第二个点的3D坐标（顶点）
            point3: 第三个点的3D坐标
            
        Returns:
            float: 角度值（弧度）
        """
        # 向量计算
        v1 = np.array(point1) - np.array(point2)
        v2 = np.array(point3) - np.array(point2)
        
        # 单位向量
        v1_unit = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_unit = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
        
        # 点积计算角度
        dot_product = np.dot(v1_unit, v2_unit)
        # 限制在[-1, 1]范围内，避免浮点误差
        dot_product = max(-1.0, min(1.0, dot_product))
        
        return np.arccos(dot_product)
    
    def calculate_thumb_index_angle(self, landmarks_3d: List[List[float]]) -> float:
        """
        计算大拇指和食指之间的角度
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            
        Returns:
            float: 角度值（弧度）
        """
        if not landmarks_3d or len(landmarks_3d) < 9:
            return 0.0
        
        # 大拇指指尖
        thumb_tip = landmarks_3d[4]
        # 食指指尖
        index_tip = landmarks_3d[8]
        # 手腕点
        wrist = landmarks_3d[0]
        
        return self.calculate_finger_angle(thumb_tip, wrist, index_tip)
    
    def estimate_hand_pose(self, landmarks_3d: List[List[float]]) -> Dict[str, float]:
        """
        估计手部姿态
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            
        Returns:
            Dict[str, float]: 手部姿态信息
        """
        if not landmarks_3d or len(landmarks_3d) < 21:
            return {
                'palm_angle': 0.0,
                'hand_size': 0.1,
                'thumb_index_angle': 0.0
            }
        
        # 计算手掌角度（手腕到中指MCP关节的向量与垂直方向的夹角）
        wrist = np.array(landmarks_3d[0])
        middle_mcp = np.array(landmarks_3d[9])
        
        # 计算向量
        palm_vector = middle_mcp - wrist
        vertical_vector = np.array([0, -1, 0])  # 垂直向上的向量
        
        # 计算手掌角度
        palm_vector_unit = palm_vector / np.linalg.norm(palm_vector) if np.linalg.norm(palm_vector) > 0 else palm_vector
        dot_product = np.dot(palm_vector_unit, vertical_vector)
        dot_product = max(-1.0, min(1.0, dot_product))
        palm_angle = np.arccos(dot_product)
        
        # 计算手部大小
        hand_size = self.calculate_hand_size(landmarks_3d)
        
        # 计算大拇指和食指之间的角度
        thumb_index_angle = self.calculate_thumb_index_angle(landmarks_3d)
        
        return {
            'palm_angle': palm_angle,
            'hand_size': hand_size,
            'thumb_index_angle': thumb_index_angle
        }
    
    def calculate_relative_distance(self, distance: float, hand_size: float) -> float:
        """
        计算相对距离（相对于手部大小）
        
        Args:
            distance: 实际距离
            hand_size: 手部大小
            
        Returns:
            float: 相对距离比例
        """
        if hand_size <= 0:
            return 0.0
        
        return distance / hand_size
    
    def get_adaptive_pinch_threshold(self, hand_size: float, palm_angle: float) -> float:
        """
        获取自适应捏合阈值
        
        Args:
            hand_size: 手部大小
            palm_angle: 手掌角度（弧度）
            
        Returns:
            float: 自适应捏合阈值
        """
        # 基础阈值（相对于手部大小的比例）- 调大阈值，使检测不那么灵敏
        base_threshold = hand_size * 0.4  # 从0.15调大到0.22，降低灵敏度
        
        # 根据手掌角度调整阈值
        # 当手掌角度接近垂直时，阈值稍大；当手掌角度接近水平时，阈值稍小
        angle_factor = 1.0 + 0.1 * abs(math.cos(palm_angle))  # 进一步减小角度因子，使阈值更稳定
        
        return base_threshold * angle_factor
    
    def is_thumb_index_pinch(self, landmarks_3d: List[List[float]]) -> Tuple[bool, Dict[str, float]]:
        """
        检测大拇指和食指的捏合动作
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            
        Returns:
            Tuple[bool, Dict[str, float]]: (是否捏合, 调试信息)
        """
        if not landmarks_3d or len(landmarks_3d) < 21:
            return False, {"error": "关键点不足"}
        
        # 获取关键点
        thumb_tip = landmarks_3d[4]      # 大拇指指尖
        index_tip = landmarks_3d[8]       # 食指指尖
        thumb_mcp = landmarks_3d[2]       # 大拇指掌指关节
        index_mcp = landmarks_3d[5]       # 食指掌指关节
        
        # 计算3D距离
        thumb_index_distance_3d = self.calculate_3d_distance(thumb_tip, index_tip)
        thumb_index_distance_2d = self.calculate_2d_distance(thumb_tip[:2], index_tip[:2])
        
        # 估计手部姿态
        hand_pose = self.estimate_hand_pose(landmarks_3d)
        hand_size = hand_pose['hand_size']
        palm_angle = hand_pose['palm_angle']
        thumb_index_angle = hand_pose['thumb_index_angle']
        
        # 计算相对距离
        relative_distance = self.calculate_relative_distance(thumb_index_distance_3d, hand_size)
        
        # 获取自适应阈值
        pinch_threshold = self.get_adaptive_pinch_threshold(hand_size, palm_angle)
        
        # 计算手指角度约束
        # 大拇指和食指的角度应该在合理范围内
        thumb_angle = self.calculate_finger_angle(landmarks_3d[0], landmarks_3d[2], landmarks_3d[4])
        index_angle = self.calculate_finger_angle(landmarks_3d[0], landmarks_3d[5], landmarks_3d[8])
        
        # 进一步放宽角度约束，几乎允许所有角度
        # 手指角度约束：0.1到3.0弧度之间（几乎覆盖所有可能的角度）
        thumb_angle_valid = 0.1 < thumb_angle < 3.0
        index_angle_valid = 0.1 < index_angle < 3.0
        
        # 大拇指和食指之间的角度约束：0.0到120度之间
        angle_valid = 0.0 < thumb_index_angle < 2 * math.pi / 3  # 0到120度
        
        # 综合判断 - 进一步放宽约束，只考虑距离和基本角度有效性
        # 移除大拇指和食指之间的角度约束，允许更灵活的手势
        is_pinch = (
            thumb_index_distance_3d < pinch_threshold and
            (thumb_angle_valid or index_angle_valid)  # 只需要其中一个手指角度有效
        )
        
        # 计算捏合强度
        pinch_strength = max(0, min(1, 1 - (thumb_index_distance_3d / pinch_threshold)))
        
        debug_info = {
            'thumb_index_distance_3d': thumb_index_distance_3d,
            'thumb_index_distance_2d': thumb_index_distance_2d,
            'hand_size': hand_size,
            'palm_angle': palm_angle,
            'thumb_index_angle': thumb_index_angle,
            'relative_distance': relative_distance,
            'pinch_threshold': pinch_threshold,
            'pinch_strength': pinch_strength,
            'thumb_angle': thumb_angle,
            'index_angle': index_angle,
            'thumb_angle_valid': thumb_angle_valid,
            'index_angle_valid': index_angle_valid,
            'angle_valid': angle_valid,
            'is_pinch': is_pinch
        }
        
        return is_pinch, debug_info
    
    def transform_coordinates(self, landmarks_3d: List[List[float]], transform_matrix: np.ndarray) -> List[List[float]]:
        """
        对3D关键点进行仿射变换
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            transform_matrix: 4x4仿射变换矩阵
            
        Returns:
            List[List[float]]: 变换后的3D关键点坐标
        """
        transformed = []
        
        for point in landmarks_3d:
            # 转换为齐次坐标
            point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
            # 应用变换
            transformed_point = np.dot(transform_matrix, point_homogeneous)
            # 转换回3D坐标
            transformed.append(transformed_point[:3].tolist())
        
        return transformed
    
    def calculate_joint_angle(self, landmarks_3d: List[List[float]], joint_indices: Tuple[int, int, int]) -> float:
        """
        计算关节角度
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            joint_indices: 三个点的索引 (点1, 关节点, 点2)
            
        Returns:
            float: 关节角度（弧度）
        """
        if not landmarks_3d or len(landmarks_3d) < max(joint_indices) + 1:
            return 0.0
        
        point1 = landmarks_3d[joint_indices[0]]
        joint_point = landmarks_3d[joint_indices[1]]
        point2 = landmarks_3d[joint_indices[2]]
        
        return self.calculate_finger_angle(point1, joint_point, point2)
    
    def get_finger_orientation(self, landmarks_3d: List[List[float]], finger: str) -> List[float]:
        """
        获取手指的方向向量
        
        Args:
            landmarks_3d: 3D关键点坐标列表
            finger: 手指名称 ('thumb', 'index', 'middle', 'ring', 'pinky')
            
        Returns:
            List[float]: 手指方向向量
        """
        finger_indices = {
            'thumb': (2, 4),    # 大拇指掌指关节到指尖
            'index': (5, 8),    # 食指掌指关节到指尖
            'middle': (9, 12),  # 中指掌指关节到指尖
            'ring': (13, 16),   # 无名指掌指关节到指尖
            'pinky': (17, 20)   # 小指掌指关节到指尖
        }
        
        if finger not in finger_indices or not landmarks_3d or len(landmarks_3d) < 21:
            return [0, 0, 0]
        
        start_idx, end_idx = finger_indices[finger]
        start_point = np.array(landmarks_3d[start_idx])
        end_point = np.array(landmarks_3d[end_idx])
        
        direction = end_point - start_point
        # 归一化
        direction_unit = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
        
        return direction_unit.tolist()
    
    def calculate_orientation_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的方向相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 相似度（-1到1之间，1表示方向完全相同）
        """
        # 转换为numpy数组
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算单位向量
        v1_unit = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_unit = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
        
        # 计算点积
        dot_product = np.dot(v1_unit, v2_unit)
        
        # 确保在-1到1之间
        return max(-1.0, min(1.0, dot_product))
