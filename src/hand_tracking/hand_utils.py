"""
手部检测实用工具

该模块提供了手部检测相关的实用工具函数，包括：
- 图像预处理
- 关键点坐标转换
- 距离和角度计算
- 手部区域提取
- 可视化工具
- 数据导出功能
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import math
from typing import List, Tuple, Dict, Any, Optional, Union
import os
from datetime import datetime


class HandUtils:
    """
    手部检测实用工具类
    
    提供手部检测相关的各种实用功能和工具方法。
    """
    
    # MediaPipe初始化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    def __init__(self):
        """初始化工具类"""
        pass
    
    @staticmethod
    def preprocess_image(image: np.ndarray, 
                        target_size: Tuple[int, int] = (640, 480),
                        enhance: bool = False) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            enhance: 是否增强图像对比度
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 调整图像尺寸
        if image.shape[:2][::-1] != target_size:
            image = cv2.resize(image, target_size)
        
        # 图像增强
        if enhance:
            # 增强对比度
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 轻微的高斯模糊减少噪声
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    @staticmethod
    def normalize_coordinates(landmarks: List, image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        将归一化坐标转换为像素坐标
        
        Args:
            landmarks: MediaPipe归一化关键点
            image_shape: 图像形状 (height, width)
            
        Returns:
            List[Tuple[float, float]]: 像素坐标列表
        """
        h, w = image_shape[:2]
        pixel_coords = []
        
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            pixel_coords.append((x, y))
        
        return pixel_coords
    
    @staticmethod
    def denormalize_coordinates(pixel_coords: List[Tuple[int, int]], 
                              image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        将像素坐标转换为归一化坐标
        
        Args:
            pixel_coords: 像素坐标列表
            image_shape: 图像形状 (height, width)
            
        Returns:
            List[Tuple[float, float]]: 归一化坐标列表
        """
        h, w = image_shape[:2]
        normalized_coords = []
        
        for x, y in pixel_coords:
            norm_x = x / w
            norm_y = y / h
            normalized_coords.append((norm_x, norm_y))
        
        return normalized_coords
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            point1: 第一个点坐标
            point2: 第二个点坐标
            
        Returns:
            float: 距离值
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_angle(point1: Tuple[float, float],
                       vertex: Tuple[float, float], 
                       point2: Tuple[float, float]) -> float:
        """
        计算三点之间的角度
        
        Args:
            point1: 第一个点
            vertex: 顶点 (角度在此处测量)
            point2: 第二个点
            
        Returns:
            float: 角度 (度)
        """
        # 计算向量
        vector1 = (point1[0] - vertex[0], point1[1] - vertex[1])
        vector2 = (point2[0] - vertex[0], point2[1] - vertex[1])
        
        # 计算角度
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        # 限制cos_angle在[-1, 1]范围内避免数值误差
        cos_angle = max(-1, min(1, cos_angle))
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    @staticmethod
    def extract_hand_region(image: np.ndarray, 
                           hand_center: Tuple[float, float],
                           hand_size: float = 150) -> np.ndarray:
        """
        提取手部区域
        
        Args:
            image: 输入图像
            hand_center: 手部中心坐标 (归一化)
            hand_size: 手部区域大小
            
        Returns:
            np.ndarray: 提取的手部区域图像
        """
        h, w, c = image.shape
        center_x, center_y = int(hand_center[0] * w), int(hand_center[1] * h)
        
        # 计算边界框
        x1 = max(0, center_x - int(hand_size // 2))
        y1 = max(0, center_y - int(hand_size // 2))
        x2 = min(w, center_x + int(hand_size // 2))
        y2 = min(h, center_y + int(hand_size // 2))
        
        # 提取区域
        hand_region = image[y1:y2, x1:x2]
        
        return hand_region
    
    @staticmethod
    def draw_hand_skeleton(image: np.ndarray, 
                          landmarks: List,
                          connections: Optional[List[Tuple[int, int]]] = None,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        绘制手部骨架
        
        Args:
            image: 输入图像
            landmarks: 关键点列表
            connections: 连线列表，默认为MediaPipe标准连接
            color: 线条颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        if connections is None:
            connections = HandUtils.mp_hands.HAND_CONNECTIONS
        
        # 绘制连线
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = HandUtils.normalize_coordinates([landmarks[start_idx]], image.shape[:2])[0]
                end_point = HandUtils.normalize_coordinates([landmarks[end_idx]], image.shape[:2])[0]
                
                cv2.line(image, start_point, end_point, color, thickness)
        
        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            point = HandUtils.normalize_coordinates([landmark], image.shape[:2])[0]
            cv2.circle(image, point, 4, (255, 0, 0), -1)
            cv2.putText(image, str(i), (point[0] + 5, point[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return image
    
    @staticmethod
    def create_hand_mask(image: np.ndarray, 
                        landmarks: List,
                        dilation: int = 5) -> np.ndarray:
        """
        创建手部掩膜
        
        Args:
            image: 输入图像
            landmarks: 关键点列表
            dilation: 膨胀程度
            
        Returns:
            np.ndarray: 手部掩膜
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 获取关键点像素坐标
        points = HandUtils.normalize_coordinates(landmarks, image.shape[:2])
        
        # 创建凸包
        hull = cv2.convexHull(np.array(points))
        
        # 填充凸包
        cv2.fillPoly(mask, [hull], 255)
        
        # 膨胀以扩大区域
        kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    @staticmethod
    def smooth_landmarks(landmarks_history: List[List],
                        alpha: float = 0.3) -> List:
        """
        平滑关键点序列
        
        Args:
            landmarks_history: 关键点历史列表
            alpha: 平滑因子 (0-1)
            
        Returns:
            List: 平滑后的关键点
        """
        if len(landmarks_history) == 0:
            return []
        
        if len(landmarks_history) == 1:
            return landmarks_history[0]
        
        # 使用指数移动平均进行平滑
        smoothed = []
        
        for i in range(len(landmarks_history[0])):
            values = [frame[i] for frame in landmarks_history]
            
            if i == 0:  # x坐标
                smoothed_values = []
                for j, value in enumerate(values):
                    if j == 0:
                        smoothed_values.append(value)
                    else:
                        smoothed_value = alpha * value + (1 - alpha) * smoothed_values[-1]
                        smoothed_values.append(smoothed_value)
                smoothed.append(smoothed_values[-1])
            else:  # y坐标
                smoothed_values = []
                for j, value in enumerate(values):
                    if j == 0:
                        smoothed_values.append(value)
                    else:
                        smoothed_value = alpha * value + (1 - alpha) * smoothed_values[-1]
                        smoothed_values.append(smoothed_value)
                smoothed.append(smoothed_values[-1])
        
        return smoothed
    
    @staticmethod
    def export_landmarks_data(landmarks_data: Dict[str, Any],
                            filename: Optional[str] = None,
                            format: str = 'json') -> str:
        """
        导出关键点数据
        
        Args:
            landmarks_data: 关键点数据
            filename: 输出文件名
            format: 输出格式 ('json', 'csv', 'txt')
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hand_landmarks_{timestamp}"
        
        if format == 'json':
            filepath = f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(landmarks_data, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            filepath = f"{filename}.csv"
            
            # 转换为DataFrame
            rows = []
            for frame_idx, frame_data in enumerate(landmarks_data.get('frames', [])):
                for hand_id, hand_data in frame_data.get('hands', {}).items():
                    row = {
                        'frame': frame_idx,
                        'hand_id': hand_id,
                        'gesture': hand_data.get('gesture', 'unknown')
                    }
                    
                    # 添加关键点坐标
                    for i, landmark in enumerate(hand_data.get('landmarks', [])):
                        row[f'landmark_{i}_x'] = landmark[0]
                        row[f'landmark_{i}_y'] = landmark[1]
                        row[f'landmark_{i}_z'] = landmark[2]
                    
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        elif format == 'txt':
            filepath = f"{filename}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("Hand Landmarks Data Export\n")
                f.write("=" * 50 + "\n")
                
                for frame_idx, frame_data in enumerate(landmarks_data.get('frames', [])):
                    f.write(f"Frame {frame_idx}:\n")
                    
                    for hand_id, hand_data in frame_data.get('hands', {}).items():
                        f.write(f"  Hand {hand_id}:\n")
                        f.write(f"    Gesture: {hand_data.get('gesture', 'unknown')}\n")
                        f.write(f"    Center: {hand_data.get('center', 'unknown')}\n")
                        
                        landmarks = hand_data.get('landmarks', [])
                        if landmarks:
                            f.write(f"    Landmarks ({len(landmarks)} points):\n")
                            for i, landmark in enumerate(landmarks):
                                f.write(f"      Point {i}: ({landmark[0]:.4f}, {landmark[1]:.4f}, {landmark[2]:.4f})\n")
        
        return filepath
    
    @staticmethod
    def analyze_hand_movement(landmarks_sequence: List[List],
                            window_size: int = 10) -> Dict[str, Any]:
        """
        分析手部运动模式
        
        Args:
            landmarks_sequence: 关键点序列
            window_size: 分析窗口大小
            
        Returns:
            Dict: 运动分析结果
        """
        if len(landmarks_sequence) < window_size:
            return {}
        
        analysis = {
            'movement_type': 'unknown',
            'speed': 0,
            'direction': (0, 0),
            'stability': 0,
            'trajectory_length': 0
        }
        
        # 计算中心点轨迹
        center_trajectory = []
        for landmarks in landmarks_sequence:
            center_x = sum(lm[0] for lm in landmarks) / len(landmarks)
            center_y = sum(lm[1] for lm in landmarks) / len(landmarks)
            center_trajectory.append((center_x, center_y))
        
        # 计算轨迹长度
        trajectory_length = 0
        for i in range(1, len(center_trajectory)):
            distance = HandUtils.calculate_distance(center_trajectory[i-1], center_trajectory[i])
            trajectory_length += distance
        
        analysis['trajectory_length'] = trajectory_length
        
        # 计算平均速度
        if len(center_trajectory) > 1:
            speeds = []
            for i in range(1, len(center_trajectory)):
                distance = HandUtils.calculate_distance(center_trajectory[i-1], center_trajectory[i])
                speed = distance  # 假设每帧时间间隔为1
                speeds.append(speed)
            
            analysis['speed'] = np.mean(speeds)
            
            # 计算主要方向
            if speeds:
                dx = center_trajectory[-1][0] - center_trajectory[0][0]
                dy = center_trajectory[-1][1] - center_trajectory[0][1]
                analysis['direction'] = (dx, dy)
                
                # 判断运动类型
                if analysis['speed'] < 0.01:
                    analysis['movement_type'] = 'stationary'
                elif analysis['speed'] < 0.05:
                    analysis['movement_type'] = 'slow'
                else:
                    analysis['movement_type'] = 'fast'
        
        # 计算稳定性
        if len(center_trajectory) > 1:
            distances_from_mean = []
            mean_x = sum(p[0] for p in center_trajectory) / len(center_trajectory)
            mean_y = sum(p[1] for p in center_trajectory) / len(center_trajectory)
            mean_point = (mean_x, mean_y)
            
            for point in center_trajectory:
                distance = HandUtils.calculate_distance(point, mean_point)
                distances_from_mean.append(distance)
            
            analysis['stability'] = 1.0 / (1.0 + np.std(distances_from_mean))
        
        return analysis
    
    @staticmethod
    def create_hand_heatmap(image: np.ndarray, 
                           landmarks_sequence: List[List],
                           alpha: float = 0.3) -> np.ndarray:
        """
        创建手部运动热力图
        
        Args:
            image: 输入图像
            landmarks_sequence: 关键点序列
            alpha: 透明度
            
        Returns:
            np.ndarray: 热力图图像
        """
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        
        # 累加所有帧的关键点位置
        for landmarks in landmarks_sequence:
            for landmark in landmarks:
                x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
                
                # 创建高斯热点
                cv2.circle(heatmap, (x, y), 20, 1.0, -1)
        
        # 归一化
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = np.clip(heatmap, 0, 1)
        
        # 创建彩色热力图
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 与原图混合
        result = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        
        return result
    
    @staticmethod
    def measure_hand_diameter(landmarks: List) -> float:
        """
        测量手部直径
        
        Args:
            landmarks: 手部关键点
            
        Returns:
            float: 手部直径 (基于手腕到中指的距离)
        """
        try:
            # 使用手腕到中指尖的距离作为参考
            wrist = (landmarks[0].x, landmarks[0].y)
            middle_tip = (landmarks[12].x, landmarks[12].y)  # 中指尖
            
            distance = HandUtils.calculate_distance(wrist, middle_tip)
            return distance
        except:
            return 0.0
    
    @staticmethod
    def is_hand_visible(landmarks: List, 
                       visibility_threshold: float = 0.5) -> bool:
        """
        判断手部是否可见
        
        Args:
            landmarks: 手部关键点
            visibility_threshold: 可见性阈值
            
        Returns:
            bool: 手部是否可见
        """
        visible_points = 0
        total_points = len(landmarks)
        
        for landmark in landmarks:
            if hasattr(landmark, 'visibility') and landmark.visibility > visibility_threshold:
                visible_points += 1
            elif hasattr(landmark, 'presence') and landmark.presence > visibility_threshold:
                visible_points += 1
        
        # 如果超过50%的关键点可见，则认为手部可见
        return (visible_points / total_points) > 0.5