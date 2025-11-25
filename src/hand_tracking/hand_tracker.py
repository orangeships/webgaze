"""
实时手部跟踪器

该模块提供了基于MediaPipe的实时手部跟踪功能，支持：
- 实时手部检测和跟踪
- 手部轨迹记录和分析
- 手部运动状态检测
- 多目标手部跟踪
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from hand_detector import HandDetector


class HandTracker:
    """
    实时手部跟踪器类
    
    提供实时手部检测、跟踪、轨迹记录和运动分析功能。
    """
    
    def __init__(self, 
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        初始化手部跟踪器
        
        Args:
            max_num_hands: 最大跟踪手部数量
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        # 初始化手部检测器
        self.detector = HandDetector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 跟踪参数
        self.tracking_history = {}
        self.hand_trails = {}
        self.max_trail_length = 30  # 最大轨迹长度
        
        # 运动检测参数
        self.motion_threshold = 0.01  # 运动阈值
        self.stationary_frames = 0
        self.stationary_threshold = 15
        
        # 时间戳
        self.last_detection_time = {}
        
        # FPS计算
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)  # 保存最近30帧的FPS
        self.last_fps_time = time.time()
        self.fps = 0.0
        
    def start_tracking(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        开始手部跟踪
        
        Args:
            image: 输入图像
            
        Returns:
            List[Dict]: 跟踪结果
        """
        # 检测手部
        hands_data = self.detector.detect_hands(image, draw=False)
        
        # 更新跟踪信息
        self._update_tracking_data(hands_data)
        
        # 更新轨迹
        self._update_hand_trails(hands_data)
        
        return hands_data
    
    def _update_tracking_data(self, hands_data: List[Dict[str, Any]]):
        """
        更新手部跟踪数据
        
        Args:
            hands_data: 手部检测结果
        """
        current_time = time.time()
        
        # 更新已存在的手部跟踪
        for hand_info in hands_data:
            hand_id = hand_info['hand_id']
            
            if hand_id not in self.tracking_history:
                self.tracking_history[hand_id] = {
                    'positions': deque(maxlen=10),
                    'velocities': deque(maxlen=10),
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'confidence_history': deque(maxlen=5)
                }
            
            # 更新位置历史
            position = hand_info['center']
            self.tracking_history[hand_id]['positions'].append(position)
            self.tracking_history[hand_id]['last_seen'] = current_time
            
            # 计算速度
            if len(self.tracking_history[hand_id]['positions']) >= 2:
                current_pos = self.tracking_history[hand_id]['positions'][-1]
                previous_pos = self.tracking_history[hand_id]['positions'][-2]
                
                # 计算速度 (像素/秒)
                dt = 1/30.0  # 假设30fps
                velocity = ((current_pos[0] - previous_pos[0]) / dt,
                           (current_pos[1] - previous_pos[1]) / dt)
                self.tracking_history[hand_id]['velocities'].append(velocity)
            
            # 更新置信度历史
            confidence = hand_info.get('confidence', 1.0)
            self.tracking_history[hand_id]['confidence_history'].append(confidence)
        
        # 清理消失的手部
        self._cleanup_missing_hands(current_time)
    
    def _cleanup_missing_hands(self, current_time: float):
        """
        清理消失的手部
        
        Args:
            current_time: 当前时间戳
        """
        timeout = 2.0  # 2秒超时
        
        to_remove = []
        for hand_id in self.tracking_history:
            if current_time - self.tracking_history[hand_id]['last_seen'] > timeout:
                to_remove.append(hand_id)
        
        for hand_id in to_remove:
            del self.tracking_history[hand_id]
            if hand_id in self.hand_trails:
                del self.hand_trails[hand_id]
    
    def _update_hand_trails(self, hands_data: List[Dict[str, Any]]):
        """
        更新手部轨迹
        
        Args:
            hands_data: 手部检测结果
        """
        for hand_info in hands_data:
            hand_id = hand_info['hand_id']
            center = hand_info['center']
            
            if hand_id not in self.hand_trails:
                self.hand_trails[hand_id] = deque(maxlen=self.max_trail_length)
            
            self.hand_trails[hand_id].append(center)
    
    def get_hand_velocity(self, hand_id: int) -> Optional[Tuple[float, float]]:
        """
        获取手部速度
        
        Args:
            hand_id: 手部ID
            
        Returns:
            Tuple[float, float]: 速度向量 (vx, vy)，如果未找到则返回None
        """
        if hand_id in self.tracking_history:
            velocities = self.tracking_history[hand_id]['velocities']
            if velocities:
                # 返回最近的平均速度
                recent_velocities = list(velocities)[-3:]  # 最近3帧
                avg_vx = sum(v[0] for v in recent_velocities) / len(recent_velocities)
                avg_vy = sum(v[1] for v in recent_velocities) / len(recent_velocities)
                return (avg_vx, avg_vy)
        return None
    
    def get_hand_acceleration(self, hand_id: int) -> Optional[Tuple[float, float]]:
        """
        获取手部加速度
        
        Args:
            hand_id: 手部ID
            
        Returns:
            Tuple[float, float]: 加速度向量 (ax, ay)，如果未找到则返回None
        """
        if hand_id in self.tracking_history:
            velocities = self.tracking_history[hand_id]['velocities']
            if len(velocities) >= 2:
                recent_velocities = list(velocities)[-2:]
                dt = 1/30.0
                ax = (recent_velocities[1][0] - recent_velocities[0][0]) / dt
                ay = (recent_velocities[1][1] - recent_velocities[0][1]) / dt
                return (ax, ay)
        return None
    
    def detect_hand_motion(self, hand_id: int) -> str:
        """
        检测手部运动状态
        
        Args:
            hand_id: 手部ID
            
        Returns:
            str: 运动状态 ('moving', 'stationary', 'accelerating', 'decelerating')
        """
        velocity = self.get_hand_velocity(hand_id)
        acceleration = self.get_hand_acceleration(hand_id)
        
        if velocity is None:
            return 'unknown'
        
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        if speed < self.motion_threshold:
            self.stationary_frames += 1
            if self.stationary_frames > self.stationary_threshold:
                return 'stationary'
            else:
                return 'moving'
        else:
            self.stationary_frames = 0
            
            if acceleration is not None:
                acc_magnitude = np.sqrt(acceleration[0]**2 + acceleration[1]**2)
                if acc_magnitude > self.motion_threshold * 2:
                    return 'accelerating' if np.dot(velocity, acceleration) > 0 else 'decelerating'
            
            return 'moving'
    
    def draw_hand_trail(self, image: np.ndarray, hand_id: int, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        绘制手部轨迹
        
        Args:
            image: 输入图像
            hand_id: 手部ID
            color: 轨迹颜色 (B, G, R)
        """
        if hand_id in self.hand_trails and len(self.hand_trails[hand_id]) > 1:
            trail = list(self.hand_trails[hand_id])
            h, w, c = image.shape
            
            # 绘制轨迹线
            for i in range(1, len(trail)):
                start_point = (int(trail[i-1][0] * w), int(trail[i-1][1] * h))
                end_point = (int(trail[i][0] * w), int(trail[i][1] * h))
                
                # 根据轨迹长度调整透明度
                alpha = i / len(trail)
                thickness = int(2 + alpha * 3)
                
                cv2.line(image, start_point, end_point, color, thickness)
    
    def draw_tracking_info(self, image: np.ndarray, hands_data: List[Dict[str, Any]]):
        """
        绘制跟踪信息
        
        Args:
            image: 输入图像
            hands_data: 手部检测结果
        """
        for hand_info in hands_data:
            hand_id = hand_info['hand_id']
            center = hand_info['center']
            
            if center:
                h, w, c = image.shape
                center_x, center_y = int(center[0] * w), int(center[1] * h)
                
                # 获取运动状态
                motion_status = self.detect_hand_motion(hand_id)
                
                # 获取速度和加速度
                velocity = self.get_hand_velocity(hand_id)
                acceleration = self.get_hand_acceleration(hand_id)
                
                # 绘制跟踪信息
                info_texts = [
                    f"Hand {hand_id}",
                    f"Motion: {motion_status}"
                ]
                
                if velocity:
                    speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                    info_texts.append(f"Speed: {speed:.2f}")
                
                if acceleration:
                    acc_mag = np.sqrt(acceleration[0]**2 + acceleration[1]**2)
                    info_texts.append(f"Acc: {acc_mag:.2f}")
                
                # 绘制信息文本
                y_offset = 30
                for i, text in enumerate(info_texts):
                    cv2.putText(image, text, (center_x + 15, center_y + y_offset + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 绘制速度矢量
                if velocity:
                    scale = 10  # 速度矢量缩放
                    end_x = center_x + int(velocity[0] * scale)
                    end_y = center_y + int(velocity[1] * scale)
                    cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), 
                                   (0, 255, 255), 2, tipLength=0.3)
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """
        获取跟踪统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        stats = {
            'active_hands': len(self.tracking_history),
            'total_hands_seen': len(self.tracking_history),
            'hand_details': {}
        }
        
        current_time = time.time()
        
        for hand_id, history in self.tracking_history.items():
            # 计算跟踪时长
            tracking_duration = current_time - history['first_seen']
            
            # 计算平均速度
            velocities = list(history['velocities'])
            avg_speed = 0
            if velocities:
                speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
                avg_speed = np.mean(speeds)
            
            # 计算平均置信度
            confidences = list(history['confidence_history'])
            avg_confidence = np.mean(confidences) if confidences else 0
            
            stats['hand_details'][hand_id] = {
                'tracking_duration': tracking_duration,
                'avg_speed': avg_speed,
                'avg_confidence': avg_confidence,
                'frames_tracked': len(history['positions'])
            }
        
        return stats
    
    def reset_tracking(self):
        """重置跟踪状态"""
        self.tracking_history.clear()
        self.hand_trails.clear()
        self.stationary_frames = 0
    
    @property
    def active_tracks(self) -> List[int]:
        """获取当前活跃的手部跟踪ID列表"""
        return list(self.tracking_history.keys())
    
    def get_average_fps(self) -> float:
        """获取平均FPS"""
        # 计算当前帧的FPS
        current_time = time.time()
        dt = current_time - self.last_fps_time
        
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps_history.append(current_fps)
            self.fps = sum(self.fps_history) / len(self.fps_history)
        
        self.last_fps_time = current_time
        self.frame_count += 1
        
        return self.fps