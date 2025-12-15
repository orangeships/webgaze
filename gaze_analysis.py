#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼手交互系统注视点分析模块

该模块负责注视点数据的分析、平滑和处理，包括卡尔曼滤波和注视离散度分析。

主要功能：
1. 注视点平滑处理（使用轻量卡尔曼滤波器）
2. 注视离散度分析（计算注视点的分布情况）
3. 注视点触发条件检测（判断是否满足注视稳定条件）
4. 多屏环境下的注视点坐标转换支持
5. 注视点历史数据管理

核心类：
- LightweightKalmanFilter: 轻量级卡尔曼滤波器，用于注视点平滑
- GazeDispersionAnalyzer: 注视离散度分析器，用于检测注视稳定状态

关键算法：
1. 卡尔曼滤波算法：用于平滑注视点数据，减少噪声影响
2. 离散度分析算法：基于角度和像素离散度判断注视稳定性
3. 几何中心计算：用于确定注视点的中心位置

使用场景：
1. 平滑实时注视点数据
2. 检测用户是否在某个区域稳定注视
3. 触发交互事件（如鼠标自动移动）
4. 分析注视行为模式

作者：TraeAI
版本：1.0.0
日期：2025-12-09
"""
import numpy as np
import time
from collections import deque
import math

class LightweightKalmanFilter:
    """轻量卡尔曼滤波器，用于注视点平滑"""
    
    def __init__(self, process_noise=0.6, measurement_noise=0.1, error_estimate=1.0):
        # 状态向量: [x, y, vx, vy] (位置和速度)
        self.x = None  # 状态估计
        self.P = None  # 状态协方差矩阵
        
        # 过程噪声协方差矩阵 Q (优化以增强平滑效果，减少抖动)
        self.Q = np.array([
            [process_noise, 0, 0, 0],         # x位置噪声（减小以增强位置平滑）
            [0, process_noise, 0, 0],         # y位置噪声（减小以增强位置平滑）
            [0, 0, process_noise * 0.5, 0],   # x速度噪声（减小，减少速度波动）
            [0, 0, 0, process_noise * 0.5]    # y速度噪声（减小，减少速度波动）
        ])
        
        # 测量噪声协方差矩阵 R (增加以减少对噪声测量的信任)
        self.R = np.array([
            [measurement_noise, 0],            # x位置测量噪声（增加以增强稳定性）
            [0, measurement_noise]             # y位置测量噪声（增加以增强稳定性）
        ])
        
        # 状态转移矩阵 F
        self.F = np.array([
            [1, 0, 1, 0],   # x = x + vx * dt
            [0, 1, 0, 1],   # y = y + vy * dt
            [0, 0, 1, 0],   # vx = vx
            [0, 0, 0, 1]    # vy = vy
        ])
        
        # 测量矩阵 H
        self.H = np.array([
            [1, 0, 0, 0],   # 测量x位置
            [0, 1, 0, 0]    # 测量y位置
        ])
        
        # 初始状态协方差矩阵
        self.initial_P = np.eye(4) * error_estimate
    
    def update(self, measurement):
        """更新卡尔曼滤波器
        
        Args:
            measurement: 测量值 (x, y)
            
        Returns:
            滤波后的值 (x, y)
        """
        # 初始化状态
        if self.x is None:
            self.x = np.array([measurement[0], measurement[1], 0, 0])
            self.P = self.initial_P.copy()
            return tuple(self.x[:2])
        
        # 预测步骤
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 更新步骤
        y = np.array(measurement) - self.H @ x_pred  # 残差
        S = self.H @ P_pred @ self.H.T + self.R  # 残差协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        
        self.x = x_pred + K @ y
        I = np.eye(4)  # 单位矩阵
        self.P = (I - K @ self.H) @ P_pred
        
        return tuple(self.x[:2])

class GazeDispersionAnalyzer:
    """注视点离散度分析器 - 基于最近6帧眼动数据进行注视行为判断"""
    
    def __init__(self, frame_count=5, angle_threshold=5.0, pixel_threshold=150):
        self.frame_count = frame_count  # 使用帧数而非时间窗口，减少所需帧数
        self.angle_threshold = angle_threshold  # 增加角度阈值，使条件更宽松
        self.pixel_threshold = pixel_threshold  # 增加像素阈值，使条件更宽松
        self.gaze_points = deque(maxlen=frame_count)  # 存储 (timestamp, x, y) 元组，限制为最近frame_count帧
        self.last_trigger_time = 0
        self.trigger_cooldown = 800  # 减少冷却时间，加快触发频率
    
    def add_gaze_point(self, x, y):
        """添加新的注视点"""
        current_time = time.time() * 1000  # 转换为毫秒
        self.gaze_points.append((current_time, x, y))
        # 由于使用了maxlen，deque会自动移除最旧的点，保持最近frame_count帧的数据
    
    def calculate_dispersion(self):
        """计算最近frame_count帧内注视点的离散度"""
        if len(self.gaze_points) < 2:
            return {
                'angle_dispersion': 0,
                'pixel_dispersion': 0,
                'point_count': len(self.gaze_points),
                'geometric_center': None
            }
        
        # 提取坐标
        points = [(x, y) for _, x, y in self.gaze_points]
        
        # 计算几何中心
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        geometric_center = (center_x, center_y)
        
        # 计算像素离散度（标准差）
        pixel_distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points]
        pixel_dispersion = math.sqrt(sum(d**2 for d in pixel_distances) / len(pixel_distances))
        
        # 计算角度离散度（相对于屏幕中心的角度变化）
        if hasattr(self, 'screen_width') and hasattr(self, 'screen_height'):
            screen_center_x = self.screen_width / 2
            screen_center_y = self.screen_height / 2
            
            # 计算每个点相对于屏幕中心的角度
            angles = []
            for x, y in points:
                dx = x - screen_center_x
                dy = y - screen_center_y
                angle = math.degrees(math.atan2(dy, dx))
                angles.append(angle)
            
            # 计算角度离散度
            if angles:
                angles.sort()
                max_angle_diff = max(angles[-1] - angles[0], 360 - (angles[-1] - angles[0]))
                angle_dispersion = min(max_angle_diff, 180)  # 最大角度差不超过180度
            else:
                angle_dispersion = 0
        else:
            # 如果没有屏幕尺寸信息，使用像素离散度估算角度离散度
            # 假设屏幕距离和尺寸，转换为近似角度
            screen_diagonal_pixels = 1920  # 假设
            angle_dispersion = (pixel_dispersion / screen_diagonal_pixels) * 180
        
        return {
            'angle_dispersion': angle_dispersion,
            'pixel_dispersion': pixel_dispersion,
            'point_count': len(self.gaze_points),
            'geometric_center': geometric_center
        }
    
    def check_trigger_conditions(self):
        """检查是否满足触发条件"""
        dispersion_info = self.calculate_dispersion()
        
        current_time = time.time() * 1000
        
        # 检查是否在冷却期内
        if current_time - self.last_trigger_time < self.trigger_cooldown:
            return False, None
        
        # 检查触发条件
        angle_triggered = dispersion_info['angle_dispersion'] < self.angle_threshold
        pixel_triggered = dispersion_info['pixel_dispersion'] < self.pixel_threshold
        
        if angle_triggered or pixel_triggered:
            self.last_trigger_time = current_time
            return True, dispersion_info['geometric_center']
        
        return False, None
    
    def set_screen_dimensions(self, width, height):
        """设置屏幕尺寸用于角度计算"""
        self.screen_width = width
        self.screen_height = height