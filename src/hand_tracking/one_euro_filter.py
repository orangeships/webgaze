#!/usr/bin/env python3
"""
One Euro Filter 实现

基于 "One Euro Filter: A Simple Speed-based Low-pass Filter" 论文实现。
这是一种自适应低通滤波器，能够根据运动速度自动调整截止频率，
在减少抖动的同时保持良好的响应性。

参考论文: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import math
from typing import List, Tuple, Optional


class LowPassFilter:
    """低通滤波器"""
    
    def __init__(self, alpha: float):
        """
        初始化低通滤波器
        
        Args:
            alpha: 滤波系数 (0-1)，越大越平滑
        """
        self.alpha = alpha
        self.x = None  # 上一次滤波值
    
    def __call__(self, value: float) -> float:
        """
        应用低通滤波
        
        Args:
            value: 输入值
            
        Returns:
            float: 滤波后的值
        """
        if self.x is None:
            self.x = value
        else:
            self.x = self.alpha * value + (1 - self.alpha) * self.x
        return self.x


class OneEuroFilter:
    """One Euro Filter 单点滤波器"""
    
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        初始化 One Euro Filter
        
        Args:
            freq: 信号频率 (Hz)，默认30fps
            min_cutoff: 最小截止频率 (Hz)，越大越平滑但延迟越高
            beta: 速度系数，越大对高速运动越敏感
            d_cutoff: 导数截止频率 (Hz)
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(d_cutoff))
        
        self.last_time = None
        self.last_value = None
    
    def _alpha(self, cutoff: float) -> float:
        """
        计算滤波系数 alpha
        
        Args:
            cutoff: 截止频率
            
        Returns:
            float: 滤波系数
        """
        te = 1.0 / self.freq  # 采样周期
        tau = 1.0 / (2 * math.pi * cutoff)  # 时间常数
        return te / (te + tau)
    
    def __call__(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        应用 One Euro Filter
        
        Args:
            value: 输入值
            timestamp: 时间戳（可选）
            
        Returns:
            float: 滤波后的值
        """
        if self.last_value is None:
            self.last_value = value
            return value
        
        # 计算时间差
        if timestamp is None:
            if self.last_time is None:
                self.last_time = 0
                dt = 1.0 / self.freq
            else:
                dt = 1.0 / self.freq  # 假设恒定帧率
        else:
            if self.last_time is None:
                self.last_time = timestamp
                dt = 1.0 / self.freq
            else:
                dt = timestamp - self.last_time
                self.last_time = timestamp
        
        if dt <= 0:
            dt = 1.0 / self.freq
        
        # 计算导数（速度）
        dx = (value - self.last_value) / dt
        
        # 滤波导数
        dx_hat = self.dx_filter(dx)
        
        # 自适应截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # 更新滤波器
        self.x_filter.alpha = self._alpha(cutoff)
        
        # 滤波值
        value_hat = self.x_filter(value)
        
        self.last_value = value
        
        return value_hat


class HandOneEuroFilter:
    """手部关键点 One Euro Filter 滤波器"""
    
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        """
        初始化手部 One Euro Filter
        
        Args:
            freq: 信号频率 (Hz)，默认30fps
            min_cutoff: 最小截止频率 (Hz)，推荐值1.0-2.5
            beta: 速度系数，推荐值0.002-0.01，越大对高速运动越敏感
            d_cutoff: 导数截止频率 (Hz)，推荐值1.0
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # 为每个关键点创建滤波器 (21个关键点 × 2坐标)
        self.filters = []
        for i in range(21):
            self.filters.append([
                OneEuroFilter(freq, min_cutoff, beta, d_cutoff),  # x坐标
                OneEuroFilter(freq, min_cutoff, beta, d_cutoff)   # y坐标
            ])
    
    def smooth(self, landmarks: List[Tuple[float, float]], timestamp: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        平滑手部关键点
        
        Args:
            landmarks: 原始关键点列表 [(x1, y1), (x2, y2), ...]
            timestamp: 时间戳（可选）
            
        Returns:
            List[Tuple]: 平滑后的关键点列表
        """
        if not landmarks:
            return landmarks
        
        smoothed_landmarks = []
        
        for i, (x, y) in enumerate(landmarks):
            if i < len(self.filters):
                # 应用滤波器
                smoothed_x = self.filters[i][0](x, timestamp)
                smoothed_y = self.filters[i][1](y, timestamp)
                smoothed_landmarks.append((smoothed_x, smoothed_y))
            else:
                # 如果关键点数量超过预期，使用原始值
                smoothed_landmarks.append((x, y))
        
        return smoothed_landmarks
    
    def reset(self):
        """重置所有滤波器"""
        # 重新创建滤波器
        self.filters = []
        for i in range(21):
            self.filters.append([
                OneEuroFilter(self.freq, self.min_cutoff, self.beta, self.d_cutoff),
                OneEuroFilter(self.freq, self.min_cutoff, self.beta, self.d_cutoff)
            ])


# 测试代码
if __name__ == "__main__":
    import numpy as np
    
    # 创建滤波器
    filter = HandOneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.007)
    
    # 模拟抖动数据
    def generate_noisy_signal(t):
        return 0.5 + 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn()
    
    # 测试滤波效果
    print("One Euro Filter 测试:")
    print("时间\t原始值\t\t滤波值")
    print("-" * 40)
    
    for i in range(100):
        t = i / 30.0  # 30fps
        original = generate_noisy_signal(t)
        
        # 创建单个关键点进行测试
        landmarks = [(original, original)]
        filtered = filter.smooth(landmarks, timestamp=t)
        
        if i % 10 == 0:  # 每10帧输出一次
            print(f"{t:.2f}\t{original:.4f}\t\t{filtered[0][0]:.4f}")