"""
手部关键点检测系统 - 简化版

只包含手部关键点检测和实时可视化功能。
基于MediaPipe实现，支持实时手部关键点检测和可视化。

主要功能：
- 实时手部检测和关键点提取
- 关键点可视化
- 边界框绘制
- 简单的实时显示
- 食指指尖桌面接触检测

"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import os
from typing import Optional, List, Tuple, Dict
import time

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_detector import HandDetector
from one_euro_filter import HandOneEuroFilter
from affine_transformer_3d import AffineTransformer3D


class SimpleHandDetectionSystem:
    """
    简化版手部关键点检测系统
    
    只包含核心手部检测、关键点平滑和事件检测功能。
    """
    
    def __init__(self, camera_id: int = 2):
        """
        初始化手部跟踪系统
        
        Args:
            camera_id: 摄像头ID，默认为2
        """
        self.camera_id = camera_id
        self.cap = None
        self.hand_detector = HandDetector()
        
        # 初始化One Euro Filter平滑（自适应抖动抑制）
        self.smoother = HandOneEuroFilter(freq=30.0, min_cutoff=2.4, beta=2, d_cutoff=1.0)
        
        # 初始化3D仿射变换模块
        self.affine_transformer_3d = AffineTransformer3D()
        
        # 事件状态跟踪（只保留必要的冷却时间功能）
        self.event_states = {
            'last_pinch_time': 0,
            'last_click_time': 0,
            'click_cooldown': 0.5,  # 点击冷却时间（秒）
            'pinch_cooldown': 0.2   # 捏合冷却时间（秒）
        }
        
        # 速度追踪相关变量（只保留必要的）
        self.velocity_tracker = {
            'prev_distance': None,       # 上一帧指尖距离
            'prev_time': None,           # 上一帧时间
            'distance_velocity': 0.0,  # 距离变化速度
            'distance_history': [],      # 距离历史记录
            'approach_history': [],      # 接近状态历史记录
            'max_history': 5            # 历史记录最大长度
        }
        
        # 初始化检测引擎
        self._pinch_detection_engine = PinchDetectionEngine(self.affine_transformer_3d)
        self._click_detection_engine = ClickDetectionEngine(self.affine_transformer_3d, self.velocity_tracker)
        
        print("手部关键点检测系统初始化完成（One Euro Filter已启用）")
    
    def start_camera(self, camera_index: int = 2, width: int = 1280, height: int = 720):
        """
        启动摄像头
        
        Args:
            camera_index: 摄像头索引，默认使用2号摄像头
            width: 图像宽度
            height: 图像高度
            
        Returns:
            bool: 启动是否成功
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            return False
        
        # 设置摄像头参数 - 优化为较低分辨率以提高启动速度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 降低FPS以减少码率
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 使用MJPEG压缩格式
        
        print(f"摄像头启动成功 (索引: {camera_index}, 分辨率: {width}x{height}, FPS: {30})")
        return True
    
    def start_video_file(self, video_path: str):
        """
        启动视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            bool: 启动是否成功
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return False
        
        # 获取视频信息
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"视频文件启动成功 (路径: {video_path}, 分辨率: {width}x{height}, FPS: {fps})")
        return True
    
    def stop_camera(self):
        """停止摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
            print("摄像头已停止")
    
    def detect_hand_events(self, frame: np.ndarray) -> dict:
        """
        处理单帧图像，返回手部检测结果和事件信息
        
        Args:
            frame: 输入帧
            
        Returns:
            dict: 包含以下信息的字典
                - hand_results: 手部检测结果列表，每个结果包含：
                    - landmarks_2d: 平滑后的2D关键点
                    - landmarks_3d: 3D关键点
                    - center: 手部中心点
                    - handedness: 手部类型
                - events: 事件信息字典，包含：
                    - pinch_events: 捏合事件列表，每个事件包含：
                        - is_pinching: 是否捏合
                        - pinch_info: 捏合信息
                        - hand_index: 手部索引
                    - click_events: 点击事件列表，每个事件包含：
                        - is_click: 是否点击
                        - click_info: 点击信息
                        - hand_index: 手部索引
        """
        # 检测手部 (不绘制原始关键点，只使用平滑后的点)
        hand_results = self.hand_detector.detect_hands(frame, draw=False)
        
        # 使用简单平滑器处理每个手部
        for hand_info in hand_results:
            # 提取原始关键点
            landmarks_2d = hand_info['landmarks_2d']
            landmarks_3d = hand_info.get('landmarks', [])
            
            # 应用简单平滑
            smoothed_landmarks = self.smoother.smooth(landmarks_2d)
            
            # 更新关键点数据
            hand_info['landmarks_2d'] = smoothed_landmarks
            hand_info['landmarks_3d'] = landmarks_3d  # 保存3D关键点数据
            
            # 重新计算中心点
            if smoothed_landmarks:
                center_x = sum(point[0] for point in smoothed_landmarks) / len(smoothed_landmarks)
                center_y = sum(point[1] for point in smoothed_landmarks) / len(smoothed_landmarks)
                hand_info['center'] = (center_x, center_y)
        
        # 事件信息初始化
        events = {
            'pinch_events': [],
            'click_events': []
        }
        
        # 事件状态管理
        current_time = time.time()
        
        # 处理每个手部的事件
        for hand_index, hand_info in enumerate(hand_results):
            # 从hand_info字典中提取数据
            landmarks_2d = hand_info['landmarks_2d']  # 这里已经是平滑后的
            landmarks_3d = hand_info.get('landmarks_3d', [])
            
            # 检测食指和大拇指捏合
            is_pinching, pinch_info = self._detect_thumb_index_pinch(landmarks_2d, landmarks_3d)
            
            # 检测食指点击大拇指事件
            is_click, click_info = self._detect_index_thumb_click(landmarks_2d, landmarks_3d)
            
            # 每次检测到的事件都添加到events列表中，不受冷却时间限制
            if is_click:
                events['click_events'].append({
                    'is_click': is_click,
                    'click_info': click_info,
                    'hand_index': hand_index,
                    'thumb_tip': landmarks_2d[4] if len(landmarks_2d) > 4 else None
                })
                
                # 冷却时间只用于控制print输出频率
                if current_time - self.event_states['last_click_time'] > self.event_states['click_cooldown']:
                    self.event_states['last_click_time'] = current_time
            
            # 每次检测到的捏合事件都添加到events列表中，不受冷却时间限制
            if is_pinching:
                events['pinch_events'].append({
                    'is_pinching': is_pinching,
                    'pinch_info': pinch_info,
                    'hand_index': hand_index,
                    'thumb_tip': landmarks_2d[4] if len(landmarks_2d) > 4 else None
                })
                
                # 冷却时间只用于控制print输出频率
                if current_time - self.event_states['last_pinch_time'] > self.event_states['pinch_cooldown']:
                    self.event_states['last_pinch_time'] = current_time
        
        return {
            'hand_results': hand_results,
            'events': events
        }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像（保留原方法用于向后兼容）
        
        Args:
            frame: 输入帧
            
        Returns:
            np.ndarray: 处理后的帧
        """
        # 调用新的detect_hand_events方法获取检测结果
        result = self.detect_hand_events(frame)
        hand_results = result['hand_results']
        events = result['events']
        
        # 绘制检测结果
        for hand_index, hand_info in enumerate(hand_results):
            # 从hand_info字典中提取数据
            landmarks_2d = hand_info['landmarks_2d']  # 这里已经是平滑后的
            center = hand_info['center']
            handedness = hand_info.get('handedness', 'Unknown')
            
            # 计算边界框
            if landmarks_2d:
                x_coords = [point[0] for point in landmarks_2d]
                y_coords = [point[1] for point in landmarks_2d]
                x_min, x_max = int(min(x_coords) * frame.shape[1]), int(max(x_coords) * frame.shape[1])
                y_min, y_max = int(min(y_coords) * frame.shape[0]), int(max(y_coords) * frame.shape[0])
                bbox = (x_min, y_min, x_max, y_max)
            else:
                bbox = None
            
            # 绘制边界框
            self._draw_bounding_box(frame, bbox, handedness)
            
            # 绘制关键点
            self._draw_landmarks(frame, landmarks_2d)
            
            # 检查当前手部是否有捏合或点击事件
            is_pinching = False
            is_click = False
            pinch_info = {}
            click_info = {}
            
            # 检查捏合事件
            for pinch_event in events['pinch_events']:
                if pinch_event['hand_index'] == hand_index:
                    is_pinching = pinch_event['is_pinching']
                    pinch_info = pinch_event['pinch_info']
                    break
            
            # 检查点击事件
            for click_event in events['click_events']:
                if click_event['hand_index'] == hand_index:
                    is_click = click_event['is_click']
                    click_info = click_event['click_info']
                    break
            
            # 绘制事件检测信息
            self._draw_event_info(frame, is_pinching, is_click, pinch_info, click_info)
            
            # 如果检测到事件，绘制特殊效果
            if is_click:
                self._draw_click_effect(frame, landmarks_2d, handedness)
            elif is_pinching:
                self._draw_pinch_effect(frame, landmarks_2d, handedness)
        
        # 显示检测信息
        self._display_detection_info(frame, len(hand_results))
        
        return frame
    
    def _draw_bounding_box(self, frame: np.ndarray, bbox: Optional[Tuple], handedness: str):
        """绘制边界框"""
        if bbox:
            x1, y1, x2, y2 = bbox
            # 左手用绿色，右手用蓝色
            color = (0, 255, 0) if "Left" in handedness else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, handedness, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks_2d: Optional[List[Tuple]]):
        """绘制手部关键点"""
        if not landmarks_2d:
            return
        
        # MediaPipe手部关键点连接
        # 定义关键点之间的连接关系
        connections = [
            # 大拇指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (5, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (9, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (13, 17), (17, 18), (18, 19), (19, 20),
            # 手掌连接
            (5, 9), (9, 13), (13, 17)
        ]
        
        # 绘制连接线
        for connection in connections:
            if connection[0] < len(landmarks_2d) and connection[1] < len(landmarks_2d):
                point1 = (int(landmarks_2d[connection[0]][0] * frame.shape[1]),
                         int(landmarks_2d[connection[0]][1] * frame.shape[0]))
                point2 = (int(landmarks_2d[connection[1]][0] * frame.shape[1]),
                         int(landmarks_2d[connection[1]][1] * frame.shape[0]))
                cv2.line(frame, point1, point2, (0, 255, 255), 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(landmarks_2d):
            point = (int(x * frame.shape[1]), int(y * frame.shape[0]))
            # 手腕点用红色，其他点用黄色
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.circle(frame, point, 3, color, -1)
            cv2.circle(frame, point, 3, (0, 0, 0), 1)
    
    def _output_keypoints_info(self, hand_results: List[dict]):
        """每秒输出一次手部关键点信息（只输出平滑后的点）"""
        # 移除对last_output_time和output_interval属性的依赖
        # 简化为只在检测到手部时输出信息
        if hand_results:
            current_time = time.time()
            print(f"\n=== 手部关键点信息（已平滑）===")
            print(f"输出时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
            print(f"检测到手部数量: {len(hand_results)}")
            
            for i, hand_info in enumerate(hand_results):
                print(f"\n--- 手部 {i+1} ---")
                print(f"手部类型: {hand_info.get('handedness', 'Unknown')}")
                
                # 输出平滑后的点0（手腕）的坐标
                landmarks_2d = hand_info.get('landmarks_2d', [])
                
                if landmarks_2d and len(landmarks_2d) >= 1:
                    wrist_x, wrist_y = landmarks_2d[0]
                    print(f"手腕点0坐标（平滑后）: ({wrist_x:.6f}, {wrist_y:.6f})")
                else:
                    print("手腕点0坐标（平滑后）: 无法获取")
            
            print("=" * 40)

    def _display_detection_info(self, frame: np.ndarray, hand_count: int):
        """显示检测信息"""
        # 移除对last_output_time和output_interval属性的依赖
        # 每次都显示检测信息
        info_text = [
            f"检测到手部: {hand_count}",
            "按 'q' 退出程序",
            f"时间: {time.strftime('%H:%M:%S', time.localtime(time.time()))}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_event_info(self, frame: np.ndarray, is_pinching: bool, is_click: bool, 
                        pinch_info: dict, click_info: dict):
        """
        绘制事件检测信息
        """
        # 在左上角显示事件状态
        y_offset = 30
        
        if is_click:
            cv2.putText(frame, "CLICK! 食指点击大拇指", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 25
            if click_info:
                cv2.putText(frame, f"捏合强度: {click_info.get('pinch_strength', 0):.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
        elif is_pinching:
            cv2.putText(frame, "PINCH 食指大拇指捏合", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 25
            if pinch_info:
                cv2.putText(frame, f"距离: {pinch_info.get('thumb_index_distance', 0):.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                cv2.putText(frame, f"强度: {pinch_info.get('pinch_strength', 0):.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                # 显示速度信息（如果可用）
                thumb_vel = pinch_info.get('thumb_velocity', 0)
                index_vel = pinch_info.get('index_velocity', 0)
                if thumb_vel > 0 or index_vel > 0:
                    cv2.putText(frame, f"拇指速度: {thumb_vel:.3f}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20
                    cv2.putText(frame, f"食指速度: {index_vel:.3f}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20

    
    def _draw_click_effect(self, frame: np.ndarray, landmarks_2d: List[Tuple], handedness: str):
        """绘制点击效果"""
        h, w = frame.shape[:2]
        
        # 获取关键点
        thumb_tip = landmarks_2d[4]
        index_tip = landmarks_2d[8]
        
        # 计算中点
        center_x = int((thumb_tip[0] + index_tip[0]) * w / 2)
        center_y = int((thumb_tip[1] + index_tip[1]) * h / 2)
        
        # 绘制点击圆圈动画
        cv2.circle(frame, (center_x, center_y), 20, (0, 255, 255), 3)
        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 255), -1)
        
        # 绘制连接线
        thumb_pos = (int(thumb_tip[0] * w), int(thumb_tip[1] * h))
        index_pos = (int(index_tip[0] * w), int(index_tip[1] * h))
        cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 3)
        
        # 显示点击文字
        cv2.putText(frame, "CLICK!", (center_x - 30, center_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    def _draw_pinch_effect(self, frame: np.ndarray, landmarks_2d: List[Tuple], handedness: str):
        """绘制捏合效果"""
        h, w = frame.shape[:2]
        
        # 获取关键点
        thumb_tip = landmarks_2d[4]
        index_tip = landmarks_2d[8]
        
        # 计算中点
        center_x = int((thumb_tip[0] + index_tip[0]) * w / 2)
        center_y = int((thumb_tip[1] + index_tip[1]) * h / 2)
        
        # 绘制捏合圆圈
        cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), 2)
        
        # 绘制连接线
        thumb_pos = (int(thumb_tip[0] * w), int(thumb_tip[1] * h))
        index_pos = (int(index_tip[0] * w), int(index_tip[1] * h))
        cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 2)
        
        # 显示捏合强度
        cv2.putText(frame, "PINCH", (center_x - 20, center_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _detect_thumb_index_pinch(self, landmarks_2d: Optional[List[Tuple]], landmarks_3d: Optional[List[List[float]]] = None) -> tuple[bool, dict]:
        """
        检测食指和大拇指的捏合动作（接触事件）
        
        Args:
            landmarks_2d: 2D关键点列表
            landmarks_3d: 3D关键点列表
            
        Returns:
            tuple: (是否捏合, 调试信息)
        """
        return self._pinch_detection_engine.detect_pinch(landmarks_2d, landmarks_3d)
    
    def _detect_index_thumb_click(self, landmarks_2d: Optional[List[Tuple]], 
                                 landmarks_3d: Optional[List[List[float]]] = None,
                                 thumb_movement_threshold: float = 0.02) -> tuple[bool, dict]:
        """
        检测食指点击大拇指事件（基于快速捏合序列检测）
        
        Args:
            landmarks_2d: 2D关键点列表
            landmarks_3d: 3D关键点列表
            thumb_movement_threshold: 大拇指移动阈值
            
        Returns:
            tuple: (是否点击, 调试信息)
        """
        return self._click_detection_engine.detect_click(landmarks_2d, landmarks_3d)
    
    def cleanup(self):
        """清理资源"""
        self.stop_camera()
        print("系统已清理")


class PinchDetectionEngine:
    """捏合检测引擎 - 封装捏合检测逻辑"""
    
    def __init__(self, affine_transformer_3d: AffineTransformer3D):
        """
        初始化捏合检测引擎
        
        Args:
            affine_transformer_3d: 3D仿射变换模块
        """
        self.affine_transformer_3d = affine_transformer_3d
    
    def detect_pinch(self, landmarks_2d: Optional[List[Tuple]], landmarks_3d: Optional[List[List[float]]] = None) -> tuple[bool, dict]:
        """
        检测食指和大拇指的捏合动作
        
        Args:
            landmarks_2d: 2D关键点列表
            landmarks_3d: 3D关键点列表
            
        Returns:
            tuple: (是否捏合, 调试信息)
        """
        # 如果有3D关键点数据，使用3D仿射变换进行检测
        if landmarks_3d and len(landmarks_3d) >= 21:
            # print("3D关键点检测")
            return self.affine_transformer_3d.is_thumb_index_pinch(landmarks_3d)
        
        # fallback到2D检测
        if not landmarks_2d or len(landmarks_2d) < 21:
            return False, {"error": "关键点不足"}
    
        # 获取关键点索引
        thumb_tip = landmarks_2d[4]      # 大拇指指尖
        index_tip = landmarks_2d[8]       # 食指指尖
        thumb_mcp = landmarks_2d[2]      # 大拇指掌指关节
        index_mcp = landmarks_2d[5]      # 食指掌指关节
        
        # 计算指尖距离
        thumb_index_distance = np.sqrt(
            (thumb_tip[0] - index_tip[0])**2 + 
            (thumb_tip[1] - index_tip[1])**2
        )
        
        # 计算手掌大小（用作相对距离参考）
        palm_size = np.sqrt(
            (thumb_mcp[0] - index_mcp[0])**2 + 
            (thumb_mcp[1] - index_mcp[1])**2
        )
        
        # 判断捏合的阈值（相对手掌大小的比例）
        pinch_threshold = palm_size * 0.45  # 手掌大小的45%（调大阈值，降低灵敏度）
        is_pinching = thumb_index_distance < pinch_threshold
        
        # 精简调试信息
        debug_info = {
            'is_pinching': is_pinching
        }
        
        return is_pinching, debug_info


class ClickDetectionEngine:
    """点击检测引擎 - 封装点击检测逻辑"""
    
    def __init__(self, affine_transformer_3d: AffineTransformer3D, velocity_tracker: dict):
        """
        初始化点击检测引擎
        
        Args:
            affine_transformer_3d: 3D仿射变换模块
            velocity_tracker: 速度追踪器
        """
        self.affine_transformer_3d = affine_transformer_3d
        self.velocity_tracker = velocity_tracker
    
    def detect_click(self, landmarks_2d: Optional[List[Tuple]], landmarks_3d: Optional[List[List[float]]] = None) -> tuple[bool, dict]:
        """
        检测食指点击大拇指事件
        
        Args:
            landmarks_2d: 2D关键点列表
            landmarks_3d: 3D关键点列表
            
        Returns:
            tuple: (是否点击, 调试信息)
        """
        # 使用快速捏合序列检测
        is_fast_pinch, fast_pinch_info = self._detect_fast_pinch_sequence(landmarks_2d, landmarks_3d)
        
        # 精简调试信息，只保留核心数据
        debug_info = {
            'is_click': is_fast_pinch,
            **fast_pinch_info
        }
        
        return is_fast_pinch, debug_info
    
    def _calculate_hand_size(self, landmarks_2d: Optional[List[Tuple]], landmarks_3d: Optional[List[List[float]]] = None) -> float:
        """计算手部大小"""
        hand_size = 0.1
        if landmarks_3d and len(landmarks_3d) >= 21:
            # print("3D手部大小计算")
            hand_size = self.affine_transformer_3d.calculate_hand_size(landmarks_3d)
        elif landmarks_2d and len(landmarks_2d) >= 21:
            palm_size = np.sqrt(
                (landmarks_2d[2][0] - landmarks_2d[5][0])**2 + 
                (landmarks_2d[2][1] - landmarks_2d[5][1])**2
            )
            hand_size = palm_size
        return hand_size
    
    def _detect_fast_pinch_sequence(self, landmarks_2d: Optional[List[Tuple]], 
                                   landmarks_3d: Optional[List[List[float]]] = None) -> tuple[bool, dict]:
        """检测快速捏合序列"""
        current_distance, distance_velocity = self._calculate_pinch_distance_velocity(landmarks_2d, landmarks_3d)
        hand_size = self._calculate_hand_size(landmarks_2d, landmarks_3d)
        
        pinch_threshold = hand_size * 0.4  # 调大阈值，降低灵敏度
        is_pinching = current_distance < pinch_threshold
        
        # 检测快速接近（调高速度阈值）
        is_approaching = distance_velocity > hand_size * 3
        
        # 更新状态历史
        current_time = time.time()
        self.velocity_tracker['approach_history'].append({
            'approaching': is_approaching,
            'pinching': is_pinching,
            'distance': current_distance,
            'velocity': distance_velocity,
            'timestamp': current_time
        })
        
        # 保持历史记录
        if len(self.velocity_tracker['approach_history']) > 10:
            self.velocity_tracker['approach_history'].pop(0)
        
        # 分析快速捏合序列
        is_fast_pinch = False
        
        if len(self.velocity_tracker['approach_history']) >= 3:
            recent_history = self.velocity_tracker['approach_history'][-5:]
            approach_frames = sum(1 for frame in recent_history if frame['approaching'])
            recent_pinch = any(frame['pinching'] for frame in recent_history[-3:])
            
            is_fast_pinch = (approach_frames >= 2 and 
                           recent_pinch and 
                           distance_velocity > hand_size * 0.9)
        
        # 精简调试信息
        debug_info = {
            'is_fast_pinch': is_fast_pinch
        }
        
        return is_fast_pinch, debug_info
    
    def _calculate_pinch_distance_velocity(self, landmarks_2d: Optional[List[Tuple]], 
                                         landmarks_3d: Optional[List[List[float]]] = None) -> tuple[float, float]:
        """计算捏合距离变化速度"""
        current_time = time.time()
        current_distance = 0.0
        distance_velocity = 0.0
        
        # 计算当前帧的指尖距离
        if landmarks_3d and len(landmarks_3d) >= 21:
            # print("3D指尖距离计算")
            thumb_tip = landmarks_3d[4]
            index_tip = landmarks_3d[8]
            current_distance = self.affine_transformer_3d.calculate_3d_distance(thumb_tip, index_tip)
        elif landmarks_2d and len(landmarks_2d) >= 21:
            thumb_tip = landmarks_2d[4]
            index_tip = landmarks_2d[8]
            current_distance = np.sqrt(
                (thumb_tip[0] - index_tip[0])**2 + 
                (thumb_tip[1] - index_tip[1])**2
            )
        
        # 计算距离变化速度
        if (self.velocity_tracker['prev_distance'] is not None and 
            self.velocity_tracker['prev_time'] is not None):
            
            dt = current_time - self.velocity_tracker['prev_time']
            if dt > 0:
                distance_change = self.velocity_tracker['prev_distance'] - current_distance
                distance_velocity = distance_change / dt
                
                # 添加到历史记录
                self.velocity_tracker['distance_history'].append({
                    'distance': current_distance,
                    'velocity': distance_velocity,
                    'timestamp': current_time
                })
                
                if len(self.velocity_tracker['distance_history']) > self.velocity_tracker['max_history']:
                    self.velocity_tracker['distance_history'].pop(0)
        
        # 更新上一帧数据
        self.velocity_tracker['prev_distance'] = current_distance
        self.velocity_tracker['prev_time'] = current_time
        
        # 计算平均速度（平滑处理）
        if self.velocity_tracker['distance_history']:
            avg_distance_vel = np.mean([d['velocity'] for d in self.velocity_tracker['distance_history']])
            self.velocity_tracker['distance_velocity'] = avg_distance_vel
        else:
            self.velocity_tracker['distance_velocity'] = distance_velocity
        
        return current_distance, self.velocity_tracker['distance_velocity']
    

    

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手部关键点检测系统 (简化版)')
    parser.add_argument('--camera', type=int, default=2, help='摄像头索引')
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--max_hands', type=int, default=1, help='最大检测手部数')
    parser.add_argument('--detection_conf', type=float, default=0.7, help='检测置信度')
    
    args = parser.parse_args()
    
    # 创建系统实例
    hand_system = SimpleHandDetectionSystem(camera_id=args.camera)
    
    try:
        # 启动摄像头
        if not hand_system.start_camera(args.camera):
            return
        
        print("\n=== 手部关键点检测系统 (简化版) ===")
        print("功能:")
        print("- 实时手部关键点检测")
        print("- 关键点和边界框可视化")
        print("- 捏合和点击事件检测")
        print("控制:")
        print("  'q' - 退出程序")
        
        is_running = True
        while is_running:
            ret, frame = hand_system.cap.read()
            if not ret:
                continue
            
            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 处理帧
            processed_frame = hand_system.process_frame(frame)
            
            # 显示帧
            cv2.imshow('手部关键点检测系统 (简化版)', processed_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                is_running = False
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        hand_system.cleanup()
        print("系统已清理")


if __name__ == "__main__":
    main()