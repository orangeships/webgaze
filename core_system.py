#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼手交互系统核心协调器 (EyeHandInteractionSystem)

该模块是整个眼手交互系统的核心协调器，负责管理和协调各个子模块，实现注视点检测、校准、多屏支持和交互模式运行。

主要功能：
1. 系统初始化和资源管理
2. 校准流程控制（单屏/多屏）
3. 交互模式运行和状态管理
4. 注视点数据处理和分发
5. 多屏环境下的屏幕切换和坐标转换
6. 手眼协调机制的集成

核心类：
- EyeHandInteractionSystem: 主系统协调类，管理所有子模块

关键依赖：
- PyQt5: UI组件和事件处理
- OpenCV: 摄像头视频捕获
- pygame: 校准UI界面
- gaze_tracking.model: 眼动追踪模型
- 自定义模块: gaze_analysis, screen_management, hand_eye_coordination, ui_components

使用流程：
1. 创建EyeHandInteractionSystem实例
2. 调用initialize()初始化系统
3. 调用show_menu()显示主菜单
4. 调用run_calibration()执行校准
5. 调用run_interaction_mode()进入交互模式

作者：TraeAI
版本：1.0.0
日期：2025-12-09
"""
import os
import sys
import cv2
import time
import numpy as np
import json
from collections import deque
import math
import ctypes
from ctypes import wintypes
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox

# 导入其他模块
from ui_components import EyeHandInteractionUI
from gaze_analysis import LightweightKalmanFilter, GazeDispersionAnalyzer
from hand_eye_coordination import HandEyeCoordinator
from screen_management import ScreenManager
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel

class EyeHandInteractionSystem:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.ui = None
        self.model = None
        self.homtrans = None
        self.cap = None
        self.calibration_data = None
        
        # SfM启用状态
        self.sfm_enabled = False  # 默认启用SfM
        # 双屏支持相关
        self.calibration_mode = "single"  # 校准模式：single 或 multi
        self.is_dual_screen_mode = False  # 标志位，指示是否为双屏模式
        
        # 跨屏坐标转换支持
        self.screen_switching = False  # 标志位，防止屏幕切换过程中重复触发
        self.last_gaze_point_abs = None  # 存储上一帧的绝对坐标注视点，用于平滑过渡
        self.screen_switching_adaptation_frames = 0  # 屏幕切换后的卡尔曼滤波适应帧数
        
        
        
        # 统一的阈值配置管理器
        self.threshold_config = {
            'distance_multiplier': 1.0,      # 距离阈值倍数，用于跨屏调整
            'auto_move_distance': 400,       # 鼠标右键触发移动的距离阈值（像素）
            'auto_move_cooldown': 1000,      # 自动移动冷却时间（毫秒）
            'gaze_dispersion_frames': 6,     # 注视点分析使用的帧数
            'angle_threshold': 3.0,          # 角度离散度阈值（度） 
            'pixel_threshold': 100,          # 像素离散度阈值（像素）
            'boundary_buffer': 20,          # 屏幕边界缓冲区（像素）- 增加到100像素，避免边缘区域失效
            'interaction_zone_duration': 500,  # 交互区域显示持续时间（毫秒）
            # 鼠标移动触发相关配置
            'mouse_movement_threshold': 300,  # 鼠标移动触发阈值（像素）
            'teleport_circle_radius': 100,     # 传送圆周半径（像素）
            'teleport_cooldown_duration': 1000,  # 传送冷却时间（毫秒）
            'sliding_window_time_limit': 350,   # 滑动窗口时间限制（毫秒）
            'mad_threshold': 50,               # 稳定凝视MAD阈值（像素）
            'perc95_threshold': 120,           # 稳定凝视95%百分位阈值（像素）
            'sliding_window_angle_threshold': 3.0,  # 滑动窗口角度阈值
            # UI辅助跳转相关配置
            'ui_assisted_jump_enabled': True,         # UI辅助跳转功能开关
            'ui_control_max_size': 500,               # UI控件最大尺寸（像素），超过则忽略
            'scatter_detection_radius': 100,          # 散布检测半径（像素）
        }
        
        # 初始化注视点分析器，使用统一配置
        self.dispersion_analyzer = GazeDispersionAnalyzer(
            frame_count=self.threshold_config['gaze_dispersion_frames'],
            angle_threshold=self.threshold_config['angle_threshold'],
            pixel_threshold=self.threshold_config['pixel_threshold']
        )
        
        # 注视点平滑相关
        self.smoothing_enabled = True  # 平滑开关
        # 使用轻量卡尔曼滤波器替代原来的平滑机制
        self.kalman_filter = LightweightKalmanFilter(
            process_noise=0.8,       # 过程噪声，进一步提高响应速度
            measurement_noise=0.4,   # 测量噪声，更信任新测量值，平滑更轻微
            error_estimate=50.0 
        )
        
        # 注视点显示控制
        self.show_gaze_point = True  # 控制是否显示注视点（红点）
        
        # 系统运行状态
        self.running = False
    
    def initialize(self):
        """初始化系统"""
        # 初始化UI
        self.ui = EyeHandInteractionUI()
        self.ui.initialize_display()
        
        # 设置屏幕尺寸用于角度计算
        self.dispersion_analyzer.set_screen_dimensions(self.ui.screen_width, self.ui.screen_height)
        
        # 初始化模型
        self.model = EyeModel(self.project_dir)
        
        # 初始化HomTransform（基础实例，稍后应用校准数据）
        self.homtrans = HomTransform(self.project_dir)
        
        # 初始化屏幕管理器
        self.screen_manager = ScreenManager(self.project_dir, self.ui)
        
        # 初始化手眼协调器
        self.hand_eye_coordinator = HandEyeCoordinator(self.ui, self.screen_manager, self.threshold_config)

        # 将注视点分析器切换为绝对坐标基准
        self._configure_absolute_dispersion()
        
        # 初始化摄像头
        self.cap = None
        for device_id in [1, 0, 2]:
            try:
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    self.cap.release()
                    self.cap = None
            except Exception:
                continue
        print(f"使用摄像头设备ID: {device_id}")
        if self.cap is None:
            return False
            
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
    
    def switch_to_monitor_with_coordinate_update(self, target_screen):
        """切换到目标屏幕并更新坐标"""
        # 检查是否已经是最优显示器
        if target_screen == self.screen_manager.current_monitor_index or target_screen >= len(self.ui.monitors_info):
            return False
            
        try:
            # 设置屏幕切换标志并保存当前状态
            self.screen_switching = True
            last_gaze_point = self.last_gaze_point_abs
            
            # 切换显示器 - 使用带坐标更新的方法
            if self.screen_manager.switch_to_monitor_with_coordinate_update(target_screen, self.homtrans, self.dispersion_analyzer, self.kalman_filter):
                # 重新初始化卡尔曼滤波器以适应新屏幕并更新坐标转换
                self.kalman_filter = LightweightKalmanFilter(
                    process_noise=0.8,  # 增加过程噪声，加速适应新屏幕
                    measurement_noise=0.4,  # 增加测量噪声，减少拖拽效应
                    error_estimate=2.0  # 增加初始误差估计
                )
                self.screen_switching_adaptation_frames = 5  # 屏幕切换后5帧内快速适应
                
                # 切换后重新配置屏幕边界与绝对坐标分析基准
                target_monitor = self.ui.monitors_info[target_screen]
                self._configure_absolute_dispersion()
                
                # 如果有上一帧注视点，将其转换为新屏幕的坐标
                if last_gaze_point:
                    new_gaze_x = last_gaze_point[0] - target_monitor['x']
                    new_gaze_y = last_gaze_point[1] - target_monitor['y']
                    print(f"[DEBUG] 屏幕切换到{target_screen}，卡尔曼滤波器已重置，将快速适应新坐标")
                
                self.screen_switching = False
                return True
            else:
                self.screen_switching = False
                return False
            
        except Exception as e:
            print(f"切换屏幕失败: {e}")
            import traceback
            traceback.print_exc()
            self.screen_switching = False
            return False
    
    def _smooth_gaze_point(self, raw_gaze_point):
        """使用卡尔曼滤波平滑注视点"""
        return self.kalman_filter.update(raw_gaze_point)

    def _configure_absolute_dispersion(self):
        """
        将注视点分析器切换为绝对坐标体系。
        统一记录桌面原点和范围，便于后续做距离计算和坐标归一化。
        """
        if not self.ui or not getattr(self.ui, "monitors_info", None):
            return

        xs = [m['x'] for m in self.ui.monitors_info]
        ys = [m['y'] for m in self.ui.monitors_info]
        rights = [m['x'] + m['width'] for m in self.ui.monitors_info]
        bottoms = [m['y'] + m['height'] for m in self.ui.monitors_info]

        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(rights), max(bottoms)

        self.desktop_origin = (min_x, min_y)
        self.desktop_size = (max_x - min_x, max_y - min_y)
        self.dispersion_analyzer.set_screen_dimensions(self.desktop_size[0], self.desktop_size[1])

    def _convert_rel_to_abs(self, rel_point, monitor):
        """将当前屏幕的相对坐标转成绝对坐标"""
        return (int(rel_point[0] + monitor['x']), int(rel_point[1] + monitor['y']))

    def _convert_abs_to_rel(self, abs_point, monitor):
        """将绝对坐标转换回指定屏幕的相对坐标"""
        return (abs_point[0] - monitor['x'], abs_point[1] - monitor['y'])

    def _clamp_relative_to_monitor(self, rel_point, monitor):
        """约束相对坐标在屏幕内，避免视觉元素溢出"""
        clamped_x = max(0, min(int(rel_point[0]), monitor['width'] - 1))
        clamped_y = max(0, min(int(rel_point[1]), monitor['height'] - 1))
        return clamped_x, clamped_y

    def _smooth_absolute_point(self, abs_point):
        """
        在绝对坐标系下做平滑。
        屏幕切换适配帧内跳过滤波，避免卡尔曼状态拖拽。
        """
        if getattr(self, "screen_switching_adaptation_frames", 0) > 0:
            self.screen_switching_adaptation_frames -= 1
            return abs_point
        return self._smooth_gaze_point(abs_point)

    def _is_near_screen_edge(self, abs_point, monitor, threshold=5):
        """检查绝对坐标距离当前屏幕边界是否在阈值内"""
        left = monitor['x']
        right = monitor['x'] + monitor['width']
        top = monitor['y']
        bottom = monitor['y'] + monitor['height']
        return (
            abs_point[0] - left < threshold or
            right - abs_point[0] < threshold or
            abs_point[1] - top < threshold or
            bottom - abs_point[1] < threshold
        )

    def _resolve_target_screen_from_point(self, abs_point, monitor, threshold=5):
        """
        在接近边界时根据绝对坐标推断目标屏幕。
        优先用当前位置判定，若仍在当前屏幕则沿最近边界外推1像素再次判定。
        """
        if len(self.ui.monitors_info) <= 1:
            return None

        direct_target = self.screen_manager.get_target_screen(abs_point[0], abs_point[1])
        if direct_target != self.screen_manager.current_monitor_index:
            return direct_target

        distances = {
            "left": abs_point[0] - monitor['x'],
            "right": monitor['x'] + monitor['width'] - abs_point[0],
            "top": abs_point[1] - monitor['y'],
            "bottom": monitor['y'] + monitor['height'] - abs_point[1]
        }
        nearest_edge = min(distances, key=distances.get)
        projected_point = list(abs_point)
        if nearest_edge == "left":
            projected_point[0] = monitor['x'] - 1
        elif nearest_edge == "right":
            projected_point[0] = monitor['x'] + monitor['width'] + 1
        elif nearest_edge == "top":
            projected_point[1] = monitor['y'] - 1
        elif nearest_edge == "bottom":
            projected_point[1] = monitor['y'] + monitor['height'] + 1

        projected_target = self.screen_manager.get_target_screen(projected_point[0], projected_point[1])
        if projected_target != self.screen_manager.current_monitor_index:
            return projected_target
        return None

    def _add_absolute_gaze_to_analyzer(self, abs_point):
        """
        将绝对坐标的注视点送入离散度分析器。
        统一以桌面原点为基准做偏移，保证计算基于绝对坐标系。
        """
        if not abs_point or not hasattr(self, "desktop_origin") or not hasattr(self, "desktop_size"):
            return
        normalized_x = abs_point[0] - self.desktop_origin[0]
        normalized_y = abs_point[1] - self.desktop_origin[1]
        self.dispersion_analyzer.add_gaze_point(normalized_x, normalized_y)
    
    def show_menu(self):
        """显示主菜单"""
        # 首先选择交互模式
        single_button, multi_button = self.ui.show_interaction_mode_selection()
        
        # 等待用户操作
        loop = QEventLoop()
        
        result = {'choice': None, 'mode': None}
        
        def on_single_clicked():
            result['choice'] = 'calibrate'
            result['mode'] = 'single'
            loop.quit()
        
        def on_multi_clicked():
            result['choice'] = 'calibrate'
            result['mode'] = 'multi'
            loop.quit()
        
        single_button.clicked.connect(on_single_clicked)
        if multi_button:
            multi_button.clicked.connect(on_multi_clicked)
        
        # 等待用户选择
        loop.exec_()
        
        self.ui.close_current_widget()
        
        # 返回用户的模式和选择
        if result['choice'] == 'calibrate':
            return result['mode'], 'calibrate'
        else:
            return None, 'quit'
    
    def run_calibration(self, interaction_mode, auto_load=False):
        """运行校准
        
        Args:
            interaction_mode: 'single' 或 'multi'
            auto_load: 是否自动加载校准数据（多屏模式）
        """
        print(f"开始{interaction_mode}模式校准...")
        
        # 如果是自动加载模式，直接尝试加载校准数据
        if auto_load and interaction_mode == 'multi':
            print("自动加载模式：尝试加载已存在的校准数据...")
            load_success = self.load_calibration_data(interaction_mode, auto_select=True)
            if load_success:
                print("自动加载校准数据成功！")
                return True
            else:
                print("自动加载失败，将进行新校准...")
                # 自动加载失败，继续执行校准流程
        
        # 根据交互模式显示不同的校准选择界面
        if interaction_mode == 'single':
            single_button, load_button = self.ui.show_single_screen_calibration_choice()
        else:  # multi
            dual_button, load_button = self.ui.show_multi_screen_calibration_choice()
        
        # 等待用户选择
        loop = QEventLoop()
        
        result = {'choice': None}
        
        def on_calibrate_clicked():
            result['choice'] = 'calibrate'
            loop.quit()
        
        def on_load_clicked():
            result['choice'] = 'load'
            loop.quit()
        
        if interaction_mode == 'single':
            single_button.clicked.connect(on_calibrate_clicked)
        else:
            dual_button.clicked.connect(on_calibrate_clicked)
        load_button.clicked.connect(on_load_clicked)
        
        # 等待用户选择
        loop.exec_()  # 阻塞等待事件循环结束
        
        # 检查ESC键是否被按下（在事件循环结束后检查）
        if self.hand_eye_coordinator.check_escape_key():
            result['choice'] = None
        
        choice = result['choice']
        self.ui.close_current_widget()
        
        if choice == 'load':
            # 多屏模式下可以选择自动或手动选择显示器
            auto_select = auto_load  # 如果是自动加载模式，优先使用自动选择
            return self.load_calibration_data(interaction_mode, auto_select=auto_select)
        elif choice == 'calibrate':
            if interaction_mode == 'single':
                return self.perform_single_screen_calibration()
            else:
                return self.perform_dual_screen_calibration()
        else:
            print("用户取消校准")
            return False
    
    def perform_single_screen_calibration(self):
        """执行单屏幕校准"""
        print("执行单屏幕校准...")
        print("校准开始...")
        print("模型预热中，请等待...")
        print("模型预热完成，开始正式校准流程")
        try:
            # 在校准过程中检测ESC键
            while True:
                if self.hand_eye_coordinator.check_escape_key():
                    return False
                
                STransG = self.homtrans.calibrate(self.model, self.cap, sfm=False)
                if STransG is not None:
                    self.calibration_data = STransG
                    return True
                else:
                    return False
        except Exception:
            return False
    
    def perform_dual_screen_calibration(self):
        """执行双屏幕校准"""
        # 执行双屏幕校准
        success = self.screen_manager.perform_dual_screen_calibration(self.model, self.cap)
        if success:
            # 校准成功后，将第一个显示器的校准结果应用到 self.homtrans 实例上
            # 这样在进入交互模式时，self.homtrans 实例就会有正确的校准数据
            if self.screen_manager.calibration_results:
                # 优先选择主显示器 0
                monitor_index = 0
                if monitor_index not in self.screen_manager.calibration_results:
                    # 如果主显示器没有校准数据，选择第一个有校准数据的显示器
                    monitor_index = next(iter(self.screen_manager.calibration_results.keys()))
                
                # 统一使用switch_to_monitor方法，减少重复代码
                success = self.screen_manager.switch_to_monitor(monitor_index, self.homtrans, self.dispersion_analyzer)
                if success:
                    self.calibration_data = self.homtrans.STransG
                    print(f"✓ 双屏校准成功！已将显示器 {monitor_index} 的校准结果应用到当前系统")
                    return True
                else:
                    print(f"✗ 双屏校准成功，但无法应用显示器 {monitor_index} 的校准结果")
                    return False
        return success
    
    def load_calibration_data(self, interaction_mode, auto_select=False):
        """根据交互模式加载对应的校准数据
        
        Args:
            interaction_mode: 'single' 或 'multi'
            auto_select: 是否自动选择显示器（多屏模式）
        """
        print(f"开始加载 {interaction_mode} 模式的校准数据...")
        
        try:
            if interaction_mode == 'single':
                # 单屏模式保护：确保只使用主校准文件，绝不涉及多屏逻辑
                print("单屏模式：加载主校准文件")
                
                calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
                current_monitor_index = 0  # 单屏模式下固定为0
                
                # 确保单屏模式下不会尝试切换显示器或使用多屏校准文件
                self.is_dual_screen_mode = False
                self.screen_manager.current_monitor_index = 0  # 强制设为0
                
                if os.path.exists(calibration_file):
                    print(f"正在加载校准文件: {calibration_file}")
                    
                    # 使用 homtransform 内置的方法来正确加载校准数据
                    if self.homtrans.load_calibration_results(calibration_file, self.sfm_enabled):
                        self.calibration_data = self.homtrans.STransG
                        print("✓ 单屏校准数据加载成功！")

                        return True
                    else:
                        print("✗ 校准数据加载失败，将进行新校准")
                        return False
                else:
                    print(f"单屏校准文件不存在: {calibration_file}")
                    print("将进行新校准")
                    return False
                    
            else:  # dual screen mode
                # 多屏模式：加载所有校准数据，以便注视点在不同屏幕间切换
                print("多屏模式：加载所有显示器校准数据")
                self.is_dual_screen_mode = True
                
                # 加载所有校准结果
                self.screen_manager._load_all_calibration_results()
                
                if self.screen_manager.calibration_results:
                    # 自动选择最佳显示器
                    monitor_index = self.screen_manager.auto_select_best_monitor()
                    if monitor_index is not None:
                        # 统一使用switch_to_monitor方法，避免重复调用
                        success = self.screen_manager.switch_to_monitor(monitor_index, self.homtrans, self.dispersion_analyzer)
                        if success:
                            self.calibration_data = self.homtrans.STransG
                            print(f"✓ 多屏校准数据加载成功！")
                            print(f"✓ 已加载 {len(self.screen_manager.calibration_results)} 个显示器的校准数据")
                            return True
                    print("✗ 无法选择合适的显示器")
                    return False
                else:
                    print("✗ 没有找到任何校准文件")
                    return False
                
        except Exception as e:
            print(f"加载校准数据时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def run_interaction_mode(self):
        """运行交互模式 - 支持多屏幕环境"""
        
        # 根据交互模式设置显示模式
        is_multi_mode = self.is_dual_screen_mode
        
        # 多屏模式下仍允许显示交互覆盖层（用于渐变圆圈）
        if is_multi_mode:
            print("=== 多屏模式：输出注视点坐标，同时支持渐变圆圈展示 ===")
            show_ui = True  # 允许创建交互覆盖层
            output_interval = 1.0  # 每秒输出一次
        else:
            print("=== 单屏模式：输出注视点坐标，同时支持渐变圆圈展示 ===")
            show_ui = True
            output_interval = 1.0  # 单屏模式：每秒输出一次
        
        # 初始化屏幕边界信息
        self.screen_manager._update_screen_boundaries()
        # 注视点分析切换到绝对坐标基准
        self._configure_absolute_dispersion()
        self.alpha = 0.8  # 平滑因子，越大越信任新值
        self.previous_gaze_3d = None  # 保存上一次的gaze 3D数据
        
        # α-β滤波器参数（用于PnP delta_t平滑）
        self.alpha_beta_filter_enabled = True  # 是否启用α-β滤波
        self.alpha_beta_alpha = 0.7  # α参数，控制位置平滑
        self.beta = 0.3  # β参数，控制速度平滑
        self.delta_t_filtered = None  # 滤波后的delta_t
        self.previous_delta_t = None  # 上一帧的delta_t
        current_gaze_point_abs = None  # 绝对坐标注视点

        # 创建定时器用于更新界面（仅UI模式）
        if show_ui:
            timer = QTimer()
            timer.timeout.connect(self.update_interaction_frame)
            timer.start(16)  # 约60FPS
        else:
            timer = None
        
        # 创建多屏模式下的定时输出
        last_output_time = 0.0  # 记录上次输出时间
        
        # 主循环标志
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检测人脸和眼动
            try:
                eye_info ,landmarks, pnp_info = self.model.get_gaze(frame=frame, imshow=False)
            except Exception:
                eye_info = None

            if eye_info is not None:
                gaze = eye_info['gaze']
                if pnp_info['pnp_tvec'] is not None:
                    tvec_curr = pnp_info['pnp_tvec']
               
                # 简化指数平均处理gaze三维数据（保持numpy数组格式）
                if self.previous_gaze_3d is None:
                    self.previous_gaze_3d = gaze.copy()

                else:
                    # 简单指数平均：当前值 = α * 新值 + (1-α) * 旧值（使用numpy保持数组格式）
                    gaze = self.alpha * gaze + (1 - self.alpha) * self.previous_gaze_3d
                    self.previous_gaze_3d = gaze.copy()

                try:
                    if self.homtrans.calibrate_pnp is not None and tvec_curr is not None:
                        delta_t = tvec_curr - self.homtrans.calibrate_pnp
                        if self.alpha_beta_filter_enabled:
                            delta_t_filtered = self._apply_alpha_beta_filter(delta_t)
                            # 输出调试信息（可选）
                            delta_t = delta_t_filtered
                        delta_t=delta_t.flatten()
                        SRW=[[1,0,0],[0,1,0],[0,0,-1]]

                        delta_t=SRW @ delta_t 
                        # 注释掉频繁输出
                        # print(f"delta_t:{delta_t}")
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze, delta_t)
                    else:
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    

                    
                    
                    # 使用main_pygame_dual_screen.py的注视点计算逻辑
                    if FSgaze is not None and len(FSgaze) >= 2:
                        # 将毫米坐标转换为像素坐标
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        gaze_x = screen_pos_px[0]
                        gaze_y = screen_pos_px[1]

                        if self.homtrans.calibrate_pnp is not None and tvec_curr is not None:
                            delta_t = tvec_curr - self.homtrans.calibrate_pnp
                        
                            # 对delta_t进行α-β滤波
                            if self.alpha_beta_filter_enabled:
                                delta_t_filtered = self._apply_alpha_beta_filter(delta_t)
                                # 输出调试信息（可选）
                                delta_t = delta_t_filtered
                        
                            delta_screen_3d = self.homtrans.STransG[:3, :3] @ delta_t
                                # 输出调试信息（可选）
                            delta_mm = delta_screen_3d[:2].flatten()
                            delta_px = self.homtrans._mm2pixel(delta_mm)
                            gaze_x += delta_px[0]
                            gaze_y += delta_px[1]   

                        # gaze_x = max(0, min(gaze_x, self.screen_width))
                        # gaze_y = max(0, min(gaze_y, self.screen_height))
                        # 计算绝对屏幕坐标（考虑显示器位置偏移）
                        # gaze_x_abs = int(screen_pos_px[0] + monitor['x'])
                        # gaze_y_abs = int(screen_pos_px[1] + monitor['y'])
                        
                        # 计算相对当前屏幕的坐标（作为转换基础）
                        gaze_x_rel_original = gaze_x
                        gaze_y_rel_original = gaze_y
                        gaze_point_rel_original = (gaze_x_rel_original, gaze_y_rel_original)
                        
                        # a) 相对 -> 绝对
                        monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                        raw_abs_point = self._convert_rel_to_abs(gaze_point_rel_original, monitor)

                        # b) 绝对坐标卡尔曼滤波
                        filtered_abs_point = self._smooth_absolute_point(raw_abs_point)
                        current_gaze_point_abs = filtered_abs_point
                        self.last_gaze_point_abs = filtered_abs_point

                        # 注视点分析器使用绝对坐标（统一桌面偏移）
                        self._add_absolute_gaze_to_analyzer(filtered_abs_point)

                        # c) 5px 边界判定，触发屏幕切换
                        if (not self.screen_switching and len(self.ui.monitors_info) > 1 and 
                            hasattr(self, 'is_dual_screen_mode') and self.is_dual_screen_mode and
                            self._is_near_screen_edge(filtered_abs_point, monitor, threshold=5)):
                            target_screen = self._resolve_target_screen_from_point(filtered_abs_point, monitor, threshold=5)
                            if target_screen is not None and target_screen != self.screen_manager.current_monitor_index:
                                self.switch_to_monitor_with_coordinate_update(target_screen)
                                monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]

                        # 显示用相对坐标：统一从绝对坐标回算并裁剪到屏幕范围
                        smoothed_rel_point = self._convert_abs_to_rel(filtered_abs_point, monitor)
                        display_rel_point = self._clamp_relative_to_monitor(smoothed_rel_point, monitor)
                        self.fade_circle_x = int(display_rel_point[0] + monitor['x'])
                        self.fade_circle_y = int(display_rel_point[1] + monitor['y'])

                        # 确保视觉元素显示在屏幕内（显示坐标改为裁剪后的绝对值）
                        current_gaze_point_abs = (self.fade_circle_x, self.fade_circle_y)
                        self.last_gaze_point_abs = current_gaze_point_abs

                        # 手眼协调使用平滑且裁剪后的相对坐标
                        self.hand_eye_coordinator._process_hand_eye_coordination(display_rel_point)
                        
                except Exception as e:
                    print(f"注视点处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 设置默认注视点，确保界面正常显示
                    monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                    current_gaze_point_abs = (monitor['x'] + monitor['width'] // 2, monitor['y'] + monitor['height'] // 2)
            
            # 更新交互界面（用于渐变圆圈）
            if show_ui:
                # 输出坐标信息
                current_time = time.time()
                if current_time - last_output_time >= output_interval:
                    if current_gaze_point_abs:
                        # 获取当前显示器信息
                        current_monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                        
                        # 计算相对于当前屏幕的坐标
                        rel_x = current_gaze_point_abs[0] - current_monitor['x']
                        rel_y = current_gaze_point_abs[1] - current_monitor['y']
                        
                        # 确保坐标在屏幕边界内
                        rel_x = max(0, min(rel_x, current_monitor['width']))
                        rel_y = max(0, min(rel_y, current_monitor['height']))
                        
                        # 输出当前注视点位置
                        if is_multi_mode:
                            print(f"[多屏模式] 当前注视点 - 屏幕{self.screen_manager.current_monitor_index}: 绝对坐标({current_gaze_point_abs[0]:.0f}, {current_gaze_point_abs[1]:.0f}), 相对坐标({rel_x:.0f}, {rel_y:.0f})")
                        else:
                            print(f"[单屏模式] 当前注视点 - 屏幕{self.screen_manager.current_monitor_index}: 绝对坐标({current_gaze_point_abs[0]:.0f}, {current_gaze_point_abs[1]:.0f}), 相对坐标({rel_x:.0f}, {rel_y:.0f})")
                    else:
                        if is_multi_mode:
                            print(f"[多屏模式] 当前注视点 - 屏幕{self.screen_manager.current_monitor_index}: 正在检测中...")
                        else:
                            print(f"[单屏模式] 当前注视点 - 屏幕{self.screen_manager.current_monitor_index}: 正在检测中...")
                    last_output_time = current_time

                # 对于当前屏幕，使用相对坐标；对于其他屏幕，使用绝对坐标
                if self.hand_eye_coordinator.current_interaction_zone or self.hand_eye_coordinator.previous_interaction_zone or current_gaze_point_abs:
                    # 显示交互界面
                    self.ui.show_interaction_screen(
                        interaction_zone=self.hand_eye_coordinator.current_interaction_zone,
                        current_gaze_point=current_gaze_point_abs,  # 使用绝对坐标，以便在所有屏幕上显示
                        show_gaze_point=self.show_gaze_point  # 传递注视点显示标志位
                    )
                self.hand_eye_coordinator.previous_interaction_zone = self.hand_eye_coordinator.current_interaction_zone

                # 处理Qt事件
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
            else:
                # 多屏模式：使用main_pygame_dual_screen.py的注视点计算逻辑
                current_time = time.time()
                if current_time - last_output_time >= output_interval:
                    if current_gaze_point_abs:
                        # 重新计算当前注视点实际所在的屏幕
                        abs_x, abs_y = current_gaze_point_abs
                        actual_screen_index = self.screen_manager.get_target_screen(abs_x, abs_y)
                        
                        # 使用实际屏幕索引获取显示器信息
                        actual_monitor = self.ui.monitors_info[actual_screen_index]
                        
                        # 计算相对实际屏幕的坐标
                        rel_x = abs_x - actual_monitor['x']
                        rel_y = abs_y - actual_monitor['y']
                        
                        # 确保坐标在屏幕边界内
                        rel_x = max(0, min(rel_x, actual_monitor['width']))
                        rel_y = max(0, min(rel_y, actual_monitor['height']))
                        
                        # 输出当前注视点位置
                        print(f"[多屏模式] 当前注视点 - 屏幕{actual_screen_index}: 绝对坐标({abs_x:.0f}, {abs_y:.0f}), 相对坐标({rel_x:.0f}, {rel_y:.0f})")
                    else:
                        print(f"[多屏模式] 当前注视点 - 屏幕{self.screen_manager.current_monitor_index}: 正在检测中...")
                    last_output_time = current_time
            
            # 检测ESC键退出
            if self.hand_eye_coordinator.check_escape_key():
                self.running = False
                break
            
            # 检测空格键切换注视点显示
            if self.hand_eye_coordinator.check_space_key():
                self.show_gaze_point = not self.show_gaze_point
                # 延迟一小段时间，避免连续触发
                time.sleep(0.2)
            
            
        # 清理资源
        if timer:
            timer.stop()
        if show_ui:
            self.ui.close_current_widget()
        
        if is_multi_mode:
            print("=== 多屏模式交互结束 ===")
        else:
            print("=== 单屏模式交互结束 ===")
    
    def update_interaction_frame(self):
        """更新交互帧（由定时器调用）"""
        pass  # 界面更新在run_interaction_mode中处理

    def _apply_alpha_beta_filter(self, current_delta_t):
        """对一阶α-β滤波器应用于delta_t进行平滑"""
        if self.previous_delta_t is None:
            # 第一次初始化
            self.delta_t_filtered = current_delta_t.copy()
            self.previous_delta_t = current_delta_t.copy()
            return current_delta_t
        
        # 计算速度（当前测量值 - 上一次测量值）
        velocity = current_delta_t - self.previous_delta_t
        
        # α-β滤波预测
        if self.delta_t_filtered is None:
            predicted_delta_t = current_delta_t
        else:
            predicted_delta_t = self.delta_t_filtered + velocity
        
        # 更新（校正步骤）
        measurement_residual = current_delta_t - predicted_delta_t
        self.delta_t_filtered = predicted_delta_t + self.alpha_beta_alpha * measurement_residual
        
        # 更新速度估计
        velocity_estimate = velocity + self.beta * measurement_residual
        
        # 保存当前状态用于下一帧
        self.previous_delta_t = current_delta_t.copy()
        
        return self.delta_t_filtered
