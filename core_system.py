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

class EyeHandInteractionSystem:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.ui = None
        self.model = None
        self.homtrans = None
        self.cap = None
        self.calibration_data = None
        
        # 双屏支持相关
        self.calibration_mode = "single"  # 校准模式：single 或 dual
        self.is_dual_screen_mode = False  # 标志位，指示是否为双屏模式
        
        # 上一帧的绝对坐标注视点，用于平滑过渡
        self.last_gaze_point_abs = None
        
        # 统一的阈值配置管理器
        self.threshold_config = {
            'distance_multiplier': 1.0,      # 距离阈值倍数，用于跨屏调整
            'auto_move_distance': 400,       # 鼠标右键触发移动的距离阈值（像素）
            'auto_move_cooldown': 1000,      # 自动移动冷却时间（毫秒）
            'gaze_dispersion_frames': 6,     # 注视点分析使用的帧数
            'angle_threshold': 3.0,          # 角度离散度阈值（度） 
            'pixel_threshold': 100,          # 像素离散度阈值（像素）
            'boundary_buffer': 20,           # 屏幕边界缓冲区（像素）
            'interaction_zone_duration': 500,  # 交互区域显示持续时间（毫秒）
            # 鼠标移动触发相关配置
            'mouse_movement_threshold': 300,  # 鼠标移动触发阈值（像素）
            'teleport_circle_radius': 100,     # 传送圆周半径（像素）
            'teleport_cooldown_duration': 1000,  # 传送冷却时间（毫秒）
            'sliding_window_time_limit': 350,   # 滑动窗口时间限制（毫秒）
            'mad_threshold': 50,               # 稳定凝视MAD阈值（像素）
            'perc95_threshold': 120,           # 稳定凝视95%百分位阈值（像素）
            'sliding_window_angle_threshold': 3.0  # 滑动窗口角度阈值
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
        from gaze_tracking.model import EyeModel
        self.model = EyeModel(self.project_dir)
        
        # 初始化HomTransform（基础实例，稍后应用校准数据）
        from gaze_tracking.homtransform import HomTransform
        self.homtrans = HomTransform(self.project_dir)
        
        # 初始化屏幕管理器
        self.screen_manager = ScreenManager(self.project_dir, self.ui)
        
        # 初始化手眼协调器
        self.hand_eye_coordinator = HandEyeCoordinator(self.ui, self.screen_manager, self.threshold_config)
        
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        return True
    
    def switch_to_monitor_with_coordinate_update(self, target_screen):
        """切换到目标屏幕并更新坐标"""
        return self.screen_manager.switch_to_monitor_with_coordinate_update(target_screen, self.homtrans, self.dispersion_analyzer, self.kalman_filter)
    
    def _smooth_gaze_point(self, raw_gaze_point):
        """使用卡尔曼滤波平滑注视点"""
        return self.kalman_filter.update(raw_gaze_point)
    
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
            interaction_mode: 'single' 或 'dual'
            auto_load: 是否自动加载校准数据（多屏模式）
        """
        print(f"开始{interaction_mode}模式校准...")
        
        # 如果是自动加载模式，直接尝试加载校准数据
        if auto_load and interaction_mode == 'dual':
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
                
                STransG = self.homtrans.calibrate(self.model, self.cap, sfm=True)
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
                
                # 切换到目标显示器，应用校准数据
                if self.screen_manager.switch_to_monitor(monitor_index, self.homtrans, self.dispersion_analyzer):
                    self.calibration_data = self.homtrans.STransG
                    print(f"✓ 双屏校准成功！已将显示器 {monitor_index} 的校准结果应用到当前系统")
                else:
                    print(f"✗ 双屏校准成功，但无法应用显示器 {monitor_index} 的校准结果")
        return success
    
    def load_calibration_data(self, interaction_mode, auto_select=False):
        """根据交互模式加载对应的校准数据
        
        Args:
            interaction_mode: 'single' 或 'dual'
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
                    if self.homtrans.load_calibration_results(calibration_file):
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
                        # 切换到目标显示器
                        if self.screen_manager.switch_to_monitor(monitor_index, self.homtrans, self.dispersion_analyzer):
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
    
    def show_monitor_selection(self):
        """显示显示器选择界面（用于多屏模式加载校准数据）"""
        # 如果校准结果为空，先尝试扫描可用的校准文件
        if not self.screen_manager.calibration_results:
            print("正在扫描可用的校准文件...")
            self.screen_manager._load_all_calibration_results()
            
        if not self.screen_manager.calibration_results:
            # 如果仍然没有校准结果，检查文件系统
            results_dir = os.path.join(self.project_dir, "results")
            available_files = []
            
            for i in range(len(self.ui.monitors_info)):
                calibration_file = os.path.join(results_dir, f"calibration_results_screen_{i}.json")
                if os.path.exists(calibration_file):
                    available_files.append((i, calibration_file))
            
            if not available_files:
                QMessageBox.information(None, "提示", "没有找到任何校准文件")
                return None
            
            # 创建显示器选择界面
            dialog = QDialog()
            dialog.setWindowTitle("选择显示器")
            dialog.setModal(True)
            dialog.resize(400, 300)
            
            layout = QVBoxLayout(dialog)
            
            # 添加说明文字
            label = QLabel("请选择要加载校准数据的显示器:")
            layout.addWidget(label)
            
            # 为每个有校准文件的显示器创建按钮
            for monitor_index, calibration_file in available_files:
                try:
                    with open(calibration_file, 'r', encoding='utf-8') as f:
                        calibration_data = json.load(f)
                    
                    # 获取显示器信息
                    monitor = self.ui.monitors_info[monitor_index]
                    button_text = f"显示器 {monitor_index}: {monitor['width']}x{monitor['height']}"
                    button = QPushButton(button_text)
                    
                    def make_callback(idx):
                        def callback():
                            dialog.accept()
                            dialog.selected_monitor = idx
                        return callback
                    
                    button.clicked.connect(make_callback(monitor_index))
                    layout.addWidget(button)
                except Exception as e:
                    print(f"读取显示器 {monitor_index} 校准文件失败: {e}")
            
            # 添加取消按钮
            cancel_button = QPushButton("取消")
            cancel_button.clicked.connect(dialog.reject)
            layout.addWidget(cancel_button)
            
            # 显示对话框
            result = dialog.exec_()
            if result == QDialog.Accepted and hasattr(dialog, 'selected_monitor'):
                return dialog.selected_monitor
            else:
                return None
        else:
            # 创建显示器选择界面
            dialog = QDialog()
            dialog.setWindowTitle("选择显示器")
            dialog.setModal(True)
            dialog.resize(400, 300)
            
            layout = QVBoxLayout(dialog)
            
            # 添加说明文字
            label = QLabel("请选择要加载校准数据的显示器:")
            layout.addWidget(label)
            
            # 为每个有校准数据的显示器创建按钮
            buttons = []
            for monitor_index in sorted(self.screen_manager.calibration_results.keys()):
                monitor_data = self.screen_manager.calibration_results[monitor_index]
                button_text = f"显示器 {monitor_index}: {monitor_data['width']}x{monitor_data['height']}"
                button = QPushButton(button_text)
                
                def make_callback(idx):
                    def callback():
                        dialog.accept()
                        dialog.selected_monitor = idx
                    return callback
                
                button.clicked.connect(make_callback(monitor_index))
                layout.addWidget(button)
                buttons.append(button)
            
            # 添加取消按钮
            cancel_button = QPushButton("取消")
            cancel_button.clicked.connect(dialog.reject)
            layout.addWidget(cancel_button)
            
            # 显示对话框
            result = dialog.exec_()
            if result == QDialog.Accepted and hasattr(dialog, 'selected_monitor'):
                return dialog.selected_monitor
            else:
                return None
    
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
        
        # 用于SfM的前一帧
        frame_prev = None
        current_gaze_point_rel = None  # 当前屏幕相对坐标注视点
        current_gaze_point_abs = None  # 绝对坐标注视点
        previous_gaze_point = None  # 用于快速移动检测
        
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
                face_boxes = self.model.face_detection.predict(frame)
                if not face_boxes:  # 如果没有检测到人脸
                    eye_info = None
                else:
                    # 获取眼动信息
                    try:
                        eye_info = self.model.get_gaze(frame=frame, face_boxes=face_boxes, imshow=False)
                    except Exception:
                        eye_info = None
            except Exception:
                face_boxes = None
                eye_info = None
            
            if eye_info is not None:
                gaze = eye_info['gaze']
                
                # 使用SfM进行视线映射
                try:
                    if frame_prev is not None:
                        face_features_curr = self.model.get_FaceFeatures(frame, face_boxes=face_boxes)
                        cached_prev_features = self.homtrans.sfm.get_cached_face_features('curr')
                        if cached_prev_features is not None:
                            face_features_prev = cached_prev_features
                        else:
                            face_features_prev = self.model.get_FaceFeatures(frame_prev, face_boxes=face_boxes)
                        
                        WTransG1, WTransG2, W_P = self.homtrans.sfm.get_GazeToWorld(
                            self.model, frame_prev, frame, 
                            face_features_prev=face_features_prev, 
                            face_features_curr=face_features_curr
                        )
                        
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG1)
                        self.homtrans.sfm.update_caches(
                            frame_prev_features=face_features_prev,
                            frame_curr_features=face_features_curr
                        )
                    else:
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 使用main_pygame_dual_screen.py的注视点计算逻辑
                    if FSgaze is not None and len(FSgaze) >= 2:
                        # 将毫米坐标转换为像素坐标
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        
                        # 计算绝对屏幕坐标（考虑显示器位置偏移）
                        monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                        gaze_x_abs = int(screen_pos_px[0] + monitor['x'])
                        gaze_y_abs = int(screen_pos_px[1] + monitor['y'])
                        
                        # 计算相对当前屏幕的坐标
                        gaze_x_rel = max(0, min(screen_pos_px[0], self.ui.screen_width))
                        gaze_y_rel = max(0, min(screen_pos_px[1], self.ui.screen_height))
                    else:
                        # 默认使用屏幕中心
                        monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                        gaze_x_abs = monitor['x'] + monitor['width'] // 2
                        gaze_y_abs = monitor['y'] + monitor['height'] // 2
                        gaze_x_rel = self.ui.screen_width // 2
                        gaze_y_rel = self.ui.screen_height // 2

                    # 应用卡尔曼滤波平滑算法 - 使用相对坐标进行平滑
                    raw_gaze_point = (gaze_x_rel, gaze_y_rel)

                    # 使用轻量卡尔曼滤波算法
                    if self.smoothing_enabled:
                        # 屏幕切换后，短期内禁用平滑以快速适应新坐标
                        if hasattr(self.screen_manager, 'screen_switching_adaptation_frames') and self.screen_manager.screen_switching_adaptation_frames > 0:
                            # 直接使用原始注视点，快速适应新屏幕
                            gaze_point_rel = raw_gaze_point
                            # 减少适应帧数
                            self.screen_manager.screen_switching_adaptation_frames -= 1
                        else:
                            # 正常使用卡尔曼滤波平滑
                            gaze_point_rel = self._smooth_gaze_point(raw_gaze_point)
                    else:
                        gaze_point_rel = raw_gaze_point

                    current_gaze_point_rel = gaze_point_rel

                    # 使用平滑后的相对坐标重新计算绝对坐标
                    monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                    smoothed_gaze_x_abs = int(gaze_point_rel[0] + monitor['x'])
                    smoothed_gaze_y_abs = int(gaze_point_rel[1] + monitor['y'])
                    current_gaze_point_abs = (smoothed_gaze_x_abs, smoothed_gaze_y_abs)
                    self.last_gaze_point_abs = current_gaze_point_abs

                    # 实时监测注视点位置，实现屏幕自动切换（仅在多屏模式下启用）
                    if (not self.screen_manager.screen_switching and len(self.ui.monitors_info) > 1 and self.is_dual_screen_mode):
                        # 检查注视点是否超出当前屏幕边界
                        if self.screen_manager.is_gaze_out_of_screen(smoothed_gaze_x_abs, smoothed_gaze_y_abs, self.screen_manager.current_monitor_index):
                            # 获取目标屏幕索引
                            target_screen = self.screen_manager.get_target_screen(smoothed_gaze_x_abs, smoothed_gaze_y_abs)

                            # 切换到目标屏幕
                            if target_screen != self.screen_manager.current_monitor_index:
                                self.switch_to_monitor_with_coordinate_update(target_screen)

                                # 重新计算相对坐标
                                gaze_x_rel, gaze_y_rel = self.screen_manager.convert_abs_to_rel_coordinate(
                                    smoothed_gaze_x_abs, smoothed_gaze_y_abs, target_screen)
                    
                    # 手眼协调机制处理（使用相对坐标）
                    self.hand_eye_coordinator._process_hand_eye_coordination(gaze_point_rel, previous_gaze_point, self.is_dual_screen_mode)
                    
                    # 更新前一注视点
                    previous_gaze_point = current_gaze_point_rel
                    
                    # 添加到注视点分析器（使用相对坐标）
                    self.dispersion_analyzer.add_gaze_point(gaze_point_rel[0], gaze_point_rel[1])
                    
                    # 检查触发条件
                    triggered, center_point = self.dispersion_analyzer.check_trigger_conditions()
                        
                except Exception as e:
                    print(f"注视点处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 设置默认注视点，确保界面正常显示
                    monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
                    current_gaze_point_abs = (monitor['x'] + monitor['width'] // 2, monitor['y'] + monitor['height'] // 2)
            
            # 获取离散度信息
            dispersion_info = self.dispersion_analyzer.calculate_dispersion()
            
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
            
            # 更新前一帧
            frame_prev = frame.copy()
            
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