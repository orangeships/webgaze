#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼手交互系统屏幕管理模块

该模块负责多屏环境下的屏幕管理、校准数据处理和屏幕切换功能。

主要功能：
1. 多屏检测和信息获取
2. 校准数据的加载、保存和管理
3. 屏幕切换和坐标系统更新
4. 绝对坐标与相对坐标的转换
5. 跨屏距离计算
6. 屏幕边界检测和目标屏幕判断

核心类：
- ScreenManager: 屏幕管理类，负责所有屏幕相关功能

关键功能点：
1. 自动加载所有屏幕的校准数据
2. 智能选择最佳显示器进行校准
3. 屏幕切换时的坐标系统更新
4. 跨屏注视点检测和屏幕自动切换
5. 单屏/多屏模式的自动适应

使用场景：
1. 多屏环境下的屏幕自动切换
2. 校准数据的管理和应用
3. 跨屏坐标转换和距离计算
4. 多屏校准流程控制

"""
import os
import json
import cv2

class ScreenManager:
    """屏幕管理类，负责多屏检测、切换和校准数据管理"""
    
    def __init__(self, project_dir, ui):
        self.project_dir = project_dir
        self.ui = ui
        self.calibration_results = {}  # 存储所有显示器的校准结果
        self.current_monitor_index = 0
        self.screen_boundaries = []  # 存储每个屏幕的边界信息 [left, right, top, bottom]
        self.screen_switching = False  # 标志位，防止屏幕切换过程中重复触发
        self.screen_switching_adaptation_frames = 0  # 屏幕切换后的卡尔曼滤波适应帧数
        
        # 初始化屏幕边界信息
        self._update_screen_boundaries()
    
    def _update_screen_boundaries(self):
        """更新屏幕边界信息"""
        self.screen_boundaries = []
        for monitor in self.ui.monitors_info:
            left = monitor['x']
            right = monitor['x'] + monitor['width']
            top = monitor['y']
            bottom = monitor['y'] + monitor['height']
            self.screen_boundaries.append([left, right, top, bottom])
    
    def _load_all_calibration_results(self):
        """加载所有显示器的校准结果"""
        results_dir = os.path.join(self.project_dir, "results")
        
        # 初始化校准结果存储
        self.calibration_results = {}
        
        for monitor_index in range(len(self.ui.monitors_info)):
            calibration_file = os.path.join(results_dir, f"calibration_results_screen_{monitor_index}.json")
            if os.path.exists(calibration_file):
                print(f"正在加载显示器 {monitor_index} 的校准结果: {calibration_file}")
                try:
                    # 创建当前显示器的HomTransform实例
                    monitor = self.ui.monitors_info[monitor_index]
                    from gaze_tracking.homtransform import HomTransform
                    current_homtrans = HomTransform(self.project_dir, custom_width=monitor['width'], custom_height=monitor['height'])
                    
                    # 使用HomTransform内置的加载方法加载校准结果
                    if current_homtrans.load_calibration_results(calibration_file):
                        self.calibration_results[monitor_index] = {
                            'STransG': current_homtrans.STransG,
                            'homtrans': current_homtrans,
                            'width': monitor['width'],
                            'height': monitor['height'],
                            'x': monitor['x'],
                            'y': monitor['y']
                        }
                        print(f"显示器 {monitor_index} 校准结果加载成功")
                    else:
                        print(f"显示器 {monitor_index} 校准结果加载失败")
                except Exception as e:
                    print(f"加载显示器 {monitor_index} 校准结果失败: {e}")
            else:
                print(f"未找到显示器 {monitor_index} 的校准文件")
    
    def save_calibration_results(self, monitor_index):
        """保存特定显示器的校准结果"""
        try:
            # 获取校准结果
            result = self.calibration_results[monitor_index]
            homtrans = result['homtrans']
            
            # 直接保存校准结果到指定的文件名
            calibration_filename = f"calibration_results_screen_{monitor_index}.json"
            
            homtrans._save_calibration_results(
                result['STransG'],
                homtrans.df if hasattr(homtrans, 'df') and homtrans.df is not None else [],
                homtrans.SetVal if hasattr(homtrans, 'SetVal') else None,
                homtrans.gaze if hasattr(homtrans, 'gaze') else None,
                sfm=True,
                STransW=homtrans.STransW if hasattr(homtrans, 'STransW') else None,
                scaleWtG=homtrans.scaleWtG if hasattr(homtrans, 'scaleWtG') else None,
                filename=calibration_filename
            )
            
            print(f"显示器 {monitor_index} 的校准结果已保存到: calibration_results_screen_{monitor_index}.json")
            return True
            
        except Exception as e:
            print(f"保存显示器 {monitor_index} 校准结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_current_monitor_calibration(self, homtrans):
        """应用当前显示器的校准数据"""
        current_monitor_index = self.ui.current_monitor_index
        target_monitor = self.ui.monitors_info[current_monitor_index]
        
        # 先尝试从已有的校准结果中应用
        if (current_monitor_index in self.calibration_results and 
            self.calibration_results[current_monitor_index] is not None):
            calibration = self.calibration_results[current_monitor_index]
            
            # 正确访问校准数据结构，从保存的homtrans实例复制属性
            if hasattr(homtrans, 'STransG') and 'STransG' in calibration:
                homtrans.STransG = calibration['STransG']
            
            # 从保存的HomTransform实例复制其他属性
            source_homtrans = calibration['homtrans']
            
            # 设置其他必要属性
            if hasattr(source_homtrans, 'scaleWtG'):
                homtrans.scaleWtG = source_homtrans.scaleWtG
            if hasattr(source_homtrans, 'StG'):
                homtrans.StG = source_homtrans.StG
            if hasattr(source_homtrans, 'StW'):
                homtrans.StW = source_homtrans.StW
            if hasattr(source_homtrans, 'STransW'):
                homtrans.STransW = source_homtrans.STransW
            if hasattr(source_homtrans, 'scaleWtG2'):
                homtrans.scaleWtG2 = source_homtrans.scaleWtG2
            if hasattr(source_homtrans, 'df') and source_homtrans.df is not None:
                homtrans.df = source_homtrans.df
            if hasattr(source_homtrans, 'SetVal'):
                homtrans.SetVal = source_homtrans.SetVal
            if hasattr(source_homtrans, 'gaze'):
                homtrans.gaze = source_homtrans.gaze
            
            print(f"已应用显示器 {current_monitor_index+1} 的校准数据")
            return homtrans.STransG
        else:
            # 如果没有校准结果，尝试从文件加载
            calibration_file = os.path.join(self.project_dir, "results", 
                                          f"calibration_results_screen_{current_monitor_index}.json")
            
            if os.path.exists(calibration_file):
                try:
                    # 使用HomTransform内置的加载方法加载校准结果
                    if homtrans.load_calibration_results(calibration_file):
                        print(f"已从文件应用显示器 {current_monitor_index+1} 的校准数据")
                        return homtrans.STransG
                    else:
                        print(f"加载显示器 {current_monitor_index+1} 校准数据失败: 内置加载方法返回False")
                        self.calibration_results[current_monitor_index] = None
                except Exception as e:
                    print(f"加载显示器 {current_monitor_index+1} 校准数据失败: {e}")
                    self.calibration_results[current_monitor_index] = None
            else:
                print(f"显示器 {current_monitor_index+1} 没有可用的校准文件")
        
        return None
    
    def switch_to_monitor(self, monitor_index, homtrans, dispersion_analyzer):
        """切换到指定显示器并加载对应的校准数据"""
        if self.ui.switch_to_monitor(monitor_index):
            self.current_monitor_index = monitor_index
            
            # 更新屏幕尺寸用于角度计算
            dispersion_analyzer.set_screen_dimensions(self.ui.screen_width, self.ui.screen_height)
            
            # 加载目标显示器的校准数据
            calibration_data = self._apply_current_monitor_calibration(homtrans)
            
            print(f"已切换到显示器 {monitor_index+1}")
            return True
        return False
    
    def switch_to_monitor_with_coordinate_update(self, target_screen, homtrans, dispersion_analyzer, kalman_filter=None):
        """切换到目标屏幕并更新坐标系统 - 移植正确实现（单屏模式保护）
        
        Args:
            target_screen: 目标屏幕索引
            homtrans: 当前的HomTransform实例
            dispersion_analyzer: 离散度分析器实例
            kalman_filter: 卡尔曼滤波器实例，可选
            
        Returns:
            bool: 切换成功返回True，否则返回False
        """
        # 单屏模式保护：绝不允许切换屏幕
        if len(self.ui.monitors_info) <= 1:
            if hasattr(self, 'current_monitor_index'):
                # 确保单屏模式下始终使用屏幕0
                self.current_monitor_index = 0
            return False
            
        if target_screen == self.current_monitor_index or target_screen >= len(self.ui.monitors_info):
            return False
            
        try:
            self.screen_switching = True
            
            # 保存当前状态
            last_gaze_point = None  # 可以考虑从外部传入上一帧注视点
            
            # 关键修复：使用目标显示器的物理位置来设置窗口位置
            target_monitor = self.ui.monitors_info[target_screen]
            window_pos_x = target_monitor['x']
            window_pos_y = target_monitor['y']
            
            # 重新初始化UI以适应当前显示器，同时确保窗口在正确位置
            self.ui.initialize_display_at_position(target_monitor['x'], target_monitor['y'])
            
            # 关键修复：创建新的HomTransform实例以匹配目标显示器的分辨率
            # 这与校准阶段的行为一致，确保视线计算使用正确的坐标系统
            from gaze_tracking.homtransform import HomTransform
            new_homtrans = HomTransform(self.project_dir, custom_width=target_monitor['width'], custom_height=target_monitor['height'])
            
            # 加载并应用对应屏幕的校准结果到新的HomTransform实例
            if target_screen in self.calibration_results:
                calibration = self.calibration_results[target_screen]
                
                # 正确访问校准数据结构
                new_homtrans.STransG = calibration['STransG']
                
                # 从 HomTransform 实例复制其他属性
                source_homtrans = calibration['homtrans']
                
                # 设置其他必要属性
                if hasattr(source_homtrans, 'scaleWtG'):
                    new_homtrans.scaleWtG = source_homtrans.scaleWtG
                if hasattr(source_homtrans, 'StG'):
                    new_homtrans.StG = source_homtrans.StG
                if hasattr(source_homtrans, 'StW'):
                    new_homtrans.StW = source_homtrans.StW
                if hasattr(source_homtrans, 'STransW'):
                    new_homtrans.STransW = source_homtrans.STransW
                if hasattr(source_homtrans, 'scaleWtG2'):
                    new_homtrans.scaleWtG2 = source_homtrans.scaleWtG2
                if hasattr(source_homtrans, 'df') and source_homtrans.df is not None:
                    new_homtrans.df = source_homtrans.df
                if hasattr(source_homtrans, 'SetVal'):
                    new_homtrans.SetVal = source_homtrans.SetVal
                if hasattr(source_homtrans, 'gaze'):
                    new_homtrans.gaze = source_homtrans.gaze
                
            else:
                # 如果没有校准结果，尝试从文件加载
                calibration_file = os.path.join(self.project_dir, "results", f"calibration_results_screen_{target_screen}.json")
                if os.path.exists(calibration_file):
                    new_homtrans.load_calibration_results(calibration_file)
            
            # 关键修复：同步更新HomTransform的物理尺寸参数，确保_mm2pixel使用正确的屏幕尺寸
            new_homtrans.width = target_monitor['width']
            new_homtrans.height = target_monitor['height']
            
            # 切换UI
            if self.ui.switch_to_monitor(target_screen):
                # 更新当前屏幕索引
                self.current_monitor_index = target_screen
                
                # 更新屏幕边界信息
                self._update_screen_boundaries()
                
                # 更新屏幕尺寸用于离散度分析
                dispersion_analyzer.set_screen_dimensions(target_monitor['width'], target_monitor['height'])
                
                # 关键修复：重置卡尔曼滤波器状态，避免跨屏幕坐标混淆
                if kalman_filter is not None:
                    # 重新初始化卡尔曼滤波器，清除旧状态
                    from gaze_analysis import LightweightKalmanFilter
                    kalman_filter = LightweightKalmanFilter(
                        process_noise=0.8,  # 增加过程噪声，加速适应新屏幕
                        measurement_noise=0.4,  # 增加测量噪声，减少拖拽效应
                        error_estimate=2.0  # 增加初始误差估计
                    )
                    # 设置屏幕切换标志，短期内禁用平滑以快速适应新坐标
                    self.screen_switching_adaptation_frames = 5  # 屏幕切换后5帧内快速适应
                    print(f"[DEBUG] 屏幕切换到{target_screen}，卡尔曼滤波器已重置，将快速适应新坐标")
                
                # 关键修复：将新的HomTransform实例属性复制到传入的实例中
                # 这样可以确保外部使用的实例包含正确的校准数据
                homtrans.__dict__.update(new_homtrans.__dict__)
                
                self.screen_switching = False
                print(f"已切换到显示器 {target_screen+1}")
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
    
    def is_gaze_out_of_screen(self, gaze_x, gaze_y, monitor_index):
        """检查注视点是否超出当前屏幕边界"""
        if monitor_index < 0 or monitor_index >= len(self.screen_boundaries):
            return False
        
        left, right, top, bottom = self.screen_boundaries[monitor_index]
        boundary_buffer = 20  # 屏幕边界缓冲区（像素）
        
        # 检查是否超出边界（带缓冲区）
        if (gaze_x < left + boundary_buffer or gaze_x > right - boundary_buffer or
            gaze_y < top + boundary_buffer or gaze_y > bottom - boundary_buffer):
            return True
        return False
    
    def get_target_screen(self, gaze_x, gaze_y):
        """获取注视点所在的目标屏幕索引"""
        for i, (left, right, top, bottom) in enumerate(self.screen_boundaries):
            if left <= gaze_x <= right and top <= gaze_y <= bottom:
                return i
        return self.current_monitor_index  # 默认返回当前屏幕
    
    def convert_abs_to_rel_coordinate(self, abs_x, abs_y, target_screen):
        """将绝对坐标转换为目标屏幕的相对坐标"""
        if target_screen < 0 or target_screen >= len(self.ui.monitors_info):
            return abs_x, abs_y
        
        monitor = self.ui.monitors_info[target_screen]
        rel_x = abs_x - monitor['x']
        rel_y = abs_y - monitor['y']
        return rel_x, rel_y
    
    def convert_rel_to_abs_coordinate(self, rel_x, rel_y, source_screen):
        """将相对坐标转换为绝对坐标"""
        if source_screen < 0 or source_screen >= len(self.ui.monitors_info):
            return rel_x, rel_y
        
        monitor = self.ui.monitors_info[source_screen]
        abs_x = rel_x + monitor['x']
        abs_y = rel_y + monitor['y']
        return abs_x, abs_y
    
    def auto_select_best_monitor(self):
        """自动选择最佳显示器进行校准数据加载
        
        优先级：
        1. 主显示器 (monitor 0)
        2. 第一个可用的有校准文件的显示器
        
        Returns:
            int: 选择的显示器索引，如果没有找到可用的则返回 None
        """
        print("正在自动选择最佳显示器...")
        
        # 首先尝试加载所有校准结果
        if not self.calibration_results:
            print("正在扫描可用的校准文件...")
            self._load_all_calibration_results()
        
        # 策略1：优先选择主显示器0
        if 0 in self.calibration_results:
            print("自动选择主显示器 (monitor 0)")
            return 0
        
        # 策略2：选择第一个可用的显示器
        results_dir = os.path.join(self.project_dir, "results")
        available_monitors = []
        
        # 检查文件系统中的校准文件
        for i in range(len(self.ui.monitors_info)):
            calibration_file = os.path.join(results_dir, f"calibration_results_screen_{i}.json")
            if os.path.exists(calibration_file):
                available_monitors.append(i)
        
        if available_monitors:
            selected_monitor = available_monitors[0]
            print(f"自动选择第一个可用显示器 (monitor {selected_monitor})")
            return selected_monitor
        
        print("没有找到任何可用的校准数据")
        return None
    
    def perform_dual_screen_calibration(self, model, cap):
        """执行双屏幕校准"""
        print("执行双屏幕校准...")
        
        if len(self.ui.monitors_info) < 2:
            print("警告：只检测到1个显示器，无法进行双屏幕校准")
            return False
        
        try:
            # 关键修复：为每个显示器创建独立的校准结果存储
            self.calibration_results = {}
            monitors = self.ui.monitors_info
            
            for i, monitor in enumerate(monitors):
                print(f"\n开始校准显示器 {i}: {monitor['width']}x{monitor['height']} (位置: {monitor['x']}, {monitor['y']})")
                
                # 关键修复：为当前显示器创建独立的PygameUI和HomTransform实例
                # 参考main_pygame_dual_screen.py的正确实现
                
                # 创建针对当前显示器的PygameUI实例 - 从正确位置导入
                from src.main_pygame_dual_screen import PygameUI
                current_ui = PygameUI(width=monitor['width'], height=monitor['height'], display_index=i)
                current_ui.initialize_display()
                
                # 创建针对当前显示器的HomTransform实例，传入当前显示器的分辨率
                from gaze_tracking.homtransform import HomTransform
                current_homtrans = HomTransform(self.project_dir, custom_width=monitor['width'], custom_height=monitor['height'])
                
                print(f"模型预热中，请等待...")
                
                # 生成针对当前显示器的校准文件名
                calibration_filename = f"calibration_results_screen_{i}.json"
                
                # 校准当前显示器
                STransG = current_homtrans.calibrate(model, cap, sfm=True, filename=calibration_filename)
                if STransG is not None:
                    # 关键修复：存储每个显示器的独立校准结果和对应实例
                    self.calibration_results[i] = {
                        'STransG': STransG,
                        'homtrans': current_homtrans,
                        'ui': current_ui,
                        'width': monitor['width'],
                        'height': monitor['height'],
                        'x': monitor['x'],
                        'y': monitor['y']
                    }
                    
                    # 校准结果已在校准过程中自动保存到 calibration_results_screen_{i}.json
                    print(f"显示器 {i} 校准成功")
                else:
                    print(f"显示器 {i} 校准失败或被用户取消")
                    return False
                
                # 检查ESC键
                # 注意：这里需要从外部传入ESC键检查函数或使用全局变量
            
            print("\n双屏幕校准全部完成！")
            
            # 初始化校准结果加载
            self._load_all_calibration_results()
            
            return True
                
        except Exception as e:
            print(f"双屏幕校准过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            # 重新初始化显示
            self.ui.initialize_display()
            return False