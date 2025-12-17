#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼手交互系统手眼协调模块

该模块负责实现眼手协调机制，包括鼠标自动移动、右键触发检测和交互区域管理。

主要功能：
1. 鼠标自动移动到注视点位置（基于鼠标右键触发）
2. 注视稳定性检测（Dwell检测）
3. 交互区域管理和显示
4. 键盘和鼠标事件检测
5. 跨屏环境下的手眼协调支持
6. 自动移动冷却机制和距离阈值管理

核心类：
- HandEyeCoordinator: 手眼协调器，负责所有手眼协调功能

关键功能点：
1. 鼠标右键触发的手眼协调机制
2. 基于Dwell检测的注视稳定性判断
3. 跨屏环境下的距离计算和阈值调整
4. 交互区域的动态显示和管理
5. 自动移动冷却机制，防止频繁触发

使用场景：
1. 鼠标右键触发的自动鼠标移动
2. 基于注视点的交互区域显示
3. 多屏环境下的跨屏鼠标移动
4. 注视稳定性检测和触发控制

"""
import numpy as np
import time
from collections import deque
import win32api
import win32con
import ctypes
from ctypes import wintypes
import uiautomation as auto

# ---------- UI控件检测相关常量和函数 ----------
# 跳过文字/图标类型
SKIP_TYPES = {
    "TextControl",
    "ImageControl",
    "GlyphControl",
    "ListControl"
}

def find_non_leaf_control(ctrl):
    """
    如果鼠标命中了文字、图标，向上爬找到真正的 UI 容器控件
    （例如按钮、面板、菜单项等）
    
    Args:
        ctrl: 原始UI控件
        
    Returns:
        真正的UI容器控件，如果未找到则返回None
    """
    while ctrl:
        if ctrl.ControlTypeName not in SKIP_TYPES:
            return ctrl
        ctrl = ctrl.GetParentControl()
    return None

class HandEyeCoordinator:
    """手眼协调器，负责处理手眼协调机制、鼠标自动移动"""
    
    def __init__(self, ui, screen_manager, threshold_config=None):
        self.ui = ui
        self.screen_manager = screen_manager
        
        # 初始化阈值配置 - 必须从core_system.py传入，不提供默认值
        if threshold_config is None:
            raise ValueError("threshold_config 参数必须从 CoreSystem 传入，不能为空")
        
        # 验证必需的阈值参数是否存在
        required_keys = [
            'distance_multiplier', 'auto_move_distance', 'auto_move_cooldown',
            'gaze_dispersion_frames', 'angle_threshold', 'pixel_threshold',
            'boundary_buffer', 'interaction_zone_duration', 'mouse_movement_threshold',
            'teleport_circle_radius', 'teleport_cooldown_duration', 'sliding_window_time_limit',
            'mad_threshold', 'perc95_threshold', 'sliding_window_angle_threshold'
        ]
        
        missing_keys = [key for key in required_keys if key not in threshold_config]
        if missing_keys:
            raise ValueError(f"threshold_config 中缺少必需的参数: {missing_keys}")
        
        self.threshold_config = threshold_config
        
        # 检测到的UI控件信息，用于在UI界面上绘制边框
        self.detected_ui_controls = []
        
        # 手眼协调机制相关状态变量
        self.hand_eye_coordination_enabled = True  # 手眼协调机制总开关（鼠标右键触发）
        
        # 鼠标自动移动相关 - 基于鼠标右键检测
        self.last_auto_mouse_move_time = 0  # 上次自动鼠标移动的时间
        
        # 当前交互状态
        self.current_interaction_zone = None
        self.previous_interaction_zone = None
        self.interaction_zone_start_time = 0
        
        # 滑动窗口机制相关 - 鼠标移动触发专用
        self.sliding_window_gaze_points = deque(maxlen=8)  # 滑动窗口，最多8个注视点
        
        # 鼠标滑动窗口检测相关
        self.mouse_movement_window = deque(maxlen=5)  # 鼠标滑动窗口，最多5个位置点
        self.initial_mouse_position = None  # 鼠标初始位置
        
        # 传送冷却机制相关
        self.last_teleport_trigger_time = 0  # 上次传送触发时间
        
        # 传送后速度阻尼相关变量
        self.last_teleport_time = 0  # 上次传送的时间戳
        self.post_teleport_damping_enabled = True  # 传送后阻尼开关
        self.post_teleport_duration = 200  # 阻尼持续时间（毫秒）
        self.damping_factor = 0.01  # 阻尼系数：0.01表示速度降低到1%
        
        # 系统级鼠标速度控制
        self.user32 = ctypes.windll.user32
        self.SPI_SETMOUSESPEED = 0x0071
        self.SPI_GETMOUSESPEED = 0x0070
        self.SPIF_SENDCHANGE = 0x0002
        
        self.TARGET_LOW_SPEED = 2  # 目标低速度
        self.RESTORE_TARGET_SPEED = 10  # 恢复目标速度
        self.RESTORE_TIME = 0.3  # 恢复时间（秒）
        self.RESTORE_STEP_DELAY = 0.01  # 恢复步长延迟
        self.EASING_POWER = 3  # 非线性缓出指数
        
        self.restoring = False  # 恢复状态标记
        self.original_mouse_speed = self.get_mouse_speed()  # 记录原始鼠标速度
    
    def check_right_mouse_button(self):
        """检查鼠标右键是否被按下"""
        return win32api.GetKeyState(win32con.VK_RBUTTON) < 0
    
    def check_escape_key(self):
        """检查ESC键是否被按下"""
        return win32api.GetKeyState(win32con.VK_ESCAPE) < 0
    
    def check_space_key(self):
        """检查空格键是否被按下"""
        return win32api.GetKeyState(win32con.VK_SPACE) < 0
    
    def _calculate_gaze_center(self):
        """
        计算凝视点中心点
        
        Returns:
            tuple: 凝视点中心点坐标 (x, y) 绝对坐标
        """
        points = np.array([(x, y) for _, x, y in self.sliding_window_gaze_points])
        center = np.mean(points, axis=0)
        return (center[0], center[1])
    
    def _check_sliding_window_distribution(self):
        """更稳健的滑动窗口视线稳定性判断
        
        Returns:
            bool: 视线是否稳定
        """
        try:
            points = np.array([(x, y) for _, x, y in self.sliding_window_gaze_points])
            # ===== 1) 使用中位数作为中心（比均值稳健得多） =====
            center = np.median(points, axis=0)

            # ===== 2) 使用绝对偏差( MAD )代替 max_distance / avg_distance =====
            # MAD 是鲁棒统计中特别常用的“抗噪”指标
            distances = np.sqrt(np.sum((points - center)**2, axis=1))
            mad = np.median(np.abs(distances - np.median(distances)))

            # ===== 3) 判断分布范围（比你现在的 max/avg 更稳定）=====
            perc95 = np.percentile(distances, 95)

            # 使用配置中的阈值
            mad_threshold = self.threshold_config.get('mad_threshold', 50)
            perc95_threshold = self.threshold_config.get('perc95_threshold', 120)
            
            stable_spatial = (mad < mad_threshold) and (perc95 < perc95_threshold)

            # ===== 4) 加入时间稳定性：至少 300ms 都稳定 =====
            timestamps = [t for t, _, _ in self.sliding_window_gaze_points]
            duration = timestamps[-1] - timestamps[0]
            time_limit = self.threshold_config.get('sliding_window_time_limit', 350)
            stable_time = duration > 0.3   # 300 ms 以上才算真正凝视

            # ======= 最终判定 =======
            if stable_spatial and stable_time:
                # 视线稳定，检查鼠标移动
                # 找到中心点所在的屏幕
                gaze_monitor_index = self.screen_manager.get_target_screen(center[0], center[1])
                gaze_monitor = self.ui.monitors_info[gaze_monitor_index]
                
                # 将绝对坐标转换为目标屏幕的相对坐标
                rel_x = center[0] - gaze_monitor['x']
                rel_y = center[1] - gaze_monitor['y']
                
                # 检查鼠标移动并触发传送
                self._check_mouse_movement_and_trigger_cursor(rel_x, rel_y)
                return True
            return False
        except Exception:
            return False
    
    def _detect_ui_control_at_point(self, x, y):
        """
        检测指定坐标点的UI控件
        
        Args:
            x: 检测点X坐标（绝对坐标）
            y: 检测点Y坐标（绝对坐标）
        
        Returns:
            tuple: (UI控件, 边界矩形, 状态)，如果未检测到则返回 (None, None, "")
        """
        try:
            # 将坐标转换为整数
            x = int(x)
            y = int(y)
            
            raw_ctrl = auto.ControlFromPoint(x, y)
            if raw_ctrl is None:
                return None, None, ""
            
            ctrl = find_non_leaf_control(raw_ctrl)
            if ctrl is None:
                return None, None, ""
            
            rect = ctrl.BoundingRectangle
            if rect is None:
                return None, None, ""
            
            # 检查控件大小是否超过阈值
            ui_control_max_size = self.threshold_config.get('ui_control_max_size', 350)
            # rect.width 和 rect.height 是方法，需要调用
            if rect.width() > ui_control_max_size or rect.height() > ui_control_max_size:
                return ctrl, rect, "无效框（控件太大）"
            
            return ctrl, rect, "命中了UI控件"
        except Exception as e:
            print(f"[DEBUG] UI控件检测异常: {e}")
            import traceback
            traceback.print_exc()
            return None, None, ""
    
    def _auto_move_mouse_to_gaze(self, x, y, use_abs_coords=False):
        """自动移动鼠标到注视点位置"""
        current_time = time.time() * 1000
        
        # 检查冷却时间
        if current_time - self.last_auto_mouse_move_time < self.threshold_config['auto_move_cooldown']:
            return
        
        # 设置冷却时间
        self.last_auto_mouse_move_time = current_time
        
        if use_abs_coords:
            # 使用绝对坐标直接移动
            win32api.SetCursorPos((int(x), int(y)))
            
            # 在传送位置添加渐变圆圈效果
            display_x, display_y = x, y
        else:
            # 使用相对坐标移动（基于当前屏幕）
            current_monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
            abs_x = int(x + current_monitor['x'])
            abs_y = int(y + current_monitor['y'])
            win32api.SetCursorPos((abs_x, abs_y))
            
            # 在传送位置添加渐变圆圈效果
            display_x, display_y = abs_x, abs_y
        
        # 找到目标位置所在的显示器
        target_monitor_index = None
        for i, monitor in enumerate(self.ui.monitors_info):
            if monitor['x'] <= display_x < monitor['x'] + monitor['width'] and \
               monitor['y'] <= display_y < monitor['y'] + monitor['height']:
                target_monitor_index = i
                break
        
        if target_monitor_index is not None:
            # 计算相对于目标显示器的坐标
            monitor = self.ui.monitors_info[target_monitor_index]
            relative_x = display_x - monitor['x']
            relative_y = display_y - monitor['y']
            
            # 在双屏模式下，使用interaction_overlays
            if self.ui.interaction_overlays and len(self.ui.interaction_overlays) > target_monitor_index:
                self.ui.interaction_overlays[target_monitor_index].add_fade_circle(relative_x, relative_y, radius=100, duration=1500)
            # 兼容单屏模式
            elif hasattr(self.ui, 'current_widget') and self.ui.current_widget:
                self.ui.current_widget.add_fade_circle(display_x, display_y, radius=100, duration=1500)
    
    def _check_mouse_movement_and_trigger_cursor(self, target_x, target_y):
        """检查鼠标移动条件并触发光标跳转（改进的方向判断方法）
        
        Args:
            target_x: 目标注视点X坐标（相对当前活动屏幕）
            target_y: 目标注视点Y坐标（相对当前活动屏幕）
        """
        try:
            # 获取当前鼠标位置
            current_cursor_pos = win32api.GetCursorPos()
            cursor_x, cursor_y = current_cursor_pos
            
            # 获取当前时间戳（毫秒）
            current_time = time.time() * 1000
            
            # 计算当前鼠标与注视点的距离（第三个判定条件）
            # 转换注视点为绝对坐标
            # 关键修复：使用当前活动屏幕的偏移量计算注视点的绝对坐标
            current_monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
            
            # 首先使用当前活动屏幕计算注视点的绝对坐标
            gaze_x_abs = target_x + current_monitor['x']
            gaze_y_abs = target_y + current_monitor['y']
            
            # 确定注视点实际所在的屏幕
            gaze_monitor_index = None
            for i, monitor in enumerate(self.ui.monitors_info):
                if (monitor['x'] <= gaze_x_abs < monitor['x'] + monitor['width'] and
                    monitor['y'] <= gaze_y_abs < monitor['y'] + monitor['height']):
                    gaze_monitor_index = i
                    break
            
            # 如果注视点不在任何已知屏幕内，使用当前活动屏幕
            if gaze_monitor_index is None:
                gaze_monitor_index = self.screen_manager.current_monitor_index
            
            # 记录注视点所在的屏幕
            self.gaze_monitor_index = gaze_monitor_index
            
            gaze_cursor_distance = np.sqrt((cursor_x - gaze_x_abs)**2 + (cursor_y - gaze_y_abs)**2)
            
            # 记录当前鼠标位置到滑动窗口
            self.mouse_movement_window.append((current_time, cursor_x, cursor_y))
            
            # 使用滑动窗口第一个和最后一个位置计算移动距离和方向
            if len(self.mouse_movement_window) >= 2:
                # 使用滑动窗口第一个和最后一个位置计算移动距离和方向
                first_pos = self.mouse_movement_window[0]  # 第一个位置 (timestamp, x, y)
                last_pos = self.mouse_movement_window[-1]   # 最后一个位置 (timestamp, x, y)
                move_dx = last_pos[1] - first_pos[1]  # x坐标差值
                move_dy = last_pos[2] - first_pos[2]  # y坐标差值
                mouse_move_distance = np.sqrt(move_dx**2 + move_dy**2)
                
                # 计算余弦相似度
                cosine_similarity = self._calculate_cosine_similarity_to_gaze(
                    move_dx, move_dy, cursor_x, cursor_y, gaze_x_abs, gaze_y_abs)
                
                # 距离变化方法（直接检查距离是否在减少）
                distance_change = self._calculate_distance_change(
                    first_pos[1], first_pos[2], cursor_x, cursor_y, gaze_x_abs, gaze_y_abs)
                
                # 综合判断：使用余弦相似度和距离变化的组合
                direction_valid = self._combined_direction_check(
                    cosine_similarity, distance_change, mouse_move_distance)
                
                # 检查是否达到所有触发条件：降低阈值提高灵敏度
                mouse_movement_threshold = self.threshold_config.get('mouse_movement_threshold', 200)
                if (mouse_move_distance >= mouse_movement_threshold and 
                    gaze_cursor_distance > 300 and  # 降低最小距离要求到300像素
                    direction_valid):
                    print(f"mouse_move_distance: {mouse_move_distance:.2f}, mouse_movement_threshold: {mouse_movement_threshold:.2f}")
                    
                    # 检查传送冷却时间
                    current_time = time.time() * 1000  # 毫秒时间戳
                    cooldown = self.threshold_config.get('teleport_cooldown_duration', 1000)
                    if current_time - self.last_teleport_trigger_time >= cooldown:
                        # 所有条件都满足且不在冷却期：执行传送
                        print(f"[DEBUG] 鼠标移动触发传送: 移动{mouse_move_distance:.0f}px, 距离{gaze_cursor_distance:.0f}px")
                        print(f"[DEBUG] 方向判断 - 余弦相似度: {cosine_similarity:.3f}, 距离变化: {distance_change:.1f}")
                        self.last_teleport_trigger_time = current_time  # 更新上次传送时间
                        # 触发传送和视觉反馈
                        self._trigger_fade_circle_cursor_move_and_reset(gaze_x_abs, gaze_y_abs)
            else:
                # 如果是第一次记录，不需要设置初始位置，滑动窗口会自动维护
                pass
                    
        except Exception as e:
            print(f"[DEBUG] 鼠标移动检测异常: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    def _generate_scatter_points(self, center_x, center_y, radius, count=8):
        """
        生成散布检测点
        
        Args:
            center_x: 圆心X坐标（绝对坐标）
            center_y: 圆心Y坐标（绝对坐标）
            radius: 圆半径（像素）
            count: 生成的检测点数量
        
        Returns:
            list: 检测点列表，每个元素为 (x, y) 绝对坐标
        """
        points = []
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((int(x), int(y)))
        return points
    
    def calculate_cross_screen_distance(self, point1, point2):
        """计算跨屏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_cosine_similarity_to_gaze(self, move_dx, move_dy, cursor_x, cursor_y, target_x, target_y):
        """计算鼠标移动方向与指向注视点方向的余弦相似度
        
        Args:
            move_dx: 鼠标移动X方向分量
            move_dy: 鼠标移动Y方向分量
            cursor_x: 当前鼠标X坐标
            cursor_y: 当前鼠标Y坐标
            target_x: 目标注视点X坐标（绝对坐标）
            target_y: 目标注视点Y坐标（绝对坐标）
            
        Returns:
            float: 余弦相似度值，范围[-1, 1]
        """
        # 鼠标移动向量
        move_vector = np.array([move_dx, move_dy])
        move_magnitude = np.linalg.norm(move_vector)
        
        if move_magnitude < 1e-6:  # 移动距离太小
            return 0.0
        
        # 指向注视点的向量
        gaze_vector = np.array([target_x - cursor_x, target_y - cursor_y])
        gaze_magnitude = np.linalg.norm(gaze_vector)
        
        if gaze_magnitude < 1e-6:  # 距离太近
            return 0.0
        
        # 计算余弦相似度
        cosine_similarity = np.dot(move_vector, gaze_vector) / (move_magnitude * gaze_magnitude)
        # 限制在[-1, 1]范围内
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        return cosine_similarity
    
    def _calculate_distance_change(self, first_x, first_y, current_x, current_y, target_x, target_y):
        """计算鼠标与注视点距离的变化（减少为正数）
        
        Args:
            first_x: 初始鼠标X坐标
            first_y: 初始鼠标Y坐标
            current_x: 当前鼠标X坐标
            current_y: 当前鼠标Y坐标
            target_x: 目标注视点X坐标（绝对坐标）
            target_y: 目标注视点Y坐标（绝对坐标）
            
        Returns:
            float: 距离变化值，正数表示接近，负数表示远离
        """
        # 初始距离
        initial_distance = np.sqrt((first_x - target_x)**2 + (first_y - target_y)**2)
        # 当前距离
        current_distance = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
        
        # 距离变化：正数表示接近，负数表示远离
        distance_change = initial_distance - current_distance
        
        return distance_change
    
    def _combined_direction_check(self, cosine_similarity, distance_change, move_distance):
        """优化的方向判断：降低阈值提高灵敏度
        
        Args:
            cosine_similarity: 余弦相似度值
            distance_change: 距离变化值
            move_distance: 鼠标移动距离
            
        Returns:
            bool: 方向是否有效
        """
        # 余弦相似度阈值：>0.3表示朝注视点方向移动（降低阈值）
        cosine_threshold = 0.3
        cosine_valid = cosine_similarity > cosine_threshold
        
        # 距离变化阈值：>20像素表示确实在接近（降低阈值）
        distance_threshold = 20.0
        distance_valid = distance_change > distance_threshold
        
        # 两个条件都满足时才触发传送
        return cosine_valid and distance_valid
    
    def _calculate_circular_teleport(self, center_x, center_y, move_dx, move_dy):
        """根据鼠标移动方向计算圆周传送目标点
        
        Args:
            center_x: 中心X坐标（绝对坐标）
            center_y: 中心Y坐标（绝对坐标）
            move_dx: 鼠标移动X方向分量
            move_dy: 鼠标移动Y方向分量
            
        Returns:
            tuple: 传送目标点 (x, y) 绝对坐标，确保在目标屏幕范围内
        """
        # 归一化鼠标移动方向向量
        move_magnitude = np.sqrt(move_dx**2 + move_dy**2)
        if move_magnitude < 1e-6:  # 防止除零
            return center_x, center_y
            
        # 归一化方向向量
        normalized_dx = move_dx / move_magnitude
        normalized_dy = move_dy / move_magnitude
        
        # 计算传送位置：在相反方向（对侧）的圆周上
        radius = self.threshold_config.get('teleport_circle_radius', 100)
        teleport_x = center_x - normalized_dx * radius
        teleport_y = center_y - normalized_dy * radius
        
        # 确保传送目标位置在目标屏幕范围内
        # 首先确定注视点所在的屏幕
        gaze_monitor_index = self.screen_manager.get_target_screen(center_x, center_y)
        gaze_monitor = self.ui.monitors_info[gaze_monitor_index]
        
        # 将传送目标位置限制在目标屏幕范围内
        teleport_x = max(gaze_monitor['x'], min(teleport_x, gaze_monitor['x'] + gaze_monitor['width'] - 1))
        teleport_y = max(gaze_monitor['y'], min(teleport_y, gaze_monitor['y'] + gaze_monitor['height'] - 1))
        
        return teleport_x, teleport_y
    
    def _trigger_fade_circle_cursor_move_and_reset(self, fade_circle_x, fade_circle_y):
        """触发传送到渐变圆圈边界的光标移动并重置所有相关状态
        
        Args:
            fade_circle_x: 渐变圆圈中心X坐标（绝对坐标）
            fade_circle_y: 渐变圆圈中心Y坐标（绝对坐标）
        """
        try:
            # 保存传送前的鼠标滑动窗口数据，防止传送后窗口数据被污染
            saved_window = list(self.mouse_movement_window)
            
            # 获取当前鼠标位置作为参考点
            current_cursor_pos = win32api.GetCursorPos()
            cursor_x, cursor_y = current_cursor_pos
            
            # 确定注视点所在的屏幕
            gaze_monitor_index = self.screen_manager.get_target_screen(fade_circle_x, fade_circle_y)
            gaze_monitor = self.ui.monitors_info[gaze_monitor_index]
            
            # ===== UI辅助跳转点确认功能 =====
            ui_target_x, ui_target_y = fade_circle_x, fade_circle_y
            ui_status = "未使用UI辅助"
            
            # 检查是否启用了UI辅助跳转功能
            ui_assisted_jump_enabled = self.threshold_config.get('ui_assisted_jump_enabled', True)
            print(f"[DEBUG] UI辅助跳转功能: {'已启用' if ui_assisted_jump_enabled else '已禁用'}")
            print(f"[DEBUG] 滑动窗口点数量: {len(self.sliding_window_gaze_points)}/8")
            
            if ui_assisted_jump_enabled and len(self.sliding_window_gaze_points) >= 8:
                # 计算凝视点中心点
                gaze_center = self._calculate_gaze_center()
                print(f"[DEBUG] 凝视点中心点: ({gaze_center[0]:.2f}, {gaze_center[1]:.2f})")
                
                # 查找最优的UI目标控件
                optimal_target, status = self._find_optimal_ui_target(gaze_center)
                if optimal_target:
                    ui_target_x, ui_target_y = optimal_target
                    ui_status = status
                    print(f"[DEBUG] 找到最优UI目标: ({optimal_target[0]}, {optimal_target[1]}), 状态: {status}")
                else:
                    print(f"[DEBUG] 未找到有效UI目标，状态: {status}")
            
            # 决定跳转位置
            if ui_status == "命中了UI控件":
                # 情况1：初始UI检测命中有效UI控件，跳转到UI中心
                target_x, target_y = ui_target_x, ui_target_y
                print(f"[DEBUG] UI辅助跳转: 初始检测命中UI控件，跳转到中心 ({int(target_x)}, {int(target_y)})")
                
                # 绿色渐变圆圈也绘制在UI控件中心
                fade_circle_x = ui_target_x
                fade_circle_y = ui_target_y
            elif ui_status == "随机抽的UI框":
                # 情况2：散布检测找到UI控件，需要跳转到边界位置
                print(f"[DEBUG] UI辅助跳转: 散布检测找到UI控件，需要跳转到边界位置")
                
                # 使用滑动窗口第一个和最后一个位置计算移动方向和距离
                if len(self.mouse_movement_window) >= 2:
                    # 使用滑动窗口第一个和最后一个位置
                    first_pos = self.mouse_movement_window[0]  # (timestamp, x, y)
                    last_pos = self.mouse_movement_window[-1]   # (timestamp, x, y)
                    move_dx = last_pos[1] - first_pos[1]  # x坐标差值
                    move_dy = last_pos[2] - first_pos[2]  # y坐标差值
                    move_distance = np.sqrt(move_dx**2 + move_dy**2)  # 滑动窗口总移动距离
                    
                    # 使用滑动窗口移动方向计算传送目标到UI边界
                    if move_distance > 1e-6:
                        # 计算圆周传送目标点
                        target_x, target_y = self._calculate_circular_teleport(
                            ui_target_x, ui_target_y, move_dx, move_dy)
                    else:
                        # 如果移动距离太小，使用固定偏移
                        radius = self.threshold_config.get('teleport_circle_radius', 100)
                        target_x = ui_target_x - radius
                        target_y = ui_target_y - radius
                else:
                    # 如果没有足够的鼠标移动数据，直接使用UI目标坐标
                    target_x, target_y = ui_target_x, ui_target_y
                    
                # 绿色渐变圆圈绘制在UI控件中心
                fade_circle_x = ui_target_x
                fade_circle_y = ui_target_y
            else:
                # 情况3：未命中UI控件，使用原有默认传送逻辑
                print(f"[DEBUG] UI辅助跳转: 未命中UI控件，使用默认传送逻辑")
                # 使用滑动窗口第一个和最后一个位置计算移动方向和距离
                if len(self.mouse_movement_window) >= 2:
                    # 使用滑动窗口第一个和最后一个位置
                    first_pos = self.mouse_movement_window[0]  # (timestamp, x, y)
                    last_pos = self.mouse_movement_window[-1]   # (timestamp, x, y)
                    move_dx = last_pos[1] - first_pos[1]  # x坐标差值
                    move_dy = last_pos[2] - first_pos[2]  # y坐标差值
                    move_distance = np.sqrt(move_dx**2 + move_dy**2)  # 滑动窗口总移动距离
                    
                    # 使用滑动窗口移动方向计算传送目标
                    if move_distance > 1e-6:
                        # 计算圆周传送目标点
                        target_x, target_y = self._calculate_circular_teleport(
                            ui_target_x, ui_target_y, move_dx, move_dy)
                    else:
                        # 如果移动距离太小，使用固定偏移
                        radius = self.threshold_config.get('teleport_circle_radius', 100)
                        target_x = ui_target_x - radius
                        target_y = ui_target_y - radius
                else:
                    # 如果没有足够的鼠标移动数据，直接使用原始凝视点中心
                    target_x, target_y = ui_target_x, ui_target_y
                    
                # 绿色渐变圆圈绘制在原始凝视点中心
                fade_circle_x = ui_target_x
                fade_circle_y = ui_target_y
            
            # 移动鼠标到目标位置（使用绝对坐标）
            win32api.SetCursorPos((int(target_x), int(target_y)))
            
            # 输出UI信息使用的状态
            print(f"[DEBUG] 跳转目标: ({int(target_x)}, {int(target_y)}), UI状态: {ui_status}")
            
            # 确保绿色渐变圆圈的坐标在屏幕内
            # 找到UI控件所在的显示器
            target_monitor_index = 0
            fade_circle_x_clamped = fade_circle_x
            fade_circle_y_clamped = fade_circle_y
            
            # 遍历所有显示器，找到fade_circle所在的显示器
            for i, monitor in enumerate(self.ui.monitors_info):
                if (monitor['x'] <= fade_circle_x < monitor['x'] + monitor['width'] and
                    monitor['y'] <= fade_circle_y < monitor['y'] + monitor['height']):
                    target_monitor_index = i
                    break
            
            # 获取目标显示器信息
            target_monitor = self.ui.monitors_info[target_monitor_index]
            
            # 计算相对于目标显示器的坐标
            relative_x = int(fade_circle_x - target_monitor['x'])
            relative_y = int(fade_circle_y - target_monitor['y'])
            
            # 在双屏模式下，使用interaction_overlays
            if self.ui.interaction_overlays and len(self.ui.interaction_overlays) > target_monitor_index:
                self.ui.interaction_overlays[target_monitor_index].add_fade_circle(relative_x, relative_y, radius=100, duration=1500)
            # 兼容单屏模式
            elif hasattr(self.ui, 'current_widget') and self.ui.current_widget:
                self.ui.current_widget.add_fade_circle(int(fade_circle_x_clamped), int(fade_circle_y_clamped), radius=100, duration=1500)
            
            # 应用传送后阻尼效果
            self._apply_post_teleport_damping()
            
        except Exception as e:
            print(f"[DEBUG] 传送执行异常: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_damping_factor(self, current_time):
        """计算当前阻尼系数（简化的指数阻尼）
        
        Args:
            current_time: 当前时间戳（毫秒）
            
        Returns:
            float: 阻尼系数（0.0-1.0）
        """
        if not self.post_teleport_damping_enabled:
            return 1.0
        
        # 检查是否在传送后阻尼期间
        time_since_teleport = current_time - self.last_teleport_time
        if time_since_teleport > self.post_teleport_duration or self.last_teleport_time == 0:
            return 1.0
        
        # 计算阻尼期间的进度（0.0-1.0）
        progress = time_since_teleport / self.post_teleport_duration
        
        # 使用指数阻尼：快速衰减然后逐渐平缓
        factor = self.damping_factor + (1.0 - self.damping_factor) * np.exp(-4 * progress)
        
        return max(self.damping_factor, min(1.0, factor))
    
    def set_mouse_speed(self, speed):
        """设置鼠标速度（使用Windows API）
        
        Args:
            speed: 鼠标速度值（1-20）
        """
        speed = int(max(1, min(20, speed)))
        self.user32.SystemParametersInfoW(self.SPI_SETMOUSESPEED, 0, speed, self.SPIF_SENDCHANGE)
    
    def get_mouse_speed(self):
        """获取当前鼠标速度
        
        Returns:
            int: 当前鼠标速度值
        """
        v = ctypes.c_int()
        self.user32.SystemParametersInfoW(self.SPI_GETMOUSESPEED, 0, ctypes.byref(v), 0)
        return v.value
    
    def restore_speed_ease_out(self, start_speed):
        """非线性缓出式恢复鼠标速度
        
        Args:
            start_speed: 起始速度
        """
        self.restoring = True
        steps = int(self.RESTORE_TIME / self.RESTORE_STEP_DELAY)
        
        for i in range(steps + 1):
            if not self.restoring:
                break
            
            # 计算当前步骤的进度（0.0-1.0）
            progress = i / steps
            
            # 使用缓出函数：进度的立方，使得恢复速度先慢后快
            ease_progress = progress ** self.EASING_POWER
            
            # 计算当前目标速度
            current_speed = int(start_speed + (self.RESTORE_TARGET_SPEED - start_speed) * ease_progress)
            
            # 设置鼠标速度
            self.set_mouse_speed(current_speed)
            
            # 等待一段时间
            time.sleep(self.RESTORE_STEP_DELAY)
        
        self.restoring = False
    
    def _apply_post_teleport_damping(self):
        """应用传送后阻尼效果"""
        if not self.post_teleport_damping_enabled:
            return
        
        # 更新上次传送时间
        self.last_teleport_time = time.time() * 1000
        
        # 保存当前速度
        current_speed = self.get_mouse_speed()
        
        # 设置低速度
        self.set_mouse_speed(self.TARGET_LOW_SPEED)
        
        # 启动恢复线程
        self.restoring = False
        import threading
        threading.Thread(target=self.restore_speed_ease_out, args=(self.TARGET_LOW_SPEED,)).start()
    
    def _update_detected_ui_controls(self, rects):
        """
        更新检测到的UI控件列表
        
        Args:
            rects: UI控件矩形列表，每个元素为 (left, top, width, height) 绝对坐标
        """
        # 清空现有列表
        self.detected_ui_controls.clear()
        
        # 添加新的UI控件矩形
        for rect in rects:
            self.detected_ui_controls.append(rect)
        
        # 将UI控件矩形传递给UI组件，用于绘制边框
        if hasattr(self.ui, 'interaction_overlays') and self.ui.interaction_overlays:
            # 多屏模式：为每个屏幕准备对应的UI控件列表
            screen_ui_controls = [[] for _ in range(len(self.ui.interaction_overlays))]
            
            for rect in self.detected_ui_controls:
                left, top, width, height = rect
                # 计算UI控件的中心点，用于确定所在屏幕
                center_x = left + width // 2
                center_y = top + height // 2
                
                # 找到UI控件所在的屏幕
                target_screen_index = None
                for i, monitor in enumerate(self.ui.monitors_info):
                    if (monitor['x'] <= center_x < monitor['x'] + monitor['width'] and
                        monitor['y'] <= center_y < monitor['y'] + monitor['height']):
                        target_screen_index = i
                        break
                
                if target_screen_index is not None:
                    # 将绝对坐标转换为目标屏幕的相对坐标
                    monitor = self.ui.monitors_info[target_screen_index]
                    rel_left = left - monitor['x']
                    rel_top = top - monitor['y']
                    
                    # 添加到对应屏幕的UI控件列表
                    screen_ui_controls[target_screen_index].append((rel_left, rel_top, width, height))
            
            # 为每个屏幕更新对应的UI控件
            for i, overlay in enumerate(self.ui.interaction_overlays):
                overlay.update_ui_controls(screen_ui_controls[i])
        elif hasattr(self.ui, 'current_widget') and self.ui.current_widget:
            # 单屏模式：直接传递绝对坐标
            self.ui.current_widget.update_ui_controls(self.detected_ui_controls)
    
    def _find_optimal_ui_target(self, gaze_center):
        """
        查找最优的UI目标控件，并返回UI信息使用的状态
        
        Args:
            gaze_center: 凝视点中心点坐标 (x, y) 绝对坐标
        
        Returns:
            tuple: (最优UI控件的中心坐标, UI信息使用的状态)，如果未找到则返回 (None, "")
        """
        # 初始化状态变量
        status = ""
        optimal_rect = None
        
        # 1. 初始UI控件检测
        x, y = gaze_center
        print(f"[DEBUG] 初始UI控件检测: 坐标 ({x}, {y})")
        ctrl, rect, status = self._detect_ui_control_at_point(x, y)
        if ctrl and rect:
                print(f"[DEBUG] 初始UI控件检测: 找到控件, 状态: {status}")
                if status != "无效框（控件太大）":
                    # 计算UI控件的几何中心
                    center_x = rect.left + rect.width() // 2
                    center_y = rect.top + rect.height() // 2
                    # 设置最优UI控件矩形
                    optimal_rect = (rect.left, rect.top, rect.width(), rect.height())
                    # 更新检测到的UI控件
                    self._update_detected_ui_controls([optimal_rect])
                    print(f"[DEBUG] 找到UI控件，中心坐标: ({center_x}, {center_y})")
                    return (center_x, center_y), status
        else:
            print(f"[DEBUG] 初始UI控件检测: 未找到有效控件")
        
        # 2. 散布检测机制
        scatter_radius = self.threshold_config.get('scatter_detection_radius', 100)
        scatter_points = self._generate_scatter_points(x, y, scatter_radius)
        print(f"[DEBUG] 散布检测机制: 圆心 ({x}, {y}), 半径 {scatter_radius}, 检测点数量 {len(scatter_points)}")
        
        valid_controls = []
        valid_rects = []
        
        for i, (px, py) in enumerate(scatter_points):
            print(f"[DEBUG] 散布检测点 {i+1}: 坐标 ({px}, {py})")
            ctrl, rect, point_status = self._detect_ui_control_at_point(px, py)
            if ctrl and rect:
                print(f"[DEBUG] 散布检测点 {i+1}: 找到控件, 状态: {point_status}")
                if point_status != "无效框（控件太大）":
                    # 计算UI控件的几何中心
                    center_x = rect.left + rect.width() // 2
                    center_y = rect.top + rect.height() // 2
                    
                    # 计算到凝视点中心点的距离
                    distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
                    
                    valid_controls.append((center_x, center_y, distance, point_status, rect))
                    valid_rects.append((rect.left, rect.top, rect.width(), rect.height()))
                    print(f"[DEBUG] 散布检测点 {i+1}: 添加到有效控件列表, 中心坐标 ({center_x}, {center_y}), 距离 {distance:.2f}")
            else:
                print(f"[DEBUG] 散布检测点 {i+1}: 未找到有效控件")
        
        if valid_controls:
            # 选择距离最近的UI控件
            valid_controls.sort(key=lambda x: x[2])
            optimal_target = valid_controls[0][0], valid_controls[0][1]
            # 获取最优UI控件的矩形
            optimal_rect = (valid_controls[0][4].left, valid_controls[0][4].top, 
                          valid_controls[0][4].width(), valid_controls[0][4].height())
            # 更新检测到的UI控件，只显示最优的一个
            self._update_detected_ui_controls([optimal_rect])
            print(f"[DEBUG] 散布检测机制: 找到 {len(valid_controls)} 个有效控件, 最优目标: ({optimal_target[0]}, {optimal_target[1]})")
            return optimal_target, "随机抽的UI框"
        
        # 3. 无效框处理
        if status == "无效框（控件太大）":
            # 显示无效框
            if rect:
                invalid_rect = (rect.left, rect.top, rect.width(), rect.height())
                self._update_detected_ui_controls([invalid_rect])
            return None, status
        
        # 没有找到任何UI控件，清空显示
        self._update_detected_ui_controls([])
        print(f"[DEBUG] 所有检测均未找到有效UI控件")
        return None, ""
    
    def _process_mouse_movement_triggered_teleport(self, gaze_point):
        """处理鼠标移动触发的传送逻辑主流程
        
        Args:
            gaze_point: 当前凝视点 (x, y) 相对坐标
        """
        if not gaze_point:
            return
        
        current_time = time.time() * 1000  # 转换为毫秒
        
        # 将相对坐标转换为绝对坐标
        current_monitor = self.ui.monitors_info[self.screen_manager.current_monitor_index]
        gaze_x_abs = gaze_point[0] + current_monitor['x']
        gaze_y_abs = gaze_point[1] + current_monitor['y']
        
        # 持续收集绝对坐标注视点到滑动窗口（自动维护最多8个点）
        self.sliding_window_gaze_points.append((current_time, gaze_x_abs, gaze_y_abs))
        
        # 当滑动窗口收集满8个点时进行检测
        if len(self.sliding_window_gaze_points) >= 8:
            self._check_sliding_window_distribution()
            # 检查完成后不清空，让滑动窗口继续工作（移除最老点，加入新点）
    
    def _process_hand_eye_coordination(self, gaze_point, previous_gaze_point, is_dual_screen_mode=False):
        """处理手眼协调机制 - 基于鼠标右键检测，支持多屏距离计算和自动屏幕切换
        同时支持鼠标移动触发的传送逻辑
        """
        if not self.hand_eye_coordination_enabled or not self.ui or self.screen_manager.screen_switching:
            return
        
        # ===== 新增：鼠标移动触发传送逻辑（完整传送逻辑） =====
        self._process_mouse_movement_triggered_teleport(gaze_point)
        
        # ===== 原有：鼠标右键触发逻辑 =====
        # 检测鼠标右键是否被按下
        right_button_pressed = self.check_right_mouse_button()
        
        # ===== 冷却时间检查：500ms 内不进行右键交互检测 =====
        current_time = time.time() * 1000
        if right_button_pressed and current_time - self.last_auto_mouse_move_time < 500:
            return
        
        if right_button_pressed:
            # 使用统一的滑动窗口分布检测进行注视稳定性检查
            
            print("右键按下，即将进行跳转！！！！！！！！！！！！！！！！！")
            
            # 获取当前鼠标位置
            try:
                current_cursor_pos = win32api.GetCursorPos()
                cursor_x, cursor_y = current_cursor_pos
                gaze_distance = np.sqrt((gaze_point[0] - cursor_x)**2 + (gaze_point[1] - cursor_y)**2)
                # 如果距离大于阈值，则移动鼠标到注视点
                distance_threshold = self.threshold_config['auto_move_distance'] * self.threshold_config['distance_multiplier']
                if gaze_distance > distance_threshold:
                    print(f"Distance: {gaze_distance:.2f}, Threshold: {distance_threshold:.2f}")
                    self._auto_move_mouse_to_gaze(gaze_point[0], gaze_point[1], use_abs_coords=False)
                    
            except Exception:
                pass