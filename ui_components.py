#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼手交互系统UI组件模块

该模块负责眼手交互系统的所有UI组件和动画效果，包括透明窗口、交互覆盖层、渐变圆圈动画等。

主要功能：
1. 透明窗口创建和管理（用于覆盖在屏幕上显示视觉效果）
2. 交互覆盖层实现（支持多屏环境）
3. 渐变圆圈动画效果（用于标记注视点位置和鼠标移动）
4. 主菜单和交互模式选择UI
5. 校准界面和进度显示
6. 多屏环境下的UI布局管理

核心类：
- TransparentWindow: 透明窗口基类，提供基础的透明窗口功能
- EyeHandInteractionUI: 主UI类，负责管理所有UI组件
- InteractionOverlay: 交互覆盖层，用于显示渐变圆圈等视觉效果
- FadeOutCircle: 渐变圆圈动画类，实现圆圈的淡出效果

关键依赖：
- PyQt5: UI组件和动画框架
- OpenCV: 图像处理（用于创建透明背景）
- win32api/win32con: 屏幕信息获取和窗口管理

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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QMessageBox, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QThread, pyqtSlot, QEasingCurve, QVariantAnimation, QEvent
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap, QPainterPath, QRegion, QKeyEvent, QLinearGradient
from PyQt5.QtCore import QSize
import win32api
import win32con

# 颜色定义
WHITE = QColor(255, 255, 255)
BLACK = QColor(0, 0, 0)
GRAY = QColor(200, 200, 200)
BLUE = QColor(52, 152, 219)
GREEN = QColor(46, 204, 113)
RED = QColor(231, 76, 60)
LIGHT_GREEN_TRANSPARENT = QColor(144, 238, 144, 128)

class TransparentWindow(QWidget):
    """透明窗口基类"""
    
    def __init__(self):
        super().__init__()
        self.setup_window()
        
    def setup_window(self):
        """设置窗口属性"""
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        
        # 设置窗口为无边框、置顶、全屏
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 鼠标穿透
        self.setAttribute(Qt.WA_ShowWithoutActivating)  # 不激活窗口
        
        # 设置窗口大小为屏幕大小
        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        
        # 设置窗口透明度
        self.setWindowOpacity(0.99)  # 几乎完全透明
        
    def paintEvent(self, event):
        """重写绘制事件，实现透明背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 完全透明，不绘制任何背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

class EyeHandInteractionUI:
    def __init__(self):
        self.windows = []  # 存储所有屏幕的透明窗口
        self.app = None
        self.current_widget = None
        self.screen_width = 0
        self.screen_height = 0
        
        # 多显示器支持
        self.monitors_info = []
        self.current_monitor_index = 0  # 当前激活的显示器索引
        self.primary_monitor_index = 0  # 主显示器索引
        
        # 交互区域覆盖层
        self.interaction_overlays = []  # 存储所有屏幕的交互覆盖层
        
        # 添加定时器用于更新渐变圆圈动画
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.update_fade_animation)
        self.fade_timer.start(16)  # 约60FPS
        
    def initialize_display(self):
        """初始化显示"""
        # 获取多显示器信息
        self._detect_monitors()
        
        # 获取当前显示器的屏幕尺寸
        current_monitor = self.monitors_info[self.current_monitor_index]
        self.screen_width = current_monitor['width']
        self.screen_height = current_monitor['height']
        
        # 为每个显示器创建透明窗口
        self.windows = []
        for monitor in self.monitors_info:
            window = TransparentWindow()
            # 设置窗口位置和大小为对应显示器
            window.setGeometry(monitor['x'], monitor['y'], monitor['width'], monitor['height'])
            self.windows.append(window)
        
        # 设置字体
        self.font_large = QFont("Microsoft YaHei", 30)
        self.font_medium = QFont("Microsoft YaHei", 20)
        self.font_small = QFont("Microsoft YaHei", 20)
        
        # 返回主显示器的窗口
        return self.windows[self.primary_monitor_index]
    
    def initialize_display_at_position(self, x, y):
        """在指定位置初始化显示
        
        Args:
            x: 窗口x坐标
            y: 窗口y坐标
        """
        # 设置SDL环境变量确保窗口在指定位置
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

        # 调用标准初始化方法
        self.initialize_display()
    
    def _detect_monitors(self):
        """检测所有显示器信息"""
        try:
            # 使用PyQt5的QDesktopWidget获取显示器信息
            desktop = QDesktopWidget()
            monitor_count = desktop.screenCount()
            
            self.monitors_info = []
            
            for i in range(monitor_count):
                monitor_geometry = desktop.screenGeometry(i)
                
                monitor_info = {
                    'index': i,
                    'x': monitor_geometry.x(),
                    'y': monitor_geometry.y(),
                    'width': monitor_geometry.width(),
                    'height': monitor_geometry.height(),
                    'is_primary': i == desktop.primaryScreen(),
                    'name': f"显示器 {i+1}"
                }
                
                self.monitors_info.append(monitor_info)
            
            # 按x坐标排序，确保主显示器在左边
            self.monitors_info.sort(key=lambda m: m['x'])
            
            # 设置当前显示器为主显示器
            for i, monitor in enumerate(self.monitors_info):
                if monitor['is_primary']:
                    self.current_monitor_index = i
                    self.primary_monitor_index = i
                    break
            else:
                # 如果没有找到主显示器，使用第一个
                self.current_monitor_index = 0
                self.primary_monitor_index = 0
                
        except Exception as e:
            # 降级到单显示器模式
            screen = QApplication.primaryScreen()
            screen_geometry = screen.geometry()
            
            self.monitors_info = [{
                'index': 0,
                'x': screen_geometry.x(),
                'y': screen_geometry.y(),
                'width': screen_geometry.width(),
                'height': screen_geometry.height(),
                'is_primary': True,
                'name': "主显示器"
            }]
            
            self.current_monitor_index = 0
            self.primary_monitor_index = 0
    
    def get_monitors_info(self):
        """获取所有显示器信息"""
        return self.monitors_info
    
    def get_current_monitor(self):
        """获取当前激活的显示器信息"""
        if self.monitors_info:
            return self.monitors_info[self.current_monitor_index]
        return None
    
    def get_primary_monitor(self):
        """获取主显示器信息"""
        if self.monitors_info:
            return self.monitors_info[self.primary_monitor_index]
        return None
    
    def switch_to_monitor(self, monitor_index):
        """切换到指定显示器"""
        if 0 <= monitor_index < len(self.monitors_info):
            self.current_monitor_index = monitor_index
            current_monitor = self.monitors_info[monitor_index]
            self.screen_width = current_monitor['width']
            self.screen_height = current_monitor['height']
            return True
        return False
    
    def create_start_widget(self):
        """创建开始界面组件"""
        widget = QWidget()
        widget.setStyleSheet("background-color: white;")
        widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # 标题
        title_label = QLabel("眼手协同交互系统")
        title_label.setFont(self.font_large)
        title_label.setStyleSheet("color: black; background-color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 分辨率信息
        if len(self.monitors_info) > 1:
            monitors_text = "多显示器检测:"
            for monitor in self.monitors_info:
                status = "主显示器" if monitor['is_primary'] else f"显示器 {monitor['index']+1}"
                monitors_text += f"\n{status}: 位置({monitor['x']}, {monitor['y']}) 尺寸: {monitor['width']}x{monitor['height']}"
            
            resolution_label = QLabel(monitors_text)
            resolution_label.setFont(self.font_small)
            resolution_label.setStyleSheet("color: gray; background-color: white;")
            resolution_label.setAlignment(Qt.AlignCenter)
        else:
            resolution_label = QLabel(f"分辨率: {self.screen_width}x{self.screen_height}")
            resolution_label.setFont(self.font_small)
            resolution_label.setStyleSheet("color: gray; background-color: white;")
            resolution_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(resolution_label)
        
        # 说明文字
        instruction_label1 = QLabel("按 'S' 键开始校准")
        instruction_label1.setFont(self.font_medium)
        instruction_label1.setStyleSheet("color: black; background-color: white;")
        instruction_label1.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label1)
        
        instruction_label2 = QLabel("按 'ESC' 键退出")
        instruction_label2.setFont(self.font_medium)
        instruction_label2.setStyleSheet("color: black; background-color: white;")
        instruction_label2.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label2)
        
        instruction_label3 = QLabel("右键按下 + 注视距离鼠标400px以上时自动移动鼠标")
        instruction_label3.setFont(self.font_small)
        instruction_label3.setStyleSheet("color: blue; background-color: white;")
        instruction_label3.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label3)
        
        # 开始按钮
        start_button = QPushButton("开始")
        start_button.setFont(self.font_medium)
        start_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(46, 204, 113);
                color: white;
                border: 2px solid black;
                border-radius: 10px;
                padding: 20px 40px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: rgb(39, 174, 96);
            }
            QPushButton:pressed {
                background-color: rgb(33, 150, 83);
            }
        """)
        layout.addWidget(start_button)
        
        widget.setLayout(layout)
        widget.setGeometry(self.screen_width//4, self.screen_height//4, 
                          self.screen_width//2, self.screen_height//2)
        
        return widget, start_button
    
    def create_interaction_mode_widget(self):
        """创建交互模式选择界面组件"""
        widget = QWidget()
        widget.setStyleSheet("background-color: white;")
        widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # 标题
        title_label = QLabel("选择交互模式")
        title_label.setFont(self.font_large)
        title_label.setStyleSheet("color: black; background-color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 系统介绍
        instruction_label = QLabel("请选择您要使用的交互模式")
        instruction_label.setFont(self.font_medium)
        instruction_label.setStyleSheet("color: blue; background-color: white;")
        instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label)
        
        # 单屏模式按钮
        single_button = QPushButton("单屏交互模式")
        single_button.setFont(self.font_medium)
        single_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(52, 152, 219);
                color: white;
                border: 2px solid black;
                border-radius: 10px;
                padding: 15px 30px;
                min-width: 300px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: rgb(41, 128, 185);
            }
        """)
        layout.addWidget(single_button)
        
        # 多屏模式按钮（只有多显示器时才显示）
        if len(self.monitors_info) > 1:
            multi_button = QPushButton("多屏交互模式")
            multi_button.setFont(self.font_medium)
            multi_button.setStyleSheet("""
                QPushButton {
                    background-color: rgb(155, 89, 182);
                    color: white;
                    border: 2px solid black;
                    border-radius: 10px;
                    padding: 15px 30px;
                    min-width: 300px;
                    margin: 10px;
                }
                QPushButton:hover {
                    background-color: rgb(142, 68, 173);
                }
            """)
            layout.addWidget(multi_button)
        else:
            multi_button = None
        
        # 功能说明
        if len(self.monitors_info) > 1:
            info_text = "单屏模式：适合单显示器使用\n多屏模式：支持多显示器间切换和跳转"
        else:
            info_text = "检测到单显示器，使用单屏模式"
        
        info_label = QLabel(info_text)
        info_label.setFont(self.font_small)
        info_label.setStyleSheet("color: gray; background-color: white;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        widget.setLayout(layout)
        widget.setGeometry(self.screen_width//4, self.screen_height//4, 
                          self.screen_width//2, self.screen_height//2)
        
        return widget, single_button, multi_button
    
    def create_calibration_choice_widget(self):
        """创建校准选择界面组件"""
        widget = QWidget()
        widget.setStyleSheet("background-color: white;")
        widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # 标题
        title_label = QLabel("校准选项")
        title_label.setFont(self.font_large)
        title_label.setStyleSheet("color: black; background-color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 多显示器提示
        if len(self.monitors_info) > 1:
            multi_info = QLabel(f"检测到 {len(self.monitors_info)} 个显示器，支持多屏校准")
            multi_info.setFont(self.font_small)
            multi_info.setStyleSheet("color: blue; background-color: white; font-weight: bold;")
            multi_info.setAlignment(Qt.AlignCenter)
            layout.addWidget(multi_info)
        
        # 三个按钮
        single_button = QPushButton("单屏幕校准")
        single_button.setFont(self.font_medium)
        single_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(52, 152, 219);
                color: white;
                border: 2px solid black;
                border-radius: 10px;
                padding: 15px 30px;
                min-width: 300px;
                margin: 8px;
            }
            QPushButton:hover {
                background-color: rgb(41, 128, 185);
            }
        """)
        layout.addWidget(single_button)
        
        if len(self.monitors_info) > 1:
            dual_button = QPushButton("双屏幕校准")
            dual_button.setFont(self.font_medium)
            dual_button.setStyleSheet("""
                QPushButton {
                    background-color: rgb(155, 89, 182);
                    color: white;
                    border: 2px solid black;
                    border-radius: 10px;
                    padding: 15px 30px;
                    min-width: 300px;
                    margin: 8px;
                }
                QPushButton:hover {
                    background-color: rgb(142, 68, 173);
                }
            """)
            layout.addWidget(dual_button)
        else:
            dual_button = None
        
        load_button = QPushButton("加载历史校准数据")
        load_button.setFont(self.font_medium)
        load_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(46, 204, 113);
                color: white;
                border: 2px solid black;
                border-radius: 10px;
                padding: 15px 30px;
                min-width: 300px;
                margin: 8px;
            }
            QPushButton:hover {
                background-color: rgb(39, 174, 96);
            }
        """)
        layout.addWidget(load_button)
        
        widget.setLayout(layout)
        widget.setGeometry(self.screen_width//4, self.screen_height//4, 
                          self.screen_width//2, self.screen_height//2)
        
        return widget, single_button, dual_button, load_button
    
    def show_start_screen(self):
        """显示开始界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, button = self.create_start_widget()
        self.current_widget = widget
        widget.show()
        
        return button
    
    def show_interaction_mode_selection(self):
        """显示交互模式选择界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, single_button, multi_button = self.create_interaction_mode_widget()
        self.current_widget = widget
        widget.show()
        
        return single_button, multi_button
    
    def show_calibration_choice(self):
        """显示校准选择界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, single_button, dual_button, load_button = self.create_calibration_choice_widget()
        self.current_widget = widget
        widget.show()
        
        return single_button, dual_button, load_button
    
    def show_single_screen_calibration_choice(self):
        """显示单屏校准选择界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, calibrate_button, load_button = self.create_single_calibration_widget("单屏校准")
        self.current_widget = widget
        widget.show()
        
        return calibrate_button, load_button
    
    def show_multi_screen_calibration_choice(self):
        """显示多屏校准选择界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, calibrate_button, load_button = self.create_single_calibration_widget("多屏校准")
        self.current_widget = widget
        widget.show()
        
        return calibrate_button, load_button
    
    def create_single_calibration_widget(self, title):
        """创建单个校准选择界面组件"""
        widget = QWidget()
        widget.setStyleSheet("background-color: white;")
        widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # 标题
        title_label = QLabel(title)
        title_label.setFont(self.font_large)
        title_label.setStyleSheet("color: black; background-color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 说明文本
        if "单屏" in title:
            instruction_label = QLabel("请选择单屏校准操作")
        else:
            instruction_label = QLabel("请选择多屏校准操作")
            
        instruction_label.setFont(self.font_medium)
        instruction_label.setStyleSheet("color: blue; background-color: white;")
        instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label)
        
        # 校准按钮
        if "单屏" in title:
            color = "rgb(52, 152, 219)"  # 蓝色
        else:
            color = "rgb(155, 89, 182)"  # 紫色
            
        calibrate_button = QPushButton("开始新校准")
        calibrate_button.setFont(self.font_medium)
        calibrate_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: 2px solid black;
                border-radius: 10px;
                padding: 15px 30px;
                min-width: 300px;
                margin: 10px;
            }}
            QPushButton:hover {{
                background-color: rgb(41, 128, 185);
            }}
        """)
        layout.addWidget(calibrate_button)
        
        # 加载按钮
        load_button = QPushButton("加载历史校准数据")
        load_button.setFont(self.font_medium)
        load_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(46, 204, 113);
                color: white;
                border: 2px solid black;
                border-radius: 10px;
                padding: 15px 30px;
                min-width: 300px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: rgb(39, 174, 96);
            }
        """)
        layout.addWidget(load_button)
        
        # 功能说明
        if "单屏" in title:
            info_text = "开始新校准：进行新的单屏校准\n加载历史校准数据：使用之前的校准结果"
        else:
            info_text = "开始新校准：对所有显示器进行校准\n加载历史校准数据：使用之前的校准结果"
            
        info_label = QLabel(info_text)
        info_label.setFont(self.font_small)
        info_label.setStyleSheet("color: gray; background-color: white;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        widget.setLayout(layout)
        widget.setGeometry(self.screen_width//4, self.screen_height//4, 
                          self.screen_width//2, self.screen_height//2)
        
        return widget, calibrate_button, load_button
    
    def show_interaction_screen(self, interaction_zone=None, current_gaze_point=None, show_gaze_point=True):
        """显示交互界面（完全透明，只在需要时显示绿色区域）"""
        # 只在需要时创建或更新交互区域
        if interaction_zone or current_gaze_point:
            # 如果还没有创建交互覆盖层，为每个显示器创建一个
            if not self.interaction_overlays:
                for monitor in self.monitors_info:
                    overlay = InteractionOverlay(interaction_zone, current_gaze_point, show_gaze_point)
                    # 设置覆盖层位置和大小为对应显示器
                    overlay.setGeometry(monitor['x'], monitor['y'], monitor['width'], monitor['height'])
                    overlay.show()
                    self.interaction_overlays.append(overlay)
                # 关闭当前界面（如果有的话）
                if self.current_widget:
                    self.current_widget.close()
                    self.current_widget = None
            else:
                # 更新所有交互覆盖层
                for overlay in self.interaction_overlays:
                    # 只有在位置或显示状态真正改变时才更新
                    if (overlay.interaction_zone != interaction_zone or 
                        overlay.current_gaze_point != current_gaze_point or 
                        overlay.show_gaze_point != show_gaze_point):
                        overlay.interaction_zone = interaction_zone
                        overlay.current_gaze_point = current_gaze_point
                        overlay.show_gaze_point = show_gaze_point
                        # 强制重绘整个窗口，避免闪烁
                        overlay.repaint()
        else:
            # 没有交互区域时，关闭所有交互覆盖层
            for overlay in self.interaction_overlays:
                overlay.close()
            self.interaction_overlays = []
            # 关闭当前界面（如果有的话）
            if self.current_widget:
                self.current_widget.close()
                self.current_widget = None
    
    def close_current_widget(self):
        """关闭当前组件"""
        if self.current_widget:
            self.current_widget.close()
            self.current_widget = None
    
    def update_fade_animation(self):
        """更新渐变圆圈动画"""
        # 更新所有交互覆盖层的渐变圆圈
        for overlay in self.interaction_overlays:
            # 更新所有渐变圆圈
            for circle in overlay.fade_circles[:]:
                if circle.update():
                    pass  # 圆圈已更新
                else:
                    # 动画完成，移除圆圈
                    overlay.fade_circles.remove(circle)
            
            # 如果有活动的圆圈，重绘界面
            if overlay.fade_circles:
                overlay.update()
        
        # 兼容旧的单屏模式
        if self.current_widget and isinstance(self.current_widget, InteractionOverlay):
            # 更新所有渐变圆圈
            for circle in self.current_widget.fade_circles[:]:
                if circle.update():
                    pass  # 圆圈已更新
                else:
                    # 动画完成，移除圆圈
                    self.current_widget.fade_circles.remove(circle)
            
            # 如果有活动的圆圈，重绘界面
            if self.current_widget.fade_circles:
                self.current_widget.update()


class FadeOutCircle:
    """渐变消失的圆圈动画类"""
    
    def __init__(self, x, y, radius=100, duration=1500):
        self.x = x
        self.y = y
        self.radius = radius
        self.opacity = 255  # 初始透明度
        self.duration = duration
        self.start_time = None
        self.animation = QVariantAnimation()
        self.animation.setDuration(duration)
        self.animation.setStartValue(255)
        self.animation.setEndValue(0)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def start(self):
        """开始动画"""
        self.start_time = time.time()
        self.animation.start()
        
    def update(self):
        """更新动画状态"""
        if self.animation.state() == QVariantAnimation.Running:
            self.opacity = self.animation.currentValue()
            return True
        return False
        
    def is_finished(self):
        """检查动画是否完成"""
        return self.animation.state() == QVariantAnimation.Stopped
        
    def get_opacity(self):
        """获取当前透明度"""
        return self.opacity


class InteractionOverlay(QWidget):
    """交互区域覆盖层"""
    
    def __init__(self, interaction_zone, current_gaze_point=None, show_gaze_point=True):
        super().__init__()
        self.interaction_zone = interaction_zone
        self.current_gaze_point = current_gaze_point
        self.show_gaze_point = show_gaze_point  # 控制是否显示注视点
        self.fade_circles = []  # 存储所有渐变圆圈
        self.ui_rectangles = []  # 存储检测到的UI矩形框
        self.setup_overlay()
    
    def add_ui_rectangle(self, rect, duration=2000):
        """添加UI矩形框到覆盖层"""
        # rect格式: (x, y, width, height) 或 BoundingRectangle对象
        if hasattr(rect, 'left') and hasattr(rect, 'top') and hasattr(rect, 'right') and hasattr(rect, 'bottom'):
            # 处理uiautomation矩形对象
            x = rect.left
            y = rect.top
            width = rect.right - rect.left
            height = rect.bottom - rect.top
        else:
            # 处理(x, y, width, height)格式
            x, y, width, height = rect
        
        ui_rect = {
            'rect': (x, y, width, height),
            'start_time': time.time() * 1000,
            'duration': duration
        }
        self.ui_rectangles.append(ui_rect)
        self.update()
    
    def update_ui_rectangles(self):
        """更新UI矩形框，移除过期的矩形"""
        current_time = time.time() * 1000
        # 过滤掉过期的矩形
        self.ui_rectangles = [
            rect for rect in self.ui_rectangles 
            if current_time - rect['start_time'] < rect['duration']
        ]
    
    def draw_ui_rectangles(self, painter):
        """绘制UI矩形框"""
        # 更新UI矩形列表
        self.update_ui_rectangles()
        
        for ui_rect in self.ui_rectangles:
            x, y, width, height = ui_rect['rect']
            
            # 计算透明度（随时间衰减）
            current_time = time.time() * 1000
            elapsed_time = current_time - ui_rect['start_time']
            opacity = max(0, 255 - (elapsed_time / ui_rect['duration']) * 255)
            
            # 绘制UI矩形框（蓝色边框，半透明填充）
            painter.setBrush(QBrush(QColor(0, 100, 255, int(opacity * 0.3))))  # 半透明蓝色填充
            painter.setPen(QPen(QColor(0, 150, 255, int(opacity)), 2))  # 蓝色边框
            painter.drawRect(x, y, width, height)
            
            # 在矩形中心绘制小点
            center_x = x + width // 2
            center_y = y + height // 2
            painter.setBrush(QBrush(QColor(255, 255, 0, int(opacity))))  # 黄色点
            painter.setPen(QPen(QColor(255, 255, 0, int(opacity)), 1))
            painter.drawEllipse(QPoint(center_x, center_y), 3, 3)
        
    def setup_overlay(self):
        """设置覆盖层"""
        # 设置窗口属性
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.Tool | 
            Qt.WindowStaysOnTopHint | 
            Qt.WindowTransparentForInput | 
            Qt.WindowDoesNotAcceptFocus 
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # 覆盖整个虚拟桌面
        self._cover_virtual_desktop()
        # 让窗口对UIA不可见
        self._make_uia_invisible()
        
        # 确保窗口不会拦截任何事件
        self.setMouseTracking(False)
        
        # 安装事件过滤器确保鼠标事件被传递
        self.installEventFilter(self)
    
    def _cover_virtual_desktop(self):
        """覆盖整个虚拟桌面"""
        desktop = QApplication.desktop()
        rect = desktop.geometry()  # 整个虚拟桌面
        self.setGeometry(rect)
        self.show()

    def _make_uia_invisible(self):
        """
        使用win32 API设置窗口样式，让窗口对UIA不可见
        这样UIAutomation就能检测到下面的应用UI元素，而不是透明交互层
        """
        try:
            import win32con
            import win32gui
            
            hwnd = self.winId()
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            style |= (
                win32con.WS_EX_LAYERED |
                win32con.WS_EX_TRANSPARENT |  # 输入与命中都透明
                win32con.WS_EX_TOOLWINDOW |   # 不计入正常窗口层级
                win32con.WS_EX_NOACTIVATE     # UIA 会忽略它
            )
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
        except Exception as e:
            print(f"[DEBUG] 设置UIA不可见失败: {e}")
        
    def update_gaze_point(self, gaze_point):
        """更新注视点位置"""
        self.current_gaze_point = gaze_point
        # 实时更新显示
        self.update()
        
    def paintEvent(self, event):
        """绘制交互区域和注视点"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制UI矩形框
        self.draw_ui_rectangles(painter)
        
        # 更新并绘制所有渐变圆圈
        self.update_fade_circles(painter)
        
        # 实时显示注视点（根据show_gaze_point标志位控制）
        if self.show_gaze_point and self.current_gaze_point:
            gaze_x, gaze_y = self.current_gaze_point
            
            # 转换绝对坐标为相对于当前覆盖层的坐标
            overlay_x = self.geometry().x()
            overlay_y = self.geometry().y()
            rel_gaze_x = gaze_x - overlay_x
            rel_gaze_y = gaze_y - overlay_y
            
            # 只有当注视点在当前覆盖层范围内时才显示
            if 0 <= rel_gaze_x < self.geometry().width() and 0 <= rel_gaze_y < self.geometry().height():
                # 外圈（白色边框）
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
                painter.drawEllipse(QPoint(int(rel_gaze_x), int(rel_gaze_y)), 8, 8)
                
                # 内圈（红色填充）
                painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
                painter.setPen(QPen(QColor(255, 0, 0, 180), 1))
                painter.drawEllipse(QPoint(int(rel_gaze_x), int(rel_gaze_y)), 5, 5)
        
        # 绘制绿色圆圈交互区域
        if self.interaction_zone:
            x, y = self.interaction_zone
            
            # 转换绝对坐标为相对于当前覆盖层的坐标
            overlay_x = self.geometry().x()
            overlay_y = self.geometry().y()
            rel_x = x - overlay_x
            rel_y = y - overlay_y
            
            # 只有当交互区域在当前覆盖层范围内时才显示
            if 0 <= rel_x < self.geometry().width() and 0 <= rel_y < self.geometry().height():
                radius = 95
                
                # 绘制实心圆圈（减少闪烁）
                painter.setBrush(QBrush(QColor(50, 205, 50, 60)))  # 半透明填充
                painter.setPen(QPen(QColor(50, 205, 50, 180), 3))  # 边框
                painter.drawEllipse(QPoint(int(rel_x), int(rel_y)), radius, radius)
                
                # 绘制内层小圆圈（装饰性）
                painter.setBrush(QBrush(QColor(50, 205, 50, 30)))  # 更透明
                painter.setPen(QPen(QColor(50, 205, 50, 100), 1))  # 细边框
                painter.drawEllipse(QPoint(int(rel_x), int(rel_y)), radius//2, radius//2)
        
    def update_fade_circles(self, painter):
        """更新并绘制所有渐变圆圈"""
        # 更新现有圆圈状态
        for circle in self.fade_circles[:]:
            if circle.update():
                # 绘制渐变圆圈
                opacity = circle.get_opacity()
                painter.setBrush(QBrush(QColor(0, 255, 0, opacity // 2)))  # 绿色半透明
                painter.setPen(QPen(QColor(0, 255, 0, opacity), 2))
                painter.drawEllipse(QPoint(int(circle.x), int(circle.y)), circle.radius, circle.radius)
            else:
                # 动画完成，移除圆圈
                self.fade_circles.remove(circle)
        
        # 如果有渐变圆圈活动，定期刷新鼠标穿透属性
        if self.fade_circles:
            QTimer.singleShot(100, lambda: self.setAttribute(Qt.WA_TransparentForMouseEvents))
                
    def add_fade_circle(self, x, y, radius=100, duration=1500):
        """添加新的渐变圆圈"""
        circle = FadeOutCircle(x, y, radius, duration)
        circle.start()
        self.fade_circles.append(circle)
        # 确保鼠标穿透属性始终有效
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.update()  # 重绘界面
    
    def eventFilter(self, obj, event):
        """事件过滤器 - 确保鼠标事件被正确传递"""
        if event.type() in [QEvent.MouseButtonPress, QEvent.MouseButtonRelease, 
                           QEvent.MouseButtonDblClick, QEvent.MouseMove]:
            # 忽略所有鼠标事件，让它们传递给下层窗口
            return True
        return super().eventFilter(obj, event)
    
    def updatePosition(self):
        """防止闪烁的更新方法"""
        # 不调用update()，避免闪烁
        pass
    
    def update_ui_controls(self, ui_controls):
        """
        更新UI控件列表
        
        Args:
            ui_controls: UI控件矩形列表，每个元素为 (left, top, width, height) 绝对坐标
        """
        # 清空现有列表
        self.ui_rectangles.clear()
        
        # 添加新的UI控件矩形
        for rect in ui_controls:
            self.add_ui_rectangle(rect)
        
        # 重绘界面
        self.update()