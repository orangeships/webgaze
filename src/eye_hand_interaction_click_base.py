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
# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel

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
        from PyQt5.QtCore import QTimer
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
            
            # print(f"检测到 {monitor_count} 个显示器")
            
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
                # print(f"显示器 {i}: 位置({monitor_geometry.x()}, {monitor_geometry.y()}) 尺寸: {monitor_geometry.width()}x{monitor_geometry.height()}")
            
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
            # print(f"检测显示器信息时出错: {e}")
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
                monitors_text += f"\n{status}: {monitor['width']}x{monitor['height']} (位置: {monitor['x']}, {monitor['y']})"
            
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
        self.setup_overlay()
        
    def setup_overlay(self):
        """设置覆盖层"""
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        
        # 设置窗口属性
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_X11DoNotAcceptFocus)  # 不接受焦点
        self.setAttribute(Qt.WA_X11NetWmWindowTypeDesktop)  # 桌面窗口类型
        
        # 设置全屏大小
        self.setGeometry(screen_geometry)
        
        # 确保窗口不会拦截任何事件
        self.setMouseTracking(False)
        
        # 安装事件过滤器确保鼠标事件被传递
        self.installEventFilter(self)
        
    def update_gaze_point(self, gaze_point):
        """更新注视点位置"""
        self.current_gaze_point = gaze_point
        # 实时更新显示
        self.update()
        
    def paintEvent(self, event):
        """绘制交互区域和注视点"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
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
    
    def __init__(self, frame_count=6, angle_threshold=3.0, pixel_threshold=100):
        self.frame_count = frame_count  # 使用帧数而非时间窗口
        self.angle_threshold = angle_threshold
        self.pixel_threshold = pixel_threshold
        self.gaze_points = deque(maxlen=frame_count)  # 存储 (timestamp, x, y) 元组，限制为最近frame_count帧
        self.last_trigger_time = 0
        self.trigger_cooldown = 1000  # 触发冷却时间1000ms
    
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

class EyeHandInteractionSystem:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.ui = None
        self.model = None
        self.homtrans = None
        self.cap = None
        self.calibration_data = None
        
        # 双屏支持相关
        self.calibration_results = {}  # 存储所有显示器的校准结果
        self.current_monitor_index = 0
        self.calibration_mode = "single"  # 校准模式：single 或 dual
        
        # 跨屏坐标转换支持
        self.screen_boundaries = []  # 存储每个屏幕的边界信息 [left, right, top, bottom]
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
            'boundary_buffer': 20,           # 屏幕边界缓冲区（像素）
            'interaction_zone_duration': 2000  # 交互区域显示持续时间（毫秒）
        }
        
        # 初始化注视点分析器，使用统一配置
        self.dispersion_analyzer = GazeDispersionAnalyzer(
            frame_count=self.threshold_config['gaze_dispersion_frames'],
            angle_threshold=self.threshold_config['angle_threshold'],
            pixel_threshold=self.threshold_config['pixel_threshold']
        )
        
        # 当前交互状态
        self.current_interaction_zone = None
        self.previous_interaction_zone = None
        self.interaction_zone_start_time = 0
        
        # 手眼协调机制相关状态变量
        self.hand_eye_coordination_enabled = True  # 手眼协调机制总开关（鼠标右键触发）
        
        # 鼠标自动移动相关 - 基于鼠标右键检测
        self.last_auto_mouse_move_time = 0  # 上次自动鼠标移动的时间
        
        # 注视点平滑相关
        self.smoothing_enabled = True  # 平滑开关
        # 使用轻量卡尔曼滤波器替代原来的平滑机制
        self.kalman_filter = LightweightKalmanFilter(
            process_noise=0.8,       # 过程噪声，进一步提高响应速度
            measurement_noise=0.4,   # 测量噪声，更信任新测量值，平滑更轻微
            error_estimate=50.0 
        )
        
        # Dwell状态检测相关 - 用于右键触发前的注视稳定性检查
        self.dwell_gaze_history = deque(maxlen=5)  # 最近5帧用于dwell检测
        self.dwell_threshold = 200  # Dwell稳定性阈值（像素）
        self.dwell_enabled = True  # Dwell检测开关
        
        # 注视点显示控制
        self.show_gaze_point = True  # 控制是否显示注视点（红点）
    
    
    
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
    
    def _load_all_calibration_results(self):
        """加载所有显示器的校准结果"""

        
        results_dir = os.path.join(self.project_dir, "results")
        
        # 初始化校准结果存储
        if not hasattr(self, 'calibration_results'):
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
        
        # 关键修复：加载完成后，将主屏幕的校准结果应用到self.homtrans实例
        primary_monitor = self.ui.monitors_info[self.ui.primary_monitor_index]
        for monitor_index, result in self.calibration_results.items():
            if result['x'] == primary_monitor['x'] and result['y'] == primary_monitor['y']:
                # 更新self.homtrans实例为当前主显示器的校准结果
                self.homtrans = result['homtrans']
                self.homtrans.width = result['width']
                self.homtrans.height = result['height']
                print(f"已将主显示器的校准结果应用到self.homtrans实例")
                break
    
    def save_calibration_results(self, monitor_index):
        """保存特定显示器的校准结果 - 使用正确的文件名参数"""
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
    
    def _apply_current_monitor_calibration(self):
        """应用当前显示器的校准数据"""
        current_monitor_index = self.ui.current_monitor_index
        
        # 先尝试从已有的校准结果中应用
        if (current_monitor_index in self.calibration_results and 
            self.calibration_results[current_monitor_index] is not None):
            calibration_data = self.calibration_results[current_monitor_index]
            
            # 应用校准数据
            if hasattr(self.homtrans, 'STransG') and 'STransG' in calibration_data:
                self.homtrans.STransG = np.array(calibration_data['STransG'])
            
            if hasattr(self.homtrans, 'scaleWtG') and 'scaleWtG' in calibration_data:
                self.homtrans.scaleWtG = calibration_data['scaleWtG']
            
            if hasattr(self.homtrans, 'scaleWtG2') and 'scaleWtG2' in calibration_data:
                self.homtrans.scaleWtG2 = calibration_data['scaleWtG2']
            
            # 应用其他可能的校准参数
            for key, value in calibration_data.items():
                if hasattr(self.homtrans, key) and key != 'STransG':
                    setattr(self.homtrans, key, value)
            
            self.calibration_data = self.homtrans.STransG
            print(f"已应用显示器 {current_monitor_index+1} 的校准数据")
        else:
            # 如果没有现有的校准数据，尝试从文件加载
            calibration_file = os.path.join(self.project_dir, "results", 
                                          f"calibration_results_screen_{current_monitor_index}.json")
            
            if os.path.exists(calibration_file):
                try:
                    with open(calibration_file, 'r', encoding='utf-8') as f:
                        calibration_data = json.load(f)
                    
                    self.calibration_results[current_monitor_index] = calibration_data
                    
                    # 应用校准数据
                    if 'STransG' in calibration_data:
                        self.homtrans.STransG = np.array(calibration_data['STransG'])
                    
                    if 'scaleWtG' in calibration_data:
                        self.homtrans.scaleWtG = calibration_data['scaleWtG']
                    
                    if 'scaleWtG2' in calibration_data:
                        self.homtrans.scaleWtG2 = calibration_data['scaleWtG2']
                    
                    # 应用其他可能的校准参数
                    for key, value in calibration_data.items():
                        if hasattr(self.homtrans, key) and key != 'STransG':
                            setattr(self.homtrans, key, value)
                    
                    self.calibration_data = self.homtrans.STransG
                    print(f"已从文件应用显示器 {current_monitor_index+1} 的校准数据")
                except Exception as e:
                    print(f"加载显示器 {current_monitor_index+1} 校准数据失败: {e}")
                    self.calibration_results[current_monitor_index] = None
            else:
                print(f"显示器 {current_monitor_index+1} 没有可用的校准文件")
    
    def switch_to_monitor(self, monitor_index):
        """切换到指定显示器并加载对应的校准数据"""
        if self.ui.switch_to_monitor(monitor_index):
            self.current_monitor_index = monitor_index
            
            # 更新屏幕尺寸用于角度计算
            self.dispersion_analyzer.set_screen_dimensions(self.ui.screen_width, self.ui.screen_height)
            
            # 加载目标显示器的校准数据
            self._apply_current_monitor_calibration()
            
            print(f"已切换到显示器 {monitor_index+1}")
            return True
        return False
    
    def show_menu(self):
        """显示主菜单"""
        # 首先选择交互模式
        single_button, multi_button = self.ui.show_interaction_mode_selection()
        
        # 等待用户操作
        from PyQt5.QtCore import QEventLoop
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
        from PyQt5.QtCore import QEventLoop
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
        if self.check_escape_key():
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
                if self.check_escape_key():
                    return False
                STransG = self.homtrans.calibrate(self.model, self.cap, sfm=True)
                if STransG is not None:
                    self.calibration_data = STransG
                    return True
                else:
                    return False
        except Exception:
            return False
    
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
                if hasattr(self, 'is_dual_screen_mode'):
                    self.is_dual_screen_mode = False
                if hasattr(self, 'current_monitor_index'):
                    self.current_monitor_index = 0  # 强制设为0
                
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
                # 多屏模式：自动或手动选择显示器
                if not hasattr(self, 'is_dual_screen_mode') or not self.is_dual_screen_mode:
                    print("警告：多屏模式调用但系统未设置为多屏模式")
                    return False
                    
                if auto_select:
                    # 自动选择最佳显示器（优先选择主显示器0，如果没有则选择第一个可用的）
                    monitor_choice = self.auto_select_best_monitor()
                    if monitor_choice is None:
                        print("没有找到可用的校准数据，自动加载失败")
                        return False
                    print(f"自动选择显示器 {monitor_choice}")
                else:
                    # 让用户选择显示器
                    monitor_choice = self.show_monitor_selection()
                    if monitor_choice is None:
                        print("用户取消加载校准数据")
                        return False
                
                current_monitor_index = monitor_choice
                calibration_file = os.path.join(self.project_dir, "results", 
                                              f"calibration_results_screen_{current_monitor_index}.json")
                
                # 检查校准文件是否存在
                if os.path.exists(calibration_file):
                    print(f"正在加载校准文件: {calibration_file}")
                    
                    # 多屏模式：先切换到目标显示器
                    if self.switch_to_monitor(current_monitor_index):
                        # 使用 homtransform 内置的方法来正确加载校准数据
                        if self.homtrans.load_calibration_results(calibration_file):
                            self.calibration_data = self.homtrans.STransG
                            print(f"✓ 显示器 {current_monitor_index} 校准数据加载成功！")
                            return True
                        else:
                            print(f"✗ 显示器 {current_monitor_index} 校准数据加载失败")
                            return False
                    else:
                        print(f"✗ 切换到显示器 {current_monitor_index} 失败")
                        return False
                else:
                    print(f"显示器 {current_monitor_index} 的校准文件不存在: {calibration_file}")
                    print("将进行新校准")
                    return False
                
        except Exception as e:
            print(f"加载校准数据时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False

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
        if not hasattr(self, 'calibration_results') or not self.calibration_results:
            print("正在扫描可用的校准文件...")
            self._load_all_calibration_results()
        
        # 策略1：优先选择主显示器0
        if hasattr(self, 'calibration_results') and self.calibration_results:
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

    def show_monitor_selection(self):
        """显示显示器选择界面（用于多屏模式加载校准数据）"""
        # 如果校准结果为空，先尝试扫描可用的校准文件
        if not hasattr(self, 'calibration_results') or not self.calibration_results:
            print("正在扫描可用的校准文件...")
            self._load_all_calibration_results()
            
        if not hasattr(self, 'calibration_results') or not self.calibration_results:
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
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
            
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
             from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
        
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
        for monitor_index in sorted(self.calibration_results.keys()):
            monitor_data = self.calibration_results[monitor_index]
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

    def perform_dual_screen_calibration(self):
        """执行双屏幕校准 - 移植正确的多屏校准方法"""
        print("执行双屏幕校准...")
        
        if len(self.ui.monitors_info) < 2:
            print("警告：只检测到1个显示器，无法进行双屏幕校准")
            return self.perform_single_screen_calibration()
        
        try:
            # 关键修复：为每个显示器创建独立的校准结果存储
            self.calibration_results = {}
            monitors = self.ui.monitors_info
            
            for i, monitor in enumerate(monitors):
                print(f"\n开始校准显示器 {i}: {monitor['width']}x{monitor['height']} (位置: {monitor['x']}, {monitor['y']})")
                
                # 关键修复：为当前显示器创建独立的PygameUI和HomTransform实例
                # 参考main_pygame_dual_screen.py的正确实现
                
                # 创建针对当前显示器的PygameUI实例 - 从正确位置导入
                from main_pygame_dual_screen import PygameUI
                current_ui = PygameUI(width=monitor['width'], height=monitor['height'], display_index=i)
                current_ui.initialize_display()
                
                # 创建针对当前显示器的HomTransform实例，传入当前显示器的分辨率
                from gaze_tracking.homtransform import HomTransform
                current_homtrans = HomTransform(self.project_dir, custom_width=monitor['width'], custom_height=monitor['height'])
                
                print(f"模型预热中，请等待...")
                
                # 生成针对当前显示器的校准文件名
                calibration_filename = f"calibration_results_screen_{i}.json"
                
                # 校准当前显示器
                STransG = current_homtrans.calibrate(self.model, self.cap, sfm=True, filename=calibration_filename)
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
                if self.check_escape_key():
                    return False
            
            # 校准完成后，恢复主显示器的UI
            self.ui.initialize_display()
            print("\n双屏幕校准全部完成！")
            
            # 初始化校准结果加载
            self._load_all_calibration_results()
            
            # 关键修复：将主屏幕的校准结果应用到self.homtrans实例
            # 确保进入交互模式时使用正确的校准数据
            primary_monitor = self.ui.monitors_info[self.ui.primary_monitor_index]
            for monitor_index, result in self.calibration_results.items():
                if result['x'] == primary_monitor['x'] and result['y'] == primary_monitor['y']:
                    # 更新self.homtrans实例为当前主显示器的校准结果
                    self.homtrans = result['homtrans']
                    self.homtrans.width = result['width']
                    self.homtrans.height = result['height']
                    print(f"已将主显示器的校准结果应用到self.homtrans实例")
                    break
            
            return True
                
        except Exception as e:
            print(f"双屏幕校准过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            # 重新初始化显示
            self.ui.initialize_display()
            return False
    
    def run_interaction_mode(self):
        """运行交互模式 - 支持多屏幕环境"""
        
        # 根据交互模式设置显示模式
        is_multi_mode = (hasattr(self, 'interaction_mode') and self.interaction_mode == 'multi')
        
        # 多屏模式下仍允许显示交互覆盖层（用于渐变圆圈）
        if is_multi_mode:
            print("=== 多屏模式：输出注视点坐标，同时支持渐变圆圈展示 ===")
            show_ui = True  # 允许创建交互覆盖层
            output_interval = 1.0  # 每秒输出一次
        else:
            show_ui = True
            output_interval = 1.0  # 单屏模式：每秒输出一次
        
        # 初始化屏幕边界信息
        self.screen_boundaries = []
        for monitor in self.ui.monitors_info:
            left = monitor['x']
            right = monitor['x'] + monitor['width']
            top = monitor['y']
            bottom = monitor['y'] + monitor['height']
            self.screen_boundaries.append([left, right, top, bottom])
        
        # 用于SfM的前一帧
        frame_prev = None
        current_gaze_point_rel = None  # 当前屏幕相对坐标注视点
        current_gaze_point_abs = None  # 绝对坐标注视点
        previous_gaze_point = None  # 用于快速移动检测
        
        # 创建定时器用于更新界面（仅UI模式）
        if show_ui:
            from PyQt5.QtCore import QTimer
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
                        try:
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
                        except Exception:
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                            self.homtrans.sfm.clear_caches()
                    else:
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 使用main_pygame_dual_screen.py的注视点计算逻辑
                    if FSgaze is not None and len(FSgaze) >= 2:
                        # 将毫米坐标转换为像素坐标
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        
                        # 计算绝对屏幕坐标（考虑显示器位置偏移）
                        gaze_x_abs = int(screen_pos_px[0] + self.ui.monitors_info[self.current_monitor_index]['x'])
                        gaze_y_abs = int(screen_pos_px[1] + self.ui.monitors_info[self.current_monitor_index]['y'])
                        
                        # 计算相对当前屏幕的坐标
                        gaze_x_rel = max(0, min(screen_pos_px[0], self.ui.screen_width))
                        gaze_y_rel = max(0, min(screen_pos_px[1], self.ui.screen_height))
                    else:
                        gaze_x_abs = self.ui.monitors_info[self.current_monitor_index]['x'] + self.ui.screen_width // 2
                        gaze_y_abs = self.ui.monitors_info[self.current_monitor_index]['y'] + self.ui.screen_height // 2
                        gaze_x_rel = self.ui.screen_width // 2
                        gaze_y_rel = self.ui.screen_height // 2

                    # 应用卡尔曼滤波平滑算法 - 使用相对坐标进行平滑
                    raw_gaze_point = (gaze_x_rel, gaze_y_rel)

                    # 使用轻量卡尔曼滤波算法
                    if self.smoothing_enabled:
                        gaze_point_rel = self._smooth_gaze_point(raw_gaze_point)
                    else:
                        gaze_point_rel = raw_gaze_point

                    current_gaze_point_rel = gaze_point_rel

                    # 使用平滑后的相对坐标重新计算绝对坐标
                    smoothed_gaze_x_abs = int(gaze_point_rel[0] + self.ui.monitors_info[self.current_monitor_index]['x'])
                    smoothed_gaze_y_abs = int(gaze_point_rel[1] + self.ui.monitors_info[self.current_monitor_index]['y'])
                    current_gaze_point_abs = (smoothed_gaze_x_abs, smoothed_gaze_y_abs)
                    self.last_gaze_point_abs = current_gaze_point_abs

                    # 实时监测注视点位置，实现屏幕自动切换（仅在多屏模式下启用）
                    if (not self.screen_switching and len(self.ui.monitors_info) > 1 and
                        hasattr(self, 'is_dual_screen_mode') and self.is_dual_screen_mode):
                        # 检查注视点是否超出当前屏幕边界
                        if self.is_gaze_out_of_screen(smoothed_gaze_x_abs, smoothed_gaze_y_abs, self.current_monitor_index):
                            # 获取目标屏幕索引
                            target_screen = self.get_target_screen(smoothed_gaze_x_abs, smoothed_gaze_y_abs)

                            # 切换到目标屏幕
                            if target_screen != self.current_monitor_index:
                                self.switch_to_monitor_with_coordinate_update(target_screen)

                                # 重新计算相对坐标
                                gaze_x_rel, gaze_y_rel = self.convert_abs_to_rel_coordinate(
                                    smoothed_gaze_x_abs, smoothed_gaze_y_abs, target_screen)
                    
                    # 手眼协调机制处理（使用相对坐标）
                    self._process_hand_eye_coordination(gaze_point_rel, previous_gaze_point)
                    
                    # 更新前一注视点
                    previous_gaze_point = current_gaze_point_rel
                    
                    # 添加到注视点分析器（使用相对坐标）
                    self.dispersion_analyzer.add_gaze_point(gaze_point_rel[0], gaze_point_rel[1])
                    
                    # 检查触发条件
                    triggered, center_point = self.dispersion_analyzer.check_trigger_conditions()
                        
                except Exception:
                    pass
            

            
            # 获取离散度信息
            dispersion_info = self.dispersion_analyzer.calculate_dispersion()
            
            # 更新交互界面（用于渐变圆圈）
            if show_ui:
                # 单屏模式：输出坐标信息
                import time
                current_time = time.time()
                if current_time - last_output_time >= output_interval:
                    if current_gaze_point_abs:
                        # 获取当前显示器信息
                        current_monitor = self.ui.monitors_info[self.current_monitor_index]
                        
                        # 计算相对于当前屏幕的坐标
                        rel_x = current_gaze_point_abs[0] - current_monitor['x']
                        rel_y = current_gaze_point_abs[1] - current_monitor['y']
                        
                        # 确保坐标在屏幕边界内
                        rel_x = max(0, min(rel_x, current_monitor['width']))
                        rel_y = max(0, min(rel_y, current_monitor['height']))
                        
                        # 输出当前注视点位置
                        print(f"[单屏模式] 当前注视点 - 屏幕{self.current_monitor_index}: 绝对坐标({current_gaze_point_abs[0]:.0f}, {current_gaze_point_abs[1]:.0f}), 相对坐标({rel_x:.0f}, {rel_y:.0f})")
                    else:
                        print(f"[单屏模式] 当前注视点 - 屏幕{self.current_monitor_index}: 正在检测中...")
                    last_output_time = current_time

                # 对于当前屏幕，使用相对坐标；对于其他屏幕，使用绝对坐标
                if self.current_interaction_zone or self.previous_interaction_zone or current_gaze_point_abs:
                    # 显示交互界面
                    self.ui.show_interaction_screen(
                        interaction_zone=self.current_interaction_zone,
                        current_gaze_point=current_gaze_point_abs,  # 使用绝对坐标，以便在所有屏幕上显示
                        show_gaze_point=self.show_gaze_point  # 传递注视点显示标志位
                    )
                self.previous_interaction_zone = self.current_interaction_zone

                # 处理Qt事件
                QApplication.processEvents()
            else:
                # 多屏模式：使用main_pygame_dual_screen.py的注视点计算逻辑
                import time
                current_time = time.time()
                if current_time - last_output_time >= output_interval:
                    if current_gaze_point_abs:
                        # 重新计算当前注视点实际所在的屏幕
                        abs_x, abs_y = current_gaze_point_abs
                        actual_screen_index = self.get_target_screen(abs_x, abs_y)
                        
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
                        print(f"[多屏模式] 当前注视点 - 屏幕{self.current_monitor_index}: 正在检测中...")
                    last_output_time = current_time
            
            # 检测ESC键退出
            if self.check_escape_key():
                self.running = False
                break
            
            # 检测空格键切换注视点显示
            if self.check_space_key():
                self.show_gaze_point = not self.show_gaze_point
                # 延迟一小段时间，避免连续触发
                import time
                time.sleep(0.2)
            
            # 更新前一帧
            frame_prev = frame.copy()
            

        # 清理资源
        if timer:
            timer.stop()
        if show_ui:
            self.ui.close_current_widget()
        
        print("=== 多屏模式或单屏模式交互结束 ===")
    
    def update_interaction_frame(self):
        """更新交互帧（由定时器调用）"""
        pass  # 界面更新在run_interaction_mode中处理
    
    def _process_hand_eye_coordination(self, gaze_point, previous_gaze_point):
        """处理手眼协调机制 - 基于鼠标右键检测，支持多屏距离计算和自动屏幕切换"""
        if not self.hand_eye_coordination_enabled or not self.ui or self.screen_switching:
            return
        
        # 检测鼠标右键是否被按下
        if self.check_right_mouse_button():
            # 检查dwell状态：连续5帧注视点分布不超过150像素
            if not self._check_gaze_dwell_state():
                return  # 如果不满足dwell条件，则不执行后续操作
            # 获取当前鼠标位置
            try:
                current_cursor_pos = win32api.GetCursorPos()
                cursor_x, cursor_y = current_cursor_pos
                
                # 多屏环境下的距离计算（仅在真正的多屏模式下）
                if (len(self.ui.monitors_info) > 1 and hasattr(self, 'is_dual_screen_mode') and 
                    self.is_dual_screen_mode):
                    # 计算注视点的绝对坐标
                    gaze_x_abs, gaze_y_abs = self.convert_rel_to_abs_coordinate(
                        gaze_point[0], gaze_point[1], self.current_monitor_index)
                    
                    # 使用跨屏距离计算（传入坐标对而不是单独的x,y值）
                    gaze_distance = self.calculate_cross_screen_distance(
                        (gaze_x_abs, gaze_y_abs), (cursor_x, cursor_y))
                    
                    # 确定目标屏幕
                    target_screen = self.get_target_screen(gaze_x_abs, gaze_y_abs)
                    
                    # 跨屏模式下的阈值需要适当调整，考虑更大距离触发
                    distance_threshold = 600 * self.threshold_config['distance_multiplier']
                    
                    # 只有当距离大于阈值时才执行操作
                    if gaze_distance > distance_threshold:
                        # 如果需要切换屏幕
                        if target_screen != self.current_monitor_index:
                            # 切换到目标屏幕
                            switching_success = self.switch_to_monitor_with_coordinate_update(target_screen)
                            
                            # 如果切换成功，切换后立即移动鼠标
                            if switching_success:
                                # 切换屏幕后，直接使用绝对坐标移动鼠标
                                self._auto_move_mouse_to_gaze(gaze_x_abs, gaze_y_abs, use_abs_coords=True)
                        else:
                            # 如果在同一屏幕内，直接移动鼠标
                            self._auto_move_mouse_to_gaze(gaze_point[0], gaze_point[1], use_abs_coords=False)
                        
                else:
                    # 单屏环境下的传统距离计算（无钩子，钩子仅在双屏模式启用）
                    gaze_distance = np.sqrt((gaze_point[0] - cursor_x)**2 + (gaze_point[1] - cursor_y)**2)
                    
                    # 如果距离大于阈值，则移动鼠标到注视点
                    distance_threshold = 400 * self.threshold_config['distance_multiplier']
                    if gaze_distance > distance_threshold:
                        self._auto_move_mouse_to_gaze(gaze_point[0], gaze_point[1], use_abs_coords=False)
                    
            except Exception:
                pass

    def _smooth_gaze_point(self, raw_gaze_point):
        """使用卡尔曼滤波平滑注视点，在屏幕切换适应期内禁用平滑
        
        Args:
            raw_gaze_point: 原始注视点 (x, y)
            
        Returns:
            平滑后的注视点 (x, y)
        """
        
        # 检查是否处于屏幕切换适应期
        if (hasattr(self, 'screen_switching_adaptation_frames') and 
            self.screen_switching_adaptation_frames > 0):
            # 在适应期内，返回原始坐标以避免卡尔曼滤波的拖拽效应
            self.screen_switching_adaptation_frames -= 1
            smoothed_point = raw_gaze_point
        else:
            # 使用卡尔曼滤波器平滑注视点
            smoothed_point = self.kalman_filter.update(raw_gaze_point)
        
        # 添加到dwell状态历史记录
        if self.dwell_enabled:
            self.dwell_gaze_history.append(smoothed_point)
        
        return smoothed_point
    
    def _auto_move_mouse_to_gaze(self, x, y, use_abs_coords=False):
        """自动移动鼠标到指定的视线位置，支持多屏环境"""
        current_time = time.time() * 1000
        
        # 检查冷却时间
        if current_time - self.last_auto_mouse_move_time < self.threshold_config['auto_move_cooldown']:
            return
        
        try:
            # 多屏环境下的坐标处理
            if use_abs_coords:
                # 直接使用绝对坐标
                target_x, target_y = x, y
            else:
                # 将相对坐标转换为绝对坐标
                target_x, target_y = self.convert_rel_to_abs_coordinate(x, y, self.current_monitor_index)
            
            # 获取当前鼠标位置
            current_cursor_pos = win32api.GetCursorPos()
            cursor_x, cursor_y = current_cursor_pos
            
            # 计算移动距离
            distance = np.sqrt((target_x - cursor_x)**2 + (target_y - cursor_y)**2)
            
            # 只有当距离超过阈值时才移动
            if distance > self.threshold_config['auto_move_distance']:
                # 移动鼠标到视线中心点（绝对坐标）
                win32api.SetCursorPos((int(target_x), int(target_y)))
                self.last_auto_mouse_move_time = current_time
                
                # 在目标位置添加渐变圆圈效果（使用显示坐标）
                display_x, display_y = target_x, target_y
                
                # 找到目标位置所在的显示器
                target_monitor_index = None
                for i, monitor in enumerate(self.ui.monitors_info):
                    if monitor['x'] <= target_x < monitor['x'] + monitor['width'] and \
                       monitor['y'] <= target_y < monitor['y'] + monitor['height']:
                        target_monitor_index = i
                        break
                
                if target_monitor_index is not None:
                    # 计算相对于目标显示器的坐标
                    monitor = self.ui.monitors_info[target_monitor_index]
                    relative_x = target_x - monitor['x']
                    relative_y = target_y - monitor['y']
                    
                    # 在双屏模式下，使用interaction_overlays
                    if self.ui.interaction_overlays and len(self.ui.interaction_overlays) > target_monitor_index:
                        self.ui.interaction_overlays[target_monitor_index].add_fade_circle(relative_x, relative_y, radius=100, duration=1500)
                    # 兼容单屏模式
                    elif self.ui.current_widget and isinstance(self.ui.current_widget, InteractionOverlay):
                        self.ui.current_widget.add_fade_circle(display_x, display_y, radius=100, duration=1500)
                
        except Exception:
            pass

    def check_escape_key(self):
        """检测ESC键是否被按下"""
        try:
            # 使用win32api检测ESC键状态
            import win32con
            esc_key_state = win32api.GetAsyncKeyState(win32con.VK_ESCAPE)
            return esc_key_state & 0x8000 != 0  # 检查最高位是否为1
        except Exception:
            return False

    def check_space_key(self):
        """检测空格键是否被按下"""
        try:
            # 使用win32api检测空格键状态
            import win32con
            space_key_state = win32api.GetAsyncKeyState(win32con.VK_SPACE)
            return space_key_state & 0x8000 != 0  # 检查最高位是否为1
        except Exception:
            return False
    
    def check_right_mouse_button(self):
        """检测鼠标右键是否被按下"""
        try:
            # 使用win32api检测鼠标右键状态 (VK_RBUTTON = 0x02)
            right_button_state = win32api.GetAsyncKeyState(0x02)  # VK_RBUTTON
            return right_button_state & 0x8000 != 0  # 检查最高位是否为1
        except Exception:
            return False
    
    def _check_gaze_dwell_state(self):
        """检查注视点是否处于稳定状态（dwell状态）
        
        连续5帧滑动窗口内注视点分布不超过150像素则认为稳定
        
        Returns:
            bool: True表示注视点处于稳定状态，False表示不稳定
        """
        # 如果dwell功能未启用，直接返回True
        if not self.dwell_enabled:
            return True
            
        # 检查dwell历史记录中是否有足够的点
        if len(self.dwell_gaze_history) < 4:
            # 数据不足时，视为不稳定
            return False
            
        try:
            # 获取最近的5个注视点（不足5个则使用全部）
            recent_points = list(self.dwell_gaze_history)[-5:]
            
            # 计算注视点的中心点
            center_x = sum(p[0] for p in recent_points) / len(recent_points)
            center_y = sum(p[1] for p in recent_points) / len(recent_points)
            
            # 计算每个点到中心点的距离
            distances = []
            for point in recent_points:
                distance = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                distances.append(distance)
            
            # 计算最大距离（最远的点）
            max_distance = max(distances)
            
            # 检查是否超过dwell阈值
            if max_distance > self.dwell_threshold:
                # print(f"[DEBUG] Dwell状态检查失败: 最大距离 {max_distance:.1f}px 超过阈值 {self.dwell_threshold}px")
                return False
            else:
                # print(f"[DEBUG] Dwell状态检查通过: 最大距离 {max_distance:.1f}px 在阈值 {self.dwell_threshold}px 内")
                return True
                
        except Exception as e:
            print(f"[ERROR] Dwell状态检查出错: {e}")
            return False
    

    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'ui'):
            self.ui.close_current_widget()

    def is_gaze_out_of_screen(self, gaze_x_abs, gaze_y_abs, screen_index):
        """判断绝对坐标的注视点是否超出指定屏幕边界
        
        改进的边界判断逻辑：
        1. 使用更小的缓冲区减少误切换
        2. 在屏幕边缘区域增加连续性检查，避免瞬时抖动
        3. 对不同方向使用不同的阈值策略
        
        Args:
            gaze_x_abs: 注视点绝对x坐标
            gaze_y_abs: 注视点绝对y坐标  
            screen_index: 屏幕索引
            
        Returns:
            bool: 超出边界返回True，否则返回False
        """
        if screen_index >= len(self.screen_boundaries):
            return True
            
        left, right, top, bottom = self.screen_boundaries[screen_index]
        
        # 获取当前屏幕尺寸，用于动态调整阈值
        screen_width = right - left
        screen_height = bottom - top
        
        # 基础缓冲区：屏幕尺寸的2%，最小20像素，最大50像素
        base_buffer = max(20, min(50, int(min(screen_width, screen_height) * 0.02)))
        
        # 检查是否真的在屏幕边界外，而不是缓冲区边界外
        # 只有当注视点明显超出屏幕物理边界时才触发切换
        hard_threshold = max(self.threshold_config.get('boundary_buffer', 100), 80)
        
        # 软缓冲区：仅用于UI显示，不触发屏幕切换
        soft_buffer = base_buffer
        
        # 判断是否在屏幕边界外
        is_out_of_hard_boundary = (gaze_x_abs < left - hard_threshold or 
                                   gaze_x_abs >= right + hard_threshold or 
                                   gaze_y_abs < top - hard_threshold or 
                                   gaze_y_abs >= bottom + hard_threshold)
        
        # 额外检查：在缓冲区边缘时，检查是否在其他屏幕范围内
        if not is_out_of_hard_boundary:
            # 检查是否完全在屏幕物理边界内
            if left <= gaze_x_abs < right and top <= gaze_y_abs < bottom:
                return False  # 在屏幕内，不切换
            
            # 检查是否在其他屏幕范围内
            target_screen = self.get_target_screen(gaze_x_abs, gaze_y_abs)
            if target_screen != screen_index:
                return True  # 在其他屏幕范围内，需要切换
            
            # 在缓冲区边缘但不在其他屏幕内，返回False
            return False
        
        return True
    
    def get_target_screen(self, gaze_x_abs, gaze_y_abs):
        """根据绝对坐标的注视点位置获取目标屏幕索引
        
        Args:
            gaze_x_abs: 注视点绝对x坐标
            gaze_y_abs: 注视点绝对y坐标
            
        Returns:
            int: 目标屏幕索引，如果没有找到则动态计算最近屏幕
        """
        # 首先尝试找到完全匹配的屏幕
        for i, (left, right, top, bottom) in enumerate(self.screen_boundaries):
            if left <= gaze_x_abs <= right and top <= gaze_y_abs <= bottom:
                return i
        
        # 如果没有完全匹配，动态计算最近的屏幕
        min_distance = float('inf')
        best_screen_index = 0
        
        for i, (left, right, top, bottom) in enumerate(self.screen_boundaries):
            # 计算到屏幕中心的距离
            screen_center_x = (left + right) / 2
            screen_center_y = (top + bottom) / 2
            distance = abs(gaze_x_abs - screen_center_x) + abs(gaze_y_abs - screen_center_y)
            
            if distance < min_distance:
                min_distance = distance
                best_screen_index = i
        
        return best_screen_index
    
    def convert_abs_to_rel_coordinate(self, gaze_x_abs, gaze_y_abs, target_screen_index):
        """将绝对坐标转换为目标屏幕的相对坐标
        
        Args:
            gaze_x_abs: 绝对x坐标
            gaze_y_abs: 绝对y坐标
            target_screen_index: 目标屏幕索引
            
        Returns:
            tuple: 相对坐标 (x, y)
        """
        if target_screen_index >= len(self.ui.monitors_info):
            return gaze_x_abs, gaze_y_abs
            
        target_monitor = self.ui.monitors_info[target_screen_index]
        rel_x = gaze_x_abs - target_monitor['x']
        rel_y = gaze_y_abs - target_monitor['y']
        return rel_x, rel_y
    
    def convert_rel_to_abs_coordinate(self, gaze_x_rel, gaze_y_rel, screen_index):
        """将相对坐标转换为绝对坐标
        
        Args:
            gaze_x_rel: 相对x坐标
            gaze_y_rel: 相对y坐标
            screen_index: 屏幕索引
            
        Returns:
            tuple: 绝对坐标 (x, y)
        """
        if screen_index >= len(self.ui.monitors_info):
            return gaze_x_rel, gaze_y_rel
            
        monitor = self.ui.monitors_info[screen_index]
        abs_x = gaze_x_rel + monitor['x']
        abs_y = gaze_y_rel + monitor['y']
        return abs_x, abs_y
    
    def calculate_cross_screen_distance(self, point1_abs, point2_abs):
        """计算两个绝对坐标点之间的距离（支持跨屏）"""
        return np.sqrt((point1_abs[0] - point2_abs[0])**2 + (point1_abs[1] - point2_abs[1])**2)
    
    def calculate_relative_distance(self, gaze_point, cursor_pos):
        """计算注视点到鼠标位置的距离，支持跨屏坐标"""
        # 将两个点都转换为绝对坐标
        gaze_abs = gaze_point
        if isinstance(gaze_point, tuple) and len(gaze_point) == 2:
            # gaze_point已经是相对坐标，需要转换为绝对坐标
            gaze_abs = self.convert_rel_to_abs_coordinate(gaze_point[0], gaze_point[1], self.current_monitor_index)
        
        # cursor_pos来自win32api已经是绝对坐标
        return self.calculate_cross_screen_distance(gaze_abs, cursor_pos)
    
    def switch_to_monitor_with_coordinate_update(self, target_screen_index):
        """切换到目标屏幕并更新相关坐标系统 - 移植正确实现（单屏模式保护）
        
        Args:
            target_screen_index: 目标屏幕索引
            
        Returns:
            bool: 切换成功返回True，否则返回False
        """
        # 单屏模式保护：绝不允许切换屏幕
        if (not hasattr(self, 'is_dual_screen_mode') or not self.is_dual_screen_mode or
            len(self.ui.monitors_info) <= 1):
            if hasattr(self, 'current_monitor_index'):
                # 确保单屏模式下始终使用屏幕0
                self.current_monitor_index = 0
            return False
            
        if target_screen_index == self.current_monitor_index or target_screen_index >= len(self.ui.monitors_info):
            return False
            
        try:
            self.screen_switching = True
            
            # 保存当前状态
            last_gaze_point = self.last_gaze_point_abs
            
            # 关键修复：使用目标显示器的物理位置来设置窗口位置
            target_monitor = self.ui.monitors_info[target_screen_index]
            window_pos_x = target_monitor['x']
            window_pos_y = target_monitor['y']
            
            # 重新初始化UI以适应当前显示器，同时确保窗口在正确位置
            self.ui.initialize_display_at_position(target_monitor['x'], target_monitor['y'])
            
            # 关键修复：创建新的HomTransform实例以匹配目标显示器的分辨率
            # 这与校准阶段的行为一致，确保视线计算使用正确的坐标系统
            new_homtrans = HomTransform(self.project_dir, custom_width=target_monitor['width'], custom_height=target_monitor['height'])
            
            # 加载并应用对应屏幕的校准结果到新的HomTransform实例
            if target_screen_index in self.calibration_results:
                calibration = self.calibration_results[target_screen_index]
                
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
                calibration_file = os.path.join(self.project_dir, "results", f"calibration_results_screen_{target_screen_index}.json")
                if os.path.exists(calibration_file):
                    new_homtrans.load_calibration_results(calibration_file)
            
            # 切换UI和HomTransform实例
            self.ui.screen_width = target_monitor['width']
            self.ui.screen_height = target_monitor['height']
            self.homtrans = new_homtrans
            
            # 关键修复：同步更新HomTransform的物理尺寸参数，确保_mm2pixel使用正确的屏幕尺寸
            self.homtrans.width = target_monitor['width']
            self.homtrans.height = target_monitor['height']
            
            # 更新当前屏幕索引
            self.current_monitor_index = target_screen_index
            
            # 更新屏幕边界信息
            self.screen_boundaries = []
            for monitor in self.ui.monitors_info:
                left = monitor['x']
                right = monitor['x'] + monitor['width']
                top = monitor['y'] 
                bottom = monitor['y'] + monitor['height']
                self.screen_boundaries.append([left, right, top, bottom])
            
            # 关键修复：重置卡尔曼滤波器状态，避免跨屏幕坐标混淆
            # 当切换屏幕时，卡尔曼滤波器的状态可能包含旧屏幕的坐标信息
            if hasattr(self, 'kalman_filter') and self.kalman_filter:
                # 重新初始化卡尔曼滤波器，清除旧状态
                self.kalman_filter = LightweightKalmanFilter(
                    process_noise=0.8,  # 增加过程噪声，加速适应新屏幕
                    measurement_noise=0.4,  # 增加测量噪声，减少拖拽效应
                    error_estimate=2.0  # 增加初始误差估计
                )
                # 设置屏幕切换标志，短期内禁用平滑以快速适应新坐标
                self.screen_switching_adaptation_frames = 5  # 屏幕切换后5帧内快速适应
                print(f"[DEBUG] 屏幕切换到{target_screen_index}，卡尔曼滤波器已重置，将快速适应新坐标")
            
            # 如果有上一帧注视点，将其转换为新屏幕的坐标
            if last_gaze_point:
                # 将绝对坐标转换为新屏幕的相对坐标
                new_gaze_x = last_gaze_point[0] - self.ui.monitors_info[target_screen_index]['x']
                new_gaze_y = last_gaze_point[1] - self.ui.monitors_info[target_screen_index]['y']
                
                # 更新离散度分析器的屏幕尺寸
                self.dispersion_analyzer.set_screen_dimensions(
                    self.ui.monitors_info[target_screen_index]['width'],
                    self.ui.monitors_info[target_screen_index]['height']
                )
            
            self.screen_switching = False
            return True
            
        except Exception as e:
            print(f"切换屏幕失败: {e}")
            import traceback
            traceback.print_exc()
            self.screen_switching = False
            return False

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 获取项目根目录（src目录的父目录）
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # 创建交互系统
        system = EyeHandInteractionSystem(project_dir)
        
        # 初始化系统
        if not system.initialize():
            print("系统初始化失败")
            return
        
        # 显示主菜单
        interaction_mode, choice = system.show_menu()
        if choice == 'quit' or not interaction_mode:
            print("用户选择退出")
            return
        
        print(f"用户选择了: {interaction_mode}交互模式")
        
        # 根据交互模式设置系统配置
        if interaction_mode == 'single':
            system.interaction_mode = 'single'
            system.is_dual_screen_mode = False  # 明确设置为单屏模式
            print("设置为单屏交互模式")
        elif interaction_mode == 'multi':
            system.interaction_mode = 'multi'
            system.is_dual_screen_mode = True  # 明确设置为多屏模式
            print("设置为多屏交互模式")
        
        # 运行校准（启用自动加载功能）
        auto_load = True  # 启用自动加载校准数据
        if not system.run_calibration(interaction_mode, auto_load=auto_load):
            print("校准失败")
            return
        
        print("校准成功，准备进入交互模式...")
        
        # 运行交互模式
        system.run_interaction_mode()
        
    except Exception as e:
        print(f"程序执行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'system' in locals():
            system.cleanup()
        app.quit()

if __name__ == '__main__':
    main()