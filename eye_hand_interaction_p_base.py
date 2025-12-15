import os
import sys
import cv2
import time
import numpy as np
from collections import deque
import math
import ctypes
import threading
from pynput import keyboard
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QMessageBox, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QThread, pyqtSlot, QEasingCurve, QVariantAnimation, QEvent
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap, QPainterPath, QRegion, QKeyEvent, QLinearGradient
from PyQt5.QtCore import QSize
import win32api
import win32con
# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel
from gaze_tracking.gaze_smoothing import KalmanFilter

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
        self.window = None
        self.app = None
        self.current_widget = None
        self.screen_width = 0
        self.screen_height = 0
        # 添加定时器用于更新渐变圆圈动画
        from PyQt5.QtCore import QTimer
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.update_fade_animation)
        self.fade_timer.start(16)  # 约60FPS
        
    def initialize_display(self):
        """初始化显示"""
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()

        # 创建透明窗口
        self.window = TransparentWindow()

        # 设置字体
        self.font_large = QFont("Microsoft YaHei", 30)
        self.font_medium = QFont("Microsoft YaHei", 20)
        self.font_small = QFont("Microsoft YaHei", 20)
        return self.window
    
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
        
        # 两个按钮
        load_button = QPushButton("加载历史校准数据")
        load_button.setFont(self.font_medium)
        load_button.setStyleSheet("""
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
        layout.addWidget(load_button)
        
        new_button = QPushButton("进行新校准")
        new_button.setFont(self.font_medium)
        new_button.setStyleSheet("""
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
        layout.addWidget(new_button)
        
        widget.setLayout(layout)
        widget.setGeometry(self.screen_width//4, self.screen_height//4, 
                          self.screen_width//2, self.screen_height//2)
        
        return widget, load_button, new_button
    
    def show_start_screen(self):
        """显示开始界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, button = self.create_start_widget()
        self.current_widget = widget
        widget.show()
        
        return button
    
    def show_calibration_choice(self):
        """显示校准选择界面"""
        if self.current_widget:
            self.current_widget.close()
        
        widget, load_button, new_button = self.create_calibration_choice_widget()
        self.current_widget = widget
        widget.show()
        
        return load_button, new_button
    
    def show_interaction_screen(self, interaction_zone=None, current_gaze_point=None, show_gaze_point_ref=None):
        """显示交互界面（完全透明，只在需要时显示绿色区域）"""
        # 只在需要时创建或更新交互区域
        if interaction_zone or current_gaze_point:
            # 如果已经存在交互区域，只更新位置而不重新创建
            if self.current_widget and isinstance(self.current_widget, InteractionOverlay):
                # 只有在位置真正改变时才更新
                if (self.current_widget.interaction_zone != interaction_zone or 
                    self.current_widget.current_gaze_point != current_gaze_point):
                    self.current_widget.interaction_zone = interaction_zone
                    self.current_widget.current_gaze_point = current_gaze_point
                    self.current_widget.show_gaze_point_ref = show_gaze_point_ref
                    # 强制重绘整个窗口，避免闪烁
                    self.current_widget.repaint()
            else:
                # 关闭当前界面
                if self.current_widget:
                    self.current_widget.close()
                    self.current_widget = None
                
                # 创建新的交互区域
                overlay = InteractionOverlay(interaction_zone, current_gaze_point, show_gaze_point_ref)
                overlay.show()
                self.current_widget = overlay
        else:
            # 没有交互区域时，关闭当前界面
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
        if self.current_widget and isinstance(self.current_widget, InteractionOverlay):
            # 更新所有渐变圆圈
            for circle in self.current_widget.fade_circles[:]:
                if circle.update():
                    pass  # 圆圈已更新
                else:
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
    
    def __init__(self, interaction_zone, current_gaze_point=None, show_gaze_point_ref=None):
        super().__init__()
        self.interaction_zone = interaction_zone
        self.current_gaze_point = current_gaze_point
        self.fade_circles = []  # 存储所有渐变圆圈
        self.show_gaze_point_ref = show_gaze_point_ref  # 引用EyeHandInteractionSystem实例的show_gaze_point属性
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
        """更新注视点位置（不再实时显示）"""
        self.current_gaze_point = gaze_point
        # 不再实时更新显示，只在鼠标移动时显示渐变圆圈
        # self.update()
        
    def paintEvent(self, event):
        """绘制交互区域和注视点"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 更新并绘制所有渐变圆圈
        self.update_fade_circles(painter)
        
        # 实时显示注视点（受显示控制变量影响）
        show_gaze_point = True  # 默认显示
        if self.show_gaze_point_ref is not None:
            show_gaze_point = self.show_gaze_point_ref()
        if self.current_gaze_point and show_gaze_point:
            gaze_x, gaze_y = self.current_gaze_point
            # 外圈（白色边框）
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
            painter.drawEllipse(QPoint(int(gaze_x), int(gaze_y)), 8, 8)
            
            # 内圈（红色填充）
            painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
            painter.setPen(QPen(QColor(255, 0, 0, 180), 1))
            painter.drawEllipse(QPoint(int(gaze_x), int(gaze_y)), 5, 5)
        
        # 绘制绿色圆圈交互区域
        if self.interaction_zone:
            x, y = self.interaction_zone
            radius = 100  # 与传送距离保持一致
            
            # 绘制实心圆圈（减少闪烁）
            painter.setBrush(QBrush(QColor(50, 205, 50, 60)))  # 半透明填充
            painter.setPen(QPen(QColor(50, 205, 50, 180), 3))  # 边框
            painter.drawEllipse(QPoint(int(x), int(y)), radius, radius)
            
            # 绘制内层小圆圈（装饰性）
            painter.setBrush(QBrush(QColor(50, 205, 50, 30)))  # 更透明
            painter.setPen(QPen(QColor(50, 205, 50, 100), 1))  # 细边框
            painter.drawEllipse(QPoint(int(x), int(y)), radius//2, radius//2)
            
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
    




class EyeHandInteractionSystem:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.ui = None
        self.model = None
        self.homtrans = None
        self.cap = None
        self.calibration_data = None
        

        
        # 当前交互状态
        self.current_interaction_zone = None
        self.previous_interaction_zone = None
        self.interaction_zone_duration = 1000  # 交互区域显示持续时间1000ms
        self.interaction_zone_start_time = 0
        
        # 手眼协调机制相关状态变量
        self.hand_eye_coordination_enabled = True  # 手眼协调机制总开关
        self.hand_eye_coordination_active = False  # 当前是否在手眼协调模式
        
        # 滑动窗口机制相关
        self.sliding_window_gaze_points = deque(maxlen=8)  # 滑动窗口，最多8个注视点
        self.sliding_window_start_time = None  # 滑动窗口开始时间
        self.sliding_window_angle_threshold = 3.0  # 角度分布阈值 4°
        self.sliding_window_time_limit = 350  # 滑动窗口时间限制 350ms
        
        # 鼠标滑动窗口检测相关
        self.mouse_movement_window = deque(maxlen=5)  # 鼠标滑动窗口，最多5个位置点
        self.mouse_movement_threshold = 200  # 鼠标移动触发阈值（像素）
        self.initial_mouse_position = None  # 鼠标初始位置
        self.teleport_circle_radius = 100  # 传送圆周半径（像素）
        
        # 传送冷却机制相关
        self.last_teleport_trigger_time = 0  # 上次传送触发时间
        self.teleport_cooldown_duration = 1000  # 传送冷却时间1000ms（1秒）
        
        # 轻量级卡尔曼滤波器初始化（替换自适应移动平均算法）
        self.kalman_filter = LightweightKalmanFilter(
            process_noise=0.6,     # 过程噪声，进一步提高响应速度
            measurement_noise=0.2, # 测量噪声，更信任新测量值，平滑更轻微
            error_estimate=50.0    # 初始误差估计，最大化初始不确定性
        )
        self.smoothing_enabled = True  # 平滑开关
        
        # 传送完成标记相关
        self.first_teleport_completed = False  # 标记第一次传送是否完成
        self.teleport_info = None  # 存储第一次传送的信息
        
        # 注视点显示控制相关
        self.show_gaze_point = True  # 是否显示注视点（红色点）
        
        # 传送后速度阻尼相关变量
        self.last_teleport_time = 0  # 上次传送的时间戳
        self.post_teleport_damping_enabled = True  # 传送后阻尼开关
        self.post_teleport_duration = 1000  # 阻尼持续时间（毫秒）
        self.damping_factor = 0.01  # 阻尼系数：0.01表示速度降低到1%
        
        # 系统级鼠标速度控制（集成test.py方法）
        self.user32 = ctypes.windll.user32
        self.SPI_SETMOUSESPEED = 0x0071
        self.SPI_GETMOUSESPEED = 0x0070
        self.SPIF_SENDCHANGE = 0x0002
        
        self.TARGET_LOW_SPEED = 2  # 目标低速度
        self.RESTORE_TARGET_SPEED = 10  # 恢复目标速度
        self.RESTORE_TIME = 1.2  # 恢复时间（秒）
        self.RESTORE_STEP_DELAY = 0.01  # 恢复步长延迟
        self.EASING_POWER = 3  # 非线性缓出指数
        
        self.restoring = False  # 恢复状态标记
        self.original_mouse_speed = self.get_mouse_speed()  # 记录原始鼠标速度
        
        # 初始化时设置快速移动距离阈值
    
        
    def initialize(self):
        """初始化系统"""
        # 初始化UI
        self.ui = EyeHandInteractionUI()
        self.ui.initialize_display()
        
        # 初始化模型
        self.model = EyeModel(self.project_dir)
        
        # 初始化HomTransform
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
        
        if self.cap is None:
            return False
            
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        return True
    
    def show_menu(self):
        """显示主菜单"""
        button = self.ui.show_start_screen()
        
        # 等待用户操作
        from PyQt5.QtCore import QEventLoop
        loop = QEventLoop()
        
        result = {'choice': None}
        
        def on_button_clicked():
            result['choice'] = 'calibrate'
            loop.quit()
        
        button.clicked.connect(on_button_clicked)
        
        # 等待用户选择
        loop.exec_()
        
        self.ui.close_current_widget()
        
        return result['choice'] if result['choice'] else 'quit'
    
    def run_calibration(self):
        """运行校准"""
        load_button, new_button = self.ui.show_calibration_choice()
        
        # 等待用户选择
        from PyQt5.QtCore import QEventLoop
        loop = QEventLoop()
        
        result = {'choice': None}
        
        def on_load_clicked():
            result['choice'] = 'load'
            loop.quit()
        
        def on_new_clicked():
            result['choice'] = 'new'
            loop.quit()
        
        load_button.clicked.connect(on_load_clicked)
        new_button.clicked.connect(on_new_clicked)
        
        # 等待用户选择
        loop.exec_()
        
        choice = result['choice']
        self.ui.close_current_widget()
        
        if choice == 'load':
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
            if os.path.exists(calibration_file):
                if self.homtrans.load_calibration_results(calibration_file):
                    self.calibration_data = self.homtrans.STransG
                    return True
                else:
                    return self.perform_new_calibration()
            else:
                return self.perform_new_calibration()
        else:
            return self.perform_new_calibration()
    
    def perform_new_calibration(self):
        """执行新校准"""
        try:
            STransG = self.homtrans.calibrate(self.model, self.cap, sfm=True)
            if STransG is not None:
                self.calibration_data = STransG
                return True
            else:
                return False
        except Exception:
            return False
    
    def run_interaction_mode(self):
        """运行交互模式"""
        
        # 用于SfM的前一帧
        frame_prev = None
        current_gaze_point = None
        previous_gaze_point = None  # 用于快速移动检测
        
        # 创建定时器用于更新界面
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(self.update_interaction_frame)
        timer.start(16)  # 约60FPS
        
        # 主循环标志
        self.running = True
        
        # 导入keyboard库用于ESC键和空格键检测
        try:
            import keyboard
        except ImportError:
            print("Warning: keyboard library not found. ESC exit may not work properly.")
            keyboard = None
        
        # 用于防止空格键重复触发的状态变量
        self.space_key_was_pressed = False
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检查ESC键退出（每几帧检查一次以提高性能）
            if keyboard and keyboard.is_pressed('esc'):
                print("检测到ESC键，程序将退出...")
                self.running = False
                break
            
            # 检查空格键切换注视点显示状态（防抖处理）
            if keyboard:
                space_key_pressed = keyboard.is_pressed('space')
                if space_key_pressed and not self.space_key_was_pressed:
                    # 空格键刚被按下，切换注视点显示状态
                    self.show_gaze_point = not self.show_gaze_point
                    status = "显示" if self.show_gaze_point else "隐藏"
                    print(f"切换注视点显示状态：{status}")
                    
                    # 强制重绘界面以立即显示变化
                    if self.ui.current_widget:
                        self.ui.current_widget.repaint()
                    
                    self.space_key_was_pressed = True
                elif not space_key_pressed:
                    # 空格键被释放，允许下次按下时触发
                    self.space_key_was_pressed = False
            
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
                    
                    # 转换为像素坐标
                    if FSgaze is not None and len(FSgaze) >= 2:
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        
                        gaze_x = max(0, min(screen_pos_px[0], self.ui.screen_width))
                        gaze_y = max(0, min(screen_pos_px[1], self.ui.screen_height))
                    else:
                        gaze_x = self.ui.screen_width // 2
                        gaze_y = self.ui.screen_height // 2
                    
                    # 应用轻量级卡尔曼滤波平滑算法
                    raw_gaze_point = (gaze_x, gaze_y)
                    
                    # 使用轻量级卡尔曼滤波算法进行平滑处理
                    if self.smoothing_enabled:
                        gaze_point = self._smooth_gaze_point(raw_gaze_point)
                    else:
                        gaze_point = raw_gaze_point
                    
                    # 添加小范围约束，避免微小抖动
                    if current_gaze_point:
                        dx = abs(gaze_point[0] - current_gaze_point[0])
                        dy = abs(gaze_point[1] - current_gaze_point[1])
                        if dx < 2 and dy < 2:  # 如果移动距离很小，保持原位
                            gaze_point = current_gaze_point
                    
                    current_gaze_point = gaze_point
                    
                    # 手眼协调机制处理（基于滑动窗口）
                    self._process_hand_eye_coordination(gaze_point)
                    
                    # 更新前一注视点
                    previous_gaze_point = current_gaze_point
                        
                except Exception:
                    pass
            
            # 更新交互界面（同时显示注视点和交互区域）
            if self.current_interaction_zone or self.previous_interaction_zone or current_gaze_point:
                # 显示交互界面，传递show_gaze_point属性的引用
                self.ui.show_interaction_screen(
                    interaction_zone=self.current_interaction_zone,
                    current_gaze_point=current_gaze_point,
                    show_gaze_point_ref=lambda: self.show_gaze_point
                )
            self.previous_interaction_zone = self.current_interaction_zone
            
            # 处理Qt事件
            QApplication.processEvents()
            
            # 更新前一帧
            frame_prev = frame.copy()
        
        timer.stop()
        self.ui.close_current_widget()
    
    def update_interaction_frame(self):
        """更新交互帧（由定时器调用）"""
        pass  # 界面更新在run_interaction_mode中处理
    
    def _process_hand_eye_coordination(self, gaze_point):
        """手眼协调处理：收集注视点并进行稳定性检测"""
        if not gaze_point:
            return
        
        current_time = time.time() * 1000  # 转换为毫秒
        
        # 持续收集注视点到滑动窗口（自动维护最多8个点）
        self.sliding_window_gaze_points.append((current_time, gaze_point[0], gaze_point[1]))
   
        # 当滑动窗口收集满8个点时进行检查
        if len(self.sliding_window_gaze_points) >= 8:
            self._check_sliding_window_distribution()
            # 检查完成后不清空，让滑动窗口继续工作（移除最老点，加入新点）
        
    def _check_sliding_window_distribution(self):
        """更稳健的滑动窗口视线稳定性判断"""
        points = np.array([(x, y) for _, x, y in self.sliding_window_gaze_points])
        # ===== 1) 使用中位数作为中心（比均值稳健得多） =====
        center = np.median(points, axis=0)

        # ===== 2) 使用绝对偏差( MAD )代替 max_distance / avg_distance =====
        # MAD 是鲁棒统计中特别常用的“抗噪”指标
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        mad = np.median(np.abs(distances - np.median(distances)))

        # ===== 3) 判断分布范围（比你现在的 max/avg 更稳定）=====
        # 推荐阈值（你可调）：
        mad_threshold = 50         # 稳定凝视：散布在半径约 50 px 范围内
        perc95_threshold = 120     # 绝大部分点不要过分发散

        perc95 = np.percentile(distances, 95)

        stable_spatial = (mad < mad_threshold) and (perc95 < perc95_threshold)

        # ===== 4) 加入时间稳定性：至少 300ms 都稳定 =====
        timestamps = [t for t, _, _ in self.sliding_window_gaze_points]
        duration = timestamps[-1] - timestamps[0]
        stable_time = duration > 0.3   # 300 ms 以上才算真正凝视

    # ======= 最终判定 =======
        if stable_spatial and stable_time:
            self._check_mouse_movement_and_trigger_cursor(center[0], center[1])

    
    def _check_mouse_movement_and_trigger_cursor(self, target_x, target_y):
        """检查鼠标移动条件并触发光标跳转（改进的方向判断方法）
        
        注意：此方法只进行逻辑判断和传送触发，实际的鼠标位置控制由_auto_move_mouse_to_gaze处理
        避免应用级阻尼与系统级速度控制产生冲突，确保传送后效果平滑
        """
        try:
            # 获取当前鼠标位置
            current_cursor_pos = win32api.GetCursorPos()
            cursor_x, cursor_y = current_cursor_pos
            
            # 获取当前时间戳（毫秒）
            current_time = time.time() * 1000
            
            # 计算当前鼠标与注视点的距离（第三个判定条件）
            gaze_cursor_distance = np.sqrt((cursor_x - target_x)**2 + (cursor_y - target_y)**2)
            
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
                
                # === 改进的方向判断方法 ===
                # 方法1: 余弦相似度（更直接的相似度判断）
                cosine_similarity = self._calculate_cosine_similarity_to_gaze(
                    move_dx, move_dy, cursor_x, cursor_y, target_x, target_y)
                
                # 距离变化方法（直接检查距离是否在减少）
                distance_change = self._calculate_distance_change(
                    first_pos[1], first_pos[2], cursor_x, cursor_y, target_x, target_y)
                
                # 综合判断：使用余弦相似度和距离变化的组合
                direction_valid = self._combined_direction_check(
                    cosine_similarity, distance_change, mouse_move_distance)
                
                # 检查是否达到所有触发条件：保持300像素最小距离要求
                if (mouse_move_distance >= self.mouse_movement_threshold and 
                    gaze_cursor_distance > 300 and  # 保持300像素最小距离要求
                    direction_valid):
                    
                    # 检查传送冷却时间
                    current_time = time.time() * 1000  # 毫秒时间戳
                    time_since_last_teleport = current_time - self.last_teleport_trigger_time
                    
                    if time_since_last_teleport >= self.teleport_cooldown_duration:
                        # 所有条件都满足且不在冷却期：执行传送
                        print(f"[INFO] 鼠标传送触发: 移动{mouse_move_distance:.0f}px, 距离{gaze_cursor_distance:.0f}px")
                        print(f"[DEBUG] 方向判断 - 余弦相似度: {cosine_similarity:.3f}, 距离变化: {distance_change:.1f}")
                        self.last_teleport_trigger_time = current_time  # 更新上次传送时间
                        self._trigger_fade_circle_cursor_move_and_reset(target_x, target_y)
            else:
                # 如果是第一次记录，不需要设置初始位置，滑动窗口会自动维护
                pass
                    
        except Exception as e:
            print(f"[DEBUG] 鼠标移动检测异常: {e}")
            pass
    
    def _calculate_cosine_similarity_to_gaze(self, move_dx, move_dy, cursor_x, cursor_y, target_x, target_y):
        """计算鼠标移动方向与指向注视点方向的余弦相似度"""
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
        """计算鼠标与注视点距离的变化（减少为正数）"""
        # 初始距离
        initial_distance = np.sqrt((first_x - target_x)**2 + (first_y - target_y)**2)
        # 当前距离
        current_distance = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
        
        # 距离变化：正数表示接近，负数表示远离
        distance_change = initial_distance - current_distance
        
        return distance_change
    
    def _combined_direction_check(self, cosine_similarity, distance_change, move_distance):
        """优化的方向判断：降低阈值减少延迟"""
        # 余弦相似度阈值：>0.3表示朝注视点方向移动（降低阈值）
        cosine_threshold = 0.4
        cosine_valid = cosine_similarity > cosine_threshold
        
        # 距离变化阈值：>20像素表示确实在接近（降低阈值）
        distance_threshold = 40.0
        distance_valid = distance_change > distance_threshold
        
        # 移动距离验证（避免微小抖动）
        min_move_threshold = 15.0
        move_valid = move_distance > min_move_threshold
        
        # 只有在有足够移动距离的情况下，才进行方向判断
        if not move_valid:
            return False
        
        # 两个条件都满足时才触发传送
        return cosine_valid and distance_valid
    
    def _calculate_circular_teleport(self, center_x, center_y, move_dx, move_dy):
        """根据鼠标移动方向计算圆周传送目标点"""
        # 归一化鼠标移动方向向量
        move_magnitude = np.sqrt(move_dx**2 + move_dy**2)
        if move_magnitude < 1e-6:  # 防止除零
            return center_x, center_y
            
        # 归一化方向向量
        normalized_dx = move_dx / move_magnitude
        normalized_dy = move_dy / move_magnitude
        
        # 计算传送位置：在相反方向（对侧）的圆周上
        teleport_x = center_x - normalized_dx * self.teleport_circle_radius
        teleport_y = center_y - normalized_dy * self.teleport_circle_radius
        
        return teleport_x, teleport_y
    
    def _trigger_fade_circle_cursor_move_and_reset(self, fade_circle_x, fade_circle_y):
        """触发传送到渐变圆圈边界的光标移动并重置所有相关状态"""
        try:
            # 保存传送前的鼠标滑动窗口数据，防止传送后窗口数据被污染
            saved_window = list(self.mouse_movement_window)
            
            # 获取当前鼠标位置作为参考点
            current_cursor_pos = win32api.GetCursorPos()
            cursor_x, cursor_y = current_cursor_pos
            
            # 使用滑动窗口第一个和最后一个位置计算移动方向和距离
            if len(self.mouse_movement_window) >= 2:
                # 使用滑动窗口第一个和最后一个位置
                first_pos = self.mouse_movement_window[0]  # (timestamp, x, y)
                last_pos = self.mouse_movement_window[-1]   # (timestamp, x, y)
                move_dx = last_pos[1] - first_pos[1]  # x坐标差值
                move_dy = last_pos[2] - first_pos[2]  # y坐标差值
                move_distance = np.sqrt(move_dx**2 + move_dy**2)  # 滑动窗口总移动距离
                
                # 滑动窗口移动信息，仅在需要时输出
                
                # 使用滑动窗口移动方向计算传送目标
                if move_distance > 1e-6:
                    # 归一化方向向量
                    normalized_dx = move_dx / move_distance
                    normalized_dy = move_dy / move_distance
                    
                    # 传送到渐变圆圈的边界上（对侧方向）
                    target_x = fade_circle_x - normalized_dx * self.teleport_circle_radius
                    target_y = fade_circle_y - normalized_dy * self.teleport_circle_radius
                else:
                    # 如果移动距离太小，使用固定偏移
                    target_x = fade_circle_x - self.teleport_circle_radius
                    target_y = fade_circle_y
            else:
                # 如果滑动窗口数据不足，使用当前鼠标位置与渐变圆圈中心计算方向
                move_dx = cursor_x - fade_circle_x
                move_dy = cursor_y - fade_circle_y
                move_magnitude = np.sqrt(move_dx**2 + move_dy**2)
                
                if move_magnitude > 1e-6:
                    normalized_dx = move_dx / move_magnitude
                    normalized_dy = move_dy / move_magnitude
                    target_x = fade_circle_x - normalized_dx * self.teleport_circle_radius
                    target_y = fade_circle_y - normalized_dy * self.teleport_circle_radius
                else:
                    target_x = fade_circle_x - self.teleport_circle_radius
                    target_y = fade_circle_y
            
            # 在传送前，在注视点中心位置显示绿色圆圈（作为视觉反馈）
            if self.ui and self.ui.current_widget and isinstance(self.ui.current_widget, InteractionOverlay):
                self.ui.current_widget.add_fade_circle(fade_circle_x, fade_circle_y, radius=self.teleport_circle_radius, duration=1500)
            
            # 如果是第一次传送，存储信息并设置标记
            if not self.first_teleport_completed:
                self.first_teleport_completed = True
                self.teleport_info = {
                    'center_x': fade_circle_x,
                    'center_y': fade_circle_y,
                    'teleport_x': target_x,
                    'teleport_y': target_y
                }
                
                # 等待一小段时间让传送完成
                time.sleep(0.1)
                
                # 获取传送前的鼠标位置和传送后的光标位置
                try:
                    final_cursor_pos = win32api.GetCursorPos()
                    # 计算位置差异
                    x_diff = final_cursor_pos[0] - cursor_x
                    y_diff = final_cursor_pos[1] - cursor_y
                except Exception as e:
                    pass
            
            # 执行光标传送到渐变圆圈边界
            self._auto_move_mouse_to_gaze(target_x, target_y)
            
            # 传送后重置鼠标滑动窗口，确保下次传送基于正确的鼠标移动数据
            self.mouse_movement_window.clear()
            # 添加传送后的鼠标位置作为新起点
            final_pos = win32api.GetCursorPos()
            current_time = time.time() * 1000
            self.mouse_movement_window.append((current_time, final_pos[0], final_pos[1]))
            
        except Exception as e:
            print(f"[DEBUG] 传送执行异常: {e}")
            pass
    

    
    def _smooth_gaze_point(self, raw_gaze_point):
        """轻量级卡尔曼滤波注视点平滑算法
        
        使用轻量级卡尔曼滤波器实现轻微平滑效果，避免过度滤波导致响应延迟
        通过优化的参数设置避免过冲现象
        
        Args:
            raw_gaze_point: 原始注视点坐标 (x, y)
            
        Returns:
            平滑后的注视点坐标 (x, y)
        """
        if not self.smoothing_enabled or not raw_gaze_point:
            return raw_gaze_point
        
        # 确保raw_gaze_point是numpy数组格式
        measurement = np.array(raw_gaze_point, dtype=np.float32)
        
        # 使用卡尔曼滤波器更新状态
        filtered_state = self.kalman_filter.update(measurement)
        
        # 获取滤波后的位置
        smoothed_position = self.kalman_filter.get_position()
        
        # 返回平滑后的位置（优化的轻微平滑效果，无过冲）
        smoothed_point = (float(smoothed_position[0]), float(smoothed_position[1]))
        
        return smoothed_point
    
    def _check_gaze_dwell_state(self):
        """检查dwell状态：检测注视点稳定性（用于右键触发前的稳定性检查）
        
        Returns:
            bool: 如果dwell检测开启且最近5帧注视点分布不超过200像素阈值，返回True
        """
        if not self.dwell_enabled:
            return True  # 如果dwell检测关闭，总是返回True
        
        # 获取最近5帧注视点（如果没有足够的历史记录，也返回True）
        if len(self.dwell_gaze_history) < 5:
            return True
        
        # 计算注视点分布范围：最大距离差
        min_x = min(gaze[0] for gaze in self.dwell_gaze_history)
        max_x = max(gaze[0] for gaze in self.dwell_gaze_history)
        min_y = min(gaze[1] for gaze in self.dwell_gaze_history)
        max_y = max(gaze[1] for gaze in self.dwell_gaze_history)
        
        # 计算水平和垂直方向的最大变化距离
        max_distance = max(max_x - min_x, max_y - min_y)
        
        # 如果最大变化距离不超过阈值，认为注视状态稳定
        return max_distance <= self.dwell_threshold
    
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
        """设置鼠标速度（使用Windows API）"""
        speed = int(max(1, min(20, speed)))
        self.user32.SystemParametersInfoW(self.SPI_SETMOUSESPEED, 0, speed, self.SPIF_SENDCHANGE)

    def get_mouse_speed(self):
        """获取当前鼠标速度"""
        v = ctypes.c_int()
        self.user32.SystemParametersInfoW(self.SPI_GETMOUSESPEED, 0, ctypes.byref(v), 0)
        return v.value

    def restore_speed_ease_out(self, start_speed):
        """非线性缓出式恢复鼠标速度"""
        self.restoring = True
        steps = int(self.RESTORE_TIME / self.RESTORE_STEP_DELAY)
        
        for i in range(steps + 1):
            if not self.restoring:
                return
            
            t = i / steps  # 0 → 1
            eased = 1 - (1 - t) ** self.EASING_POWER
            
            new_speed = start_speed + (self.RESTORE_TARGET_SPEED - start_speed) * eased
            self.set_mouse_speed(new_speed)
            time.sleep(self.RESTORE_STEP_DELAY)
        
        self.restoring = False
        self.set_mouse_speed(self.RESTORE_TARGET_SPEED)

    def _auto_move_mouse_to_gaze(self, x, y):
        """瞬间传送鼠标到指定位置，记录传送时间并应用系统级速度阻尼"""
        try:
            # 瞬间传送鼠标到目标位置
            win32api.SetCursorPos((int(x), int(y)))
            
            # 记录传送时间戳，用于触发传送后速度阻尼效果
            self.last_teleport_time = time.time() * 1000  # 毫秒时间戳
            print(f"[INFO] 鼠标传送到: ({int(x)}, {int(y)})")
            
            # 传送完成后降低鼠标速度到4（应用test.py方法）
            self.restoring = False  # 终止之前的恢复过程
            self.set_mouse_speed(self.TARGET_LOW_SPEED)
            print(f"[INFO] 鼠标速度降低到: {self.TARGET_LOW_SPEED}")
            
            # 在1秒内非线性恢复鼠标速度（应用test.py方法）
            original_speed = self.TARGET_LOW_SPEED
            
            def restore_thread():
                time.sleep(0.1)  # 稍微延迟，确保传送操作完成
                start = self.get_mouse_speed()
                self.restore_speed_ease_out(start)
                print(f"[INFO] 鼠标速度非线性恢复到: {self.RESTORE_TARGET_SPEED}")
            
            restore_thread_obj = threading.Thread(target=restore_thread, daemon=True)
            restore_thread_obj.start()
            
        except Exception as e:
            pass

    def _end_program(self):
        """结束程序"""
        print("程序即将结束...")
        # 清理资源
        self.cleanup()
        # 退出应用程序
        QApplication.quit()

    def cleanup(self):
        """清理资源"""
        # 首先停止所有速度恢复过程
        self.restoring = False
        
        # 恢复原始鼠标速度
        try:
            if hasattr(self, 'original_mouse_speed') and self.original_mouse_speed is not None:
                self.set_mouse_speed(self.original_mouse_speed)
                print(f"[INFO] 鼠标速度已恢复到原始值: {self.original_mouse_speed}")
        except Exception as e:
            print(f"[WARNING] 恢复鼠标速度失败: {e}")
        
        # 释放其他资源
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'ui'):
            self.ui.close_current_widget()

        pass


class LightweightKalmanFilter:
    """轻量级卡尔曼滤波器，专门用于注视点平滑
    
    特点：
    1. 优化的参数配置，自然避免过冲现象
    2. 平衡平滑效果和响应速度，避免过度滤波导致延迟
    3. 计算开销低，适合实时应用
    """
    
    def __init__(self, process_noise=0.2, measurement_noise=0.1, error_estimate=50.0):
        """
        初始化轻量级卡尔曼滤波器
        
        Args:
            process_noise: 过程噪声，决定滤波器对状态变化的敏感度
            measurement_noise: 测量噪声，决定滤波器对测量误差的鲁棒性
            error_estimate: 初始误差估计
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.error_estimate = error_estimate
        
        # 状态向量 [x, y, vx, vy] - 位置和速度
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 状态协方差矩阵
        self.P = np.eye(4, dtype=np.float32) * error_estimate
        
        # 状态转移矩阵（假设匀速运动）
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # 观测矩阵（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0],  # 观测x
            [0, 1, 0, 0]   # 观测y
        ], dtype=np.float32)
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        
        # 观测噪声协方差矩阵
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # 初始化标志
        self.initialized = False
        
    def update(self, measurement):
        """
        更新滤波器状态
        
        Args:
            measurement: 观测值 [x, y]
            
        Returns:
            滤波后的状态估计 [x, y, vx, vy]
        """
        measurement = np.array(measurement, dtype=np.float32)
        
        if not self.initialized:
            # 初始化：设置初始位置，速度为0
            self.state[0] = measurement[0]  # x
            self.state[1] = measurement[1]  # y
            self.state[2] = 0.0  # vx
            self.state[3] = 0.0  # vy
            self.initialized = True
            return self.state.copy()
        
        # 预测步骤
        # 状态预测: x_pred = F * x
        x_pred = self.F @ self.state
        
        # 协方差预测: P_pred = F * P * F^T + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 更新步骤
        # 观测预测: z_pred = H * x_pred
        z_pred = self.H @ x_pred
        
        # 观测残差: y = z - z_pred
        y = measurement - z_pred
        
        # 残差协方差: S = H * P_pred * H^T + R
        S = self.H @ P_pred @ self.H.T + self.R
        
        # 卡尔曼增益: K = P_pred * H^T * S^(-1)
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新: x = x_pred + K * y
        self.state = x_pred + K @ y
        
        # 协方差更新: P = (I - K * H) * P_pred
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ P_pred
        
        return self.state.copy()
    
    def get_position(self):
        """获取当前滤波后的位置估计"""
        return np.array([self.state[0], self.state[1]], dtype=np.float32)
    
    def get_velocity(self):
        """获取当前速度估计"""
        return np.array([self.state[2], self.state[3]], dtype=np.float32)
    
    def reset(self):
        """重置滤波器状态"""
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * self.error_estimate
        self.initialized = False
    
    def is_moving_fast(self, velocity_threshold=50.0):
        """
        检测是否在快速移动
        
        Args:
            velocity_threshold: 速度阈值（像素/帧）
            
        Returns:
            bool: 如果当前速度超过阈值，返回True
        """
        velocity = self.get_velocity()
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        return speed > velocity_threshold
    
    def emergency_brake(self):
        """
        紧急制动：立即停止当前运动，用于无过冲精准制动控制
        """
        # 设置速度为0，但保持当前位置
        self.state[2] = 0.0  # vx = 0
        self.state[3] = 0.0  # vy = 0
        
        # 增加位置协方差，减少速度协方差，实现快速稳定
        self.P[0, 0] *= 0.5  # x位置协方差减半，更稳定
        self.P[1, 1] *= 0.5  # y位置协方差减半，更稳定
        self.P[2, 2] *= 2.0  # vx协方差加倍，减少速度影响
        self.P[3, 3] *= 2.0  # vy协方差加倍，减少速度影响


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    system = None
    
    try:
        # 创建交互系统
        system = EyeHandInteractionSystem(project_dir)
        print("EyeHandInteractionSystem创建成功")
        
        # 初始化系统
        if not system.initialize():
            print("系统初始化失败")
            return
        
        print("系统初始化成功")
        
        # 显示主菜单
        choice = system.show_menu()
        if choice == 'quit':
            print("用户选择退出")
            return
        
        # 运行校准
        if not system.run_calibration():
            print("校准失败")
            return
        
        print("校准成功")
        
        # 运行交互模式
        system.run_interaction_mode()
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            system.cleanup()
        app.quit()

if __name__ == '__main__':
    main()