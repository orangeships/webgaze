import os
import sys
import cv2
import time
import numpy as np
from collections import deque
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QMessageBox, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QThread, pyqtSlot, QEasingCurve, QVariantAnimation, QEvent
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap, QPainterPath, QRegion, QKeyEvent, QLinearGradient
from PyQt5.QtCore import QSize
import win32api
import win32con

# 添加手部检测相关导入
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'hand_tracking'))
from hand_tracking_system import SimpleHandDetectionSystem
from one_euro_filter import HandOneEuroFilter
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
    
    def show_interaction_screen(self, interaction_zone=None, current_gaze_point=None):
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
                    # 强制重绘整个窗口，避免闪烁
                    self.current_widget.repaint()
            else:
                # 关闭当前界面
                if self.current_widget:
                    self.current_widget.close()
                    self.current_widget = None
                
                # 创建新的交互区域
                overlay = InteractionOverlay(interaction_zone, current_gaze_point)
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
    
    def __init__(self, interaction_zone, current_gaze_point=None):
        super().__init__()
        self.interaction_zone = interaction_zone
        self.current_gaze_point = current_gaze_point
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
        
        # 实时显示注视点
        if self.current_gaze_point:
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
            radius = 95
            
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
    


class GazeDispersionAnalyzer:
    """注视点离散度分析器 - 与原始代码相同"""
    
    def __init__(self, time_window_ms=500, angle_threshold=3.0, pixel_threshold=100):
        self.time_window_ms = time_window_ms
        self.angle_threshold = angle_threshold
        self.pixel_threshold = pixel_threshold
        self.gaze_points = deque()  # 存储 (timestamp, x, y) 元组
        self.last_trigger_time = 0
        self.trigger_cooldown = 1000  # 触发冷却时间1000ms
    
    def add_gaze_point(self, x, y):
        """添加新的注视点"""
        current_time = time.time() * 1000  # 转换为毫秒
        self.gaze_points.append((current_time, x, y))
        
        # 移除超过时间窗口的点
        while self.gaze_points and (current_time - self.gaze_points[0][0] > self.time_window_ms):
            self.gaze_points.popleft()
    
    def set_screen_dimensions(self, width, height):
        """设置屏幕尺寸用于角度计算"""
        self.screen_width = width
        self.screen_height = height

class HandDetectionThread(QThread):
    """手部检测线程"""
    pinch_detected = pyqtSignal(tuple)  # 捏合动作检测信号
    thumb_position = pyqtSignal(tuple)  # 大拇指位置信号
    no_pinch_detected = pyqtSignal()    # 未检测到捏合动作信号
    click_detected = pyqtSignal(tuple)  # 点击动作检测信号
    
    def __init__(self, camera_id=2):
        super().__init__()
        self.camera_id = camera_id
        self.hand_system = None
        self.is_running = False
        
        # 初始化手部关键点滤波器（硬编码启用）
        self.hand_filter = HandOneEuroFilter(freq=30.0, min_cutoff=2.4, beta=2, d_cutoff=1.0)
    
    def run(self):
        """线程运行函数"""
        # 初始化手部检测系统
        self.hand_system = SimpleHandDetectionSystem(camera_id=self.camera_id)
        self.is_running = True
        
        # 启动摄像头
        if not self.hand_system.start_camera(self.camera_id):
            print("手部检测摄像头启动失败")
            self.is_running = False
            return
        
        # 主循环
        while self.is_running:
            ret, frame = self.hand_system.cap.read()
            if not ret:
                continue
            
            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 使用新的detect_hand_events方法获取手部检测结果和事件信息
            result = self.hand_system.detect_hand_events(frame)
            hand_results = result['hand_results']
            events = result['events']
            
            # 绘制手部关键点和边界框
            for hand_info in hand_results:
                landmarks_2d = hand_info['landmarks_2d']  # 已经是平滑后的关键点
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
                if bbox:
                    x1, y1, x2, y2 = bbox
                    # 左手用绿色，右手用蓝色
                    color = (0, 255, 0) if "Left" in handedness else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, handedness, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 绘制关键点
                for i, (x, y) in enumerate(landmarks_2d):
                    point = (int(x * frame.shape[1]), int(y * frame.shape[0]))
                    # 手腕点用红色，其他点用黄色
                    color = (0, 0, 255) if i == 0 else (0, 255, 255)
                    cv2.circle(frame, point, 3, color, -1)
                    cv2.circle(frame, point, 3, (0, 0, 0), 1)
                
                # 绘制连接线
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
                for connection in connections:
                    if connection[0] < len(landmarks_2d) and connection[1] < len(landmarks_2d):
                        point1 = (int(landmarks_2d[connection[0]][0] * frame.shape[1]),
                                 int(landmarks_2d[connection[0]][1] * frame.shape[0]))
                        point2 = (int(landmarks_2d[connection[1]][0] * frame.shape[1]),
                                 int(landmarks_2d[connection[1]][1] * frame.shape[0]))
                        cv2.line(frame, point1, point2, (0, 255, 255), 2)
            
            # 处理捏合事件
            is_any_pinching = False
            for pinch_event in events['pinch_events']:
                is_pinching = pinch_event['is_pinching']
                if is_pinching:
                    is_any_pinching = True
                    # 发送捏合动作信号
                    self.pinch_detected.emit((0, 0))  # 暂时使用默认位置
                    
                    # 获取大拇指位置
                    thumb_tip = pinch_event['thumb_tip']
                    if thumb_tip:
                        self.thumb_position.emit((thumb_tip[0], thumb_tip[1]))
                        
                        # 绘制捏合效果
                        hand_index = pinch_event['hand_index']
                        if hand_index < len(hand_results):
                            landmarks_2d = hand_results[hand_index]['landmarks_2d']
                            if len(landmarks_2d) > 8:
                                thumb_pos = (int(thumb_tip[0] * frame.shape[1]), int(thumb_tip[1] * frame.shape[0]))
                                index_tip = landmarks_2d[8]
                                index_pos = (int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0]))
                                cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 3)
                                cv2.putText(frame, "PINCH", (thumb_pos[0] + 10, thumb_pos[1] - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 处理未检测到捏合动作的情况
            if not is_any_pinching:
                self.no_pinch_detected.emit()
            
            # 处理点击事件
            for click_event in events['click_events']:
                is_click = click_event['is_click']
                if is_click:
                    # 发送点击动作信号
                    self.click_detected.emit((0, 0))  # 暂时使用默认位置
                    
                    # 绘制点击效果
                    hand_index = click_event['hand_index']
                    if hand_index < len(hand_results):
                        landmarks_2d = hand_results[hand_index]['landmarks_2d']
                        thumb_tip = click_event['thumb_tip']
                        if thumb_tip and len(landmarks_2d) > 8:
                            thumb_pos = (int(thumb_tip[0] * frame.shape[1]), int(thumb_tip[1] * frame.shape[0]))
                            index_tip = landmarks_2d[8]
                            index_pos = (int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0]))
                            cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 3)
                            cv2.putText(frame, "CLICK", (thumb_pos[0] + 10, thumb_pos[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 显示检测信息
            cv2.putText(frame, f"手部数量: {len(hand_results)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "按 'q' 退出", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示手部检测窗口
            cv2.imshow('手部检测', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            
            # 控制帧率
            time.sleep(0.016)  # 约60FPS
    
    def stop(self):
        """停止线程"""
        self.is_running = False
        if self.hand_system:
            self.hand_system.stop_camera()
        cv2.destroyAllWindows()
        self.wait()


class EyeHandInteractionSystem:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.ui = None
        self.model = None
        self.homtrans = None
        self.cap = None
        self.calibration_data = None
        self.kalman_filter = KalmanFilter(process_noise=0.005, measurement_noise=5.0, error_estimate=2.0)
        self.kalman_enabled = False
        
        # 初始化注视点分析器
        self.dispersion_analyzer = GazeDispersionAnalyzer(
            time_window_ms=500, 
            angle_threshold=3.0, 
            pixel_threshold=100
        )
        
        # 当前交互状态
        self.current_interaction_zone = None
        self.previous_interaction_zone = None
        self.interaction_zone_duration = 2000  # 交互区域显示持续时间2000ms
        
        # 手部检测相关变量
        self.hand_detection_thread = None  # 手部检测线程
        self.hand_control_mode = False  # 手控模式状态
        self.no_pinch_frame_count = 0  # 连续未检测到捏合动作的帧数
        self.last_thumb_position = None  # 上次大拇指位置
        self.current_gaze_point = None  # 当前注视点位置

        
        # 注视点平滑相关（替代卡尔曼滤波）
        self.gaze_history = deque(maxlen=5)  # 最近5个注视点用于平滑
        self.smoothing_enabled = True  # 平滑开关
        self.smoothing_threshold = 30  # 自适应平滑阈值（像素）

        
    def initialize(self):
        """初始化系统"""
        # 初始化UI
        self.ui = EyeHandInteractionUI()
        self.ui.initialize_display()
        # 设置屏幕尺寸用于角度计算
        self.dispersion_analyzer.set_screen_dimensions(self.ui.screen_width, self.ui.screen_height)
        
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
        
        # 初始化手部检测线程（使用摄像头2）
        self.hand_detection_thread = HandDetectionThread(camera_id=2)
        
        # 连接信号槽
        self.hand_detection_thread.pinch_detected.connect(self.on_pinch_detected)
        self.hand_detection_thread.thumb_position.connect(self.on_thumb_position_updated)
        self.hand_detection_thread.no_pinch_detected.connect(self.on_no_pinch_detected)
        self.hand_detection_thread.click_detected.connect(self.on_click_detected)
        
        # 启动手部检测线程
        self.hand_detection_thread.start()
        
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
            # 首先尝试加载 JSON 格式的主校准文件
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
            if os.path.exists(calibration_file):
                if self.homtrans.load_calibration_results(calibration_file):
                    self.calibration_data = self.homtrans.STransG
                    return True
                else:
                    return self.perform_new_calibration()
            else:
                # 如果没有主校准文件，尝试加载屏幕0的校准文件
                screen0_file = os.path.join(self.project_dir, "results", "calibration_results_screen_0.json")
                if os.path.exists(screen0_file):
                    if self.homtrans.load_calibration_results(screen0_file):
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
        # 主循环标志
        self.running = True
        
        # 导入keyboard库用于ESC键检测
        try:
            import keyboard
        except ImportError:
            print("Warning: keyboard library not found. ESC exit may not work properly.")
            keyboard = None
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检查ESC键退出（每几帧检查一次以提高性能）
            if keyboard and keyboard.is_pressed('esc'):
                print("检测到ESC键，程序将退出...")
                self.running = False
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
                    # 转换为像素坐标
                    if FSgaze is not None and len(FSgaze) >= 2:
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        
                        gaze_x = max(0, min(screen_pos_px[0], self.ui.screen_width))
                        gaze_y = max(0, min(screen_pos_px[1], self.ui.screen_height))
                    else:
                        gaze_x = self.ui.screen_width // 2
                        gaze_y = self.ui.screen_height // 2
                    
                    # 应用高效的平滑算法（替代卡尔曼滤波）
                    raw_gaze_point = (gaze_x, gaze_y)
                    
                    # 使用自适应平滑算法
                    if self.smoothing_enabled:
                        gaze_point = self._smooth_gaze_point(raw_gaze_point)
                    else:
                        gaze_point = raw_gaze_point
                    
                    # 添加小范围约束，避免微小抖动
                    if self.current_gaze_point:
                        dx = abs(gaze_point[0] - self.current_gaze_point[0])
                        dy = abs(gaze_point[1] - self.current_gaze_point[1])
                        if dx < 2 and dy < 2:  # 如果移动距离很小，保持原位
                            gaze_point = self.current_gaze_point
                    
                    self.current_gaze_point = gaze_point
                    
                    # 添加到注视点分析器
                    self.dispersion_analyzer.add_gaze_point(gaze_point[0], gaze_point[1])
                        
                except Exception:
                    pass
            
            # 更新交互界面（同时显示注视点和交互区域）
            if self.current_interaction_zone or self.previous_interaction_zone or self.current_gaze_point:
                # 显示交互界面
                self.ui.show_interaction_screen(
                    interaction_zone=self.current_interaction_zone,
                    current_gaze_point=self.current_gaze_point
                )
            self.previous_interaction_zone = self.current_interaction_zone
            
            # 处理Qt事件
            QApplication.processEvents()
            
            # 更新前一帧
            frame_prev = frame.copy()
        
        self.ui.close_current_widget()
    
    def _smooth_gaze_point(self, raw_gaze_point):
        """
        使用自适应移动平均，根据注视点稳定性动态调整平滑强度
        """
        if not self.smoothing_enabled or not raw_gaze_point:
            return raw_gaze_point
        
        # 添加原始注视点到历史记录
        self.gaze_history.append(raw_gaze_point)
        
        if len(self.gaze_history) < 2:
            return raw_gaze_point
        
        # 计算历史点的离散度（使用最近3个点）
        recent_points = list(self.gaze_history)[-3:]
        center_x = sum(p[0] for p in recent_points) / len(recent_points)
        center_y = sum(p[1] for p in recent_points) / len(recent_points)
        
        # 计算平均偏移距离
        avg_offset = sum(np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in recent_points) / len(recent_points)
        
        # 自适应平滑权重：注视点越稳定，平滑强度越大
        if avg_offset < 10:  # 非常稳定
            smoothing_weight = 0.8  # 强平滑
        elif avg_offset < 20:  # 较稳定
            smoothing_weight = 0.6  # 中等平滑
        elif avg_offset < self.smoothing_threshold:  # 一般稳定
            smoothing_weight = 0.4  # 轻度平滑
        else:  # 不稳定
            smoothing_weight = 0.2  # 弱平滑，保持响应性
        
        # 计算加权平均值（移动平均 + 当前点加权）
        history_weight = smoothing_weight
        current_weight = 1.0 - history_weight
        
        # 使用所有历史点进行加权平均
        smoothed_x = sum(p[0] for p in self.gaze_history) / len(self.gaze_history) * history_weight + raw_gaze_point[0] * current_weight
        smoothed_y = sum(p[1] for p in self.gaze_history) / len(self.gaze_history) * history_weight + raw_gaze_point[1] * current_weight
        smoothed_point = (smoothed_x, smoothed_y)
        return smoothed_point
    
    def _auto_move_mouse_to_gaze(self, x, y):
        """自动移动鼠标到指定的视线位置"""
        try:
            # 移动鼠标到指定位置
            win32api.SetCursorPos((int(x), int(y)))
            # 在目标位置添加渐变圆圈效果
            if self.ui and self.ui.current_widget and isinstance(self.ui.current_widget, InteractionOverlay):
                self.ui.current_widget.add_fade_circle(x, y, radius=100, duration=1500)
                
        except Exception:
            pass
    
    def on_pinch_detected(self, pinch_position):
        """处理捏合动作检测"""
        if not self.hand_control_mode and self.current_gaze_point:
            # 进入手控模式
            self.hand_control_mode = True
            self.no_pinch_frame_count = 0
            # 移动鼠标到当前注视点
            self._auto_move_mouse_to_gaze(self.current_gaze_point[0], self.current_gaze_point[1])
            # 显示绿色待选框
            if self.ui and self.ui.current_widget and isinstance(self.ui.current_widget, InteractionOverlay):
                # 使用当前交互区域作为待选框
                self.ui.current_widget.add_fade_circle(
                    self.current_gaze_point[0], 
                    self.current_gaze_point[1], 
                    radius=150, 
                    duration=5000
                )
            # 暂停注视追踪功能（通过设置标志位）
            self.smoothing_enabled = False
    
    def on_thumb_position_updated(self, thumb_position):
        """处理大拇指位置更新"""
        if self.hand_control_mode:
            # 转换大拇指位置到屏幕坐标，反转Y轴方向以匹配屏幕坐标系
            screen_x = int(thumb_position[0] * self.ui.screen_width)
            screen_y = int((1 - thumb_position[1]) * self.ui.screen_height)  # 反转Y轴
            if self.last_thumb_position:
                # 计算大拇指移动距离
                dx = screen_x - self.last_thumb_position[0]
                dy = screen_y - self.last_thumb_position[1]
                # 应用放大系数，使鼠标移动更明显
                scale_factor = 4.0
                dx *= scale_factor
                dy *= scale_factor
                
                # 获取当前鼠标位置
                current_mouse_pos = win32api.GetCursorPos()
                
                # 计算新的鼠标位置
                new_mouse_x = current_mouse_pos[0] + int(dx)
                new_mouse_y = current_mouse_pos[1] + int(dy)
                
                # 限制鼠标位置在屏幕范围内
                new_mouse_x = max(0, min(new_mouse_x, self.ui.screen_width))
                new_mouse_y = max(0, min(new_mouse_y, self.ui.screen_height))
                
                # 移动鼠标
                win32api.SetCursorPos((new_mouse_x, new_mouse_y))
            
            # 更新上次大拇指位置
            self.last_thumb_position = (screen_x, screen_y)
    
    def on_click_detected(self, click_position):
        """处理点击事件，模拟鼠标左键点击"""
        if self.hand_control_mode:
            try:
                # 模拟鼠标左键点击
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                print("模拟鼠标左键点击")
            except Exception as e:
                print(f"模拟鼠标点击失败: {e}")
    
    def on_no_pinch_detected(self):
        """处理未检测到捏合动作"""
        if not self.hand_control_mode:
            return
        # 增加连续未检测到捏合动作的帧数
        self.no_pinch_frame_count += 1
        
        # 如果连续8帧未检测到捏合动作，退出手控模式
        if self.no_pinch_frame_count >= 10:
            # 退出手控模式
            self.hand_control_mode = False
            self.no_pinch_frame_count = 0
            self.last_thumb_position = None
            # 恢复注视追踪功能
            self.smoothing_enabled = True
            # 隐藏绿色待选框（通过清除交互区域）
            if self.ui and self.ui.current_widget and isinstance(self.ui.current_widget, InteractionOverlay):
                self.ui.current_widget.fade_circles.clear()
                self.ui.current_widget.repaint()

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'ui'):
            self.ui.close_current_widget()
        if hasattr(self, 'hand_detection_thread') and self.hand_detection_thread:
            self.hand_detection_thread.stop()
        pass

def main():
    """主函数"""
    app = QApplication(sys.argv)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # 创建交互系统
        system = EyeHandInteractionSystem(project_dir)
        # 初始化系统
        if not system.initialize():
            return
        
        # 显示主菜单
        choice = system.show_menu()
        if choice == 'quit':
            return
        
        # 运行校准
        if not system.run_calibration():
            return
        
        # 运行交互模式
        system.run_interaction_mode()
        
    except Exception:
        pass
    finally:
        if 'system' in locals():
            system.cleanup()
        app.quit()

if __name__ == '__main__':
    main()