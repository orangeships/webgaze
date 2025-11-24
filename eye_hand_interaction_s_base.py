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
        
        # 不再实时显示注视点
        # if self.current_gaze_point:
        #     gaze_x, gaze_y = self.current_gaze_point
        #     # 外圈（白色边框）
        #     painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        #     painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
        #     painter.drawEllipse(QPoint(int(gaze_x), int(gaze_y)), 8, 8)
        #     
        #     # 内圈（红色填充）
        #     painter.setBrush(QBrush(QColor(255, 0, 0, 255)))
        #     painter.setPen(QPen(QColor(255, 0, 0, 180), 1))
        #     painter.drawEllipse(QPoint(int(gaze_x), int(gaze_y)), 5, 5)
        
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
    
    def calculate_dispersion(self):
        """计算当前时间窗口内注视点的离散度"""
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
        self.interaction_zone_start_time = 0
        
        # 手眼协调机制相关状态变量
        self.hand_eye_coordination_enabled = True  # 手眼协调机制总开关
        self.hand_eye_coordination_active = False  # 当前是否在手眼协调模式
        self.last_rapid_gaze_time = None  # 上次快速移动的时间戳
        self.rapid_gaze_distance_threshold = None  # 快速移动距离阈值
        
        # 滑动窗口机制相关
        self.sliding_window_gaze_points = deque(maxlen=8)  # 滑动窗口，最多8个注视点
        self.sliding_window_start_time = None  # 滑动窗口开始时间
        self.sliding_window_angle_threshold = 4.0  # 角度分布阈值 4°
        self.sliding_window_time_limit = 350  # 滑动窗口时间限制 350ms
        
        # 鼠标自动移动相关
        self.last_auto_mouse_move_time = 0  # 上次自动鼠标移动的时间
        self.auto_mouse_move_cooldown = 1000  # 自动移动冷却时间 1000ms
        self.auto_mouse_move_threshold = 95  # 自动移动触发距离阈值（像素）
        
        # 注视点平滑相关（替代卡尔曼滤波）
        self.gaze_history = deque(maxlen=5)  # 最近5个注视点用于平滑
        self.smoothing_enabled = True  # 平滑开关
        self.smoothing_threshold = 30  # 自适应平滑阈值（像素）
        
        # 初始化时设置快速移动距离阈值
        self._initialize_hand_eye_coordination()
    
    def _initialize_hand_eye_coordination(self):
        """初始化手眼协调机制"""
        # 设置快速移动距离阈值为屏幕尺寸的1/3
        if self.ui:
            screen_width = self.ui.screen_width
            screen_height = self.ui.screen_height
        else:
            # 如果UI尚未初始化，使用默认屏幕尺寸
            screen = QApplication.primaryScreen()
            screen_geometry = screen.geometry()
            screen_width = screen_geometry.width()
            screen_height = screen_geometry.height()
        
        # 计算屏幕对角线的1/6作为快速移动距离阈值（根据新要求修改）
        screen_diagonal = np.sqrt(screen_width**2 + screen_height**2)
        self.rapid_gaze_distance_threshold = screen_diagonal / 6
        
        # 系统初始化完成（简化输出）
        

        
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
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.pkl")
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
                    if current_gaze_point:
                        dx = abs(gaze_point[0] - current_gaze_point[0])
                        dy = abs(gaze_point[1] - current_gaze_point[1])
                        if dx < 2 and dy < 2:  # 如果移动距离很小，保持原位
                            gaze_point = current_gaze_point
                    
                    current_gaze_point = gaze_point
                    
                    # 手眼协调机制处理
                    self._process_hand_eye_coordination(gaze_point, previous_gaze_point)
                    
                    # 更新前一注视点
                    previous_gaze_point = current_gaze_point
                    
                    # 添加到注视点分析器
                    self.dispersion_analyzer.add_gaze_point(gaze_point[0], gaze_point[1])
                    
                    # 检查触发条件
                    triggered, center_point = self.dispersion_analyzer.check_trigger_conditions()
                        
                except Exception:
                    pass
            

            
            # 获取离散度信息
            dispersion_info = self.dispersion_analyzer.calculate_dispersion()
            
            # 更新交互界面（同时显示注视点和交互区域）
            if self.current_interaction_zone or self.previous_interaction_zone or current_gaze_point:
                # 显示交互界面
                self.ui.show_interaction_screen(
                    interaction_zone=self.current_interaction_zone,
                    current_gaze_point=current_gaze_point
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
    
    def _process_hand_eye_coordination(self, gaze_point, previous_gaze_point):
        """处理手眼协调机制"""
        if not self.hand_eye_coordination_enabled or not self.ui:
            return
        
        current_time = time.time() * 1000  # 转换为毫秒
        
        # 检查是否需要启动快速移动检测（0.1秒内移动超过屏幕尺寸1/6）
        if not self.hand_eye_coordination_active and previous_gaze_point:
            gaze_distance = np.sqrt((gaze_point[0] - previous_gaze_point[0])**2 + (gaze_point[1] - previous_gaze_point[1])**2)
            
            # 检查是否超过快速移动阈值
            if (gaze_distance > self.rapid_gaze_distance_threshold and 
                (self.last_rapid_gaze_time is None or current_time - self.last_rapid_gaze_time > 500)):
                self.last_rapid_gaze_time = current_time
                self.hand_eye_coordination_active = True
                self.sliding_window_gaze_points.clear()
                self.sliding_window_start_time = current_time
        
        # 如果在手眼协调模式中（滑动窗口检测）
        if self.hand_eye_coordination_active:
            # 检查是否超过350ms时间限制
            if current_time - self.sliding_window_start_time >= self.sliding_window_time_limit:
                self.hand_eye_coordination_active = False
                self.sliding_window_gaze_points.clear()
                self.sliding_window_start_time = None
                return
            
            # 添加当前注视点到滑动窗口
            self.sliding_window_gaze_points.append((current_time, gaze_point[0], gaze_point[1]))
            
            # 当滑动窗口收集满8个点时进行检查
            if len(self.sliding_window_gaze_points) >= 8:
                self._check_sliding_window_distribution()
                # 检查完成后清空窗口，准备下一轮检测
                self.sliding_window_gaze_points.clear()
                self.sliding_window_start_time = current_time
    
    def _check_sliding_window_distribution(self):
        """检查滑动窗口内的视线分布是否符合条件（基于像素距离）"""
        if len(self.sliding_window_gaze_points) < 8:
            return
        
        # 提取坐标点（只取坐标，不取时间戳）
        points = [(x, y) for _, x, y in self.sliding_window_gaze_points]
        
        # 计算几何中心
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        # 计算所有点到中心的距离
        distances = [np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points]
        
        # 计算最大距离作为分布范围（代替角度分布）
        max_distance = max(distances) if distances else 0
        avg_distance = np.mean(distances) if distances else 0
        
        # 检查是否满足触发条件（最大距离<150像素，平均距离<75像素）
        pixel_threshold = 150  # 最大分布距离阈值
        avg_pixel_threshold = 75  # 平均分布距离阈值
        
        if max_distance < pixel_threshold and avg_distance < avg_pixel_threshold:
            # 自动移动鼠标到视线中心点
            self._auto_move_mouse_to_gaze(center_x, center_y)
            # 触发后关闭滑动窗口
            self.hand_eye_coordination_active = False
            self.sliding_window_gaze_points.clear()
            self.sliding_window_start_time = None
    
    def _smooth_gaze_point(self, raw_gaze_point):
        """高效的注视点平滑算法（替代卡尔曼滤波）
        
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
        current_time = time.time() * 1000
        
        # 检查冷却时间
        if current_time - self.last_auto_mouse_move_time < self.auto_mouse_move_cooldown:
            return
        
        try:
            # 获取当前鼠标位置
            current_cursor_pos = win32api.GetCursorPos()
            cursor_x, cursor_y = current_cursor_pos
            
            # 计算移动距离
            distance = np.sqrt((x - cursor_x)**2 + (y - cursor_y)**2)
            
            # 只有当距离超过阈值时才移动
            if distance > self.auto_mouse_move_threshold:
                # 移动鼠标到视线中心点
                win32api.SetCursorPos((int(x), int(y)))
                self.last_auto_mouse_move_time = current_time
                
                # 在目标位置添加渐变圆圈效果
                if self.ui and self.ui.current_widget and isinstance(self.ui.current_widget, InteractionOverlay):
                    self.ui.current_widget.add_fade_circle(x, y, radius=100, duration=1500)
                
        except Exception:
            pass

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'ui'):
            self.ui.close_current_widget()

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