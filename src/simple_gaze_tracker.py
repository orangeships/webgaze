import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QMessageBox, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, QThread, pyqtSlot, QEasingCurve, QVariantAnimation, QEvent
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap, QPainterPath, QRegion, QKeyEvent, QLinearGradient
from PyQt5.QtCore import QSize
import win32api
import win32con
# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))
from PyQt5.QtCore import QEventLoop
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
        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        self.setWindowOpacity(0.99)  # 几乎完全透明
        
    def paintEvent(self, event):
        """重写绘制事件，实现透明背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 完全透明，不绘制任何背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

class EyeHandInteractionUI:
    def __init__(self):
        self.current_widget = None
        self.screen_width = 0
        self.screen_height = 0

        
    def initialize_display(self):
        """初始化显示"""
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        return self.screen_width, self.screen_height
    
    def show_interaction_screen(self, interaction_zone=None, current_gaze_point=None, show_gaze_point_ref=None):
        """显示交互界面（完全透明，只显示注视点）"""
        # 只在需要时创建或更新交互区域
        if current_gaze_point:
            # 如果已经存在交互区域，只更新位置而不重新创建
            if self.current_widget and isinstance(self.current_widget, InteractionOverlay):
                # 只有在位置真正改变时才更新
                if self.current_widget.current_gaze_point != current_gaze_point:
                    self.current_widget.current_gaze_point = current_gaze_point
                    self.current_widget.show_gaze_point_ref = show_gaze_point_ref
                    # 强制重绘整个窗口，避免闪烁
                    self.current_widget.repaint()
            else:
                # 关闭当前界面
                if self.current_widget:
                    self.current_widget.close()
                    self.current_widget = None
                
                # 创建新的交互区域，只显示注视点
                overlay = InteractionOverlay(None, current_gaze_point, show_gaze_point_ref)
                overlay.show()
                self.current_widget = overlay
    
    def close_current_widget(self):
        """关闭当前组件"""
        if self.current_widget:
            self.current_widget.close()
            self.current_widget = None
    


class InteractionOverlay(QWidget):
    """交互区域覆盖层 - 仅显示注视点"""
    
    def __init__(self, interaction_zone, current_gaze_point=None, show_gaze_point_ref=None):
        super().__init__()
        self.interaction_zone = interaction_zone
        self.current_gaze_point = current_gaze_point
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
        
        # 设置全屏大小
        self.setGeometry(screen_geometry)
        
        # 确保窗口不会拦截任何事件
        self.setMouseTracking(False)
        
        # 安装事件过滤器确保鼠标事件被传递
        self.installEventFilter(self)
        
    def update_gaze_point(self, gaze_point):
        """更新注视点位置"""
        self.current_gaze_point = gaze_point
        
    def paintEvent(self, event):
        """仅绘制注视点"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
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

    
    def eventFilter(self, obj, event):
        """事件过滤器 - 确保鼠标事件被正确传递"""
        if event.type() in [QEvent.MouseButtonPress, QEvent.MouseButtonRelease, 
                           QEvent.MouseButtonDblClick, QEvent.MouseMove]:
            # 忽略所有鼠标事件，让它们传递给下层窗口
            return True
        return super().eventFilter(obj, event)
    


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
   
        # 轻量级卡尔曼滤波器初始化（替换自适应移动平均算法）
        self.kalman_filter = LightweightKalmanFilter(
            process_noise=0.6,     # 过程噪声，进一步提高响应速度
            measurement_noise=0.2, # 测量噪声，更信任新测量值，平滑更轻微
            error_estimate=50.0    # 初始误差估计，最大化初始不确定性
        )
        self.smoothing_enabled = True  # 平滑开关
        
        # 注视点显示控制相关
        self.show_gaze_point = True  # 是否显示注视点（红色点）
        
        
    def initialize(self):
        """初始化系统"""
        # 初始化UI
        self.ui = EyeHandInteractionUI()
        # 获取屏幕尺寸，不再创建窗口
        self.screen_width, self.screen_height = self.ui.initialize_display()
        
        # 初始化模型
        self.model = EyeModel(self.project_dir)
        self.homtrans = HomTransform(self.project_dir)
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
            
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        return True
    
    def show_menu(self):
        """显示主菜单（终端输入）"""
        print("眼手协同交互系统")
        print("=" * 30)
        print("1. 开始校准")
        print("2. 退出程序")
        print("=" * 30)
        
        while True:
            choice = input("请输入选项 (1-2): ").strip()
            if choice == "1":
                return "calibrate"
            elif choice == "2":
                return "quit"
            else:
                print("无效选项，请重新输入")
    
    def run_calibration(self):
        """运行校准（终端输入）"""
        print("校准选项")
        print("=" * 30)
        print("1. 加载历史校准数据")
        print("2. 进行新校准")
        print("=" * 30)
        
        while True:
            choice = input("请输入选项 (1-2): ").strip()
            if choice in ["1", "2"]:
                break
            else:
                print("无效选项，请重新输入")
        
        if choice == "1":
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
            if os.path.exists(calibration_file):
                print("正在加载历史校准数据...")
                if self.homtrans.load_calibration_results(calibration_file):
                    self.calibration_data = self.homtrans.STransG
                    print("历史校准数据加载成功")
                    return True
                else:
                    print("历史校准数据加载失败，将进行新校准")
                    return self.perform_new_calibration()
            else:
                print("未找到历史校准数据，将进行新校准")
                return self.perform_new_calibration()
        else:
            print("正在进行新校准...")
            return self.perform_new_calibration()
    
    def perform_new_calibration(self):
        """执行新校准"""
        try:
            STransG = self.homtrans.calibrate(self.model, self.cap, sfm=False)
            if STransG is not None:
                self.calibration_data = STransG
                return True
            else:
                return False
        except Exception:
            return False
    
    def run_interaction_mode(self):
        """运行交互模式"""
        
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
                
                # 直接使用单帧注视点估计（关闭SfM功能）
                try:
                    FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 转换为像素坐标
                    if FSgaze is not None and len(FSgaze) >= 2:
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        
                        gaze_x = max(0, min(screen_pos_px[0], self.screen_width))
                        gaze_y = max(0, min(screen_pos_px[1], self.screen_height))
                    else:
                        gaze_x = self.screen_width // 2
                        gaze_y = self.screen_height // 2
                    
                    # 应用轻量级卡尔曼滤波平滑算法
                    raw_gaze_point = (gaze_x, gaze_y)
                    
                    # 使用轻量级卡尔曼滤波算法进行平滑处理
                    if self.smoothing_enabled:
                        gaze_point = self._smooth_gaze_point(raw_gaze_point)
                    else:
                        gaze_point = raw_gaze_point
                    
                    
                    current_gaze_point = gaze_point

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
        
        timer.stop()
        self.ui.close_current_widget()
    
    def update_interaction_frame(self):
        """更新交互帧（由定时器调用）"""
        pass  # 界面更新在run_interaction_mode中处理
    
    
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

    def _end_program(self):
        """结束程序"""
        print("程序即将结束...")
        # 清理资源
        self.cleanup()
        # 退出应用程序
        QApplication.quit()

    def cleanup(self):
        """清理资源"""
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
    
 


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 获取项目根目录（src目录的父目录）
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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