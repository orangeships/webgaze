import os
import sys
from tkinter import N
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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test'))
from PyQt5.QtCore import QEventLoop
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel
from head_pose_estimation import HeadPoseEstimator

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
        self.head_pose_estimator = None
        self.previous_rmat = None  # 保存上一个rmat状态
        
        # SfM启用状态
        self.sfm_enabled = False  # 默认禁用SfM
        
        # 当前交互状态
        self.current_interaction_zone = None
        self.previous_interaction_zone = None
   
        # 轻量级卡尔曼滤波器初始化（替换自适应移动平均算法）
        self.kalman_filter = LightweightKalmanFilter(
            pn=0.5,     # 过程噪声，进一步提高响应速度
            mn=0.3,     # 测量噪声，更信任新测量值，平滑更轻微
            ee=50.0     # 初始误差估计，最大化初始不确定性
        )
        self.smoothing_enabled = True  # 平滑开关
        
        # 指数平均参数（用于gaze三维数据平滑）
        self.alpha = 0.5  # 降低平滑因子，减少延迟（从0.8改为0.5）
        self.previous_gaze_3d = None  # 保存上一次的gaze 3D数据
        
        # α-β滤波器参数（用于PnP delta_t平滑）
        self.alpha_beta_filter_enabled = True  # 是否启用α-β滤波
        self.alpha_beta_alpha = 0.7  # α参数，控制位置平滑
        self.beta = 0.3  # β参数，控制速度平滑
        self.delta_t_filtered = None  # 滤波后的delta_t
        self.previous_delta_t = None  # 上一帧的delta_t
        
        # 注视点显示控制相关
        self.show_gaze_point = True  # 是否显示注视点（红色点）
        
        # 调试模式
        self.debug_pnp = False  # 是否输出PnP调试信息
        
        
    def initialize(self):
        """初始化系统"""
        # 初始化UI
        self.ui = EyeHandInteractionUI()
        # 获取屏幕尺寸，不再创建窗口
        self.screen_width, self.screen_height = self.ui.initialize_display()
        
        # 初始化模型
        self.model = EyeModel(self.project_dir)
        self.homtrans = HomTransform(self.project_dir)
        self.head_pose_estimator = HeadPoseEstimator()
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
                
                if self.homtrans.load_calibration_results(calibration_file, self.sfm_enabled):
                    self.calibration_data = self.homtrans.STransG
                    print("历史校准数据加载成功")
                    print(f"✓ PnP标定向量: {self.homtrans.calibrate_pnp}")
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
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
            
            STransG = self.homtrans.calibrate(self.model, self.cap, sfm=False)
            
            if STransG is not None:
                # 加载校准文件以初始化所有属性（STransG, StG, SetValues等）
                if self.homtrans.load_calibration_results(calibration_file, self.sfm_enabled):
                    self.calibration_data = self.homtrans.STransG
                    print("新校准完成，校准数据已加载")
                    print(f"✓ PnP标定向量: {self.homtrans.calibrate_pnp}")
                    return True
                
                self.calibration_data = STransG
                return True
            else:
                return False
        except Exception as e:
            print(f"校准失败: {e}")
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
        
        # 用于每隔一秒输出信息的计时变量
        import time
        last_print_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
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
            
            # 记录单帧处理开始时间
            frame_start_time = time.time()
            
            # 检测人脸和眼动并计算注视点（添加用时测量）
            gaze_processing_start_time = time.time()  # 开始计时
            
            try:
                eye_info ,landmarks, pnp_info = self.model.get_gaze(frame=frame, imshow=False)
               
            except Exception:
                eye_info = None
              
            if eye_info is not None:
                gaze = eye_info['gaze']
                if pnp_info['pnp_tvec'] is not None:
                    if self.homtrans.calibrate_pnp is None:
                        self.homtrans.calibrate_pnp = pnp_info['pnp_tvec']
                    tvec_curr = pnp_info['pnp_tvec']
               
                   
                # 简化指数平均处理gaze三维数据（保持numpy数组格式）
                if self.previous_gaze_3d is None:
                    self.previous_gaze_3d = gaze.copy()

                else:
                    # 简单指数平均：当前值 = α * 新值 + (1-α) * 旧值（使用numpy保持数组格式）
                    gaze = self.alpha * gaze + (1 - self.alpha) * self.previous_gaze_3d
                    self.previous_gaze_3d = gaze.copy()
                
                # 直接使用单帧注视点估计（关闭SfM功能）
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
                        print(f"delta_t:{delta_t}")

                        # StransG: [ 347.85566997  -31.10965641 -678.52976609]
                        # StransG: [ 333.17223292    7.79619339 -697.11498267]
                        # 对delta_t进行α-β滤波
                       
                        
                    FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze, delta_t)

                    
                    # 转换为像素坐标系统初始化失败
                    if FSgaze is not None and len(FSgaze) >= 2:
                        screen_pos_mm = FSgaze.flatten()[:2]
                        screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                        
                        gaze_x = screen_pos_px[0]
                        gaze_y = screen_pos_px[1]

                      

                    gaze_x = max(0, min(gaze_x, self.screen_width))
                    gaze_y = max(0, min(gaze_y, self.screen_height))


                    # 应用轻量级卡尔曼滤波平滑算法
                    raw_gaze_point = (gaze_x, gaze_y)
                    
                    # 使用轻量级卡尔曼滤波算法进行平滑处理
                    if self.smoothing_enabled:
                        gaze_point = self._smooth_gaze_point(raw_gaze_point)
                    else:
                        gaze_point = raw_gaze_point
                    
                    # 计算注视点估计用时并输出
                    gaze_processing_end_time = time.time()
                    gaze_processing_time = (gaze_processing_end_time - gaze_processing_start_time) * 1000  # 转换为毫秒
                    # print(f"单帧注视点估计用时: {gaze_processing_time:.2f}ms")
                    
                    current_gaze_point = gaze_point

                except Exception as e:
                    print(f"Error in gaze processing: {e}")
                    pass
            
            # 计算单帧总处理时间
            frame_end_time = time.time()
            frame_processing_time = (frame_end_time - frame_start_time) * 1000
            
            # 每隔10帧输出一次处理时间
            if current_time - last_print_time > 1:
                print(f"单帧总处理时间: {frame_processing_time:.2f}ms, FPS: {1000/frame_processing_time:.1f}")
                last_print_time = current_time
            
            # 更新交互界面（同时显示注视点和交互区域）
            if self.current_interaction_zone or self.previous_interaction_zone or current_gaze_point:
                # 显示交互界面，传递show_gaze_point属性的引用
                self.ui.show_interaction_screen(
                    interaction_zone=self.current_interaction_zone,
                    current_gaze_point=current_gaze_point,
                    show_gaze_point_ref=lambda: self.show_gaze_point
                )
            self.previous_interaction_zone = self.current_interaction_zone
            
            # 优化Qt事件处理，减少阻塞
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents | QEventLoop.ExcludeSocketNotifiers)
        
        timer.stop()
        self.ui.close_current_widget()
    
    def update_interaction_frame(self):
        """更新交互帧（由定时器调用）"""
        pass  # 界面更新在run_interaction_mode中处理
    
    
    def _smooth_gaze_point(self, raw_gaze_point):
        """轻量级卡尔曼滤波注视点平滑算法        """
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
    """轻量级卡尔曼滤波器，用于注视点平滑"""
    
    def __init__(self, pn=0.2, mn=0.1, ee=50.0):
        self.pn = pn  # 过程噪声
        self.mn = mn  # 测量噪声  
        self.ee = ee  # 初始误差估计
        
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * ee
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * pn
        self.R = np.eye(2, dtype=np.float32) * mn
        self.initialized = False
        
    def update(self, measurement):
        """更新滤波器状态"""
        measurement = np.array(measurement, dtype=np.float32)
        
        if not self.initialized:
            self.state[:2] = measurement  # 设置初始位置
            self.initialized = True
            return self.state.copy()
        
        # 预测步骤
        x_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 更新步骤  
        z_pred = self.H @ x_pred
        y = measurement - z_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # 状态和协方差更新
        self.state = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        return self.state.copy()
    
    def get_position(self):
        """获取滤波后的位置估计"""
        return self.state[:2].copy()
    
 
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