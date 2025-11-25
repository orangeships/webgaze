import os
import sys
import pygame
import cv2
import time
import numpy as np
import datetime
import platform
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel
from gaze_tracking.gaze_smoothing import KalmanFilter
from gaze_tracking.calibration_pygame import PygameCalibrationUI

# pygame显示器检测支持
PYTHON_VERSION = sys.version_info

# Pygame初始化
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
ORANGE = (255, 165, 0)

class PygameUI:
    def __init__(self):
        # 获取真实的屏幕尺寸（考虑DPI缩放）
        import ctypes
        try:
            # Windows系统下获取真实物理分辨率
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            self.width = user32.GetSystemMetrics(0)
            self.height = user32.GetSystemMetrics(1)
            print(f"Windows系统物理分辨率: {self.width}x{self.height}")
        except:
            # 备用方案：使用pygame
            screen_info = pygame.display.Info()
            self.width = screen_info.current_w
            self.height = screen_info.current_h
            print(f"Pygame检测分辨率: {self.width}x{self.height}")
        
        self.screen = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.clock = pygame.time.Clock()
        
        # 多屏幕支持相关属性
        self.screen_configs = None  # 屏幕配置，延迟初始化
        
    def initialize_display(self):
        """初始化显示"""
        # 多屏幕环境下的显示初始化
        if hasattr(self, 'screen_configs') and self.screen_configs is not None and len(self.screen_configs) > 1:
            # 独立屏幕模式：使用主显示器的分辨率作为显示窗口
            # 注视点计算时会映射到正确的物理屏幕
            primary_screen = self.screen_configs[0] if self.screen_configs else {'width': 1920, 'height': 1080}
            
            display_width = primary_screen.get('width', 1920)
            display_height = primary_screen.get('height', 1080)
            
            print(f"独立屏幕模式: 主显示器 {display_width}x{display_height}")
            self.screen = pygame.display.set_mode((display_width, display_height), pygame.FULLSCREEN | pygame.NOFRAME)
            
            # 更新当前显示参数为主屏幕尺寸
            self.width = display_width
            self.height = display_height
            
            # 记录各屏幕配置为独立屏幕
            self.multi_screen_mode = True
            self.independent_screens = True  # 新增：标记为独立屏幕模式
            for config in self.screen_configs:
                print(f"独立屏幕 {config['index'] + 1}: {config['width']}x{config['height']} at ({config['left']}, {config['top']})")
        else:
            # 单屏幕模式：使用检测到的物理分辨率
            print(f"单屏幕模式: {self.width}x{self.height}")
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN | pygame.NOFRAME)
            self.multi_screen_mode = False
            self.independent_screens = False
        
        pygame.display.set_caption("视线追踪系统")
        
        # 验证实际显示分辨率
        actual_width, actual_height = self.screen.get_size()
        print(f"实际显示分辨率: {actual_width}x{actual_height}")
        
        # 如果实际分辨率与预期不符，更新为实际值
        if (actual_width, actual_height) != (self.width, self.height):
            print(f"分辨率不匹配，更新为: {actual_width}x{actual_height}")
            self.width, self.height = actual_width, actual_height
        
        # 初始化字体
        try:
            self.font_large = pygame.font.Font("C:\\Windows\\Fonts\\simhei.ttf", 48)  # 中文字体
            self.font_medium = pygame.font.Font("C:\\Windows\\Fonts\\simhei.ttf", 32)
            self.font_small = pygame.font.Font("C:\\Windows\\Fonts\\simhei.ttf", 24)
        except:
            # 如果中文字体不可用，使用默认字体
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
    
    def draw_button(self, text, x, y, width, height, color, text_color=WHITE):
        """绘制按钮"""
        button_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, button_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, button_rect, 2, border_radius=10)
        
        # 绘制文字
        text_surface = self.font_medium.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        return button_rect
    
    def show_start_screen(self):
        """显示开始界面"""
        self.screen.fill(WHITE)
        
        # 标题
        title_text = self.font_large.render("视线追踪系统", True, BLACK)
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title_text, title_rect)
        
        # 分辨率信息
        resolution_text = self.font_small.render(f"分辨率: {self.width}x{self.height}", True, GRAY)
        resolution_rect = resolution_text.get_rect(center=(self.width // 2, self.height // 4 + 80))
        self.screen.blit(resolution_text, resolution_rect)
        
        # 说明文字
        instruction_text1 = self.font_medium.render("按 'S' 键开始校准", True, BLACK)
        instruction_rect1 = instruction_text1.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(instruction_text1, instruction_rect1)
        
        instruction_text2 = self.font_medium.render("按 'ESC' 键退出", True, BLACK)
        instruction_rect2 = instruction_text2.get_rect(center=(self.width // 2, self.height // 2 + 80))
        self.screen.blit(instruction_text2, instruction_rect2)
        
        # 开始按钮
        button_rect = self.draw_button("开始校准", self.width // 3, 2 * self.height // 3, 
                                      self.width // 3, 100, GREEN)
        
        pygame.display.flip()
        
        return button_rect
    
    def show_calibration_screen(self, calibration_point, point_index, total_points):
        """显示校准界面"""
        self.screen.fill(WHITE)
        
        # 绘制校准点（增大到30像素）
        pygame.draw.circle(self.screen, RED, calibration_point, 30)
        
        # 可选：显示进度信息（小字体，不遮挡）
        progress_text = self.font_small.render(f"校准点 {point_index + 1}/{total_points}", True, GRAY)
        progress_rect = progress_text.get_rect(center=(self.width // 2, 50))
        self.screen.blit(progress_text, progress_rect)
        
        pygame.display.flip()
    
    def show_warmup_screen(self, progress, total):
        """显示预热界面"""
        self.screen.fill(WHITE)
        
        # 预热文字
        warmup_text = self.font_medium.render(f"模型预热中... {progress}/{total}", True, BLACK)
        text_rect = warmup_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(warmup_text, text_rect)
        
        # 进度条
        bar_width = 400
        bar_height = 30
        bar_x = (self.width - bar_width) // 2
        bar_y = self.height // 2 + 100
        
        # 背景条
        pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_height), border_radius=15)
        
        # 进度条
        progress_width = int(bar_width * progress / total)
        pygame.draw.rect(self.screen, BLUE, (bar_x, bar_y, progress_width, bar_height), border_radius=15)
        
        pygame.display.flip()
    
    def show_gaze_tracking_on_screen(self, screen_index, gaze_point=None, kalman_enabled=True, visual_3d_enabled=False, screen_info=None):
        """在指定屏幕上显示视线追踪界面"""
        # 获取屏幕配置
        screen_config = self.screen_configs[screen_index]
        
        # 获取屏幕位置信息，如果没有left/top字段，使用默认值
        left = screen_config.get('left', 0)
        top = screen_config.get('top', 0)
        width = screen_config.get('width', 1920)
        height = screen_config.get('height', 1080)
        
        # 设置窗口位置到指定屏幕
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{left},{top}"
        
        # 如果需要在不同屏幕上显示，重新创建窗口
        if screen_index != self.current_display_screen:
            # 重新初始化显示
            pygame.display.quit()
            pygame.display.init()
            self.screen = pygame.display.set_mode((width, height))
            self.current_display_screen = screen_index
        
        # 显示追踪界面
        return self.show_gaze_tracking_screen(None, gaze_point, kalman_enabled, visual_3d_enabled, screen_info)
    
    def show_gaze_tracking_screen(self, frame, gaze_point=None, kalman_enabled=True, visual_3d_enabled=False, screen_info=None):
        """显示视线追踪界面"""
        # 显示简单的追踪界面，不显示摄像头画面
        self.screen.fill(WHITE)
        
        # 标题
        title_text = self.font_large.render("视线追踪模式", True, BLACK)
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title_text, title_rect)
        
        
        # 如果有视线点，绘制它
        if gaze_point:
            pygame.draw.circle(self.screen, RED, gaze_point, 15)
        
        # 双屏幕信息显示
        if screen_info:
            dual_screen_text = self.font_small.render(f"双屏幕模式: 当前注视屏幕 {screen_info['current_screen']}/{screen_info['total_screens']}", True, BLUE)
            dual_screen_rect = dual_screen_text.get_rect(center=(self.width // 2, self.height // 2 + 160))
            self.screen.blit(dual_screen_text, dual_screen_rect)
            
            # 绘制屏幕中心点（调试用）
            pygame.draw.circle(self.screen, GREEN, screen_info['screen1_center'], 5)
            pygame.draw.circle(self.screen, ORANGE, screen_info['screen2_center'], 5)
        
        # 卡尔曼滤波状态显示
        kalman_status = "启用" if kalman_enabled else "禁用"
        kalman_color = GREEN if kalman_enabled else RED
        kalman_text = self.font_small.render(f"卡尔曼滤波: {kalman_status}", True, kalman_color)
        kalman_rect = kalman_text.get_rect(center=(self.width // 2, self.height // 2 + 80))
        self.screen.blit(kalman_text, kalman_rect)
        
        # 3D可视化状态显示
        visual_3d_status = "启用" if visual_3d_enabled else "禁用"
        visual_3d_color = BLUE if visual_3d_enabled else GRAY
        visual_3d_text = self.font_small.render(f"3D姿态显示: {visual_3d_status}", True, visual_3d_color)
        visual_3d_rect = visual_3d_text.get_rect(center=(self.width // 2, self.height // 2 + 120))
        self.screen.blit(visual_3d_text, visual_3d_rect)
        
        # 创建两个按钮
        toggle_button_rect = self.draw_button(
            "切换滤波", 
            self.width // 2 - 100, 
            self.height // 2 + 200, 
            200, 50, 
            kalman_color
        )
        
        toggle_3d_button_rect = self.draw_button(
            "切换3D显示", 
            self.width // 2 - 100, 
            self.height // 2 + 260, 
            200, 50, 
            visual_3d_color
        )
        
        # 快捷键说明
        shortcut_text = self.font_small.render("快捷键: F-滤波, D-3D显示, ESC-退出", True, GRAY)
        shortcut_rect = shortcut_text.get_rect(center=(self.width // 2, 3 * self.height // 4 + 40))
        self.screen.blit(shortcut_text, shortcut_rect)
        
        # 退出说明
        exit_text = self.font_small.render("按ESC键退出", True, GRAY)
        exit_rect = exit_text.get_rect(center=(self.width // 2, 3 * self.height // 4 + 80))
        self.screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()
        
        return toggle_button_rect, toggle_3d_button_rect
    
    def move_window_to_screen(self, screen_x, screen_y, screen_width, screen_height):
        """将窗口移动到指定屏幕位置
        
        Args:
            screen_x: 目标屏幕的x坐标
            screen_y: 目标屏幕的y坐标  
            screen_width: 目标屏幕的宽度
            screen_height: 目标屏幕的高度
        """
        try:
            if platform.system() == "Windows":
                # 使用Windows API移动窗口
                import ctypes
                from ctypes import wintypes
                
                # 获取窗口句柄
                hwnd = pygame.display.get_wm_info()['window']
                
                # 设置窗口位置
                SWP_NOZORDER = 0x0004
                SWP_NOSIZE = 0x0001
                SWP_SHOWWINDOW = 0x0040
                
                # 先设置窗口位置，然后调整大小
                ctypes.windll.user32.SetWindowPos(
                    hwnd, 0, screen_x, screen_y, screen_width, screen_height,
                    SWP_NOZORDER | SWP_SHOWWINDOW
                )
                
                # 移除重复的调试输出
                
        except Exception as e:
            print(f"移动窗口失败: {e}")
            # 回退到SDL环境变量方法
            if screen_x is not None and screen_y is not None:
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"

    def show_gaze_tracking_on_screen(self, screen_index, gaze_point=None, kalman_enabled=True, visual_3d_enabled=False, screen_info=None, dual_screen_mode=False):
        """在指定屏幕上显示视线追踪界面"""
        # 如果是双屏幕模式，移动到对应屏幕
        if dual_screen_mode and len(self.screen_configs) > screen_index:
            screen_config = self.screen_configs[screen_index]
            # 移动窗口到对应屏幕
            self.move_window_to_screen(
                screen_config['left'], screen_config['top'],
                screen_config['width'], screen_config['height']
            )
        
        # 调用常规的显示方法
        return self.show_gaze_tracking_screen(None, gaze_point, kalman_enabled, visual_3d_enabled, screen_info)
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                elif event.key == pygame.K_s or event.key == pygame.K_SPACE:
                    return 'start'
                elif event.key == pygame.K_f:
                    return 'toggle_kalman'
                elif event.key == pygame.K_d:
                    return 'toggle_3d'
                elif event.key == pygame.K_m:
                    return 'toggle_dual_screen'
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return 'click'
        return None

class PygameGazeSystem:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.ui = None
        self.model = None
        self.homtrans = None
        self.cap = None
        self.calibration_data = None
        # 初始化卡尔曼滤波器用于平滑视线点
        # 参数调整: process_noise越小，滤波越平滑；measurement_noise越小，越信任测量值
        self.kalman_filter = KalmanFilter(process_noise=0.01, measurement_noise=2.0, error_estimate=1.0)
        self.kalman_enabled = True  # 卡尔曼滤波开关，默认为启用
        
        # 3D可视化器
        self.visual_3d = None
        self.visual_3d_enabled = False  # 3D可视化开关，默认为禁用
        
        # 双屏幕支持
        self.dual_screen_mode = False  # 双屏幕模式开关
        self.screen_configs = []  # 存储多个屏幕的配置信息
        self.current_screen_index = 0  # 当前使用的屏幕索引
        
    def initialize(self):
        """初始化系统"""
        # 首先检测屏幕配置
        print("正在检测屏幕配置...")
        self.screen_configs = self.get_screen_configurations()
        
        if len(self.screen_configs) >= 2:
            print(f"检测到 {len(self.screen_configs)} 个屏幕，启用双屏幕模式")
            self.dual_screen_mode = True
        else:
            print(f"检测到 {len(self.screen_configs)} 个屏幕，使用单屏幕模式")
            self.dual_screen_mode = False
        
        # 初始化UI（延迟显示初始化）
        self.ui = PygameUI()
        
        # 设置UI的屏幕配置（在initialize_display之前）
        if self.dual_screen_mode:
            self.ui.current_display_screen = 0  # 当前显示屏幕索引
            self.ui.screen_configs = self.screen_configs  # 保存屏幕配置
        
        # 初始化显示（现在screen_configs已经设置好了）
        self.ui.initialize_display()
        
        print(f"UI初始化完成，分辨率: {self.ui.width}x{self.ui.height}")
        
        # 初始化模型
        print("正在加载模型...")
        start_time = time.time()
        self.model = EyeModel(self.project_dir)
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {1000*load_time:.1f}ms")
        
        # 初始化HomTransform
        self.homtrans = HomTransform(self.project_dir)
        
        # 初始化摄像头（尝试多个设备ID）
        self.cap = None
        for device_id in [1, 0, 2]:
            try:
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    # 测试读取一帧来确认摄像头正常工作
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"成功打开并测试摄像头设备 {device_id}")
                        print(f"摄像头分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        break
                    else:
                        print(f"摄像头设备 {device_id} 打开但无法读取帧")
                        self.cap.release()
                        self.cap = None
                else:
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                print(f"摄像头设备 {device_id} 打开失败: {e}")
                continue
        
        if self.cap is None:
            print("无法打开任何摄像头设备")
            return False
            
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        return True
    
    def show_menu(self):
        """显示主菜单"""
        print("显示主菜单...")
        while True:
            button_rect = self.ui.show_start_screen()
            event = self.ui.handle_events()
            
            if event == 'quit':
                print("用户选择退出")
                return 'quit'
            elif event == 'start' or event == 'click':
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos) or event == 'start':
                    print("用户选择开始校准")
                    return 'calibrate'
            
            self.ui.clock.tick(60)
    
    def run_calibration(self):
        """运行校准"""
        # 显示选择菜单
        choice = self.show_calibration_choice()
        
        if choice == 'load':
            # 加载历史校准数据
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
            if os.path.exists(calibration_file):
                print(f"正在加载历史校准数据: {calibration_file}")
                if self.dual_screen_mode:
                    # 双屏幕模式：尝试加载主屏幕校准数据
                    print("双屏幕模式：尝试加载屏幕1的校准数据")
                    if self.homtrans.load_calibration_results(screen_index=0):
                        print("屏幕1校准数据加载成功！")
                        # 设置当前屏幕索引为0（主屏幕）
                        self.current_screen_index = 0
                        self.calibration_data = self.homtrans.STransG
                        return True
                    else:
                        print("屏幕1校准数据加载失败，尝试加载通用校准数据")
                        if self.homtrans.load_calibration_results(calibration_file):
                            print("通用校准数据加载成功！")
                            self.calibration_data = self.homtrans.STransG
                            return True
                        else:
                            print("所有校准数据加载失败，将进行新校准")
                            return self.perform_new_calibration()
                else:
                    # 单屏幕模式：加载通用校准数据
                    if self.homtrans.load_calibration_results(calibration_file):
                        print("历史校准数据加载成功！")
                        self.calibration_data = self.homtrans.STransG
                        return True
                    else:
                        print("历史校准数据加载失败，将进行新校准")
                        return self.perform_new_calibration()
            else:
                print(f"历史校准文件不存在: {calibration_file}")
                print("将进行新校准")
                return self.perform_new_calibration()
        else:
            # 进行新校准
            print("开始新校准...")
            return self.perform_new_calibration()
    
    def show_calibration_choice(self):
        """显示校准选择界面"""
        while True:
            self.ui.screen.fill(WHITE)
            
            # 标题
            title_text = self.ui.font_large.render("校准选项", True, BLACK)
            title_rect = title_text.get_rect(center=(self.ui.width // 2, self.ui.height // 6))
            self.ui.screen.blit(title_text, title_rect)
            
            # 显示检测到的屏幕信息
            y_offset = self.ui.height // 6 + 80
            for i, screen in enumerate(self.screen_configs):
                screen_info = f"屏幕{i+1}: {screen['name']} ({screen['width']}x{screen['height']})"
                screen_text = self.ui.font_small.render(screen_info, True, BLUE)
                screen_rect = screen_text.get_rect(center=(self.ui.width // 2, y_offset + i * 30))
                self.ui.screen.blit(screen_text, screen_rect)
            
            # 独立屏幕模式选择
            independent_screen_text = self.ui.font_medium.render("独立屏幕模式", True, BLACK)
            independent_screen_rect = independent_screen_text.get_rect(center=(self.ui.width // 2, y_offset + len(self.screen_configs) * 30 + 40))
            self.ui.screen.blit(independent_screen_text, independent_screen_rect)
            
            # 独立屏幕模式开关按钮
            independent_status = "启用" if hasattr(self, 'independent_screens') and self.independent_screens else "禁用"
            independent_color = GREEN if independent_status == "启用" else RED
            independent_button_rect = self.ui.draw_button(f"独立屏幕: {independent_status}", self.ui.width // 4, y_offset + len(self.screen_configs) * 30 + 80,
                                                  self.ui.width // 2, 60, independent_color)
            
            # 校准选项按钮
            button1_rect = self.ui.draw_button("加载历史校准数据", self.ui.width // 4, self.ui.height // 2 + 80, 
                                              self.ui.width // 2, 60, BLUE)
            button2_rect = self.ui.draw_button("进行新校准", self.ui.width // 4, self.ui.height // 2 + 160, 
                                              self.ui.width // 2, 60, GREEN)
            
            # 提示信息
            independent_enabled = hasattr(self, 'independent_screens') and self.independent_screens
            if independent_enabled:
                hint_text = self.ui.font_small.render("当前：独立屏幕模式 ", True, GREEN)
            elif self.dual_screen_mode:
                hint_text = self.ui.font_small.render("当前：双屏幕模式", True, ORANGE)
            else:
                hint_text = self.ui.font_small.render("当前：单屏幕模式", True, BLUE)
            hint_rect = hint_text.get_rect(center=(self.ui.width // 2, y_offset + len(self.screen_configs) * 30 + 160))
            self.ui.screen.blit(hint_text, hint_rect)
            
            pygame.display.flip()
            
            event = self.ui.handle_events()
            if event == 'quit':
                return 'quit'
            elif event == 'click':
                mouse_pos = pygame.mouse.get_pos()
                if independent_button_rect.collidepoint(mouse_pos):
                    # 切换独立屏幕模式
                    if not hasattr(self, 'independent_screens'):
                        self.independent_screens = False
                    self.independent_screens = not self.independent_screens
                    print(f"独立屏幕模式: {'启用' if self.independent_screens else '禁用'}")
                    # 立即更新显示，不等待下次循环
                    continue
                elif button1_rect.collidepoint(mouse_pos):
                    return 'load'
                elif button2_rect.collidepoint(mouse_pos):
                    return 'new'
            elif event == 'toggle_independent_screen':
                # 添加快捷键支持
                if not hasattr(self, 'independent_screens'):
                    self.independent_screens = False
                self.independent_screens = not self.independent_screens
                print(f"独立屏幕模式: {'启用' if self.independent_screens else '禁用'}")
                continue
            
            self.ui.clock.tick(60)
    
    def perform_new_calibration(self):
        """执行新校准"""
        independent_enabled = hasattr(self, 'independent_screens') and self.independent_screens
        if independent_enabled:
            print("执行独立屏幕校准模式")
            return self.perform_dual_screen_calibration()  # 复用双屏幕校准逻辑，但使用独立屏幕处理
        elif self.dual_screen_mode:
            print("执行虚拟双屏幕校准模式")
            return self.perform_dual_screen_calibration()
        else:
            return self.perform_single_screen_calibration()
    
    def perform_single_screen_calibration(self):
        """执行单屏幕校准"""
        # 移除预热步骤，直接调用homtransform的calibrate方法，让它负责预热
        
        # 使用homtransform的calibrate方法进行校准
        # 注意：homtransform的calibrate方法会使用Pygame显示界面
        try:
            STransG = self.homtrans.calibrate(self.model, self.cap, sfm=True)
            if STransG is not None:
                print("校准成功完成")
                self.calibration_data = STransG
                # 重新初始化Pygame显示，因为校准界面可能已清理
                self.ui.initialize_display()
                return True
            else:
                print("校准失败或被用户取消")
                # 重新初始化Pygame显示
                self.ui.initialize_display()
                return False
        except Exception as e:
            print(f"校准过程中出错: {e}")
            # 重新初始化Pygame显示
            self.ui.initialize_display()
            return False
    
    def perform_dual_screen_calibration(self):
        """执行双屏幕校准"""
        print("开始双屏幕校准...")
        
        # 如果已经检测到屏幕配置，直接使用检测到的配置
        if len(self.screen_configs) >= 2:
            print(f"使用检测到的屏幕配置进行校准")
            for i, screen_config in enumerate(self.screen_configs[:2]):
                print(f"屏幕{i+1}: {screen_config['name']} ({screen_config['width']}x{screen_config['height']})")
        else:
            print("未检测到足够的屏幕，使用默认配置")
            # 创建默认的双屏幕配置
            self.screen_configs = [
                {
                    'index': 0,
                    'name': '主显示器',
                    'width': self.ui.width // 2,
                    'height': self.ui.height,
                    'center_x': self.ui.width // 4,
                    'center_y': self.ui.height // 2,
                    'position': 'front'
                },
                {
                    'index': 1,
                    'name': '第二显示器',
                    'width': self.ui.width // 2,
                    'height': self.ui.height,
                    'center_x': 3 * self.ui.width // 4,
                    'center_y': self.ui.height // 2,
                    'position': 'right'
                }
            ]
        
        # 校准第一个屏幕
        print("\n=== 校准屏幕1 ===")
        screen1_config = self.calibrate_single_screen("屏幕1", 1, self.screen_configs[0])
        if screen1_config is None:
            print("屏幕1校准失败")
            self.ui.initialize_display()
            return False
        
        # 校准第二个屏幕
        print("\n=== 校准屏幕2 ===")
        screen2_config = self.calibrate_single_screen("屏幕2", 2, self.screen_configs[1])
        if screen2_config is None:
            print("屏幕2校准失败")
            self.ui.initialize_display()
            return False
        
        # 更新屏幕配置
        self.screen_configs[0] = screen1_config
        self.screen_configs[1] = screen2_config
        
        # 保存双屏幕校准配置
        self.save_dual_screen_config()
        
        print(f"\n双屏幕校准完成！")
        print(f"屏幕1: {screen1_config['name']} ({screen1_config['width']}x{screen1_config['height']}) 位置: {screen1_config['position']}")
        print(f"屏幕2: {screen2_config['name']} ({screen2_config['width']}x{screen2_config['height']}) 位置: {screen2_config['position']}")
        
        # 重新初始化Pygame显示
        self.ui.initialize_display()
        return True
    
    def save_dual_screen_config(self):
        """保存双屏幕配置信息"""
        import json
        import os
        
        dual_screen_config = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_screens': len(self.screen_configs),
            'screens': {}
        }
        
        for i, screen_config in enumerate(self.screen_configs):
            dual_screen_config['screens'][str(i)] = {
                'index': screen_config['index'],
                'name': screen_config['name'],
                'width': screen_config['width'],
                'height': screen_config['height'],
                'left': screen_config.get('left', 0),
                'top': screen_config.get('top', 0),
                'right': screen_config.get('right', screen_config['width']),
                'bottom': screen_config.get('bottom', screen_config['height']),
                'center_x': screen_config['center_x'],
                'center_y': screen_config['center_y'],
                'position': screen_config['position'],
                'calibration_data': screen_config.get('calibration_data', None) is not None
            }
        
        config_file = os.path.join(self.project_dir, "results", "dual_screen_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(dual_screen_config, f, indent=2, ensure_ascii=False)
        
        print(f"双屏幕配置已保存到: {config_file}")
    
    def calibrate_single_screen(self, screen_name, screen_index, screen_config):
        """校准单个屏幕"""
        print(f"请确保您的视线朝向{screen_name}")
        print(f"按任意键开始{screen_name}校准...")
        
        # 为当前屏幕创建专用的UI实例
        screen_ui = PygameCalibrationUI(screen_config['width'], screen_config['height'])
        
        # 显示准备界面 - 直接在目标屏幕上显示
        screen_ui.initialize(
            x=screen_config.get('left', 0), 
            y=screen_config.get('top', 0), 
            fullscreen=False,
            screen_index=screen_config.get('index', 0)
        )
        
        # 设置屏幕偏移信息，确保校准点在正确位置显示
        screen_ui.screen_offset = {
            'x': screen_config.get('left', 0),
            'y': screen_config.get('top', 0),
            'width': screen_config['width'],
            'height': screen_config['height']
        }
        
        # 显示准备界面
        while True:
            screen_ui.screen.fill(WHITE)
            
            # 标题
            title_text = screen_ui.font_large.render(f"准备校准{screen_name}", True, BLACK)
            title_rect = title_text.get_rect(center=(screen_ui.width // 2, screen_ui.height // 3))
            screen_ui.screen.blit(title_text, title_rect)
            
            # 说明文字
            instruction_text1 = screen_ui.font_medium.render(f"请确保您的视线朝向{screen_name}", True, BLACK)
            instruction_rect1 = instruction_text1.get_rect(center=(screen_ui.width // 2, screen_ui.height // 2))
            screen_ui.screen.blit(instruction_text1, instruction_rect1)
            
            instruction_text2 = screen_ui.font_medium.render("准备好后按空格键开始校准", True, BLACK)
            instruction_rect2 = instruction_text2.get_rect(center=(screen_ui.width // 2, screen_ui.height // 2 + 60))
            screen_ui.screen.blit(instruction_text2, instruction_rect2)
            
            pygame.display.flip()
            
            # 确保窗口获得焦点
            pygame.event.pump()
            
            event = screen_ui.handle_events()
            if event == 'quit':
                screen_ui.cleanup()
                return None
            elif event == 'start':  # 空格键
                break
            
            screen_ui.clock.tick(60)
        
        # 清理准备界面的资源
        screen_ui.cleanup()
        
        # 执行校准 - 使用专门的屏幕校准方法
        try:
            # 使用屏幕特定的校准方法，传入屏幕UI配置
            STransG = self.homtrans.calibrate_screen(self.model, self.cap, screen_config, sfm=True)
            if STransG is not None:
                # 获取屏幕信息 - 确保包含所有必要的屏幕配置字段
                screen_info = {
                    'index': screen_index,
                    'name': screen_name,
                    'width': screen_config['width'],
                    'height': screen_config['height'],
                    'left': screen_config.get('left', 0),
                    'top': screen_config.get('top', 0),
                    'right': screen_config.get('right', screen_config['width']),
                    'bottom': screen_config.get('bottom', screen_config['height']),
                    'center_x': screen_config['center_x'],
                    'center_y': screen_config['center_y'],
                    'calibration_data': STransG,
                    'position': self.estimate_screen_position(screen_index)
                }
                print(f"{screen_name}校准成功")
                return screen_info
            else:
                print(f"{screen_name}校准失败")
                return None
        except Exception as e:
            print(f"{screen_name}校准出错: {e}")
            return None
    
    def estimate_screen_position(self, screen_index):
        """估计屏幕位置（相对于用户）"""
        # 这里可以根据实际情况调整，比如通过用户输入或预设配置
        # 简化处理：假设屏幕1在正前方，屏幕2在右侧
        positions = {
            1: "front",  # 正前方
            2: "right"   # 右侧
        }
        return positions.get(screen_index, "unknown")
    
    def detect_physical_monitors_pygame(self):
        """使用pygame检测显示器信息"""
        try:
            print("使用pygame显示器检测...")
            
            monitors = []
            display_count = pygame.display.get_num_displays()
            
            if display_count > 0:
                print(f"pygame检测到 {display_count} 个显示器")
                
                # 获取主显示器信息
                info = pygame.display.Info()
                main_width = info.current_w
                main_height = info.current_h
                
                monitors.append({
                    'index': 0,
                    'name': '主显示器',
                    'width': main_width,
                    'height': main_height,
                    'left': 0,
                    'top': 0,
                    'right': main_width,
                    'bottom': main_height,
                    'center_x': main_width // 2,
                    'center_y': main_height // 2,
                    'position': 'front'
                })
                
                print(f"  主显示器: {main_width}x{main_height}")
                
                # 尝试检测其他显示器
                for i in range(1, display_count):
                    try:
                        modes = pygame.display.list_modes(i)
                        if modes:
                            # 获取最大分辨率
                            max_width, max_height = modes[0]
                            
                            # 根据显示器索引推算位置
                            if i == 1:
                                # 假设是横向扩展
                                left_offset = main_width
                                top_offset = 0
                                position = 'right'
                            else:
                                # 继续向右扩展
                                left_offset = main_width + 1920 * (i-1)
                                top_offset = 0
                                position = 'right'
                            
                            monitors.append({
                                'index': i,
                                'name': f'显示器{i+1}',
                                'width': max_width,
                                'height': max_height,
                                'left': left_offset,
                                'top': top_offset,
                                'right': left_offset + max_width,
                                'bottom': top_offset + max_height,
                                'center_x': left_offset + max_width // 2,
                                'center_y': top_offset + max_height // 2,
                                'position': position
                            })
                            
                            print(f"  显示器{i+1}: {max_width}x{max_height} 位置({left_offset},{top_offset})")
                        else:
                            print(f"  显示器{i+1}: 无可用模式")
                    except Exception as e:
                        print(f"  显示器{i+1}: 检测错误 - {e}")
                
                return monitors
            else:
                print("pygame未检测到任何显示器")
                return None
                
        except Exception as e:
            print(f"pygame显示器检测失败: {e}")
            return None
    
    def get_screen_configurations(self):
        """获取屏幕配置，优先使用pygame检测"""
        # 首先尝试使用pygame检测
        monitors = self.detect_physical_monitors_pygame()
        if monitors and len(monitors) >= 1:
            return monitors
        
        # 如果pygame检测失败，使用基础检测
        print("使用基础屏幕检测方法")
        return self.get_basic_screen_configurations()
    
    def get_basic_screen_configurations(self):
        """独立屏幕检测 - 不依赖虚拟屏幕，每个屏幕独立检测真实分辨率和位置"""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            
            # 定义Windows API结构
            class RECT(ctypes.Structure):
                _fields_ = [
                    ('left', ctypes.c_long),
                    ('top', ctypes.c_long), 
                    ('right', ctypes.c_long),
                    ('bottom', ctypes.c_long)
                ]
            
            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', ctypes.c_ulong),
                    ('rcMonitor', RECT),
                    ('rcWork', RECT),
                    ('dwFlags', ctypes.c_ulong)
                ]
            
            # 定义窗口类型
            HWND = ctypes.c_void_p
            HDC = ctypes.c_void_p
            LPARAM = ctypes.c_long
            
            # 获取主显示器信息
            primary_width = user32.GetSystemMetrics(0)   # SM_CXSCREEN
            primary_height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
            
            print(f"独立屏幕检测开始...")
            print(f"主显示器: {primary_width}x{primary_height}")
            
            # 创建回调函数来枚举所有显示器
            monitors = []
            
            def enum_display_monitors_proc(hmon, hdc, lprect, lparam):
                monitor_info = MONITORINFO()
                monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
                
                # 获取显示器信息
                if user32.GetMonitorInfoW(hmon, ctypes.byref(monitor_info)):
                    monitor_rect = monitor_info.rcMonitor
                    
                    width = monitor_rect.right - monitor_rect.left
                    height = monitor_rect.bottom - monitor_rect.top
                    
                    monitor_data = {
                        'index': len(monitors),
                        'width': width,
                        'height': height,
                        'left': monitor_rect.left,
                        'top': monitor_rect.top,
                        'right': monitor_rect.right,
                        'bottom': monitor_rect.bottom,
                        'work_left': monitor_info.rcWork.left,
                        'work_top': monitor_info.rcWork.top,
                        'work_right': monitor_info.rcWork.right,
                        'work_bottom': monitor_info.rcWork.bottom
                    }
                    
                    monitors.append(monitor_data)
                
                return True
            
            # 尝试枚举显示器
            try:
                # 定义更简单的枚举方法
                user32.EnumDisplayMonitors.restype = wintypes.BOOL
                
                # 使用ctypes创建回调函数
                MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_void_p)
                
                # 创建回调函数实例
                callback_func = MONITORENUMPROC(enum_display_monitors_proc)
                
                result = user32.EnumDisplayMonitors(0, None, callback_func, 0)
                
                if len(monitors) > 1:
                    print(f"检测到 {len(monitors)} 个独立显示器:")
                    
                    # 处理显示器信息
                    configs = []
                    for i, monitor in enumerate(monitors):
                        # 按位置排序（左到右，上到下）
                        monitor['sort_key'] = (monitor['top'], monitor['left'])
                    
                    monitors.sort(key=lambda x: x['sort_key'])
                    
                    for i, monitor in enumerate(monitors):
                        monitor['index'] = i
                        
                        # 计算屏幕中心和名称
                        center_x = monitor['left'] + monitor['width'] // 2
                        center_y = monitor['top'] + monitor['height'] // 2
                        
                        # 确定屏幕名称和位置关系
                        if i == 0:
                            name = '主显示器'
                            position = 'front'
                        else:
                            # 根据位置关系确定名称
                            prev_monitor = monitors[i-1]
                            if monitor['left'] > prev_monitor['right']:
                                name = f'显示器{i+1} (右侧)'
                                position = 'right'
                            elif monitor['left'] < prev_monitor['left']:
                                name = f'显示器{i+1} (左侧)'
                                position = 'left'
                            elif monitor['top'] > prev_monitor['bottom']:
                                name = f'显示器{i+1} (下方)'
                                position = 'below'
                            elif monitor['top'] < prev_monitor['top']:
                                name = f'显示器{i+1} (上方)'
                                position = 'above'
                            else:
                                name = f'显示器{i+1}'
                                position = 'unknown'
                        
                        config = {
                            'index': i,
                            'name': name,
                            'width': monitor['width'],
                            'height': monitor['height'],
                            'left': monitor['left'],
                            'top': monitor['top'],
                            'right': monitor['right'],
                            'bottom': monitor['bottom'],
                            'center_x': center_x,
                            'center_y': center_y,
                            'position': position,
                            'work_area': {
                                'left': monitor['work_left'],
                                'top': monitor['work_top'], 
                                'right': monitor['work_right'],
                                'bottom': monitor['work_bottom']
                            }
                        }
                        
                        configs.append(config)
                        print(f"  屏幕{i}: {monitor['width']}x{monitor['height']} 位置({monitor['left']},{monitor['top']}) 名称: {name}")
                    
                    return configs
                    
            except Exception as api_error:
                print(f"Windows API显示器枚举失败: {api_error}")
                print("使用替代方法检测独立屏幕...")
                
                # 使用GetMonitorInfo获取主显示器的实际工作区域
                try:
                    monitor_info = MONITORINFO()
                    monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
                    
                    # 获取主显示器句柄
                    primary_monitor = user32.MonitorFromWindow(None, 1)  # MONITOR_DEFAULTTOPRIMARY
                    
                    if user32.GetMonitorInfoW(primary_monitor, ctypes.byref(monitor_info)):
                        primary_rect = monitor_info.rcMonitor
                        primary_work = monitor_info.rcWork
                        
                        primary_width = primary_rect.right - primary_rect.left
                        primary_height = primary_rect.bottom - primary_rect.top
                        primary_work_width = primary_work.right - primary_work.left
                        primary_work_height = primary_work.bottom - primary_work.top
                        
                        print(f"主显示器实际分辨率: {primary_width}x{primary_height}")
                        print(f"主显示器工作区域: {primary_work_width}x{primary_work_height}")
                        
                        # 检查是否有额外的显示器
                        # 尝试检测扩展显示器
                        virtual_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
                        virtual_height = user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
                        virtual_left = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
                        virtual_top = user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
                        
                        print(f"扩展桌面: {virtual_width}x{virtual_height} at ({virtual_left}, {virtual_top})")
                        
                        configs = []
                        
                        # 添加主显示器配置
                        configs.append({
                            'index': 0,
                            'name': '主显示器',
                            'width': primary_width,
                            'height': primary_height,
                            'work_width': primary_work_width,
                            'work_height': primary_work_height,
                            'left': primary_rect.left,
                            'top': primary_rect.top,
                            'right': primary_rect.right,
                            'bottom': primary_rect.bottom,
                            'center_x': primary_rect.left + primary_width // 2,
                            'center_y': primary_rect.top + primary_height // 2,
                            'position': 'front'
                        })
                        
                        # 如果扩展桌面比主显示器大，尝试检测其他显示器
                        if virtual_width > primary_width or virtual_height > primary_height:
                            print("检测到扩展桌面，推测存在其他显示器")
                            
                            # 尝试检测扩展显示器的实际分辨率
                            # 方法1: 尝试从注册表读取显示器信息
                            try:
                                import winreg
                                
                                # 查找其他显示器的配置
                                display_configs = []
                                try:
                                    # 检查Windows显示器配置
                                    hkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                                         r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers")
                                    
                                    # 尝试获取显示器信息
                                    monitor_info_data = []
                                    try:
                                        # 枚举所有显示器配置
                                        i = 0
                                        while True:
                                            try:
                                                subkey = winreg.EnumKey(hkey, i)
                                                monitor_info_data.append(subkey)
                                                i += 1
                                            except OSError:
                                                break
                                    except:
                                        pass
                                    
                                    winreg.CloseKey(hkey)
                                    
                                    print(f"检测到 {len(monitor_info_data)} 个显示器配置")
                                    
                                except:
                                    pass
                                    
                            except Exception as registry_error:
                                print(f"注册表读取失败: {registry_error}")
                            
                            # 方法2: 基于常见配置的智能推测（基于实际物理屏幕）
                            remaining_width = virtual_width - primary_width
                            if remaining_width > 100:  # 至少有100像素的扩展
                                
                                # 智能推测第二显示器的真实分辨率
                                # 考虑常见的双显示器配置
                                if primary_height >= 1440 and primary_width >= 2560:
                                    # 主屏幕是2K或更高，第二屏幕常见为1080P
                                    if remaining_width >= 1920:
                                        second_width = 1920
                                        second_height = 1080
                                        print(f"推测配置：主屏幕2K + 副屏幕1920x1080")
                                    elif remaining_width >= 1366:
                                        second_width = 1366
                                        second_height = 768

                                    else:
                                        second_width = remaining_width
                                        second_height = 1080

                                elif primary_height == 1080:
                                    # 主屏幕是1080P，第二屏幕也可能是1080P或更低
                                    if remaining_width >= 1920:
                                        second_width = 1920
                                        second_height = 1080

                                    elif remaining_width >= 1366:
                                        second_width = 1366
                                        second_height = 768

                                    else:
                                        second_width = remaining_width
                                        second_height = 1080
                                        print(f"推测配置：1080P + {second_width}x{second_height}")
                                else:
                                    # 其他情况保守估计
                                    second_width = remaining_width
                                    second_height = min(primary_height, 1080)

                                
                                # 添加第二显示器配置
                                configs.append({
                                    'index': 1,
                                    'name': '第二显示器',
                                    'width': second_width,
                                    'height': second_height,
                                    'work_width': second_width,
                                    'work_height': second_height,
                                    'left': primary_rect.left + primary_width,
                                    'top': primary_rect.top,
                                    'right': primary_rect.left + primary_width + second_width,
                                    'bottom': primary_rect.top + second_height,
                                    'center_x': primary_rect.left + primary_width + second_width // 2,
                                    'center_y': primary_rect.top + second_height // 2,
                                    'position': 'right'
                                })
                                

                        
                        return configs
                        
                except Exception as fallback_error:

                    return [{
                        'index': 0,
                        'name': '主显示器',
                        'width': 2560,
                        'height': 1440,
                        'work_width': 2560,
                        'work_height': 1440,
                        'left': 0,
                        'top': 0,
                        'right': 2560,
                        'bottom': 1440,
                        'center_x': 1280,
                        'center_y': 720,
                        'position': 'front'
                    }, {
                        'index': 1,
                        'name': '第二显示器',
                        'width': 1920,
                        'height': 1080,
                        'work_width': 1920,
                        'work_height': 1080,
                        'left': 2560,
                        'top': 0,
                        'right': 4480,
                        'bottom': 1080,
                        'center_x': 3520,
                        'center_y': 540,
                        'position': 'right'
                    }]
                    
        except Exception as e:
            # 独立屏幕检测失败，使用默认配置
            # 最终备用方案
            return [{
                'index': 0,
                'name': '默认主显示器',
                'width': 1920,
                'height': 1080,
                'left': 0,
                'top': 0,
                'right': 1920,
                'bottom': 1080,
                'center_x': 960,
                'center_y': 540,
                'position': 'front'
            }]
    
    # 移除简化校准方法，直接使用homtransform的calibrate方法
    
    def run_gaze_tracking(self):
        """运行视线追踪"""
        print("开始视线追踪...")
        print("按ESC键退出")
        print("SfM功能已启用")
        
        # FPS计算变量
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0
        
        # 用于SfM的前一帧
        frame_prev = None
        
        # 3D可视化器已移除
        self.visual_3d = None
        
        # 移除本地缓存变量，直接使用SFM模块内部的缓存机制
        # cached_face_features_prev = None
        # cached_face_features_curr = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 优化：每帧只检测一次人脸
            face_boxes = self.model.face_detection.predict(frame)
            
            # 使用检测到的人脸框调用get_gaze方法
            eye_info = self.model.get_gaze(frame=frame, face_boxes=face_boxes, imshow=False)
            
            if eye_info is not None:
                # 获取3D视线向量
                gaze = eye_info['gaze']
                
                # 使用SfM进行视线映射
                try:
                    if frame_prev is not None:
                        try:
                            # 优化：只在必要时重新计算人脸特征点
                            # 对于当前帧，我们总是需要计算新的特征点
                            face_features_curr = self.model.get_FaceFeatures(frame, face_boxes=face_boxes)
                            
                            # 对于前一帧，我们可以使用上一次计算的当前帧特征点
                            # 这样避免了每帧都重新计算两个帧的特征点
                            cached_prev_features = self.homtrans.sfm.get_cached_face_features('curr')
                            if cached_prev_features is not None:
                                face_features_prev = cached_prev_features
                            else:
                                face_features_prev = self.model.get_FaceFeatures(frame_prev, face_boxes=face_boxes)
                            
                            # 尝试使用SfM方法，并传入预计算的特征点
                            # 获取世界坐标系中的视线方向
                            WTransG1, WTransG2, W_P = self.homtrans.sfm.get_GazeToWorld(
                                self.model, frame_prev, frame, 
                                face_features_prev=face_features_prev, 
                                face_features_curr=face_features_curr
                            )
                            
                            # 使用SfM方法将3D视线向量转换为2D屏幕坐标
                            # 如果确定了目标屏幕，使用对应的校准数据
                            if hasattr(self, 'current_screen_index'):
                                FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG1, self.current_screen_index)
                            else:
                                FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG1)
                            
                            # 更新SfM模块的缓存
                            self.homtrans.sfm.update_caches(
                                frame_prev_features=face_features_prev,
                                frame_curr_features=face_features_curr
                            )
                        except Exception as sfm_error:
                            # SfM计算失败，回退到普通方法
                            print(f"SfM计算失败，回退到普通方法: {sfm_error}")
                            # 如果确定了目标屏幕，使用对应的校准数据
                            if hasattr(self, 'current_screen_index'):
                                FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze, self.current_screen_index)
                            else:
                                FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                            # 清除缓存以避免错误数据影响下一次计算
                            self.homtrans.sfm.clear_caches()
                    else:
                        # 初始帧使用普通方法
                        # 如果确定了目标屏幕，使用对应的校准数据
                        if hasattr(self, 'current_screen_index'):
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze, self.current_screen_index)
                        else:
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 将毫米坐标转换为像素坐标
                    screen_pos_mm = FSgaze.flatten()[:2]
                    screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                    
                    # 独立屏幕模式下的注视点判断
                    if hasattr(self, 'independent_screens') and self.independent_screens and len(self.screen_configs) >= 2:
                        # 初始化当前屏幕索引（如果尚未设置）
                        if not hasattr(self, 'current_screen_index'):
                            self.current_screen_index = 0  # 默认使用主屏幕
                        
                        # 获取当前屏幕配置
                        current_screen = self.screen_configs[self.current_screen_index]
                        
                        # 直接使用屏幕的实际坐标范围进行判断
                        screen_left = current_screen.get('left', 0)
                        screen_top = current_screen.get('top', 0)
                        screen_width = current_screen.get('width', 1920)
                        screen_height = current_screen.get('height', 1080)
                        
                        # 检查坐标是否在当前屏幕的物理范围内
                        gaze_in_screen = (
                            screen_left <= screen_pos_px[0] <= screen_left + screen_width and
                            screen_top <= screen_pos_px[1] <= screen_top + screen_height
                        )
                        
                        # 如果在当前屏幕范围内，直接使用当前屏幕的坐标
                        if gaze_in_screen:
                            gaze_x = screen_pos_px[0] - screen_left
                            gaze_y = screen_pos_px[1] - screen_top
                            
                            # 修复屏幕2向上偏移问题（屏幕2的top=148）
                            if screen_top == 148:  # 屏幕2的垂直偏移补偿
                                gaze_y = max(0, min(gaze_y + 148, screen_height))
                        
                        # 如果不在当前屏幕，检查是否在其他屏幕上
                        else:
                            # 遍历所有屏幕，找到包含该坐标的屏幕
                            found_screen = False
                            for i, screen in enumerate(self.screen_configs):
                                screen_left_i = screen.get('left', 0)
                                screen_top_i = screen.get('top', 0)
                                screen_width_i = screen.get('width', 1920)
                                screen_height_i = screen.get('height', 1080)
                                
                                if (screen_left_i <= screen_pos_px[0] <= screen_left_i + screen_width_i and
                                    screen_top_i <= screen_pos_px[1] <= screen_top_i + screen_height_i):
                                    
                                    # 找到匹配的屏幕，切换到该屏幕
                                    if i != self.current_screen_index:
                                        old_screen_index = self.current_screen_index
                                        self.current_screen_index = i
                                        current_screen = screen
                                        
                                        # 加载该屏幕的校准数据
                                        try:
                                            self.homtrans.load_screen_calibration(self.current_screen_index)
                                            print(f"自动切换到屏幕 {self.current_screen_index + 1} (从屏幕 {old_screen_index + 1})")
                                        except Exception as e:
                                            print(f"切换校准数据失败: {e}")
                                    
                                    # 计算该屏幕内的坐标
                                    gaze_x = screen_pos_px[0] - screen_left_i
                                    gaze_y = screen_pos_px[1] - screen_top_i
                                    found_screen = True
                                    break
                            
                            # 如果没有找到任何匹配的屏幕，使用当前屏幕并在边界限制
                            if not found_screen:
                                gaze_x = max(0, min(screen_pos_px[0] - screen_left, screen_width))
                                gaze_y = max(0, min(screen_pos_px[1] - screen_top, screen_height))
                        

                        
                    else:
                        # 单屏幕模式或传统模式，使用默认屏幕
                        gaze_x = max(0, min(screen_pos_px[0], self.ui.width))
                        gaze_y = max(0, min(screen_pos_px[1], self.ui.height))
                    
                    # 应用卡尔曼滤波平滑视线点
                    if self.kalman_enabled:
                        smoothed_gaze = self.kalman_filter.update([gaze_x, gaze_y])
                        gaze_point = (int(smoothed_gaze[0]), int(smoothed_gaze[1]))
                    else:
                        gaze_point = (int(gaze_x), int(gaze_y))
                except Exception as e:
                    print(f"视线映射出错: {e}")
                    gaze_point = None
            else:
                gaze_point = None
            
            # 更新前一帧
            frame_prev = frame.copy()
            
            # 计算FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            if elapsed_time >= 1.0:  # 每秒更新一次FPS
                current_fps = frame_count / elapsed_time
                print(f"FPS: {current_fps:.1f}")
                frame_count = 0
                fps_start_time = current_time
            
            # 3D可视化功能已移除
            pass
            
            # 显示结果
            # 在双屏幕模式下传递当前屏幕信息
            screen_info = None
            if self.dual_screen_mode and len(self.screen_configs) >= 2 and hasattr(self, 'current_screen_index'):
                # 获取当前屏幕配置
                current_screen_config = self.screen_configs[self.current_screen_index]
                
                # 计算屏幕中心点相对于当前显示窗口的坐标（局部坐标）
                screen1_center_local = (
                    self.screen_configs[0]['center_x'] - current_screen_config['left'],
                    self.screen_configs[0]['center_y'] - current_screen_config['top']
                )
                screen2_center_local = (
                    self.screen_configs[1]['center_x'] - current_screen_config['left'],
                    self.screen_configs[1]['center_y'] - current_screen_config['top']
                )
                
                screen_info = {
                    'current_screen': self.current_screen_index + 1,
                    'total_screens': len(self.screen_configs),
                    'screen1_center': screen1_center_local,
                    'screen2_center': screen2_center_local,
                    'current_screen_center': (current_screen_config['center_x'] - current_screen_config['left'], 
                                             current_screen_config['center_y'] - current_screen_config['top'])
                }
                
                # 在正确的屏幕上显示注视点
                toggle_button_rect, toggle_3d_button_rect = self.ui.show_gaze_tracking_on_screen(
                    self.current_screen_index, gaze_point, self.kalman_enabled, self.visual_3d_enabled, screen_info, self.dual_screen_mode)
            else:
                # 单屏幕模式
                toggle_button_rect, toggle_3d_button_rect = self.ui.show_gaze_tracking_screen(
                    frame, gaze_point, self.kalman_enabled, self.visual_3d_enabled, screen_info)
            
            event = self.ui.handle_events()
            if event == 'quit':
                break
            elif event == 'toggle_kalman':
                # 切换卡尔曼滤波状态
                self.kalman_enabled = not self.kalman_enabled
                print(f"卡尔曼滤波已{'启用' if self.kalman_enabled else '禁用'}")
            elif event == 'toggle_3d':
                # 3D可视化功能已移除
                pass
            elif event == 'click':
                mouse_pos = pygame.mouse.get_pos()
                if toggle_button_rect.collidepoint(mouse_pos):
                    # 切换卡尔曼滤波状态
                    self.kalman_enabled = not self.kalman_enabled
                    print(f"卡尔曼滤波已{'启用' if self.kalman_enabled else '禁用'}")
                elif toggle_3d_button_rect.collidepoint(mouse_pos):
                    # 3D可视化功能已移除
                    pass
            
            # 提高帧率限制，以展示缓存机制带来的性能提升
            # 移除固定帧率限制，让程序以最大可能速度运行
            # self.ui.clock.tick(30)
            pass
    
    def run(self):
        """运行主程序"""
        try:
            # 初始化
            print("开始初始化...")
            if not self.initialize():
                print("初始化失败")
                return
            print("初始化成功")
            
            # 显示主菜单
            print("进入主菜单...")
            menu_result = self.show_menu()
            if menu_result == 'quit':
                print("用户从主菜单退出")
                return
            print(f"主菜单选择结果: {menu_result}")
            
            # 运行校准
            print("开始校准...")
            if not self.run_calibration():
                print("校准失败或用户取消")
                return
            print("校准完成")
            
            # 运行视线追踪
            print("开始视线追踪...")
            self.run_gaze_tracking()
            print("视线追踪结束")
            
        except Exception as e:
            print(f"运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        # 3D可视化功能已移除
        if self.cap:
            self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    # 获取项目根目录
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        project_dir = os.path.dirname(current_dir)
    else:
        project_dir = current_dir
    
    print(f"项目目录: {project_dir}")
    
    # 创建并运行系统
    gaze_system = PygameGazeSystem(project_dir)
    gaze_system.run()

if __name__ == '__main__':
    main()