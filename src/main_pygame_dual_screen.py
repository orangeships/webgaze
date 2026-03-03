import os
import sys
import pygame
import cv2
import time
import numpy as np
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel
from gaze_tracking.gaze_smoothing import KalmanFilter

# Pygame初始化
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
RED = (231, 76, 60)

# 导入screeninfo用于多显示器检测
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    print("警告：screeninfo未安装，可能无法正确检测多显示器")

class PygameUI:
    def __init__(self, width=None, height=None, display_index=0):
        # 如果提供了宽度和高度，使用它们，否则自动检测
        if width and height:
            self.width = width
            self.height = height
            self.display_index = display_index
            print(f"使用指定的分辨率和显示器: {width}x{height}, 显示器索引: {display_index}")
        else:
            # 获取真实的屏幕尺寸（考虑DPI缩放）
            import ctypes
            try:
                # Windows系统下获取真实物理分辨率
                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                self.width = user32.GetSystemMetrics(0)
                self.height = user32.GetSystemMetrics(1)
                self.display_index = 0  # 默认主显示器
                print(f"Windows系统物理分辨率: {self.width}x{self.height}")
            except:
                # 备用方案：使用pygame
                screen_info = pygame.display.Info()
                self.width = screen_info.current_w
                self.height = screen_info.current_h
                self.display_index = 0  # 默认主显示器
                print(f"Pygame检测分辨率: {self.width}x{self.height}")
        
        self.screen = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.clock = pygame.time.Clock()
        
    def initialize_display(self):
        """初始化pygame显示
        """
        window_x, window_y = 0, 0
        target_monitor_info = None
        
        # 详细的多显示器检测和显示信息
        print("原始显示器顺序:")
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
            # 重要：显示器必须按x坐标排序，确保索引0对应左侧主显示器
            sorted_monitors = sorted(monitors, key=lambda m: m.x)
            
            for i, monitor in enumerate(sorted_monitors):
                monitor_type = " (主显示器)" if monitor.x == 0 and monitor.y == 0 else ""
                print(f"  原始索引{i}: 分辨率{monitor.width}x{monitor.height}, 位置({monitor.x}, {monitor.y}){monitor_type}")
            
            if self.display_index < len(sorted_monitors):
                target_monitor = sorted_monitors[self.display_index]
                # 在目标显示器上创建窗口，位置设为显示器的起始坐标
                window_x = target_monitor.x
                window_y = target_monitor.y
                target_monitor_info = target_monitor
                print(f"使用指定的分辨率和显示器: {self.width}x{self.height}, 显示器索引: {self.display_index}")
                print(f"在显示器 {self.display_index} (位置: {target_monitor.x}, {target_monitor.y}, 分辨率: {target_monitor.width}x{target_monitor.height}) 创建窗口")
            else:
                print(f"显示器索引 {self.display_index} 无效，在默认位置创建窗口")
        except Exception as e:
            # 如果无法获取显示器信息，使用默认位置
            print(f"无法获取显示器位置信息，在默认位置创建窗口: {e}")
        
        # 在创建pygame窗口之前设置SDL环境变量
        # 始终设置SDL位置，确保窗口在正确的显示器创建
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
        
        
        # 初始化pygame
        pygame.init()
        pygame.display.init()
        
        try:
            # 在指定显示器上创建窗口
            self.screen = pygame.display.set_mode((self.width, self.height), 
                                                 pygame.NOFRAME | pygame.RESIZABLE, 
                                                 display=self.display_index)
            pygame.display.set_caption("视线追踪系统")
            

                
        except Exception as e:
            print(f"在显示器 {self.display_index} 创建窗口失败，回退到主显示器: {e}")
            # 回退到主显示器
            try:
                self.screen = pygame.display.set_mode((self.width, self.height), 
                                                     pygame.NOFRAME | pygame.RESIZABLE)
                pygame.display.set_caption("视线追踪系统")
                print("已回退到主显示器")
            except Exception as e2:
                print(f"回退创建窗口失败: {e2}")
                raise e2
        
        # 验证实际显示分辨率
        actual_width, actual_height = self.screen.get_size()
        print(f"实际显示分辨率: {actual_width}x{actual_height}")
        print(f"使用指定的分辨率: {self.width}x{self.height}")
        
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
        
        # 初始化时钟
        self.clock = pygame.time.Clock()
    
    def initialize_display_at_position(self, x, y):
        """在指定位置初始化pygame显示
        
        Args:
            x: 窗口x坐标
            y: 窗口y坐标
        """
        # 设置SDL环境变量确保窗口在指定位置
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

        # 调用标准初始化方法
        self.initialize_display()
    
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
    
    def show_calibration_choice_screen(self):
        """显示校准选择界面"""
        while True:
            self.screen.fill(WHITE)
            
            # 标题
            title_text = self.font_large.render("校准选项", True, BLACK)
            title_rect = title_text.get_rect(center=(self.width // 2, self.height // 4))
            self.screen.blit(title_text, title_rect)
            
            # 说明文字
            instruction_text = self.font_medium.render("请选择校准模式", True, BLACK)
            instruction_rect = instruction_text.get_rect(center=(self.width // 2, self.height // 3 + 40))
            self.screen.blit(instruction_text, instruction_rect)
            
            # 按钮
            single_screen_button = self.draw_button("单屏幕校准", self.width // 3, self.height // 2 - 60, 
                                                  self.width // 3, 80, BLUE)
            dual_screen_button = self.draw_button("双屏幕校准", self.width // 3, self.height // 2 + 70, 
                                                self.width // 3, 80, GREEN)
            load_history_button = self.draw_button("加载历史校准", self.width // 3, self.height // 2 + 200, 
                                                self.width // 3, 80, RED)
            
            pygame.display.flip()
            
            event = self.handle_events()
            if event == 'quit':
                return 'quit'
            elif event == 'click':
                mouse_pos = pygame.mouse.get_pos()
                if single_screen_button.collidepoint(mouse_pos):
                    return 'single'
                elif dual_screen_button.collidepoint(mouse_pos):
                    return 'dual'
                elif load_history_button.collidepoint(mouse_pos):
                    return 'load'
            
            self.clock.tick(60)
    
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
    
    def show_gaze_tracking_screen(self, frame, gaze_point=None, kalman_enabled=True):
        """显示视线追踪界面"""
        # 显示简单的追踪界面，不显示摄像头画面
        self.screen.fill(WHITE)
        
        # 标题
        title_text = self.font_large.render("视线追踪模式", True, BLACK)
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title_text, title_rect)
        
        # 说明文字
        instruction_text = self.font_medium.render("正在追踪您的视线...", True, BLACK)
        instruction_rect = instruction_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(instruction_text, instruction_rect)
        
        # 如果有视线点，绘制它
        if gaze_point:
            pygame.draw.circle(self.screen, RED, gaze_point, 15)
        
        # 卡尔曼滤波状态显示
        kalman_status = "启用" if kalman_enabled else "禁用"
        kalman_color = GREEN if kalman_enabled else RED
        kalman_text = self.font_small.render(f"卡尔曼滤波: {kalman_status}", True, kalman_color)
        kalman_rect = kalman_text.get_rect(center=(self.width // 2, self.height // 2 + 80))
        self.screen.blit(kalman_text, kalman_rect)
        
        # 创建切换滤波按钮
        toggle_button_rect = self.draw_button(
            "切换滤波", 
            self.width // 2 - 100, 
            self.height // 2 + 160, 
            200, 50, 
            kalman_color
        )
        
        # 快捷键说明
        shortcut_text = self.font_small.render("快捷键: F-滤波, ESC-退出", True, GRAY)
        shortcut_rect = shortcut_text.get_rect(center=(self.width // 2, 3 * self.height // 4))
        self.screen.blit(shortcut_text, shortcut_rect)
        
        # 退出说明
        exit_text = self.font_small.render("按ESC键退出", True, GRAY)
        exit_rect = exit_text.get_rect(center=(self.width // 2, 3 * self.height // 4 + 40))
        self.screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()
        
        return toggle_button_rect
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                elif event.key == pygame.K_s:
                    return 'start'
                elif event.key == pygame.K_f:
                    return 'toggle_kalman'

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
        
        # 双屏幕注视点显示相关状态变量
        self.current_screen_index = 0  # 当前显示的屏幕索引，默认为主屏幕
        self.screen_boundaries = []  # 存储每个屏幕的边界信息 [left, right, top, bottom]
        self.screen_switching = False  # 标志位，防止屏幕切换过程中重复触发
        self.last_gaze_point = None  # 存储上一帧的注视点，用于平滑过渡
        
        # 双屏幕校准相关
        self.monitors = []
        self.calibration_results = {}  # 存储每个显示器的校准结果
        
    def get_monitors_info(self):
        """获取所有显示器信息，按物理位置排序（左到右）"""
        if SCREENINFO_AVAILABLE:
            raw_monitors = get_monitors()
            
            # 关键修复：按x坐标排序显示器，确保左侧显示器索引为0
            # 这符合用户期望：主显示器在左，索引为0；副显示器在右，索引为1
            sorted_monitors = sorted(raw_monitors, key=lambda m: m.x)
            
            # 记录原始映射关系，用于调试
            print(f"原始显示器顺序:")
            for i, monitor in enumerate(raw_monitors):
                is_primary = " (主显示器)" if hasattr(monitor, 'is_primary') and monitor.is_primary else ""
                print(f"  原始索引{i}: 分辨率{monitor.width}x{monitor.height}, 位置({monitor.x}, {monitor.y}){is_primary}")
            
            self.monitors = sorted_monitors
            
            # 计算并存储每个屏幕的边界信息 [left, right, top, bottom]
            self.screen_boundaries = []
            for i, monitor in enumerate(self.monitors):
                # 计算屏幕边界
                left = monitor.x
                right = monitor.x + monitor.width
                top = monitor.y
                bottom = monitor.y + monitor.height
                self.screen_boundaries.append([left, right, top, bottom])
        else:
            # 如果screeninfo不可用，仅使用主显示器
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            self.monitors = [type('Monitor', (), {'width': width, 'height': height, 'x': 0, 'y': 0, 'is_primary': True})()]
            print(f"仅检测到主显示器: 分辨率 {width}x{height}")
            
            # 计算主屏幕边界
            self.screen_boundaries = [[0, width, 0, height]]
        
    def initialize(self):
        """初始化系统"""
        # 检测所有显示器
        self.get_monitors_info()
        
        # 修复：确保系统默认在主显示器（左侧2K显示器，索引0）启动
        # 使用主显示器的分辨率和索引
        main_monitor = self.monitors[0]  # 重新排序后索引0是主显示器（左侧）
        
        # 初始化UI，明确指定使用主显示器
        self.ui = PygameUI(width=main_monitor.width, height=main_monitor.height, display_index=0)
        self.ui.initialize_display()
        
        # 设置当前屏幕索引为主显示器
        self.current_screen_index = 0
        
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
        # 显示校准模式选择
        calibration_mode = self.ui.show_calibration_choice_screen()
        if calibration_mode == 'quit':
            return False
        
        if calibration_mode == 'load':
            # 加载历史校准结果
            self.load_calibration_results()
            if len(self.calibration_results) > 0:
                print("成功加载历史校准结果，跳过校准流程")
                return True
            else:
                print("未找到有效的历史校准结果，开始校准")
                # 加载失败，回退到单屏幕校准
                return self.perform_single_screen_calibration()
        elif calibration_mode == 'single':
            # 单屏幕校准
            return self.perform_single_screen_calibration()
        else:
            # 双屏幕校准
            return self.perform_dual_screen_calibration()
    
    def perform_single_screen_calibration(self):
        """执行单屏幕校准"""
        try:
            STransG = self.homtrans.calibrate(self.model, self.cap, sfm=True, display_index=0)
            if STransG is not None:
                print("单屏幕校准成功完成")
                self.calibration_data = STransG
                self.calibration_results[0] = STransG
                # 重新初始化Pygame显示，因为校准界面可能已清理
                self.ui.initialize_display()
                return True
            else:
                print("单屏幕校准失败或被用户取消")
                # 重新初始化Pygame显示
                self.ui.initialize_display()
                return False
        except Exception as e:
            print(f"单屏幕校准过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 重新初始化Pygame显示
            self.ui.initialize_display()
            return False
    
    def perform_dual_screen_calibration(self):
        """执行双屏幕校准"""
        if len(self.monitors) < 2:
            print("未检测到双显示器，回退到单屏幕校准")
            return self.perform_single_screen_calibration()
        
        try:
            # 依次在每个显示器上进行校准
            for monitor_index in range(len(self.monitors)):
                monitor = self.monitors[monitor_index]
                print(f"\n开始校准显示器 {monitor_index}: {monitor.width}x{monitor.height}")
                
                # 重新初始化UI以适应当前显示器
                current_ui = PygameUI(width=monitor.width, height=monitor.height, display_index=monitor_index)
                current_ui.initialize_display()
                
                # 创建针对当前显示器的HomTransform实例，传入当前显示器的分辨率
                current_homtrans = HomTransform(self.project_dir, custom_width=monitor.width, custom_height=monitor.height)
                
                # 生成针对当前显示器的校准文件名
                calibration_filename = f"calibration_results_screen_{monitor_index}.json"
                
                # 执行校准
                STransG = current_homtrans.calibrate(self.model, self.cap, sfm=True, display_index=monitor_index, filename=calibration_filename)
                
                if STransG is not None:
                    print(f"显示器 {monitor_index} 校准成功完成")
                    self.calibration_results[monitor_index] = {
                        'STransG': STransG,
                        'homtrans': current_homtrans,
                        'width': monitor.width,
                        'height': monitor.height,
                        'x': monitor.x,
                        'y': monitor.y
                    }
                    # 校准结果已在校准过程中自动保存到 calibration_results_screen_{monitor_index}.json
                else:
                    print(f"显示器 {monitor_index} 校准失败或被用户取消")
                    return False
            
            # 校准完成后，恢复主显示器的UI
            self.ui.initialize_display()
            print("\n双屏幕校准全部完成！")
            return True
        except Exception as e:
            print(f"双屏幕校准过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 重新初始化Pygame显示
            self.ui.initialize_display()
            return False
    
    def save_calibration_results(self, monitor_index):
        """保存特定显示器的校准结果"""
        
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
    
    def load_calibration_results(self):
        """加载所有显示器的校准结果"""
        import json
        import numpy as np
        
        results_dir = os.path.join(self.project_dir, "results")
        
        for monitor_index in range(len(self.monitors)):
            calibration_file = os.path.join(results_dir, f"calibration_results_screen_{monitor_index}.json")
            if os.path.exists(calibration_file):
                print(f"正在加载显示器 {monitor_index} 的校准结果: {calibration_file}")
                try:
                    # 创建当前显示器的HomTransform实例
                    monitor = self.monitors[monitor_index]
                    current_homtrans = HomTransform(self.project_dir, custom_width=monitor.width, custom_height=monitor.height)
                    
                    # 使用HomTransform内置的加载方法加载校准结果
                    if current_homtrans.load_calibration_results(calibration_file):
                        self.calibration_results[monitor_index] = {
                            'STransG': current_homtrans.STransG,
                            'homtrans': current_homtrans,
                            'width': monitor.width,
                            'height': monitor.height,
                            'x': monitor.x,
                            'y': monitor.y
                        }
                        print(f"显示器 {monitor_index} 校准结果加载成功")
                    else:
                        print(f"显示器 {monitor_index} 校准结果加载失败")
                except Exception as e:
                    print(f"加载显示器 {monitor_index} 校准结果失败: {e}")
    
    def is_gaze_out_of_screen(self, gaze_x, gaze_y, screen_index):
        """判断注视点是否超出指定屏幕边界
        
        Args:
            gaze_x: 注视点x坐标
            gaze_y: 注视点y坐标
            screen_index: 屏幕索引
            
        Returns:
            bool: 超出边界返回True，否则返回False
        """
        if screen_index >= len(self.screen_boundaries):
            return True
            
        left, right, top, bottom = self.screen_boundaries[screen_index]
        
        # 考虑一定的边界阈值，避免频繁切换
        threshold = 20  # 20像素的边界阈值
        
        return gaze_x < left - threshold or gaze_x > right + threshold or gaze_y < top - threshold or gaze_y > bottom + threshold
    
    def get_target_screen(self, gaze_x, gaze_y):
        """根据注视点位置获取目标屏幕索引
        
        Args:
            gaze_x: 注视点x坐标
            gaze_y: 注视点y坐标
            
        Returns:
            int: 目标屏幕索引，如果没有找到则返回当前屏幕索引
        """
        for i, (left, right, top, bottom) in enumerate(self.screen_boundaries):
            if left <= gaze_x <= right and top <= gaze_y <= bottom:
                return i
        return self.current_screen_index
    
    def switch_screen(self, target_screen_index):
        """切换到目标屏幕
        
        Args:
            target_screen_index: 目标屏幕索引
            
        Returns:
            bool: 切换成功返回True，否则返回False
        """
        if target_screen_index == self.current_screen_index or target_screen_index >= len(self.monitors):
            return False
            
        try:
            print(f"开始切换到显示器 {target_screen_index}")
            self.screen_switching = True
            
            # 保存当前状态
            kalman_enabled = self.kalman_enabled
            last_gaze_point = self.last_gaze_point
            
            # 关键修复：使用目标显示器的物理位置来设置窗口位置
            target_monitor = self.monitors[target_screen_index]
            window_pos_x = target_monitor.x
            window_pos_y = target_monitor.y
            
            # 在创建pygame窗口之前设置SDL环境变量确保窗口在正确位置
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_pos_x},{window_pos_y}"
            
            # 创建黑色过渡屏幕，然后创建新窗口在目标位置
            pygame.display.set_mode((100, 100), pygame.NOFRAME, display=target_screen_index)
            pygame.display.flip()
            pygame.time.delay(100)
            
            # 重新初始化UI以适应当前显示器，同时确保窗口在正确位置
            new_ui = PygameUI(width=target_monitor.width, height=target_monitor.height, display_index=target_screen_index)
            new_ui.initialize_display_at_position(target_monitor.x, target_monitor.y)
            
            # 关键修复：创建新的HomTransform实例以匹配目标显示器的分辨率
            # 这与校准阶段的行为一致，确保视线计算使用正确的坐标系统
            new_homtrans = HomTransform(self.project_dir, custom_width=target_monitor.width, custom_height=target_monitor.height)
            
            # 加载并应用对应屏幕的校准结果到新的HomTransform实例
            if target_screen_index in self.calibration_results:
                calibration = self.calibration_results[target_screen_index]
                new_homtrans.STransG = calibration['STransG']
                
                # 设置其他必要属性
                if hasattr(calibration['homtrans'], 'scaleWtG'):
                    new_homtrans.scaleWtG = calibration['homtrans'].scaleWtG
                if hasattr(calibration['homtrans'], 'StG'):
                    new_homtrans.StG = calibration['homtrans'].StG
                if hasattr(calibration['homtrans'], 'StW'):
                    new_homtrans.StW = calibration['homtrans'].StW
                if hasattr(calibration['homtrans'], 'STransW'):
                    new_homtrans.STransW = calibration['homtrans'].STransW
                
                print(f"成功切换到显示器 {target_screen_index}，加载了对应校准结果")
            else:
                print(f"警告：显示器 {target_screen_index} 没有校准结果")
            
            # 切换UI和HomTransform实例
            self.ui = new_ui
            self.homtrans = new_homtrans
            
            # 更新当前屏幕索引
            self.current_screen_index = target_screen_index
            
            # 恢复卡尔曼滤波状态
            self.kalman_enabled = kalman_enabled
            
            # 如果有上一帧注视点，将其转换为新屏幕的坐标并更新卡尔曼滤波器
            if last_gaze_point:
                # 将绝对坐标转换为新屏幕的相对坐标
                new_gaze_x = last_gaze_point[0] - self.monitors[target_screen_index].x
                new_gaze_y = last_gaze_point[1] - self.monitors[target_screen_index].y
                
                # 更新卡尔曼滤波器，确保切换后注视点平滑过渡
                self.kalman_filter.update([last_gaze_point[0], last_gaze_point[1]])
            
            # 再次短暂延迟，确保显示稳定
            pygame.time.delay(50)
            
            # 清除SDL环境变量
            if 'SDL_VIDEO_WINDOW_POS' in os.environ:
                del os.environ['SDL_VIDEO_WINDOW_POS']
            
            self.screen_switching = False
            print(f"成功切换到显示器 {target_screen_index}")
            return True
        except Exception as e:
            print(f"切换屏幕失败: {e}")
            import traceback
            traceback.print_exc()
            self.screen_switching = False
            return False
    
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
        
        # 如果有多个显示器的校准结果，初始化主homtrans为第一个显示器的结果
        if len(self.calibration_results) > 0:
            monitor_index = list(self.calibration_results.keys())[0]
            self.homtrans.STransG = self.calibration_results[monitor_index]['STransG']
            # 如果存在其他必要属性，也需要设置
            if hasattr(self.calibration_results[monitor_index]['homtrans'], 'scaleWtG'):
                self.homtrans.scaleWtG = self.calibration_results[monitor_index]['homtrans'].scaleWtG
            if hasattr(self.calibration_results[monitor_index]['homtrans'], 'StG'):
                self.homtrans.StG = self.calibration_results[monitor_index]['homtrans'].StG
            if hasattr(self.calibration_results[monitor_index]['homtrans'], 'StW'):
                self.homtrans.StW = self.calibration_results[monitor_index]['homtrans'].StW
            if hasattr(self.calibration_results[monitor_index]['homtrans'], 'STransW'):
                self.homtrans.STransW = self.calibration_results[monitor_index]['homtrans'].STransW
            # 校准结果已加载，继续运行视线追踪
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 优化：每帧只检测一次人脸
            face_boxes = self.model.face_detection.predict(frame)
            
            # 使用检测到的人脸框调用get_gaze方法
            eye_info ,landmarks= self.model.get_gaze(frame=frame, face_boxes=face_boxes, imshow=False)
            
            gaze_point = None
            if eye_info is not None:
                # 获取3D视线向量
                gaze = eye_info['gaze']
                
                # 根据当前homtrans计算初步视线位置，用于确定使用哪个显示器的校准结果
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
                            

                            WTransG1, WTransG2, W_P = self.homtrans.sfm.get_GazeToWorld(
                                self.model, frame_prev, frame, 
                                face_features_prev=face_features_prev, 
                                face_features_curr=face_features_curr
                            )
                            
                            # 使用SfM方法将3D视线向量转换为2D屏幕坐标
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG1)
                            
                            # 更新SfM模块的缓存
                            self.homtrans.sfm.update_caches(
                                frame_prev_features=face_features_prev,
                                frame_curr_features=face_features_curr
                            )
                        except Exception as sfm_error:
                            # SfM计算失败，回退到普通方法
                            # print(f"SfM计算失败，回退到普通方法: {sfm_error}")
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                            # 清除缓存以避免错误数据影响下一次计算
                            self.homtrans.sfm.clear_caches()
                    else:
                        # 初始帧使用普通方法
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 将毫米坐标转换为像素坐标
                    screen_pos_mm = FSgaze.flatten()[:2]
                    screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                    
                    # 计算绝对屏幕坐标（考虑显示器位置偏移）
                    gaze_x_abs = int(screen_pos_px[0] + self.monitors[self.current_screen_index].x)
                    gaze_y_abs = int(screen_pos_px[1] + self.monitors[self.current_screen_index].y)
                    
                    # 应用卡尔曼滤波平滑视线点（使用绝对坐标）
                    if self.kalman_enabled:
                        smoothed_gaze_abs = self.kalman_filter.update([gaze_x_abs, gaze_y_abs])
                        gaze_point_abs = (int(smoothed_gaze_abs[0]), int(smoothed_gaze_abs[1]))
                    else:
                        gaze_point_abs = (gaze_x_abs, gaze_y_abs)
                    
                    # 计算相对当前屏幕的注视点坐标
                    gaze_x = gaze_point_abs[0] - self.monitors[self.current_screen_index].x
                    gaze_y = gaze_point_abs[1] - self.monitors[self.current_screen_index].y
                    gaze_point = (gaze_x, gaze_y)
                    
                    # 实时监测注视点位置，实现屏幕自动切换
                    if not self.screen_switching and len(self.monitors) > 1:
                        # 检查注视点是否超出当前屏幕边界
                        if self.is_gaze_out_of_screen(gaze_point_abs[0], gaze_point_abs[1], self.current_screen_index):
                            # 获取目标屏幕索引
                            target_screen = self.get_target_screen(gaze_point_abs[0], gaze_point_abs[1])
                            
                            # 切换到目标屏幕
                            if target_screen != self.current_screen_index:
                                self.switch_screen(target_screen)
                except Exception as e:
                    print(f"视线映射出错: {e}")
                    gaze_point = None
            else:
                gaze_point = None
            
            # 更新前一帧
            frame_prev = frame.copy()
            
            # 保存当前注视点的绝对坐标，用于屏幕切换时的平滑过渡
            if gaze_point:
                # 计算当前注视点的绝对坐标
                gaze_x_abs = gaze_point[0] + self.monitors[self.current_screen_index].x
                gaze_y_abs = gaze_point[1] + self.monitors[self.current_screen_index].y
                self.last_gaze_point = (gaze_x_abs, gaze_y_abs)
            
            # 计算FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            if elapsed_time >= 1.0:  # 每秒更新一次FPS
                current_fps = frame_count / elapsed_time
                print(f"FPS: {current_fps:.1f}, 当前屏幕: {self.current_screen_index}")
                frame_count = 0
                fps_start_time = current_time
            
            
            # 显示结果
            toggle_button_rect = self.ui.show_gaze_tracking_screen(
                frame, gaze_point, self.kalman_enabled)
            
            event = self.ui.handle_events()
            if event == 'quit':
                break
            elif event == 'toggle_kalman':
                # 切换卡尔曼滤波状态
                self.kalman_enabled = not self.kalman_enabled
                print(f"卡尔曼滤波已{'启用' if self.kalman_enabled else '禁用'}")
            elif event == 'click':
                mouse_pos = pygame.mouse.get_pos()
                if toggle_button_rect.collidepoint(mouse_pos):
                    # 切换卡尔曼滤波状态
                    self.kalman_enabled = not self.kalman_enabled
                    print(f"卡尔曼滤波已{'启用' if self.kalman_enabled else '禁用'}")
    
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