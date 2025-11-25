import pygame
import numpy as np
import time
import os
import ctypes
import platform

class PygameCalibrationUI:
    """Pygame校准界面类"""
    
    def __init__(self, width, height, screen_offset=None, screen_index=None):
        self.width = width
        self.height = height
        self.screen_offset = screen_offset or {'x': 0, 'y': 0}
        self.screen_index = screen_index
        self.screen = None
        self.clock = pygame.time.Clock()
        
        # 打印屏幕信息用于调试
        print(f"创建PygameCalibrationUI - 屏幕{screen_index + 1 if screen_index is not None else 1}")
        # print(f"屏幕尺寸: {self.width}x{self.height}")
        print(f"屏幕偏移: {self.screen_offset}")
        print(f"屏幕索引: {self.screen_index}")
        
        # 颜色定义
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (231, 76, 60)
        self.BLUE = (52, 152, 219)
        self.GREEN = (46, 204, 113)
        self.GRAY = (200, 200, 200)
        
        # 字体
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        
        # 校准状态
        self.calibration_started = False
        self.current_point_idx = 0
        self.start_time = 0
        self.point_start_time = 0  # 当前校准点显示的开始时间
        self.delay_recording = 0.5  # 记录延迟时间（秒）
        self.can_record = False  # 是否可以开始记录数据
        
        # 校准点配置
        self.margin_ratio = 0.15  # 边距比例 - 增加到15%以避免校准点过于靠近屏幕边缘
        self.calib_points = self._build_grid_points()
        self.total_points = len(self.calib_points)
        
        # 动画相关变量
        self.animation_time = 0
        self.animation_speed = 4  # 提高动画速度到原来的3倍
        self.base_radius = 25  # 基础半径
        self.pulse_radius = 35  # 增加脉动范围以增强视觉效果
        
    def initialize(self, x=None, y=None, fullscreen=False, screen_index=None):
        """初始化Pygame显示
        
        Args:
            x: 窗口的x坐标（可选，屏幕绝对坐标）
            y: 窗口的y坐标（可选，屏幕绝对坐标）
            fullscreen: 是否使用全屏模式（默认False，使用窗口模式）
            screen_index: 屏幕索引，用于多屏幕校准
        """
        # 保存窗口位置信息
        self.window_x = x
        self.window_y = y
        self.screen_index = screen_index
        
        # 设置显示模式 - 使用标准窗口模式而非无边框模式
        pygame.display.init()
        
        print(f"初始化校准UI - 屏幕{screen_index + 1 if screen_index is not None else 1}")
        # print(f"窗口位置: ({x}, {y}), 尺寸: {self.width}x{self.height}")
        
        if fullscreen:
            # 全屏模式使用指定屏幕
            if screen_index is not None and hasattr(self, 'screen_offset'):
                # 计算全屏模式的偏移位置
                screen_x = self.screen_offset.get('x', 0)
                screen_y = self.screen_offset.get('y', 0)
                flags = pygame.FULLSCREEN | pygame.NOFRAME
                self.screen = pygame.display.set_mode((self.width, self.height), flags)
                # 强制移动到正确屏幕
                pygame.time.delay(200)
                self.move_window_to_screen(screen_x, screen_y)
            else:
                flags = pygame.FULLSCREEN | pygame.NOFRAME
                self.screen = pygame.display.set_mode((self.width, self.height), flags)
        else:
            # 窗口模式 - 使用标准窗口避免定位问题
            flags = 0  # 使用标准窗口模式
            
            # 在创建窗口之前设置位置
            if x is not None and y is not None:
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
                # print(f"设置SDL窗口位置到: ({x}, {y})")
            
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            
            # 创建后立即强制移动窗口到指定屏幕位置
            if x is not None and y is not None:
                pygame.time.delay(200)  # 增加延迟确保窗口创建完成
                # print(f"调用Windows API移动校准窗口到: ({x}, {y})")
                self.move_window_to_screen(x, y)
        
        # 设置窗口标题
        pygame.display.set_caption(f"视线追踪校准 - 屏幕{screen_index + 1 if screen_index is not None else 1}")
        
        # 强制更新显示
        pygame.display.flip()
        
        # 初始化字体
        try:
            self.font_large = pygame.font.Font("C:\Windows\Fonts\simhei.ttf", 48)
            self.font_medium = pygame.font.Font("C:\Windows\Fonts\simhei.ttf", 32)
            self.font_small = pygame.font.Font("C:\Windows\Fonts\simhei.ttf", 24)
        except:
            # 如果中文字体不可用，使用默认字体
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
    
    def _build_grid_points(self):
        """构建5点校准网格（支持多屏幕尺寸适配）"""
        # 计算屏幕中心点
        center_x = self.width // 2
        center_y = self.height // 2
        
        print(f"构建校准点网格 - 屏幕{self.screen_index + 1 if self.screen_index is not None else 1}")
        # print(f"屏幕尺寸: {self.width}x{self.height}")
        print(f"屏幕中心: ({center_x}, {center_y})")
        
        # 改进的边距计算 - 根据屏幕尺寸动态调整
        margin_ratio = self._calculate_adaptive_margin()
        margin_x = int(self.width * margin_ratio)
        margin_y = int(self.height * margin_ratio)
        
        print(f"自适应边距比例: {margin_ratio}")
        print(f"计算边距: ({margin_x}, {margin_y})")
        
        # 使用边距而非固定百分比来构建校准点
        points_relative = [
            (margin_x, margin_y),  # 左上角
            (self.width - margin_x, margin_y),  # 右上角
            (margin_x, self.height - margin_y),  # 左下角
            (self.width - margin_x, self.height - margin_y),  # 右下角
            (center_x, center_y)  # 中心点
        ]
        
        # print(f"基于边距的校准点: {points_relative}")
        
        # 保存局部坐标
        self.calibration_points_local = points_relative.copy()
        
        # 总是使用局部坐标在校准UI窗口内显示
        points = points_relative.copy()
        
        # 如果有屏幕偏移信息，记录调试信息
        if hasattr(self, 'screen_offset'):
            print(f"当前屏幕偏移: {self.screen_offset}")
            print(f"屏幕索引: {getattr(self, 'screen_index', 'unknown')}")
            # print(f"屏幕尺寸: {self.width}x{self.height}")
            print(f"屏幕起始位置: ({self.screen_offset['x']}, {self.screen_offset['y']})")
            # 打印校准点的局部坐标信息
            # screen_num = self.screen_index + 1 if self.screen_index is not None else 1
            # for i, (lx, ly) in enumerate(self.calibration_points_local):
            #     print(f"校准点 {i+1}: 局部坐标({lx:.1f}, {ly:.1f}) (在屏幕{screen_num}内显示)")
        else:
            print(f"未设置屏幕偏移信息，使用局部坐标")
        
        return points
    
    def _calculate_adaptive_margin(self):
        """根据屏幕尺寸计算自适应边距比例"""
        # 对于小屏幕，使用较大的边距避免校准点过于靠近边缘
        # 对于大屏幕，可以使用相对较小的边距
        min_dimension = min(self.width, self.height)
        
        if min_dimension < 800:
            return 0.1  # 小屏幕 12%边距
        elif min_dimension < 1200:
            return 0.1  # 中等屏幕 15%边距
        else:
            return 0.1  # 大屏幕 18%边距
    
    def show_start_screen(self):
        """显示开始界面"""
        self.screen.fill(self.WHITE)
        
        # 标题
        title_text = self.font_large.render("眼动追踪校准系统", True, self.BLACK)
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title_text, title_rect)
        
        # 说明文字
        instruction_text1 = self.font_medium.render("按 'S' 键开始校准", True, self.BLACK)
        instruction_rect1 = instruction_text1.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(instruction_text1, instruction_rect1)
        
        instruction_text2 = self.font_medium.render("按 'ESC' 键退出", True, self.BLACK)
        instruction_rect2 = instruction_text2.get_rect(center=(self.width // 2, self.height // 2 + 60))
        self.screen.blit(instruction_text2, instruction_rect2)
        
        # 开始按钮
        button_rect = pygame.Rect(self.width // 3, 2 * self.height // 3, self.width // 3, 80)
        pygame.draw.rect(self.screen, self.GREEN, button_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.BLACK, button_rect, 2, border_radius=10)
        
        button_text = self.font_medium.render("开始校准", True, self.WHITE)
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        pygame.display.flip()
        
        return button_rect
    
    def show_warmup_screen(self, progress, total, stable_frames=0, required_stable=10):
        """显示预热界面"""
        self.screen.fill(self.WHITE)
        
        # 预热文字
        if stable_frames > 0:
            warmup_text = self.font_medium.render(f"模型预热中... {progress}/{total} (稳定帧: {stable_frames}/{required_stable})", True, self.BLACK)
        else:
            warmup_text = self.font_medium.render(f"模型预热中... {progress}/{total}", True, self.BLACK)
        text_rect = warmup_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(warmup_text, text_rect)
        
        # 进度条
        bar_width = 400
        bar_height = 30
        bar_x = (self.width - bar_width) // 2
        bar_y = self.height // 2 + 50
        
        # 背景条
        pygame.draw.rect(self.screen, self.GRAY, (bar_x, bar_y, bar_width, bar_height), border_radius=15)
        
        # 进度条
        progress_width = int(bar_width * progress / total)
        pygame.draw.rect(self.screen, self.BLUE, (bar_x, bar_y, progress_width, bar_height), border_radius=15)
        
        pygame.display.flip()
    
    def show_calibration_point(self, point_idx, gaze_point=None):
        """显示校准点（带律动效果）"""
        self.screen.fill(self.WHITE)
        
        # 打印调试信息
        screen_num = self.screen_index + 1 if self.screen_index is not None else 1
        # 移除重复的校准点调试输出信息
        # print(f"显示校准点 {point_idx + 1} - 屏幕{screen_num}")
        # print(f"屏幕尺寸: {self.width}x{self.height}")
        # print(f"校准点总数: {self.total_points}")
        # print(f"校准点 {point_idx + 1} 局部坐标: {self.calib_points[point_idx] if point_idx < self.total_points else '超出范围'}")
        
        if point_idx < self.total_points:
            point_pos = self.calib_points[point_idx]
            
            # 更新动画时间（使用delta time确保不同帧率下动画速度一致）
            delta_time = self.clock.get_time() / 1000.0
            self.animation_time += self.animation_speed * delta_time
            
            # 计算脉动效果的半径
            # 使用正弦函数创建平滑的脉动效果
            pulse_factor = (np.sin(self.animation_time * np.pi) + 1) / 2  # 范围在0到1之间
            
            # 使用缓动函数优化动画流畅度
            eased_factor = pulse_factor * pulse_factor * (3 - 2 * pulse_factor)
            current_radius = self.base_radius + int((self.pulse_radius - self.base_radius) * eased_factor)
            
            # 优化颜色渐变效果
            # 使用HSV颜色空间实现更平滑的颜色变化
            # 在红色到橙色之间平滑过渡
            hue = 0 + 30 * eased_factor  # 0=红色，30=橙色
            saturation = 100
            value = 100 - 20 * eased_factor  # 稍微降低亮度以避免过于刺眼
            
            # 转换HSV到RGB（简化版本）
            # 这里使用简化的红色到橙色过渡
            r = 255
            g = int(100 + 155 * eased_factor)  # 绿色分量从100增加到255
            b = 0
            color = (r, g, b)
            
            # 绘制多层次光晕效果增强视觉吸引力
            # 1. 最外层光晕 - 柔和的大范围光晕
            outer_glow_radius = current_radius + 25
            outer_glow_surface = pygame.Surface((outer_glow_radius * 2, outer_glow_radius * 2), pygame.SRCALPHA)
            outer_glow_alpha = int(50 * eased_factor)
            pygame.draw.circle(outer_glow_surface, (255, 100, 100, outer_glow_alpha), 
                             (outer_glow_radius, outer_glow_radius), outer_glow_radius)
            self.screen.blit(outer_glow_surface, (point_pos[0] - outer_glow_radius, point_pos[1] - outer_glow_radius))
            
            # 2. 中层光晕 - 更集中的光晕
            mid_glow_radius = current_radius + 15
            mid_glow_surface = pygame.Surface((mid_glow_radius * 2, mid_glow_radius * 2), pygame.SRCALPHA)
            mid_glow_alpha = int(80 * eased_factor)
            pygame.draw.circle(mid_glow_surface, (255, 80, 80, mid_glow_alpha), 
                             (mid_glow_radius, mid_glow_radius), mid_glow_radius)
            self.screen.blit(mid_glow_surface, (point_pos[0] - mid_glow_radius, point_pos[1] - mid_glow_radius))
            
            # 3. 绘制主要校准点（填充圆）
            # 添加抗锯齿效果（通过多次绘制不同大小的圆）
            pygame.draw.circle(self.screen, color, point_pos, current_radius)
            # 添加一个稍小的亮色圆作为边缘高光
            highlight_color = (min(r+30, 255), min(g+30, 255), b)
            pygame.draw.circle(self.screen, highlight_color, point_pos, current_radius - 1)
            
            # 4. 在中心绘制一个小的白色亮点，增强视觉效果
            inner_radius = max(5, int(current_radius * 0.25))
            pygame.draw.circle(self.screen, self.WHITE, point_pos, inner_radius)
            # 再添加一个更小的亮黄色点，模拟光斑
            pygame.draw.circle(self.screen, (255, 255, 200), point_pos, max(1, int(inner_radius * 0.6)))
            
            # 绘制方向提示箭头（如果需要）
            self._draw_direction_arrows(point_idx, point_pos)
            
            # 显示进度信息（小字体，不遮挡）
            progress_text = self.font_small.render(f"校准点 {point_idx + 1}/{self.total_points}", True, self.GRAY)
            progress_rect = progress_text.get_rect(center=(self.width // 2, 50))
            self.screen.blit(progress_text, progress_rect)
            
            # 如果有视线点，也绘制出来
            if gaze_point is not None:
                pygame.draw.circle(self.screen, self.BLUE, gaze_point, 15)
        
        pygame.display.flip()
        
        # 确保窗口在正确的屏幕位置显示
        if hasattr(self, 'window_x') and hasattr(self, 'window_y'):
            # 强制更新窗口位置，确保校准点显示在正确的屏幕上
            pygame.display.update()
    
    def _draw_direction_arrows(self, idx, pos):
        """绘制方向提示箭头"""
        length = 40
        thickness = 4
        
        # 根据校准点索引绘制不同的方向箭头
        if idx == 0:  # 左上角 - 提示向右下看
            pygame.draw.line(self.screen, self.BLACK, (pos[0] + length, pos[1]), (pos[0] + length - 15, pos[1] - 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0] + length, pos[1]), (pos[0] + length - 15, pos[1] + 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] + length), (pos[0] - 10, pos[1] + length - 15), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] + length), (pos[0] + 10, pos[1] + length - 15), thickness)
        elif idx == 1:  # 右上角 - 提示向左下看
            pygame.draw.line(self.screen, self.BLACK, (pos[0] - length, pos[1]), (pos[0] - length + 15, pos[1] - 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0] - length, pos[1]), (pos[0] - length + 15, pos[1] + 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] + length), (pos[0] - 10, pos[1] + length - 15), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] + length), (pos[0] + 10, pos[1] + length - 15), thickness)
        elif idx == 2:  # 左下角 - 提示向右上
            pygame.draw.line(self.screen, self.BLACK, (pos[0] + length, pos[1]), (pos[0] + length - 15, pos[1] - 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0] + length, pos[1]), (pos[0] + length - 15, pos[1] + 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] - length), (pos[0] - 10, pos[1] - length + 15), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] - length), (pos[0] + 10, pos[1] - length + 15), thickness)
        elif idx == 3:  # 右下角 - 提示向左上看
            pygame.draw.line(self.screen, self.BLACK, (pos[0] - length, pos[1]), (pos[0] - length + 15, pos[1] - 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0] - length, pos[1]), (pos[0] - length + 15, pos[1] + 10), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] - length), (pos[0] - 10, pos[1] - length + 15), thickness)
            pygame.draw.line(self.screen, self.BLACK, (pos[0], pos[1] - length), (pos[0] + 10, pos[1] - length + 15), thickness)
    
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
                elif event.key == pygame.K_SPACE:
                    return 'start'
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return 'click'
        return None
    
    def get_current_calibration_point(self, time_interval=2.0):
        """获取当前校准点（带延迟记录功能）"""
        if not self.calibration_started:
            return None, np.array([0, 0])
        
        tdelta = time.time() - self.start_time
        idx = int(tdelta // time_interval)
        
        # 检查是否切换到了新的校准点
        if idx != self.current_point_idx:
            self.current_point_idx = idx
            self.point_start_time = time.time()
            self.can_record = False
        
        # 检查是否达到延迟时间
        if time.time() - self.point_start_time >= self.delay_recording:
            self.can_record = True
        
        if idx < self.total_points:
            return idx, self.calib_points[idx]
        else:
            print("校准完成！")
            return None, np.array([0, 0])
    
    def start_calibration(self):
        """开始校准"""
        self.calibration_started = True
        self.start_time = time.time()
        self.current_point_idx = 0
        self.point_start_time = time.time()
        self.can_record = False
    
    def cleanup(self):
        """清理资源"""
        pygame.display.quit()
    
    def move_window_to_screen(self, screen_x, screen_y, window_width=None, window_height=None):
        """将窗口移动到指定屏幕位置
        
        Args:
            screen_x: 目标屏幕的x坐标（绝对坐标）
            screen_y: 目标屏幕的y坐标（绝对坐标）
            window_width: 窗口宽度（可选，默认使用self.width）
            window_height: 窗口高度（可选，默认使用self.height）
        """
        try:
            if platform.system() == "Windows":
                # 使用Windows API移动窗口
                import ctypes
                from ctypes import wintypes
                
                # 获取窗口句柄
                hwnd = pygame.display.get_wm_info()['window']
                
                # 使用当前窗口尺寸（如果在initialize方法中传递了窗口尺寸）
                if window_width is None:
                    window_width = self.width
                if window_height is None:
                    window_height = self.height
                
                # 定义Windows API常量
                HWND_TOPMOST = -1
                HWND_NOTOPMOST = -2
                SWP_NOZORDER = 0x0004
                SWP_NOSIZE = 0x0001
                SWP_SHOWWINDOW = 0x0040
                
                # 窗口尺寸信息已在其他地方输出
                # print(f"窗口尺寸: {window_width}x{window_height}")
                
                # 先设置窗口为顶级窗口
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOSIZE | SWP_SHOWWINDOW
                )
                
                # 移动窗口到目标位置（只设置位置，不改变大小）
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_NOTOPMOST, screen_x, screen_y, 0, 0,
                    SWP_NOSIZE | SWP_SHOWWINDOW
                )
                
                # 校准窗口移动完成
                
                # 验证窗口位置
                pygame.time.delay(200)  # 增加延迟确保窗口移动完成
                
        except Exception as e:
            print(f"移动校准窗口失败: {e}")
            # 回退到SDL环境变量方法
            if screen_x is not None and screen_y is not None:
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_x},{screen_y}"
                # 使用SDL环境变量设置窗口位置


class PygameCalibrationTargets:
    """Pygame校准目标管理类 - 支持多屏幕校准"""
    
    def __init__(self, width, height, screen_config=None, reference_screen=None):
        self.width = width
        self.height = height
        self.screen_config = screen_config or {}
        self.reference_screen = reference_screen  # 参考屏幕（通常是第一个屏幕）
        
        # 改进的边距计算 - 根据屏幕尺寸动态调整
        if hasattr(self, '_calculate_adaptive_margin'):
            self.margin_ratio = self._calculate_adaptive_margin()
        else:
            self.margin_ratio = 0.15  # 默认边距比例
        
        self.calib_points = self._build_adaptive_grid_points()
        self.total_points = len(self.calib_points)
        self.tstart = 0
        self.current_idx = -1  # 当前校准点索引
        self.point_start_time = 0  # 当前校准点显示的开始时间
        self.delay_recording = 0.5  # 记录延迟时间（秒）
        self.can_record = False  # 是否可以开始记录数据
        
        print(f"初始化校准目标管理器:")
        print(f"  屏幕尺寸: {self.width}x{self.height}")
        print(f"  屏幕配置: {self.screen_config}")
        print(f"  参考屏幕: {reference_screen}")
        print(f"  边距比例: {self.margin_ratio}")
        print(f"  校准点数量: {self.total_points}")
        
    def _calculate_adaptive_margin(self):
        """根据屏幕尺寸计算自适应边距比例"""
        # 对于小屏幕，使用较大的边距避免校准点过于靠近边缘
        # 对于大屏幕，可以使用相对较小的边距
        min_dimension = min(self.width, self.height)
        
        if min_dimension < 800:
            return 0.12  # 小屏幕 12%边距
        elif min_dimension < 1200:
            return 0.15  # 中等屏幕 15%边距
        else:
            return 0.18  # 大屏幕 18%边距
    
    def _build_adaptive_grid_points(self):
        """构建自适应校准点网格 - 支持多屏幕尺寸差异"""
        
        # 获取屏幕信息
        screen_index = self.screen_config.get('index', 0) if self.screen_config else 0
        screen_width = self.width
        screen_height = self.height
        
        # 计算自适应边距
        margin_x = int(screen_width * self.margin_ratio)
        margin_y = int(screen_height * self.margin_ratio)
        
        print(f"屏幕{screen_index + 1}自适应校准点计算:")
        print(f"  自适应边距比例: {self.margin_ratio}")
        print(f"  自适应边距: ({margin_x}, {margin_y})")
        
        # 如果有参考屏幕，进行尺寸适配
        if self.reference_screen and self.reference_screen != self:
            ref_width = self.reference_screen.width
            ref_height = self.reference_screen.height
            
            # 计算尺寸缩放比例
            scale_x = screen_width / ref_width
            scale_y = screen_height / ref_height
            
            print(f"  参考屏幕: {ref_width}x{ref_height}")
            print(f"  缩放比例: ({scale_x:.3f}, {scale_y:.3f})")
            
            # 基于参考屏幕的校准点进行缩放适配
            ref_margin_x = int(ref_width * self.reference_screen.margin_ratio)
            ref_margin_y = int(ref_height * self.reference_screen.margin_ratio)
            
            # 计算适配后的校准点
            points = self._calculate_scaled_calibration_points(
                screen_width, screen_height, 
                ref_margin_x, ref_margin_y,
                scale_x, scale_y
            )
        else:
            # 没有参考屏幕，使用标准的5点校准
            points = self._calculate_standard_calibration_points(
                screen_width, screen_height,
                margin_x, margin_y
            )
        
        # 保存校准点到局部属性
        self.calibration_points_local = points.copy()
        
        print(f"  最终校准点: {points}")
        
        return points
    
    def _calculate_scaled_calibration_points(self, screen_width, screen_height, 
                                          ref_margin_x, ref_margin_y, scale_x, scale_y):
        """基于参考屏幕的校准点进行缩放计算"""
        
        # 参考屏幕的标准校准点位置
        ref_points = [
            [ref_margin_x, ref_margin_y],  # 左上角
            [self.reference_screen.width - ref_margin_x, ref_margin_y],  # 右上角
            [ref_margin_x, self.reference_screen.height - ref_margin_y],  # 左下角
            [self.reference_screen.width - ref_margin_x, self.reference_screen.height - ref_margin_y],  # 右下角
            [self.reference_screen.width // 2, self.reference_screen.height // 2]  # 屏幕中心
        ]
        
        # 对每个校准点进行缩放适配
        scaled_points = []
        for ref_point in ref_points:
            # 使用相对位置而非绝对位置进行缩放
            rel_x = ref_point[0] / self.reference_screen.width
            rel_y = ref_point[1] / self.reference_screen.height
            
            # 在当前屏幕上应用相同的相对位置
            scaled_x = int(rel_x * screen_width)
            scaled_y = int(rel_y * screen_height)
            
            scaled_points.append([scaled_x, scaled_y])
        
        return scaled_points
    
    def _calculate_standard_calibration_points(self, screen_width, screen_height, margin_x, margin_y):
        """计算标准5点校准网格"""
        return [
            [margin_x, margin_y],  # 左上角
            [screen_width - margin_x, margin_y],  # 右上角
            [margin_x, screen_height - margin_y],  # 左下角
            [screen_width - margin_x, screen_height - margin_y],  # 右下角
            [screen_width // 2, screen_height // 2]  # 屏幕中心
        ]
    
    def get_calibration_point_screen_coordinates(self, point_idx):
        """获取校准点在屏幕坐标系统中的位置
        
        Args:
            point_idx: 校准点索引
            
        Returns:
            tuple: 屏幕绝对坐标位置 (x, y)
        """
        if point_idx >= len(self.calib_points):
            return None
        
        local_point = self.calib_points[point_idx]
        
        # 获取屏幕偏移
        screen_offset_x = self.screen_config.get('left', 0) if self.screen_config else 0
        screen_offset_y = self.screen_config.get('top', 0) if self.screen_config else 0
        
        # 计算屏幕绝对坐标
        screen_x = screen_offset_x + local_point[0]
        screen_y = screen_offset_y + local_point[1]
        
        return (screen_x, screen_y)
    
    def adapt_to_screen_size(self, new_width, new_height, new_screen_config=None):
        """动态适配到新的屏幕尺寸
        
        Args:
            new_width: 新屏幕宽度
            new_height: 新屏幕高度
            new_screen_config: 新屏幕配置信息
        """
        print(f"适配新屏幕尺寸: {new_width}x{new_height}")
        
        self.width = new_width
        self.height = new_height
        if new_screen_config:
            self.screen_config = new_screen_config
        
        # 重新计算边距比例
        self.margin_ratio = self._calculate_adaptive_margin()
        
        # 重新构建校准点
        self.calib_points = self._build_adaptive_grid_points()
        self.total_points = len(self.calib_points)
        
        print(f"屏幕适配完成，新的校准点数量: {self.total_points}")
        
    def _build_grid_points(self):
        """构建校准点网格"""
        margin_x = int(self.width * self.margin_ratio)
        margin_y = int(self.height * self.margin_ratio)
        
        # 5点校准：四个角 + 屏幕中心
        points = [
            [margin_x, margin_y],  # 左上角
            [self.width - margin_x, margin_y],  # 右上角
            [margin_x, self.height - margin_y],  # 左下角
            [self.width - margin_x, self.height - margin_y],  # 右下角
            [self.width // 2, self.height // 2]  # 屏幕中心
        ]
        
        return points
    
    def getTargetCalibration(self, time_interval=2.0):
        """获取校准点（带延迟记录功能）"""
        tdelta = time.time() - self.tstart
        idx = int(tdelta // time_interval)
        
        # 检查是否切换到了新的校准点
        if idx != self.current_idx:
            self.current_idx = idx
            self.point_start_time = time.time()
            self.can_record = False
        
        # 检查是否达到延迟时间
        if time.time() - self.point_start_time >= self.delay_recording:
            self.can_record = True
        
        if idx < self.total_points:
            return idx, self.calib_points[idx]
        else:
            return None, np.array([0, 0])
    
    def start_timing(self):
        """开始计时"""
        self.tstart = time.time()
        self.current_idx = -1
        self.point_start_time = 0
        self.can_record = False


# 工具函数：从gui_opencv.py迁移而来
def getWhiteFrame(height, width):
    """创建纯白背景"""
    import numpy as np
    return 255 * np.ones((width, height, 3), dtype=np.uint8)


def getScreenSize():
    """获取屏幕尺寸信息（主屏幕）"""
    import screeninfo
    screen = screeninfo.get_monitors()
    for s in screen:
        if s.is_primary:
            width = s.width
            height = s.height
            width_mm = s.width_mm
            height_mm = s.height_mm
    print(f"Screen Size: {width}x{height}")
    return width, height, width_mm, height_mm

def getAllScreensInfo():
    """获取所有屏幕的详细信息"""
    import screeninfo
    screens = screeninfo.get_monitors()
    screen_info = []
    
    for i, s in enumerate(screens):
        screen_data = {
            'index': i,
            'name': f"Screen {i+1}",
            'x': s.x,
            'y': s.y,
            'width': s.width,
            'height': s.height,
            'width_mm': getattr(s, 'width_mm', 500),
            'height_mm': getattr(s, 'height_mm', 300),
            'is_primary': s.is_primary,
            'left': s.x,
            'top': s.y,
            'right': s.x + s.width,
            'bottom': s.y + s.height,
            'center_x': s.x + s.width // 2,
            'center_y': s.y + s.height // 2
        }
        screen_info.append(screen_data)
    
    print(f"检测到 {len(screen_info)} 个屏幕:")
    for screen in screen_info:
        print(f"  屏幕{screen['index']+1}: {screen['width']}x{screen['height']} 位置({screen['x']}, {screen['y']})")
    
    return screen_info


def ReadCameraCalibrationData(calibration_path, file_name = "calibration_data.txt"):
    """读取相机校准数据"""
    import numpy as np
    import os
    path = os.path.join(calibration_path, file_name)
    with open(path, 'r') as f:
        lines = f.readlines()

    # Extract the camera matrix and distortion coefficients from the file
    camera_matrix = np.array([[float(x) for x in lines[1].split()],
                            [float(x) for x in lines[2].split()],
                            [float(x) for x in lines[3].split()]])

    dist_coeffs = np.array([float(x) for x in lines[5].split()])

    return camera_matrix, dist_coeffs


def get_out_video(cap, output_path, file_name = "calibrate.mp4", scalewidth = 1):
    """创建视频输出对象"""
    import cv2
    import os
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # get frame width in pixel
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # get frame height in pixel
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 20
    out_video = cv2.VideoWriter( os.path.join(output_path,file_name), cv2.VideoWriter_fourcc(*'avc1'), fps, (scalewidth*width, height))
    return out_video, width, height


class MultiScreenCalibrationManager:
    """多屏幕校准管理器 - 处理屏幕间坐标转换和适配"""
    
    def __init__(self, screen_configs):
        """
        初始化多屏幕校准管理器
        
        Args:
            screen_configs: 屏幕配置列表，每个元素包含屏幕尺寸、位置等信息
        """
        self.screen_configs = screen_configs
        self.screen_count = len(screen_configs)
        
        # 创建每个屏幕的校准目标管理器
        self.screen_calibrators = {}
        self.reference_screen = None
        
        print(f"初始化多屏幕校准管理器 - 检测到 {self.screen_count} 个屏幕")
        
        # 初始化每个屏幕的校准器
        for i, screen_config in enumerate(screen_configs):
            screen_calibrator = PygameCalibrationTargets(
                width=screen_config['width'],
                height=screen_config['height'],
                screen_config=screen_config,
                reference_screen=None  # 稍后设置参考屏幕
            )
            self.screen_calibrators[i] = screen_calibrator
            
            print(f"  屏幕 {i+1}: {screen_config['width']}x{screen_config['height']} 位置({screen_config['left']}, {screen_config['top']})")
        
        # 设置参考屏幕（通常是第一个屏幕或主屏幕）
        self._setup_reference_screen()
        
        # 重新构建所有屏幕的校准点（基于参考屏幕进行适配）
        self._build_adaptive_calibration_points()
    
    def _setup_reference_screen(self):
        """设置参考屏幕（通常是主屏幕或第一个屏幕）"""
        # 寻找主屏幕或第一个屏幕作为参考
        reference_idx = 0
        for i, config in enumerate(self.screen_configs):
            if config.get('is_primary', False):
                reference_idx = i
                break
        
        self.reference_screen_idx = reference_idx
        self.reference_screen = self.screen_calibrators[reference_idx]
        
        print(f"设置屏幕 {reference_idx + 1} 为参考屏幕")
    
    def _build_adaptive_calibration_points(self):
        """基于参考屏幕为所有屏幕构建适配的校准点"""
        print("开始构建自适应校准点...")
        
        # 为每个屏幕创建基于参考屏幕的校准目标管理器
        for i, screen_config in enumerate(self.screen_configs):
            if i != self.reference_screen_idx:
                # 创建基于参考屏幕的校准目标管理器
                adaptive_calibrator = PygameCalibrationTargets(
                    width=screen_config['width'],
                    height=screen_config['height'],
                    screen_config=screen_config,
                    reference_screen=self.reference_screen
                )
                self.screen_calibrators[i] = adaptive_calibrator
                
                print(f"屏幕 {i+1} 已适配参考屏幕的校准点")
            else:
                print(f"屏幕 {i+1} 是参考屏幕，使用标准校准点")
        
        print("自适应校准点构建完成")
    
    def get_screen_calibrator(self, screen_index):
        """
        获取指定屏幕的校准目标管理器
        
        Args:
            screen_index: 屏幕索引
            
        Returns:
            PygameCalibrationTargets: 指定屏幕的校准目标管理器
        """
        if screen_index in self.screen_calibrators:
            return self.screen_calibrators[screen_index]
        else:
            print(f"警告：屏幕 {screen_index} 不存在，返回参考屏幕校准器")
            return self.reference_screen
    
    def get_calibration_point_for_screen(self, point_idx, screen_index):
        """
        获取指定屏幕和校准点索引的校准点位置
        
        Args:
            point_idx: 校准点索引
            screen_index: 屏幕索引
            
        Returns:
            tuple: 校准点位置 (x, y)
        """
        calibrator = self.get_screen_calibrator(screen_index)
        if calibrator:
            return calibrator.calib_points[point_idx] if point_idx < len(calibrator.calib_points) else None
        return None
    
    def get_screen_absolute_coordinates(self, point_idx, screen_index):
        """
        获取校准点在屏幕绝对坐标系统中的位置
        
        Args:
            point_idx: 校准点索引
            screen_index: 屏幕索引
            
        Returns:
            tuple: 屏幕绝对坐标 (x, y)
        """
        calibrator = self.get_screen_calibrator(screen_index)
        if calibrator:
            return calibrator.get_calibration_point_screen_coordinates(point_idx)
        return None
    
    def validate_calibration_points(self, screen_index):
        """
        验证指定屏幕的校准点是否有效
        
        Args:
            screen_index: 屏幕索引
            
        Returns:
            bool: 校准点是否有效
        """
        calibrator = self.get_screen_calibrator(screen_index)
        if calibrator and hasattr(calibrator, 'calib_points'):
            return len(calibrator.calib_points) > 0
        return False
    
    def get_screen_info(self, screen_index):
        """
        获取屏幕详细信息
        
        Args:
            screen_index: 屏幕索引
            
        Returns:
            dict: 屏幕信息
        """
        if 0 <= screen_index < self.screen_count:
            return self.screen_configs[screen_index]
        return None
    
    def debug_print_calibration_points(self):
        """调试打印所有屏幕的校准点"""
        print("=== 调试信息：所有屏幕校准点 ===")
        
        for i, calibrator in self.screen_calibrators.items():
            screen_config = self.screen_configs[i]
            print(f"\n屏幕 {i+1} ({screen_config['width']}x{screen_config['height']}):")
            print(f"  屏幕偏移: ({screen_config['left']}, {screen_config['top']})")
            
            if hasattr(calibrator, 'calib_points'):
                for j, point in enumerate(calibrator.calib_points):
                    abs_x = screen_config['left'] + point[0]
                    abs_y = screen_config['top'] + point[1]
                    print(f"  校准点 {j+1}: 局部({point[0]}, {point[1]}) -> 绝对({abs_x}, {abs_y})")
        
        print("=== 调试信息结束 ===\n")
