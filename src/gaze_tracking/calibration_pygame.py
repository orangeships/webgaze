import pygame
import numpy as np
import time
import os

class PygameCalibrationUI:
    """Pygame校准界面类"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = None
        self.clock = pygame.time.Clock()
        
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
        self.margin_ratio = 0.02  # 边距比例
        self.calib_points = self._build_grid_points()
        self.total_points = len(self.calib_points)
        
        # 动画相关变量
        self.animation_time = 0
        self.animation_speed = 4  # 提高动画速度到原来的3倍
        self.base_radius = 25  # 基础半径
        self.pulse_radius = 35  # 增加脉动范围以增强视觉效果
        
    def initialize(self):
        """初始化Pygame显示"""
        # 设置全屏显示
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN | pygame.NOFRAME)
        pygame.display.set_caption("视线追踪校准")
        
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
        """构建5点校准网格"""
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
                elif event.key == pygame.K_s or event.key == pygame.K_S:
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


class PygameCalibrationTargets:
    """Pygame校准目标管理类"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.margin_ratio = 0.02
        self.calib_points = self._build_grid_points()
        self.total_points = len(self.calib_points)
        self.tstart = 0
        self.current_idx = -1  # 当前校准点索引
        self.point_start_time = 0  # 当前校准点显示的开始时间
        self.delay_recording = 0.5  # 记录延迟时间（秒）
        self.can_record = False  # 是否可以开始记录数据
        
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
    """获取屏幕尺寸信息"""
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