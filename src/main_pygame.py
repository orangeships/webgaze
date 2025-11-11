import os
import sys
import pygame
import cv2
import time
import numpy as np
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel

# Pygame初始化
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (52, 152, 219)
GREEN = (46, 204, 113)
RED = (231, 76, 60)

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
        
    def initialize_display(self):
        """初始化显示"""
        # 设置真正的全屏显示，使用检测到的物理分辨率
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN | pygame.NOFRAME)
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
    
    def show_gaze_tracking_screen(self, frame, gaze_point=None):
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
        
        # 退出说明
        exit_text = self.font_small.render("按ESC键退出", True, GRAY)
        exit_rect = exit_text.get_rect(center=(self.width // 2, 3 * self.height // 4))
        self.screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()
    
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
        
    def initialize(self):
        """初始化系统"""
        # 初始化UI（会自动获取屏幕尺寸）
        self.ui = PygameUI()
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
            calibration_file = os.path.join(self.project_dir, "results", "calibration_results.pkl")
            if os.path.exists(calibration_file):
                print(f"正在加载历史校准数据: {calibration_file}")
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
            title_rect = title_text.get_rect(center=(self.ui.width // 2, self.ui.height // 4))
            self.ui.screen.blit(title_text, title_rect)
            
            # 两个按钮
            button1_rect = self.ui.draw_button("加载历史校准数据", self.ui.width // 4, self.ui.height // 2, 
                                              self.ui.width // 2, 80, BLUE)
            button2_rect = self.ui.draw_button("进行新校准", self.ui.width // 4, self.ui.height // 2 + 120, 
                                              self.ui.width // 2, 80, GREEN)
            
            pygame.display.flip()
            
            event = self.ui.handle_events()
            if event == 'quit':
                return 'quit'
            elif event == 'click':
                mouse_pos = pygame.mouse.get_pos()
                if button1_rect.collidepoint(mouse_pos):
                    return 'load'
                elif button2_rect.collidepoint(mouse_pos):
                    return 'new'
            
            self.ui.clock.tick(60)
    
    def perform_new_calibration(self):
        """执行新校准"""
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
        
        # 缓存变量，避免重复计算
        cached_face_features_prev = None
        cached_face_features_curr = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 获取视线信息
            eye_info = self.model.get_gaze(frame=frame, imshow=False)
            
            if eye_info is not None:
                # 获取3D视线向量
                gaze = eye_info['gaze']
                
                # 使用SfM进行视线映射
                try:
                    if frame_prev is not None:
                        try:
                            # 优化：只在必要时重新计算人脸特征点
                            # 对于当前帧，我们总是需要计算新的特征点
                            face_features_curr = self.model.get_FaceFeatures(frame)
                            
                            # 对于前一帧，我们可以使用上一次计算的当前帧特征点
                            # 这样避免了每帧都重新计算两个帧的特征点
                            if cached_face_features_curr is not None:
                                face_features_prev = cached_face_features_curr
                            else:
                                face_features_prev = self.model.get_FaceFeatures(frame_prev)
                            
                            # 更新缓存：当前帧的特征点将成为下一帧的前一帧特征点
                            cached_face_features_prev = face_features_prev
                            cached_face_features_curr = face_features_curr
                            
                            # 尝试使用SfM方法，并传入预计算的特征点
                            # 获取世界坐标系中的视线方向
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
                            print(f"SfM计算失败，回退到普通方法: {sfm_error}")
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                            # 清除缓存以避免错误数据影响下一次计算
                            self.homtrans.sfm.clear_caches()
                    else:
                        # 初始帧使用普通方法
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 将毫米坐标转换为像素坐标
                    screen_pos_mm = FSgaze.flatten()[:2]
                    screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                    
                    # 确保坐标在屏幕范围内
                    gaze_x = max(0, min(screen_pos_px[0], self.ui.width))
                    gaze_y = max(0, min(screen_pos_px[1], self.ui.height))
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
            
            # 显示结果
            self.ui.show_gaze_tracking_screen(frame, gaze_point)
            
            event = self.ui.handle_events()
            if event == 'quit':
                break
            
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