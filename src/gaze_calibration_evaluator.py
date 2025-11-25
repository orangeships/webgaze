import os
import sys
import pygame
import cv2
import numpy as np
import time
import math
import json
import os
from datetime import datetime
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import correlation
from scipy.signal import correlate
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示
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
RED = (255, 0, 0)  # 标记点颜色

class GazeCalibrationEvaluator:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.width = 0
        self.height = 0
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = None
        
        # 标记点参数
        self.target_radius = 25  # 增大标记点直径以提高可见性
        self.track_radius = 0  # 运动轨迹半径
        self.track_center = (0, 0)  # 轨迹圆心
        self.current_angle = 0  # 当前角度
        self.angular_velocity = np.pi/3  # 角速度 (1π rad/s)
        self.completed_circles = 0  # 完成的圈数
        self.start_angle = 0  # 开始角度
        
        # 数据采集
        self.gaze_data = []
        self.target_data = []
        self.test_start_time = 0
        self.is_testing = False
        
        # 模型和校准
        self.model = None
        self.homtrans = None
        self.cap = None
        
        # 延迟补偿相关参数
        self.delay_frames = 5  # 预估的延迟帧数（可调整）
        self.target_history = []  # 存储目标位置历史
        self.max_history_size = 30  # 最大历史记录大小
        
    def initialize(self):
        """初始化系统"""
        # 获取屏幕尺寸
        import ctypes
        try:
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            self.width = user32.GetSystemMetrics(0)
            self.height = user32.GetSystemMetrics(1)
        except:
            screen_info = pygame.display.Info()
            self.width = screen_info.current_w
            self.height = screen_info.current_h
        
        print(f"屏幕分辨率: {self.width}x{self.height}")
        
        # 设置全屏显示
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN | pygame.NOFRAME)
        pygame.display.set_caption("Gaze Calibration Evaluator")
        
        # 初始化字体
        try:
            
            self.font = pygame.font.SysFont('simHei',20)
        except:
            self.font = pygame.font.Font(None, 24)
        
        # 计算轨迹参数
        self.track_center = (self.width // 2, self.height // 2)
        self.track_radius = min(self.width, self.height) // 4  # 屏幕长边的1/4
        
        # 初始化模型
        print("加载模型中...")
        self.model = EyeModel(self.project_dir)
        self.homtrans = HomTransform(self.project_dir)
        
        # 初始化摄像头
        for device_id in [1, 0, 2]:
            try:
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"摄像头设备 {device_id} 初始化成功")
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                print(f"摄像头设备 {device_id} 初始化失败: {e}")
        
        if self.cap is None:
            print("无法初始化摄像头")
            return False
        
        return True
    
    def load_calibration(self):
        """加载校准数据"""
        calibration_file = os.path.join(self.project_dir, "results", "calibration_results.json")
        if os.path.exists(calibration_file):
            print(f"加载校准数据: {calibration_file}")
            if self.homtrans.load_calibration_results(calibration_file):
                print("校准数据加载成功")
                return True
            else:
                print("校准数据加载失败")
        else:
            print("校准文件不存在")
        return False
    
    def run_calibration(self):
        """运行校准流程"""
        print("开始校准流程...")
        try:
            # 使用homtransform的calibrate方法进行校准
            STransG = self.homtrans.calibrate(self.model, self.cap, sfm=True)
            if STransG is not None:
                print("校准成功完成")
                return True
            else:
                print("校准失败")
                return False
        except Exception as e:
            print(f"校准过程出错: {e}")
            return False
    
    def show_start_screen(self):
        """显示启动按钮界面，用户点击后开始校准流程"""
        button_width = 300
        button_height = 80
        button_x = (self.width - button_width) // 2
        button_y = (self.height - button_height) // 2
        
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        button_color = GREEN
        hover_color = (39, 174, 96)
        
        while True:
            self.screen.fill(WHITE)
            
            # 绘制标题
            title_text = self.font.render("眼动追踪校准评估", True, BLACK)
            title_rect = title_text.get_rect(center=(self.width // 2, 100))
            self.screen.blit(title_text, title_rect)
            
            # 绘制说明文字
            info_text1 = self.font.render("请点击下方按钮开始校准流程", True, BLACK)
            info_rect1 = info_text1.get_rect(center=(self.width // 2, button_y - 50))
            self.screen.blit(info_text1, info_rect1)
            
            info_text2 = self.font.render("校准完成后将自动开始圆环追踪测试", True, BLACK)
            info_rect2 = info_text2.get_rect(center=(self.width // 2, button_y - 20))
            self.screen.blit(info_text2, info_rect2)
            
            # 绘制退出提示
            exit_text = self.font.render("按ESC键退出", True, BLACK)
            exit_rect = exit_text.get_rect(center=(self.width // 2, self.height - 50))
            self.screen.blit(exit_text, exit_rect)
            
            # 获取鼠标位置
            mouse_pos = pygame.mouse.get_pos()
            
            # 检查鼠标是否悬停在按钮上
            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, hover_color, button_rect, border_radius=10)
            else:
                pygame.draw.rect(self.screen, button_color, button_rect, border_radius=10)
            
            # 绘制按钮文字
            button_text = self.font.render("开始校准", True, WHITE)
            button_text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, button_text_rect)
            
            pygame.display.flip()
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if button_rect.collidepoint(mouse_pos):
                        return  # 开始校准流程
            
            # 限制帧率
            self.clock.tick(60)
    
    def start_test(self):
        """开始测试"""
        print("开始眼动追踪校准评估测试...")
        self.gaze_data = []
        self.target_data = []
        self.current_angle = 0
        self.start_angle = 0
        self.completed_circles = 0
        self.test_start_time = time.time()
        self.is_testing = True
        
        # 显示开始提示
        self.screen.fill(WHITE)
        start_text1 = self.font.render("Test will start in 3 seconds", True, BLACK)
        start_text2 = self.font.render("Please follow the red dot with your eyes", True, BLACK)
        start_text3 = self.font.render("Test will stop after completing 2 full circles", True, BLACK)
        
        text_rect1 = start_text1.get_rect(center=(self.width // 2, self.height // 2 - 50))
        text_rect2 = start_text2.get_rect(center=(self.width // 2, self.height // 2))
        text_rect3 = start_text3.get_rect(center=(self.width // 2, self.height // 2 + 50))
        
        self.screen.blit(start_text1, text_rect1)
        self.screen.blit(start_text2, text_rect2)
        self.screen.blit(start_text3, text_rect3)
        pygame.display.flip()
        
        # 倒计时3秒
        for i in range(3, 0, -1):
            time.sleep(1)
            self.screen.fill(WHITE)
            countdown_text = self.font.render(f"{i}", True, BLACK)
            countdown_rect = countdown_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(countdown_text, countdown_rect)
            pygame.display.flip()
        
        time.sleep(0.5)
    
    def update_target_position(self, elapsed_time):
        """更新目标点位置，确保恒定速度的圆形轨迹"""
        # 更新角度，使用固定的角速度确保运动速度恒定
        # 使用时间增量计算角度增量，确保不同帧率下运动速度一致
        self.current_angle = (self.current_angle + self.angular_velocity * elapsed_time) % (2 * np.pi)
        
        # 计算已完成的总圈数（累计）
        total_completed_circles = int(self.current_angle / (2 * np.pi))
        if total_completed_circles > self.completed_circles:
            self.completed_circles = total_completed_circles
            print(f"完成第 {self.completed_circles} 圈")
        
        # 精确计算目标点位置，确保轨迹准确
        # 使用浮点数计算然后四舍五入，避免整数截断误差累积
        x = round(self.track_center[0] + self.track_radius * np.cos(self.current_angle))
        y = round(self.track_center[1] + self.track_radius * np.sin(self.current_angle))
        
        # 确保坐标在屏幕范围内
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        # 记录目标位置历史
        current_time = time.time()
        self.target_history.append({
            'x': x,
            'y': y,
            'timestamp': current_time
        })
        
        # 保持历史记录大小
        if len(self.target_history) > self.max_history_size:
            self.target_history.pop(0)
        
        return (int(x), int(y))
    
    def collect_gaze_data(self, target_pos, timestamp):
        """采集用户注视点数据，包含详细的错误处理和数据质量检查"""
        # 读取摄像头帧
        ret, frame = self.cap.read()
        if not ret:
            print("警告: 无法读取摄像头帧")
            return None
        
        try:
            # 获取人脸检测
            face_boxes = self.model.face_detection.predict(frame)
            
            # 检查是否检测到人脸
            if not face_boxes:
                # 不打印太多警告，避免日志过多
                if len(self.gaze_data) % 30 == 0:  # 每30帧打印一次警告
                    print("警告: 未检测到人脸")
                return None
            
            # 获取视线信息
            eye_info = self.model.get_gaze(frame=frame, face_boxes=face_boxes, imshow=False)
            
            if eye_info is not None:
                gaze = eye_info['gaze']
                
                # 视线映射
                try:
                    # 首先尝试使用SfM方法
                    try:
                        # 检查homtrans是否有sfm属性并且是否支持SfM方法
                        if hasattr(self.homtrans, 'sfm') and hasattr(self.homtrans, '_getGazeOnScreen_sfm'):
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze)
                        else:
                            # 回退到普通方法
                            FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    except:
                        # 如果SfM方法失败，使用普通方法
                        FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen(gaze)
                    
                    # 数据质量检查
                    if np.any(np.isnan(FSgaze)) or np.any(np.isinf(FSgaze)):
                        print("警告: 无效的视线数据 (NaN或无穷大)")
                        return None
                    
                    # 转换为像素坐标
                    screen_pos_mm = FSgaze.flatten()[:2]
                    screen_pos_px = self.homtrans._mm2pixel(screen_pos_mm)
                    
                    # 确保坐标在屏幕范围内
                    gaze_x = max(0, min(screen_pos_px[0], self.width))
                    gaze_y = max(0, min(screen_pos_px[1], self.height))
                    
                    # 添加额外的质量指标
                    confidence = eye_info.get('confidence', 1.0)  # 如果有置信度信息
                    
                    # 保存数据，包含更多详细信息
                    gaze_data_point = {
                        'timestamp': timestamp,
                        'gaze_x': float(gaze_x),  # 存储为浮点数以保持精度
                        'gaze_y': float(gaze_y),
                        'confidence': float(confidence),
                        'raw_gaze_vector': gaze.flatten().tolist()  # 保存原始视线向量
                    }
                    
                    target_data_point = {
                        'timestamp': timestamp,
                        'target_x': float(target_pos[0]),
                        'target_y': float(target_pos[1]),
                        'angle': float(self.current_angle),
                        'circle_count': int(self.completed_circles)
                    }
                    
                    self.gaze_data.append(gaze_data_point)
                    self.target_data.append(target_data_point)
                    
                    # 每50个数据点打印一次进度
                    if len(self.gaze_data) % 50 == 0:
                        print(f"已采集 {len(self.gaze_data)} 个数据点")
                    
                    return (int(gaze_x), int(gaze_y))
                except Exception as mapping_error:
                    # 减少错误日志频率
                    if len(self.gaze_data) % 50 == 0:
                        print(f"视线映射错误: {str(mapping_error)[:100]}...")  # 限制错误消息长度
                    return None
            else:
                # 减少警告频率
                if len(self.gaze_data) % 100 == 0:
                    print("警告: 无法获取视线信息")
                return None
        except Exception as e:
            # 减少错误日志频率
            if len(self.gaze_data) % 100 == 0:
                print(f"数据采集错误: {str(e)[:100]}...")  # 限制错误消息长度
            return None
    
    def save_collected_data(self):
        """保存采集的注视点和目标点数据，包含单独文件和合并文件"""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.project_dir, "analysis_data")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存注视点数据
            gaze_file = os.path.join(output_dir, f"gaze_data_{timestamp_str}.json")
            with open(gaze_file, 'w', encoding='utf-8') as f:
                json.dump(self.gaze_data, f, ensure_ascii=False, indent=2)
            
            # 保存目标点数据
            target_file = os.path.join(output_dir, f"target_data_{timestamp_str}.json")
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(self.target_data, f, ensure_ascii=False, indent=2)
            
            # 保存合并的数据，便于分析
            combined_data = []
            min_length = min(len(self.gaze_data), len(self.target_data))
            for i in range(min_length):
                combined_point = {
                    'timestamp': self.gaze_data[i]['timestamp'],
                    'gaze_x': self.gaze_data[i]['gaze_x'],
                    'gaze_y': self.gaze_data[i]['gaze_y'],
                    'target_x': self.target_data[i]['target_x'],
                    'target_y': self.target_data[i]['target_y'],
                    'angle': self.target_data[i]['angle'],
                    'circle_count': self.target_data[i]['circle_count']
                }
                # 添加额外的质量信息（如果存在）
                if 'confidence' in self.gaze_data[i]:
                    combined_point['confidence'] = self.gaze_data[i]['confidence']
                combined_data.append(combined_point)
            
            combined_file = os.path.join(output_dir, f"combined_data_{timestamp_str}.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            
            print(f"数据保存完成:")
            print(f"  - 注视点数据: {gaze_file}")
            print(f"  - 目标点数据: {target_file}")
            print(f"  - 合并数据: {combined_file}")
            
            return gaze_file, target_file, combined_file
        except Exception as e:
            print(f"保存数据时出错: {e}")
            return None, None, None
    
    def analyze_trajectory_similarity(self):
        """分析用户注视轨迹与目标轨迹的相似度
        
        返回:
            dict: 包含各种相似度指标的字典
        """
        if len(self.gaze_data) < 2 or len(self.target_data) < 2:
            print("警告: 数据点太少，无法进行有效分析")
            return None
        
        try:
            # 获取有效的数据点（确保两个数据集长度相同）
            min_length = min(len(self.gaze_data), len(self.target_data))
            
            # 提取X/Y坐标数据
            gaze_points = np.array([[d['gaze_x'], d['gaze_y']] for d in self.gaze_data[:min_length]])
            target_points = np.array([[d['target_x'], d['target_y']] for d in self.target_data[:min_length]])
            
            # 1. 计算每对对应点的欧氏距离
            euclidean_distances = np.sqrt(np.sum((gaze_points - target_points) ** 2, axis=1))
            
            # 基本统计指标
            metrics = {
                'mean_euclidean_distance': float(np.mean(euclidean_distances)),
                'median_euclidean_distance': float(np.median(euclidean_distances)),
                'std_euclidean_distance': float(np.std(euclidean_distances)),
                'max_euclidean_distance': float(np.max(euclidean_distances)),
                'min_euclidean_distance': float(np.min(euclidean_distances)),
                'total_data_points': min_length
            }
            
            # 2. 计算动态时间规整(DTW)距离
            try:
                dtw_distance, path = fastdtw(gaze_points, target_points, dist=euclidean)
                metrics['dtw_distance'] = float(dtw_distance)
                # 归一化DTW距离
                metrics['normalized_dtw'] = float(dtw_distance / min_length)
            except Exception as dtw_error:
                print(f"DTW计算错误: {dtw_error}")
                # 回退方案：使用简化的DTW实现
                dtw_simple = self._simple_dtw(gaze_points, target_points)
                metrics['dtw_distance_simple'] = float(dtw_simple)
            
            # 3. 计算相关系数
            try:
                # X坐标相关系数
                if np.std(gaze_points[:, 0]) > 0 and np.std(target_points[:, 0]) > 0:
                    metrics['correlation_x'] = float(correlation(gaze_points[:, 0], target_points[:, 0]))
                else:
                    metrics['correlation_x'] = 0.0
                
                # Y坐标相关系数
                if np.std(gaze_points[:, 1]) > 0 and np.std(target_points[:, 1]) > 0:
                    metrics['correlation_y'] = float(correlation(gaze_points[:, 1], target_points[:, 1]))
                else:
                    metrics['correlation_y'] = 0.0
                
                # 综合相关系数
                metrics['mean_correlation'] = float((metrics['correlation_x'] + metrics['correlation_y']) / 2)
            except Exception as corr_error:
                print(f"相关系数计算错误: {corr_error}")
                metrics['correlation_error'] = str(corr_error)
            
            # 4. 计算其他有用的指标
            # 轨迹平滑度
            metrics['trajectory_smoothness'] = float(self._calculate_smoothness(gaze_points))
            
            # 圆心偏差（与完美圆形轨迹的偏差）
            metrics['circle_center_deviation'] = float(self._calculate_center_deviation(gaze_points))
            
            # 有效追踪率（距离小于阈值的点的比例）
            threshold = metrics['mean_euclidean_distance'] * 1.5  # 使用动态阈值
            metrics['effective_tracking_rate'] = float(np.sum(euclidean_distances < threshold) / min_length)
            
            # 计算准确度分数 (0-100)
            # 基于平均距离归一化到屏幕对角线
            screen_diagonal = math.sqrt(self.width**2 + self.height**2)
            metrics['accuracy_score'] = float(max(0, 100 - (metrics['mean_euclidean_distance'] / screen_diagonal * 100)))
            
            # 计算完成圈数统计
            metrics['completed_circles'] = self.completed_circles
            
            print("轨迹相似度分析完成")
            return metrics
            
        except Exception as e:
            print(f"轨迹分析过程中发生错误: {e}")
            return None
    
    def _simple_dtw(self, series1, series2):
        """简化的DTW算法实现（备用方案）"""
        m, n = len(series1), len(series2)
        # 初始化距离矩阵
        dtw_matrix = np.full((m + 1, n + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 填充距离矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = np.sqrt(np.sum((series1[i-1] - series2[j-1]) ** 2))
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],     # 删除
                    dtw_matrix[i, j-1],     # 插入
                    dtw_matrix[i-1, j-1]    # 匹配
                )
        
        return dtw_matrix[m, n]
    
    def _calculate_smoothness(self, points):
        """计算轨迹平滑度（基于速度变化）"""
        if len(points) < 3:
            return 0.0
        
        # 计算相邻点之间的速度向量
        velocities = np.diff(points, axis=0)
        
        # 计算速度变化
        velocity_changes = np.diff(velocities, axis=0)
        
        # 计算速度变化的平均值
        avg_velocity_change = np.mean(np.sqrt(np.sum(velocity_changes ** 2, axis=1)))
        
        # 归一化
        max_possible_change = math.sqrt(self.width**2 + self.height**2) * 0.5
        smoothness = max(0, 1 - (avg_velocity_change / max_possible_change))
        
        return smoothness
    
    def _calculate_center_deviation(self, points):
        """计算轨迹与完美圆形的偏差"""
        # 计算轨迹的平均中心
        actual_center = np.mean(points, axis=0)
        
        # 理想中心
        ideal_center = np.array([self.track_center[0], self.track_center[1]])
        
        # 计算中心偏差
        center_distance = np.sqrt(np.sum((actual_center - ideal_center) ** 2))
        
        # 归一化到屏幕对角线的比例
        screen_diagonal = math.sqrt(self.width**2 + self.height**2)
        normalized_deviation = center_distance / screen_diagonal
        
        return normalized_deviation
    
    def end_evaluation(self):
        """结束评估并保存数据"""
        print(f"评估完成，共采集 {len(self.gaze_data)} 个有效数据点")
        print(f"完成了 {self.completed_circles} 圈圆形轨迹运动")
        
        # 保存采集的数据
        gaze_file, target_file, combined_file = self.save_collected_data()
        
        # 分析轨迹相似度
        print("开始进行轨迹相似度分析...")
        metrics = self.analyze_trajectory_similarity()
        
        if metrics:
            # 保存分析结果
            self.save_analysis_results(metrics, combined_file)
            print("轨迹相似度分析完成")
            return metrics
        else:
            print("警告: 无法完成轨迹相似度分析")
            return None
    
    def save_analysis_results(self, metrics, combined_file):
        """保存分析结果到JSON文件"""
        try:
            # 从合并文件路径提取时间戳
            if combined_file:
                timestamp_str = os.path.basename(combined_file).split('_')[2].split('.')[0]
            else:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_dir = os.path.join(self.project_dir, "analysis_data")
            analysis_file = os.path.join(output_dir, f"analysis_results_{timestamp_str}.json")
            
            # 添加分析时间戳
            metrics['analysis_timestamp'] = datetime.now().isoformat()
            
            # 保存结果
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            print(f"分析结果已保存至: {analysis_file}")
            
            # 打印关键指标摘要（中文）
            print("\n===== 轨迹分析关键指标 =====")
            print(f"平均欧氏距离: {metrics['mean_euclidean_distance']:.2f} 像素")
            print(f"DTW距离: {metrics.get('dtw_distance', metrics.get('dtw_distance_simple', 'N/A')):.2f}")
            print(f"相关系数: {metrics.get('mean_correlation', 'N/A'):.3f}")
            print(f"轨迹平滑度: {metrics['trajectory_smoothness']:.3f}")
            print(f"有效追踪率: {metrics['effective_tracking_rate']*100:.1f}%")
            print(f"准确度分数: {metrics['accuracy_score']:.1f}/100")
            print("==========================================\n")
            
            # 生成可视化报告
            self.generate_visual_report(metrics, timestamp_str)
            
            return analysis_file
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            return None
    
    def generate_visual_report(self, metrics, timestamp_str):
        """Generate visualization report with trajectory comparison and metrics"""
        try:
            output_dir = os.path.join(self.project_dir, "analysis_data")
            report_file = os.path.join(output_dir, f"visual_report_{timestamp_str}.png")
            
            # Get data points
            min_length = min(len(self.gaze_data), len(self.target_data))
            gaze_points = np.array([[d['gaze_x'], d['gaze_y']] for d in self.gaze_data[:min_length]])
            target_points = np.array([[d['target_x'], d['target_y']] for d in self.target_data[:min_length]])
            timestamps = np.array([d['timestamp'] for d in self.gaze_data[:min_length]])
            
            # Calculate distances
            euclidean_distances = np.sqrt(np.sum((gaze_points - target_points) ** 2, axis=1))
            
            # Create a comprehensive figure with multiple subplots
            plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 2, figure=plt.gcf())
            
            # 1. Trajectory Comparison Plot (Main Plot)
            ax1 = plt.subplot(gs[0, :])
            # Plot target trajectory (reference circle)
            circle = plt.Circle((self.center_x, self.center_y), 
                               self.trajectory_radius, 
                               fill=False, 
                               color='red', 
                               linestyle='--', 
                               linewidth=2, 
                               alpha=0.7, 
                               label='目标轨迹')
            ax1.add_artist(circle)
            
            # Plot actual target points
            ax1.scatter(target_points[:, 0], 
                        target_points[:, 1], 
                        color='red', 
                        s=10, 
                        alpha=0.5, 
                        label='目标点')
            
            # Plot gaze trajectory
            ax1.scatter(gaze_points[:, 0], 
                        gaze_points[:, 1], 
                        color='blue', 
                        s=10, 
                        alpha=0.5, 
                        label='视线点')
            
            # Plot connecting lines between gaze and target points
            for i in range(min_length):
                ax1.plot([gaze_points[i, 0], target_points[i, 0]], 
                         [gaze_points[i, 1], target_points[i, 1]], 
                         color='gray', 
                         linestyle='-', 
                         linewidth=0.5, 
                         alpha=0.3)
            
            # Mark the center
            ax1.scatter(self.center_x, 
                        self.center_y, 
                        color='green', 
                        s=50, 
                        marker='+', 
                        label='中心点')
            
            ax1.set_title('视线与目标轨迹对比', fontsize=14)
            ax1.set_xlabel('屏幕X坐标（像素）', fontsize=12)
            ax1.set_ylabel('屏幕Y坐标（像素）', fontsize=12)
            ax1.set_xlim(0, self.width)
            ax1.set_ylim(0, self.height)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            
            # 2. Error Distribution Histogram
            ax2 = plt.subplot(gs[1, 0])
            n, bins, patches = ax2.hist(euclidean_distances, 
                                       bins=30, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7)
            
            # Add mean and median lines
            ax2.axvline(metrics['mean_euclidean_distance'], 
                        color='red', 
                        linestyle='--', 
                        linewidth=2, 
                        label=f'平均值: {metrics["mean_euclidean_distance"]:.2f} 像素')
            ax2.axvline(metrics['median_euclidean_distance'], 
                        color='green', 
                        linestyle='--', 
                        linewidth=2, 
                        label=f'中位数: {metrics["median_euclidean_distance"]:.2f} 像素')
            
            ax2.set_title('欧氏距离分布', fontsize=14)
            ax2.set_xlabel('距离（像素）', fontsize=12)
            ax2.set_ylabel('频率', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. Error Over Time Plot
            ax3 = plt.subplot(gs[1, 1])
            # Normalize timestamps to start from 0
            normalized_time = (timestamps - timestamps[0]) / 1000  # Convert to seconds
            
            ax3.plot(normalized_time, 
                     euclidean_distances, 
                     color='purple', 
                     linestyle='-', 
                     linewidth=1, 
                     alpha=0.7, 
                     label='Error')
            
            # Add moving average
            window_size = max(1, min_length // 50)  # Adaptive window size
            moving_avg = np.convolve(euclidean_distances, np.ones(window_size)/window_size, mode='same')
            ax3.plot(normalized_time, 
                     moving_avg, 
                     color='orange', 
                     linestyle='-', 
                     linewidth=2, 
                     label='移动平均值')
            
            ax3.set_title('误差随时间变化', fontsize=14)
            ax3.set_xlabel('时间（秒）', fontsize=12)
            ax3.set_ylabel('欧氏距离（像素）', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. Metrics Summary
            ax4 = plt.subplot(gs[2, :])
            
            # Create a clean metrics display
            metrics_text = [
                f'视线校准评估结果',
                f'时间戳: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                f'',
                f'性能指标:',
                f'  • 平均误差: {metrics["mean_euclidean_distance"]:.2f} 像素',
                f'  • 中位数误差: {metrics["median_euclidean_distance"]:.2f} 像素',
                f'  • 误差标准差: {metrics["std_euclidean_distance"]:.2f} 像素',
                f'  • 最大误差: {metrics["max_euclidean_distance"]:.2f} 像素',
                f'  • 最小误差: {metrics["min_euclidean_distance"]:.2f} 像素',
                f'  • DTW距离: {metrics.get("dtw_distance", metrics.get("dtw_distance_simple", "N/A")):.2f}',
                f'  • 相关系数: {metrics.get("mean_correlation", "N/A"):.3f}',
                f'  • 有效追踪率: {metrics["effective_tracking_rate"]*100:.1f}%',
                f'  • 轨迹平滑度: {metrics["trajectory_smoothness"]:.3f}',
                f'  • 准确度分数: {metrics["accuracy_score"]:.1f}/100',
                f'',
                f'实验详情:',
                f'  • 总数据点: {metrics["total_data_points"]}',
                f'  • 完成圈数: {metrics["completed_circles"]}',
                f'  • 屏幕分辨率: {self.width}x{self.height}',
                f'  • 轨迹半径: {self.trajectory_radius} 像素'
            ]
            
            # Display text with formatting
            ax4.axis('off')  # Hide axes
            text_box = ax4.text(0.5, 0.5, 
                               '\n'.join(metrics_text), 
                               horizontalalignment='center',
                               verticalalignment='center',
                               fontsize=11,
                               family='monospace',
                               bbox=dict(boxstyle='round,pad=1', 
                                        facecolor='#f0f0f0', 
                                        alpha=0.7))
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.2)
            
            # Save the figure
            plt.savefig(report_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visual report generated and saved to: {report_file}")
            
            # Generate a simplified version for display in Pygame
            simplified_file = self._generate_simplified_visual(report_file, timestamp_str)
            
            return report_file, simplified_file
            
        except Exception as e:
            print(f"Error generating visual report: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _generate_simplified_visual(self, full_report_path, timestamp_str):
        """Generate a simplified visual for display in Pygame"""
        try:
            output_dir = os.path.join(self.project_dir, "analysis_data")
            simplified_file = os.path.join(output_dir, f"simplified_report_{timestamp_str}.png")
            
            # Get data points
            min_length = min(len(self.gaze_data), len(self.target_data))
            gaze_points = np.array([[d['gaze_x'], d['gaze_y']] for d in self.gaze_data[:min_length]])
            target_points = np.array([[d['target_x'], d['target_y']] for d in self.target_data[:min_length]])
            
            # Calculate basic metrics
            euclidean_distances = np.sqrt(np.sum((gaze_points - target_points) ** 2, axis=1))
            mean_error = np.mean(euclidean_distances)
            median_error = np.median(euclidean_distances)
            accuracy_score = max(0, 100 - (mean_error / math.sqrt(self.width**2 + self.height**2) * 100))
            
            # Create a simplified figure
            plt.figure(figsize=(10, 7))
            
            # Plot trajectory comparison
            plt.scatter(target_points[:, 0], target_points[:, 1], 
                       color='red', s=20, alpha=0.6, label='Target')
            plt.scatter(gaze_points[:, 0], gaze_points[:, 1], 
                       color='blue', s=20, alpha=0.6, label='Gaze')
            
            # Plot reference circle
            circle = plt.Circle((self.center_x, self.center_y), 
                               self.trajectory_radius, 
                               fill=False, color='red', 
                               linestyle='--', linewidth=1.5)
            plt.gca().add_artist(circle)
            
            # Add center mark
            plt.scatter(self.center_x, self.center_y, 
                       color='green', s=50, marker='+')
            
            # Add summary text
            summary_text = [
                f'Mean Error: {mean_error:.2f} px',
                f'Median Error: {median_error:.2f} px',
                f'Accuracy: {accuracy_score:.1f}/100',
                f'Data Points: {min_length}'
            ]
            
            plt.figtext(0.5, 0.01, '\n'.join(summary_text), 
                      ha='center', fontsize=10, 
                      bbox=dict(facecolor='lightgray', alpha=0.5))
            
            plt.title('Gaze Calibration Results', fontsize=14)
            plt.xlabel('X (pixels)', fontsize=12)
            plt.ylabel('Y (pixels)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            plt.xlim(0, self.width)
            plt.ylim(0, self.height)
            plt.gca().set_aspect('equal')
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(simplified_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return simplified_file
        except Exception as e:
            print(f"Error generating simplified visual: {e}")
            return None
    
    def run_test(self):
        """运行测试主循环"""
        self.start_test()
        
        # 检查屏幕是否有效
        if not self.screen or pygame.display.get_surface() is None:
            print("错误：显示表面无效，重新初始化显示")
            # 重新初始化显示
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("眼动追踪校准评估")
            if not self.font:
                self.font = pygame.font.SysFont("Arial", 24)
        
        prev_time = time.time()
        while self.is_testing and pygame.display.get_surface() is not None:
            # 计算帧时间
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = current_time
            timestamp = current_time - self.test_start_time
            
            # 更新目标位置
            target_pos = self.update_target_position(elapsed_time)
            
            # 采集视线数据 - 使用延迟补偿的目标位置
            compensated_target = self._get_compensated_target()
            if compensated_target:
                gaze_pos = self.collect_gaze_data(compensated_target, timestamp)
            else:
                # 如果历史数据不足，先使用当前目标位置
                gaze_pos = self.collect_gaze_data(target_pos, timestamp)
            
            try:
                # 绘制界面
                self.screen.fill(WHITE)
                
                # 绘制目标点
                pygame.draw.circle(self.screen, RED, target_pos, self.target_radius)
                
                # 不再绘制视线蓝点
                
                # 显示进度信息
                progress_text = self.font.render(f"完成圈数: {self.completed_circles}/2", True, BLACK)
                self.screen.blit(progress_text, (50, 50))
                
                # 更新显示
                pygame.display.flip()
            except pygame.error as e:
                print(f"显示错误: {e}")
                # 尝试重新创建显示表面
                try:
                    self.screen = pygame.display.set_mode((self.width, self.height))
                except:
                    self.is_testing = False
                    print("无法重新创建显示表面，退出测试")
            
            # 检查是否完成测试
            if self.completed_circles >= 2:
                self.is_testing = False
                print("测试完成")
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.is_testing = False
                    print("测试被用户中断")
            
            # 限制帧率
            self.clock.tick(60)
        
        # 结束评估并分析数据
        if len(self.gaze_data) > 0:
            try:
                metrics = self.end_evaluation()
                # 返回原始的文件路径，保持向后兼容
                return self.save_collected_data()[:2]  # 只返回前两个文件路径
            except Exception as e:
                print(f"分析数据时出错: {e}")
        return None, None
    
    def _get_compensated_target(self):
        """
        获取延迟补偿后的目标位置
        根据延迟帧数，返回历史上的目标位置
        
        Returns:
            tuple: (x, y) 延迟补偿后的目标位置坐标，如果历史数据不足则返回None
        """
        # 检查历史数据是否足够
        if len(self.target_history) > self.delay_frames:
            # 返回延迟指定帧数的历史目标位置
            # 索引0是最早的历史记录，所以使用- (self.delay_frames + 1) 来获取正确的历史位置
            delayed_index = - (self.delay_frames + 1)
            if delayed_index >= -len(self.target_history):
                history_item = self.target_history[delayed_index]
                return (history_item['x'], history_item['y'])
        
        # 如果历史数据不足，返回None
        return None
    
    def analyze_trajectory_similarity(self):
        """分析圆环整体相似度"""
        print("开始圆环整体相似度分析...")
        
        # 使用实例变量中的数据
        gaze_data = self.gaze_data
        target_data = self.target_data
        
        # 对齐数据时间戳
        aligned_data = []
        for i, gaze_point in enumerate(gaze_data):
            if i < len(target_data):
                aligned_data.append({
                    'gaze_x': gaze_point['gaze_x'],
                    'gaze_y': gaze_point['gaze_y'],
                    'target_x': target_data[i]['target_x'],
                    'target_y': target_data[i]['target_y'],
                    'timestamp': gaze_point['timestamp']
                })
        
        # 提取目标圆形参数
        target_center = self.track_center
        target_radius = self.track_radius
        
        # 提取用户注视点数据用于椭圆拟合
        gaze_points = np.array([[p['gaze_x'], p['gaze_y']] for p in aligned_data])
        
        # 拟合用户注视点形成的椭圆
        user_ellipse_params = self._fit_ellipse(gaze_points)
        
        if user_ellipse_params is None:
            print("无法拟合椭圆，使用基本统计指标")
            # 如果无法拟合椭圆，使用基本统计指标
            distances = []
            for point in aligned_data:
                dx = point['gaze_x'] - point['target_x']
                dy = point['gaze_y'] - point['target_y']
                distance = np.sqrt(dx*dx + dy*dy)
                distances.append(distance)
            
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            std_distance = np.std(distances)
            
            results = {
                'mean_distance': mean_distance,
                'median_distance': median_distance,
                'std_distance': std_distance,
                'sample_count': len(aligned_data),
                'ellipse_fit_successful': False
            }
        else:
            # 计算圆环相似度指标
            center_distance = np.sqrt((user_ellipse_params['center'][0] - target_center[0])**2 + 
                                     (user_ellipse_params['center'][1] - target_center[1])** 2)
            
            # 计算形状相似度（圆形度）
            aspect_ratio = user_ellipse_params['semi_minor'] / user_ellipse_params['semi_major']
            circularity = min(aspect_ratio, 1/aspect_ratio)  # 圆形度，越接近1越圆
            
            # 计算半径相似度
            radius_ratio = user_ellipse_params['semi_major'] / target_radius if target_radius > 0 else 0
            
            # 计算整体相似度评分（0-100分）
            position_similarity = max(0, 100 - (center_distance / target_radius) * 50)  # 位置相似度
            shape_similarity = circularity * 100  # 形状相似度（圆形度）
            size_similarity = max(0, 100 - abs(radius_ratio - 1) * 100)  # 大小相似度
            
            # 综合评分
            overall_similarity = (position_similarity * 0.4 + shape_similarity * 0.4 + size_similarity * 0.2)
            
            # 计算基本统计指标用于参考
            distances = []
            for point in aligned_data:
                # 计算点到拟合椭圆的距离
                dx = (point['gaze_x'] - user_ellipse_params['center'][0]) / user_ellipse_params['semi_major']
                dy = (point['gaze_y'] - user_ellipse_params['center'][1]) / user_ellipse_params['semi_minor']
                dist_to_ellipse = abs(dx*dx + dy*dy - 1) * (user_ellipse_params['semi_major'] + user_ellipse_params['semi_minor']) / 2
                distances.append(dist_to_ellipse)
            
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            
            # 保存分析结果
            results = {
                'target_center_x': target_center[0],
                'target_center_y': target_center[1],
                'target_radius': target_radius,
                'user_center_x': user_ellipse_params['center'][0],
                'user_center_y': user_ellipse_params['center'][1],
                'user_semi_major': user_ellipse_params['semi_major'],
                'user_semi_minor': user_ellipse_params['semi_minor'],
                'user_angle': user_ellipse_params['angle'],
                'center_distance': center_distance,
                'circularity': circularity,
                'radius_ratio': radius_ratio,
                'position_similarity': position_similarity,
                'shape_similarity': shape_similarity,
                'size_similarity': size_similarity,
                'overall_similarity': overall_similarity,
                'mean_distance_to_ellipse': mean_distance,
                'median_distance_to_ellipse': median_distance,
                'sample_count': len(aligned_data),
                'ellipse_fit_successful': True
            }
            
            # 打印分析结果
            print(f"目标圆参数: 中心({target_center[0]}, {target_center[1]}), 半径={target_radius}")
            print(f"用户椭圆参数: 中心({user_ellipse_params['center'][0]}, {user_ellipse_params['center'][1]})")
            print(f"椭圆半轴: 长={user_ellipse_params['semi_major']}, 短={user_ellipse_params['semi_minor']}")
            print(f"位置相似度: {position_similarity:.2f}/100")
            print(f"形状相似度: {shape_similarity:.2f}/100")
            print(f"大小相似度: {size_similarity:.2f}/100")
            print(f"整体相似度评分: {overall_similarity:.2f}/100")
            print(f"平均到椭圆距离: {mean_distance:.2f} px")
        
        # 保存结果
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.project_dir, "analysis_data", f"analysis_results_{timestamp_str}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 确保所有统计变量在使用前都已定义
        std_distance = float(np.std(distances))
        correlation_x = results.get('correlation_x', 0.0)
        correlation_y = results.get('correlation_y', 0.0)
        dtw_dist = results.get('dtw_distance', 0.0)
        
        print(f"分析结果: {result_file}")
        print(f"平均距离: {mean_distance:.2f} px")
        print(f"中位数距离: {median_distance:.2f} px")
        print(f"标准差: {std_distance:.2f} px")
        print(f"相关系数X: {correlation_x:.4f}")
        print(f"相关系数Y: {correlation_y:.4f}")
        print(f"DTW距离: {dtw_dist:.2f}")
        
        return results, aligned_data, result_file
    
    def _fit_ellipse(self, points):
        """
        拟合椭圆到点集
        
        Args:
            points: 二维点集数组
            
        Returns:
            包含椭圆参数的字典或None（如果无法拟合）
        """
        try:
            if len(points) < 5:
                return None
            
            # 使用OpenCV进行椭圆拟合
            # 需要将点转换为适合fitEllipse的格式
            points = points.astype(np.int32)
            ellipse = cv2.fitEllipse(points)
            
            # 解析椭圆参数
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            
            return {
                'center': (center_x, center_y),
                'semi_major': major_axis / 2,
                'semi_minor': minor_axis / 2,
                'angle': angle
            }
        except Exception as e:
            print(f"椭圆拟合错误: {e}")
            return None
    
    def visualize_results(self, aligned_data, results):
        """可视化圆环整体相似度结果"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.patches import Ellipse, Circle
            plt.rcParams['font.family'] = 'SimHei' 
            # 创建图形
            fig = plt.figure(figsize=(15, 10), dpi=100)
            
            # 1. 轨迹与拟合椭圆对比图
            ax1 = fig.add_subplot(2, 2, 1)
            
            # 绘制目标圆形 - 反转Y坐标以匹配屏幕坐标系
            target_circle = Circle(
                (results['target_center_x'], -results['target_center_y']),
                results['target_radius'],
                fill=False,
                color='red',
                linestyle='-',
                linewidth=2,
                label='Target Circle'
            )
            ax1.add_patch(target_circle)
            
            # 绘制用户注视点
            gaze_x = [p['gaze_x'] for p in aligned_data]
            gaze_y = [p['gaze_y'] for p in aligned_data]
            # 反转Y轴以匹配屏幕坐标系
            ax1.scatter(gaze_x, [-y for y in gaze_y], s=10, color='blue', alpha=0.5, label='User Gaze Points')
            
            # 绘制拟合椭圆
            if results['ellipse_fit_successful']:
                user_ellipse = Ellipse(
                    (results['user_center_x'], -results['user_center_y']),
                    2 * results['user_semi_major'],
                    2 * results['user_semi_minor'],
                    angle=results['user_angle'],
                    fill=False,
                    color='green',
                    linestyle='--',
                    linewidth=2,
                    label='Fitted Ellipse'
                )
                ax1.add_patch(user_ellipse)
            
            # 设置图表属性
            ax1.set_title('Trajectory and Fitted Shapes')
            ax1.set_xlabel('X Position (px)')
            ax1.set_ylabel('Y Position (px)')
            ax1.legend()
            ax1.grid(True)
            
            # 设置坐标轴范围以包含所有图形
            max_dim = max(self.width, self.height)
            center = (self.width // 2, self.height // 2)
            ax1.set_xlim(center[0] - max_dim // 2, center[0] + max_dim // 2)
            ax1.set_ylim(-center[1] - max_dim // 2, -center[1] + max_dim // 2)
            ax1.set_aspect('equal')
            
            # 2. 相似度评分可视化
            ax2 = fig.add_subplot(2, 2, 2)
            
            # 创建评分数据
            categories = ['Position', 'Shape', 'Size', 'Overall']
            scores = [
                results['position_similarity'],
                results['shape_similarity'],
                results['size_similarity'],
                results['overall_similarity']
            ]
            
            # 绘制条形图
            bars = ax2.bar(categories, scores, color=['blue', 'green', 'orange', 'red'])
            
            # 在条形图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom')
            ax2.set_ylim(0, 105)
            ax2.set_title('Similarity Scores')
            ax2.set_ylabel('Score (0-100)')
            ax2.grid(axis='y', alpha=0.3)
            
            # 3. 椭圆参数对比
            ax3 = fig.add_subplot(2, 2, 3)
            
            # 参数数据
            param_names = ['Center Distance', 'Circularity', 'Radius Ratio']
            param_values = [
                results['center_distance'],
                results['circularity'],
                results['radius_ratio']
            ]
            
            # 格式化显示
            formatted_values = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in param_values]
            
            # 创建表格
            table_data = [[param_names[i], formatted_values[i]] for i in range(len(param_names))]
            table = ax3.table(cellText=table_data, colLabels=['Parameter', 'Value'], loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax3.axis('off')
            
            # 4. 整体评估结果摘要
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            
            # 根据整体相似度评分给出评估等级
            if results['overall_similarity'] >= 90:
                rating = "Excellent"
                rating_color = "green"
            elif results['overall_similarity'] >= 75:
                rating = "Good"
                rating_color = "green"
            elif results['overall_similarity'] >= 60:
                rating = "Fair"
                rating_color = "orange"
            else:
                rating = "Poor"
                rating_color = "red"
            
            # 添加评估文本
            summary_text = (
                f"Evaluation Results\n"\
                f"=================\n"\
                f"Overall Rating: {rating}\n"\
                f"Overall Score: {results.get('overall_similarity', 0):.1f}/100\n"\
                f"Sample Count: {results.get('sample_count', 'N/A')}\n"\
                f"Mean Distance: {results.get('mean_distance_to_ellipse', 0):.2f} px"
            )
            
            ax4.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=1", facecolor=rating_color, alpha=0.1))
            
            plt.tight_layout()
            
            # 保存图表
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = os.path.join(self.project_dir, "analysis_data", f"evaluation_chart_{timestamp_str}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"可视化图表保存: {chart_file}")
            
            # 在Pygame中显示结果
            self.show_results_screen(chart_file, results)
            
            return chart_file
        except Exception as e:
            print(f"可视化错误: {e}")
            return None
    
    def show_results_screen(self, chart_file, results):
        """显示结果屏幕"""
        # 加载图表
        try:
            chart_surface = pygame.image.load(chart_file)
            # 缩放图表以适应屏幕
            chart_width = min(chart_surface.get_width(), self.width * 0.9)
            chart_height = min(chart_surface.get_height(), self.height * 0.7)
            chart_surface = pygame.transform.scale(chart_surface, (chart_width, chart_height))
        except:
            chart_surface = None
        
        while True:
            self.screen.fill(WHITE)
            
            # 绘制标题
            title_text = self.font.render("校准评估结果", True, BLACK)
            title_rect = title_text.get_rect(center=(self.width // 2, 50))
            self.screen.blit(title_text, title_rect)
            
            # 绘制图表
            if chart_surface:
                chart_x = (self.width - chart_width) // 2
                chart_y = 100
                self.screen.blit(chart_surface, (chart_x, chart_y))
            
            # 显示关键指标 - 改为中文
            metrics = [
                f"平均误差: {results.get('mean_distance', 0):.2f} px",
                f"中位数误差: {results.get('median_distance', 0):.2f} px",
                f"X相关系数: {results.get('correlation_x', 0.0):.4f}",
                f"Y相关系数: {results.get('correlation_y', 0.0):.4f}",
                "按ESC键退出"
            ]
            
            for i, metric in enumerate(metrics):
                text = self.font.render(metric, True, BLACK)
                text_rect = text.get_rect(center=(self.width // 2, chart_y + chart_height + 50 + i * 30))
                self.screen.blit(text, text_rect)
            
            pygame.display.flip()
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
    
    def run(self):
        """运行主程序"""
        try:
            # 初始化
            if not self.initialize():
                print("初始化失败")
                return
            
            # 添加启动按钮界面 - 确保在开始校准前显示按钮
            self.show_start_screen()
            
            # 确保先进行校准
            print("开始校准流程...")
            if not self.run_calibration():
                print("校准失败，程序退出")
                return
            
            # 校准后检查显示表面有效性
            if not pygame.display.get_init() or not self.screen or not self.screen.get_width():
                print("重新初始化显示表面...")
                pygame.display.quit()
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption('Gaze Calibration Test')
                
                # 重新创建字体
                self.font = pygame.font.SysFont(None, 36)
            
            # 清空目标历史，准备测试
            self.target_history = []
            
            # 运行测试
            gaze_file, target_file = self.run_test()
            
            # 分析结果
            if gaze_file and target_file:
                try:
                    results, aligned_data, _ = self.analyze_trajectory_similarity()
                    
                    # 可视化结果
                    self.visualize_results(aligned_data, results)
                    
                    print("测试完成！")
                except Exception as analysis_error:
                    print(f"分析或可视化错误: {analysis_error}")
        except Exception as e:
            print(f"程序运行错误: {e}")
        finally:
            # 清理资源
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            try:
                pygame.quit()
            except:
                pass

if __name__ == "__main__":
    # 获取项目目录
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 运行评估器
    evaluator = GazeCalibrationEvaluator(project_dir)
    evaluator.run()