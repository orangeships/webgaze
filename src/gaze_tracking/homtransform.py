import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R
import cv2
import os
import keyboard
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timm.models.vovnet import OsaStage
from gaze_tracking.calibration_pygame import PygameCalibrationUI, PygameCalibrationTargets, getScreenSize, getWhiteFrame, ReadCameraCalibrationData
from sfm.sfm_module import SFM
import utilities.utils as util

# Pygame支持 - 现在是必需的
try:
    import pygame
    from gaze_tracking.calibration_pygame import PygameCalibrationUI, PygameCalibrationTargets
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("警告：Pygame未安装，程序可能无法正常运行")


class HomTransform:
    """
    Calibration from gaze coordinates to screen coordinates
    """
    def __init__(self, directory, custom_width=None, custom_height=None) -> None:            
        self.dir = directory
        if custom_width and custom_height:
            # 使用自定义的屏幕尺寸
            self.width = custom_width
            self.height = custom_height
            # 尝试获取对应显示器的物理尺寸
            self.width_mm, self.height_mm = self._get_monitor_physical_size(custom_width, custom_height)
        else:
            # 使用默认的屏幕尺寸获取方法
            self.width, self.height, self.width_mm, self.height_mm = getScreenSize()
        self.df = pd.DataFrame()
        self.sfm = SFM(directory)
        self.camera_matrix, self.dist_coeffs = ReadCameraCalibrationData(os.path.join(directory, "camera_data"))
        # self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        # Tkinter渲染器（仅用于追踪页面）
        # self.renderer = None
        
        # 调试计数器
        self.debug_counter = 0  # 调试计数器
        
        # 重要：初始化校准相关的属性，防止属性不存在错误
        self.STransG = None  # 校准变换矩阵，初始化为None
        self.scaleWtG = None  # 缩放因子，初始化为None
        self.scaleWtG2 = None  # 第二个缩放因子，初始化为None
        self.STransW = None  # 世界坐标变换矩阵，初始化为None
        self.StG = None  # 注视点变换矩阵列表，初始化为None
        self.StW = None  # 世界坐标变换矩阵列表，初始化为None
        self.SetVal = None  # 校准点值，初始化为None
        self.gaze = None  # 注视数据，初始化为None
        self.calibrate_pnp = None  # PnP校准变换矩阵，初始化为None
    
    def _get_monitor_physical_size(self, target_width, target_height):
        """
        获取指定分辨率显示器的物理尺寸，添加重试机制和错误处理
        
        Args:
            target_width: 目标显示器宽度（像素）
            target_height: 目标显示器高度（像素）
            
        Returns:
            tuple: (width_mm, height_mm) 物理尺寸
        """
        import time
        
        max_retries = 3
        retry_delay = 0.1  # 100ms延迟
        
        for attempt in range(max_retries):
            try:
                print(f"尝试获取显示器物理尺寸 (第 {attempt + 1} 次)...")
                from screeninfo import get_monitors
                monitors = get_monitors()
                
                if not monitors:
                    print(f"未检测到任何显示器，重试中...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("无法检测到任何显示器")
                
                # 找到最接近的显示器
                min_distance = float('inf')
                target_monitor = None
                for monitor in monitors:
                    # 计算分辨率差的平方和作为距离
                    distance = (monitor.width - target_width)**2 + (monitor.height - target_height)**2
                    if distance < min_distance:
                        min_distance = distance
                        target_monitor = monitor
                
                if target_monitor:
                    width_mm = target_monitor.width_mm
                    height_mm = target_monitor.height_mm
                    
                    print(f"[物理尺寸] 原始数据: 显示器 {target_monitor.width}x{target_monitor.height}, 物理尺寸: {width_mm}mmx{height_mm}mm")
                    
                    # 特殊情况：1920x1080 副屏硬编码为24寸显示器尺寸
                    if target_width == 1920 and target_height == 1080:
                        print(f"[物理尺寸] 检测到1080p副屏，使用硬编码24寸屏幕尺寸")
                        width_mm = 531  # 24寸显示器标准宽度
                        height_mm = 299  # 24寸显示器标准高度
                    
                    # 验证物理尺寸数据
                    if width_mm is None or height_mm is None or width_mm <= 0 or height_mm <= 0:
                        raise Exception(f"无法获取有效物理尺寸: width_mm={width_mm}, height_mm={height_mm}")
                    
                    print(f"成功获取显示器物理尺寸: {target_width}x{target_height} -> {width_mm}mmx{height_mm}mm")
                    return width_mm, height_mm
                else:
                    raise Exception("未找到匹配的显示器")
                    
            except Exception as e:
                print(f"获取显示器物理尺寸失败 (第 {attempt + 1} 次): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print("所有重试均失败，抛出异常")
                    raise e

    def RecordGaze(self, model, cap, sfm=False):
        df = pd.DataFrame()
        frame_prev = None
        WTransG1 = np.eye(4)
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except StopIteration:
                break

            eye_info = model.get_gaze(frame)
            gaze = eye_info['gaze']

            if sfm:
                if frame_prev is not None:                
                    WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame)        # WtG1 is a unit vector, has to be scaled            
                frame_prev = frame
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
            else:
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)

            FSgaze = self._mm2pixel(FSgaze)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            df = pd.concat([ df, pd.DataFrame([np.hstack((timestamp, eye_info['gaze'], FSgaze.flatten(), eye_info['EyeRLCenterPos'], eye_info['HeadPosAnglesYPR'], eye_info['HeadPosInFrame'])) ]) ])

            cv2.waitKey(1)
            if keyboard.is_pressed('esc'):
                print("Recording stopped")
                break
        cap.release()
        df.columns = ['timestamp(hh:m:s.ms)','gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x', 'Sgaze_y', 'Sgaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll', 'HeadPos_x', 'HeadPos_y']
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.dir, "results", "MyGazeTracking.csv"))

    def RunGazeOnScreen(self, model, cap, sfm=False):
        """ Present different trajectories on screen and record gaze
        """

        # 禁用视频输出以提高性能
        out_video = None
        wc_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        wc_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 创建白色帧用于处理
        white_frame = getWhiteFrame(self.width, self.height)
        
        # 移除OpenCV渲染器，使用Pygame相关功能
        # 这里使用PygameCalibrationTargets替代OpenCV的Targets
        targets = PygameCalibrationTargets(self.width, self.height)
        frame_prev = None
        WTransG1 = np.eye(4)
        targets.setSetPos([int(self.width/8), int(self.height/8)])   # for DrawSpecificTarget()
        FSgaze = np.array([[-10],[-10],[-10]])
        
        print("开始注视跟踪...")
        print("按ESC键退出")
        # print("也可以点击窗口右上角的关闭按钮退出") # OpenCV窗口没有关闭按钮，所以不需要这个提示
        
        # 添加帧率计算变量
        frame_count = 0
        start_time = time.time()
        fps = 0
        # window_update_counter = 0  # 单独的窗口更新计数器 - OpenCV不需要这个
        frame_skip_counter = 0  # 跳帧计数器
        frame_skip_interval = 1  # 增加跳帧间隔，减少处理的帧数以提高性能
        
        # 彻底禁用数据记录功能
        save_data = False
        
        while cap.isOpened():
            
            gazeframe, SetPos = targets.DrawTargetInMiddle(white_frame.copy(), self._mm2pixel(FSgaze)) # 传入副本，避免修改原始white_frame

            try:
                ret, frame = cap.read()
            except StopIteration:
                break
            
            # 更新帧计数
            frame_count += 1
            frame_skip_counter += 1
            
            # 跳帧处理 - 降低处理频率以提高性能
            if frame_skip_counter % (frame_skip_interval + 1) != 0:
                # 显示当前帧的gazeframe，即使跳过处理，也要更新显示
                # 注意：targets.DrawTargetInMiddle已经创建了"Gaze"窗口，不需要再调用display_window
                key_pressed = cv2.waitKey(1)
                if key_pressed == 27:  # ESC键退出
                    print("退出追踪")
                    break
                continue
                
            # window_update_counter += 1 # OpenCV不需要这个
            self.debug_counter += 1
            
            # 计算帧率（每秒更新一次）
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 1.0:  # 每秒更新一次帧率
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time
                
            # 数据记录功能已完全禁用，移除相关代码
            
            # gray_image, prediction, morphedMask, falseColor, centroid = model.get_iris_Cnn(frame)
            # Undistort the image
            # frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            eye_info = model.get_gaze(frame)
            
            if eye_info is None:
                print("No eye info detected in this frame. Skipping...")
                # 即使没有眼睛信息，也要检查退出键
                # 注意：targets.DrawTargetInMiddle已经创建了"Gaze"窗口，不需要再调用display_window
                key_pressed = cv2.waitKey(1)
                if key_pressed == 27:
                    break
                continue
            
            gaze = eye_info['gaze']

            if frame_prev is not None and sfm:                
                WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame)        # WtG1 is a unit vector, has to be scaled   

            frame_prev = frame

            if sfm:
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
            else:
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)
            
            # 简化处理，只转换坐标，不生成时间戳或准备数据记录
            SetPos = self._pixel2mm(SetPos)

            # 在终端输出帧率信息
            if elapsed_time > 1.0:  # 每秒更新一次帧率
                pass  # FPS输出已移至main_pygame.py，避免重复输出
            
           # if window_update_counter % 5 == 0: # OpenCV不需要这个
            key_pressed = cv2.waitKey(1)
            if key_pressed == 27:  # ESC键退出
                print("退出追踪")
                break

            # 显示gazeframe
            # 注意：targets.DrawTargetInMiddle已经创建了"Gaze"窗口，不需要再调用display_window

        # 释放摄像头资源
        cap.release()
        
        # 释放视频输出资源（如果有）
        if out_video is not None:
            out_video.release()
        # 关闭所有OpenCV窗口
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        print("注视跟踪结束")
        return



    def calibrate(self, model, cap, sfm=False, display_index=0, filename=None):
        """
        校准方法 - 使用Pygame界面进行校准
        
        Args:
            model: 视线估计模型
            cap: 摄像头捕获对象
            sfm: 是否使用SfM（结构光）
            display_index: pygame显示器索引，指定在哪个显示器上显示校准界面
            filename: 指定校准结果文件名（可选）
            
        Returns:
            STransG: 校准变换矩阵，如果取消则返回None
        """
            
        print(f"使用Pygame校准界面 (显示器索引: {display_index})...")
        
        # 初始化Pygame（如果尚未初始化）
        if not pygame.get_init():
            pygame.init()
        
        # 创建Pygame校准界面，传入指定的显示器索引
        calib_ui = PygameCalibrationUI(self.width, self.height, display_index=display_index)
        calib_ui.initialize()
        
        # 获取摄像头尺寸
        if cap is not None:
            # 测试摄像头并获取实际帧尺寸
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                # 使用实际帧尺寸，而不是cap.get()返回的理论值
                self.WC_height, self.WC_width = test_frame.shape[:2]
            else:
                # 如果无法读取帧，使用cap.get()作为备选
                self.WC_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.WC_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # 移除开始界面和点击等待，直接开始校准流程
        print("校准开始...")
        
        warmup_frames = 30
        stable_count = 0
        required_stable_frames = 10
        WTransG1 = np.zeros((4, 4))
        
        for i in range(warmup_frames):
            try:
                ret, frame_cam = cap.read()
                if ret:
                    # 测试模型输出
                    eye_info, landmarks, pnp= model.get_gaze(frame=frame_cam, imshow=False)
                    if eye_info is not None:
                        stable_count += 1
                    else:
                        stable_count = 0
                    
                    # 如果达到要求的稳定帧数，提前结束预热
                    if stable_count >= required_stable_frames:
                        break
                    
                    # 显示预热进度
                    calib_ui.show_warmup_screen(i + 1, warmup_frames, stable_count, required_stable_frames)
                    
                    # 处理事件
                    if calib_ui.handle_events() == 'quit':
                        calib_ui.cleanup()
                        return None
                    
                    calib_ui.clock.tick(30)
            except Exception as e:
                print(f"预热过程中出错: {e}")
        
        # 显示预校准提示界面，等待用户按空格键开始
        calib_ui.show_precalibration_screen()
        
        # 等待用户按键开始校准
        waiting_for_start = True
        while waiting_for_start:
            # 处理事件
            event = calib_ui.handle_events()
            if event == 'quit':
                calib_ui.cleanup()
                return None
            elif event == 'start':  # 空格键或S键
                waiting_for_start = False
                break
            
            calib_ui.clock.tick(30)
        
        print("用户确认开始校准，正式校准流程开始")
        
        # 创建校准目标管理器
        calib_targets = PygameCalibrationTargets(self.width, self.height)
        calib_targets.start_timing()
        
        # 初始化pnp_tvec和pnp_rotations收集列表，用于存储校准阶段的所有pnp数据
        pnp_tvec_list = []
        pnp_rotations_list = []
        
        # 校准主循环
        valid_frames = 0
        calib_time_per_point = 2.2  # 每个校准点停留时间
        
        while cap.isOpened():
            # 获取当前校准点
            idx, SetPos = calib_targets.getTargetCalibration(calib_time_per_point)
            if idx is None:
                print("校准完成2！")
                break
            
            # 显示校准点
            calib_ui.show_calibration_point(idx)
            
            try:
                ret, frame_cam = cap.read()
                if not ret:
                    print("视频流结束")
                    break
                
                # SfM处理
                # 获取视线信息
                eye_info, landmarks, pnp = model.get_gaze(frame=frame_cam, imshow=False)
                # print(f"当前帧检测到的眼睛信息: {eye_info}，关键点形状: {landmarks.shape if landmarks is not None else 'None'}")
                if eye_info is None:
                    print("当前帧未检测到眼睛信息，跳过...")
                    # 处理事件
                    if calib_ui.handle_events() == 'quit':
                        break
                    calib_ui.clock.tick(60)
                    continue
                
                
                
                # 增加有效帧计数
                valid_frames += 1
                
                # 收集pnp数据用于后续计算calibrate_pnp
                if pnp is not None:
                    # 收集pnp_tvec
                    if 'pnp_tvec' in pnp:
                        pnp_tvec = pnp['pnp_tvec']
                        # 确保pnp_tvec是有效的numpy数组且形状为(3,1)
                        if isinstance(pnp_tvec, np.ndarray) and pnp_tvec.shape == (3, 1):
                            pnp_tvec_list.append(pnp_tvec)
                        elif pnp_tvec is not None:
                            # 如果不是(3,1)形状，尝试转换为标准格式
                            try:
                                pnp_tvec_arr = np.array(pnp_tvec)
                                if pnp_tvec_arr.shape == (3,):
                                    pnp_tvec_list.append(pnp_tvec_arr.reshape(3, 1))
                                elif pnp_tvec_arr.ndim == 1 and len(pnp_tvec_arr) == 3:
                                    pnp_tvec_list.append(pnp_tvec_arr.reshape(3, 1))
                            except:
                                pass
                    
                    # 收集pnp_rotations
                    if 'pnp_R' in pnp:
                        pnp_R = pnp['pnp_R']
                        # 确保pnp_R是有效的numpy数组且形状为(3,3)
                        if isinstance(pnp_R, np.ndarray) and pnp_R.shape == (3, 3):
                            pnp_rotations_list.append(pnp_R)
                        elif pnp_R is not None:
                            # 如果不是(3,3)形状，尝试转换为标准格式
                            try:
                                pnp_R_arr = np.array(pnp_R)
                                if pnp_R_arr.shape == (3, 3):
                                    pnp_rotations_list.append(pnp_R_arr)
                            except:
                                pass
                
                # 只有在can_record为True时才记录校准数据（确保用户有足够时间注视目标）
                if hasattr(calib_targets, 'can_record') and calib_targets.can_record:
                    # 处理eye_info数据
                    arr = np.array([])
                    if eye_info is not None:
                        for i in pd.Series(eye_info).values:
                            arr = np.hstack((arr, i))
                    else:
                        arr = np.zeros(19)
                    
                    # 提取pnp_tvec和pnp_R
                    pnp_arr = np.array([])
                    if pnp is not None:
                        pnp_tvec = pnp.get('pnp_tvec')
                        pnp_R = pnp.get('pnp_R')
                        if pnp_tvec is not None and isinstance(pnp_tvec, np.ndarray):
                            pnp_arr = np.hstack((pnp_arr, pnp_tvec.flatten()))
                        else:
                            pnp_arr = np.hstack((pnp_arr, np.zeros(3)))
                        if pnp_R is not None and isinstance(pnp_R, np.ndarray):
                            pnp_arr = np.hstack((pnp_arr, pnp_R.flatten()))
                        else:
                            pnp_arr = np.hstack((pnp_arr, np.zeros(9)))
                    else:
                        pnp_arr = np.hstack((pnp_arr, np.zeros(3), np.zeros(9)))
                    
                    timestamp = time.time_ns() / 1000000000
                    SetPos_mm = self._pixel2mm(SetPos)
                    self.df = pd.concat([self.df, pd.DataFrame([np.hstack((timestamp, idx, arr, SetPos_mm, 0, pnp_arr, WTransG1.flatten()))])])
                
            except Exception as e:
                pass
            
            # 处理事件
            if calib_ui.handle_events() == 'quit':
                print("用户中断校准")
                break
            
            calib_ui.clock.tick(60)
        
        # 清理Pygame资源
        calib_ui.cleanup()
        
        # 计算calibrate_pnp：校准阶段收集到的所有pnp_tvec的均值
        if len(pnp_tvec_list) > 0:
            calibrate_pnp = np.mean(pnp_tvec_list, axis=0)
            print(f"calibrate_pnp (pnp_tvec均值) 计算完成: {calibrate_pnp}")
            print(f"收集到的pnp_tvec样本数: {len(pnp_tvec_list)}")
        else:
            calibrate_pnp = None
            print("警告: 没有收集到有效的pnp_tvec数据，calibrate_pnp未定义")
        if len(pnp_rotations_list) > 0:
            print(f"收集到的pnp_R样本数: {len(pnp_rotations_list)}")
        
        print("开始处理校准数据:")
        # 保存校准数据
        self.df.columns = ['Timestamp', 'idx', 'gaze_x', 'gaze_y', 'gaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll', 'HeadBox_xmin', 'HeadBox_ymin', 'RightEyeBox_xmin', 'RightEyeBox_ymin', 'LeftEyeBox_xmin', 'LeftEyeBox_ymin', 'ROpenClose', 'LOpenClose', 'set_x', 'set_y', 'set_z', 'pnp_tvec_x', 'pnp_tvec_y', 'pnp_tvec_z', 'pnp_R_00', 'pnp_R_01', 'pnp_R_02', 'pnp_R_10', 'pnp_R_11', 'pnp_R_12', 'pnp_R_20', 'pnp_R_21', 'pnp_R_22'] + 16 * ['WTransG']
        print("self.df:", self.df)
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.dir, "results", "Calibration.csv"))
        
        # 处理数据并计算变换矩阵
        gaze, SetVal, WTransG, g, pnp_tvec_filtered, pnp_R_filtered = self._RemoveOutliers()
        
        # 尝试使用 fit_STW_with_pnp 计算 StW
        st_w_from_pnp = None
        
        if not pnp_tvec_filtered.empty and not pnp_R_filtered.empty:
            try:
                # 准备数据格式
                # calib_points_mm: 屏幕坐标 (N, 2)，单位 mm
                calib_points_mm = SetVal.to_numpy()[:, :2]
                
                # gaze_vectors: 3D 注视单位向量 (N, 3)
                gaze_vectors = gaze.to_numpy()
                
                # pnp_rotations: 头部旋转矩阵 (N, 3, 3)
                pnp_cols_R = [f'pnp_R_{i}{j}' for i in range(3) for j in range(3)]
                pnp_rotations = pnp_R_filtered[pnp_cols_R].to_numpy().reshape(-1, 3, 3)
                
                # pnp_translations: 头部平移向量 (N, 3)，单位 mm
                pnp_translations = pnp_tvec_filtered.to_numpy()
                
                # 调试输出：各输入数据维度
                print(f"[fit_STW_with_pnp 调试] calib_points_mm shape: {calib_points_mm.shape}")
                print(f"[fit_STW_with_pnp 调试] gaze_vectors shape: {gaze_vectors.shape}")
                print(f"[fit_STW_with_pnp 调试] pnp_rotations shape: {pnp_rotations.shape}")
                print(f"[fit_STW_with_pnp 调试] pnp_translations shape: {pnp_translations.shape}")
                
                # 检查数据维度是否匹配
                if (len(calib_points_mm) == len(gaze_vectors) == 
                    len(pnp_rotations) == len(pnp_translations)):
                    
                    print(f"正在使用 fit_STW_with_pnp 计算 StW...")
                    print(f"  有效样本数: {len(calib_points_mm)}")
                    
                    st_w_from_pnp = self.fit_STW_with_pnp(calib_points_mm, gaze_vectors, pnp_rotations, pnp_translations)
                    print("fit_STW_with_pnp 计算完成!")
                    print(f"  StW 计算结果:\n{st_w_from_pnp}")
                else:
                    print(f"数据维度不匹配: calib={len(calib_points_mm)}, gaze={len(gaze_vectors)}, R={len(pnp_rotations)}, t={len(pnp_translations)}")
            except Exception as e:
                print(f"fit_STW_with_pnp 调用失败: {e}")
                import traceback
                traceback.print_exc()
        
        if sfm:
            STransW, scaleWtG, STransG = self._fitSTransG_sfm(gaze, SetVal, WTransG, g)
        else:
            STransG = self._fitSTransG(gaze, SetVal, g)
        
        # 如果从 fit_STW_with_pnp 获取了 StW，可以存储供后续使用
        if st_w_from_pnp is not None:
            self.StW_from_pnp = st_w_from_pnp
        
        # 绘制结果
        # Sg, SgCalib = self._getCalibValuesOnScreen(g, STransG)
        # self._PlotGaze2D(g, Sg, SgCalib, name="GazeOnScreen")
        # self._WriteStatsInFile(STransG)
        
        # 保存校准结果
        pnp_tvec_count = len(pnp_tvec_list) if len(pnp_tvec_list) > 0 else 0
        self._save_calibration_results(STransG, g, SetVal, gaze, sfm, STransW if sfm else None, scaleWtG if sfm else None, calibrate_pnp, pnp_tvec_count, filename)
        
        return STransG
    
    def _save_calibration_results(self, STransG, g, SetVal, gaze, sfm=False, STransW=None, scaleWtG=None, calibrate_pnp=None, pnp_tvec_count=0, filename=None):
        """
        保存完整的校准结果，包括设备信息和校准点数据（仅JSON格式）
        
        Args:
            STransG: 屏幕到世界的变换矩阵
            g: 校准注视数据
            SetVal: 校准点数据
            gaze: 注视数据
            sfm: 是否使用SfM
            STransW: SfM相关的世界坐标变换矩阵
            scaleWtG: SfM缩放因子
            calibrate_pnp: 校准阶段收集的pnp_tvec均值
            pnp_tvec_count: 收集的pnp_tvec样本数量
            filename: 指定文件名
        """
        import json
        import numpy as np
        
        def convert_to_serializable(obj):
            """将对象转换为JSON可序列化的格式"""
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                # 对于其他类型，尝试转换为字符串
                return str(obj)
        
        # 创建校准结果字典
        calibration_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'device_info': {
                'screen_width': self.width,
                'screen_height': self.height,
                'screen_width_mm': self.width_mm,
                'screen_height_mm': self.height_mm,
                'webcam_width': self.WC_width,
                'webcam_height': self.WC_height,
                'camera_matrix': convert_to_serializable(self.camera_matrix),
                'dist_coeffs': convert_to_serializable(self.dist_coeffs)
            },
            'calibration_parameters': {
                'sfm_enabled': sfm,
                'total_calibration_points': len(g),
                'calibration_time_per_point': 2.5  # 默认值，可根据实际情况调整
            },
            'transformation_matrices': {
                'STransG': convert_to_serializable(STransG),
                'StG': convert_to_serializable(self.StG) if hasattr(self, 'StG') else []
            },
            'calibration_points': {
                'SetValues': convert_to_serializable(self.SetValues) if hasattr(self, 'SetValues') else [],
                'gaze_data': convert_to_serializable(g),
                'SetVal': convert_to_serializable(SetVal),
                'gaze': convert_to_serializable(gaze)
            }
        }
        
        # 如果启用了SfM，保存相关数据
        if sfm and STransW is not None and scaleWtG is not None:
            calibration_data['sfm_data'] = {
                'STransW': STransW.tolist(),
                'scaleWtG': scaleWtG,
                'StW': [stw.tolist() for stw in self.StW] if hasattr(self, 'StW') else []
            }
        
        # 保存calibrate_pnp数据（校准阶段收集的pnp_tvec均值）
        if calibrate_pnp is not None:
            calibration_data['calibrate_pnp'] = {
                'pnp_tvec_mean': convert_to_serializable(calibrate_pnp),
                'pnp_tvec_samples_count': pnp_tvec_count
            }
        
        # 保存为JSON文件 - 支持指定文件名
        if filename:
            json_file = os.path.join(self.dir, "results", filename)
        else:
            json_file = os.path.join(self.dir, "results", "calibration_results.json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"校准结果已保存到: {json_file}")

    def load_calibration_results(self, file_path=None, sfm_enabled=True):
        """
        从JSON文件加载校准结果
        
        Args:
            file_path (str, optional): 校准文件路径，默认为标准路径
            sfm_enabled (bool): 当前是否启用SfM模式，根据此参数决定是否加载SfM相关数据
        """
        import json
        import numpy as np
        
        if file_path is None:
            file_path = os.path.join(self.dir, "results", "calibration_results.json")
        
        if not os.path.exists(file_path):
            print(f"校准文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            # 恢复校准状态
            self.STransG = np.array(calibration_data['transformation_matrices']['STransG'])
            self.StG = [np.array(stg) for stg in calibration_data['transformation_matrices']['StG']]
            self.SetValues = [np.array(setval) for setval in calibration_data['calibration_points']['SetValues']]
            
            # 恢复屏幕尺寸和物理尺寸信息
            self.width = calibration_data['device_info']['screen_width']
            self.height = calibration_data['device_info']['screen_height']
            
            # 恢复物理尺寸（如果存在）
            if 'screen_width_mm' in calibration_data['device_info'] and 'screen_height_mm' in calibration_data['device_info']:
                self.width_mm = calibration_data['device_info']['screen_width_mm']
                self.height_mm = calibration_data['device_info']['screen_height_mm']
            else:
                # 如果校准数据中没有物理尺寸，使用默认值
                self.width_mm = 521
                self.height_mm = 293
            
            # 根据当前SfM启用状态决定是否加载SfM相关数据
            if sfm_enabled and 'sfm_data' in calibration_data:
                self.STransW = np.array(calibration_data['sfm_data']['STransW'])
                self.scaleWtG = calibration_data['sfm_data']['scaleWtG']
                self.StW = [np.array(stw) for stw in calibration_data['sfm_data']['StW']]
                sfm_loaded = True
            elif sfm_enabled and 'sfm_data' not in calibration_data:
                print("警告: 当前启用SfM，但校准文件中没有SfM数据")
                sfm_loaded = False
            else:
                # 当前不启用SfM，不加载SfM相关数据
                sfm_loaded = False
            
            # 加载calibrate_pnp数据（校准阶段收集的pnp_tvec均值）
            if 'calibrate_pnp' in calibration_data:
                self.calibrate_pnp = np.array(calibration_data['calibrate_pnp']['pnp_tvec_mean'])
                print(f"calibrate_pnp 已加载: {self.calibrate_pnp}")
                print(f"样本数量: {calibration_data['calibrate_pnp']['pnp_tvec_samples_count']}")
            else:
                self.calibrate_pnp = None
                print("校准文件中没有 calibrate_pnp 数据")
            
            print(f"校准结果已从 {file_path} 加载")
            print(f"屏幕尺寸: {self.width}x{self.height}")
            print(f"物理尺寸: {self.width_mm}mmx{self.height_mm}mm")
            print(f"校准点数: {calibration_data['calibration_parameters']['total_calibration_points']}")
            print(f"当前SfM启用: {sfm_enabled}, SfM数据加载: {sfm_loaded}")
            print(f"校准文件SfM状态: {calibration_data['calibration_parameters']['sfm_enabled']}")
            
            return True
            
        except Exception as e:
            print(f"加载校准结果失败: {e}")
            return False


    def _getGazeOnScreen(self, gaze, fac=None):
        Osg=self.STransG.copy()
        if fac is not None:
            self.STransG[:3,3] = self.STransG[:3,3] + fac

        scaleGaze = self._getScale(gaze, self.STransG)
        Sgaze = (self.STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """
        for i in range(len(self.StG)):
            STransG_ = np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1])))
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_                                 

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)

        """
        FSgaze = fused gaze vector, overall and for each calibration point
        Sgaze = overall gaze vector, determined over regression in screen coordinate system
        Sgaze2 = gaze vector from calibration point
        """
        self.STransG = Osg.copy()
        return FSgaze, Sgaze, Sgaze2
    

    def _getGazeOnScreen_sfm(self, gaze, WTransG):
        WTransG[:3,3] = self.scaleWtG*WTransG[:3,3]
        STransG = self.STransW @ WTransG
        scaleGaze = self._getScale(gaze, STransG)
        Sgaze = (STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """
        for i in range(len(self.StW)):
            STransG_ = np.vstack((np.hstack((SRotW, self.StW[i].reshape(3,1))), np.array([0,0,0,1]))) @ WTransG
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)
        """
        FSgaze = 融合后的注视向量，整体及各校准点均使用
        Sgaze = 在屏幕坐标系下通过回归得到的整体注视向量，已考虑头部运动
        Sgaze2 = 考虑头部运动后，从校准点得到的注视向量
        """
        return FSgaze, Sgaze, Sgaze2
      
    def fit_STW_with_pnp(self, calib_points_mm, gaze_vectors, pnp_rotations, pnp_translations):
        """
        基于 PnP 物理尺度重写的 STW 矩阵回归函数 (即改进版的 _fitSTransG)
        
        参数:
        calib_points_mm: 4个校准点的屏幕坐标 (N, 2)，单位 mm 
        gaze_vectors: 模型输出的 3D 注视单位向量 (N, 3) 
        pnp_rotations: PnP 计算得到的头部实时旋转矩阵 (N, 3, 3) 
        pnp_translations: PnP 计算得到的头部物理平移向量 (N, 3)，单位 mm 
        """
        
        def objective_function(x):
            # x = [pitch, yaw, roll, tx, ty, tz]
            # 前三个为相机相对于屏幕的旋转角度 (SRW)，后三个为平移偏置 (StW) [3, 9]
            euler_angles = x[:3]
            st_w = x[3:].reshape((3, 1))
            
            # 构造相机到屏幕的旋转矩阵 SRW
            sr_w = R.from_euler('xyz', euler_angles).as_matrix()
            
            residuals = []
            for i in range(len(calib_points_mm)):
                # 获取 PnP 提供的当前帧头部数据 (W 坐标系下)
                w_rg = pnp_rotations[i]
                w_tg = pnp_translations[i].reshape((3, 1))
                g_hat = gaze_vectors[i].reshape((3, 1))
                
                # 核心公式: Sg = SRW * (WRG * lambda * g_hat + WtG) + StW 
                # 计算 lambda (注视线与屏幕 Z=0 平面的交点) 
                # 为了使 Sg_z = 0: 
                # sr_w[2,:] @ (w_rg @ (lambda * g_hat) + w_tg) + st_w = 0
                
                direction_w = w_rg @ g_hat
                numerator = -(sr_w[2, :] @ w_tg + st_w[2])
                denominator = sr_w[2, :] @ direction_w
                
                if abs(denominator) < 1e-6:
                    lambda_i = 1000.0  # 避免除零
                else:
                    lambda_i = numerator / denominator
                
                # 计算预测的屏幕坐标 Sgi [35, Eq.25]
                p_head_w = direction_w * lambda_i + w_tg
                s_gi = sr_w @ p_head_w + st_w
                
                # 计算 x, y 方向的像素残差 (单位 mm) [26, Eq.9]
                residuals.extend((s_gi[:2, 0] - calib_points_mm[i]).tolist())
                
            return np.array(residuals)

        # 初始猜测: 假设相机镜像对齐，挂在屏幕上方 60cm 处 [11, 13]
        # x0 = [pitch (180度), yaw, roll, tx (屏幕中心), ty, tz (深度)]
        x0 = [np.pi, 0, 0, 300, -20, -600] 
        
        # 执行非线性最小二乘优化 [1, 2]
        res = opt.least_squares(objective_function, x0, method='lm')
        
        # 封装为 4x4 齐次变换矩阵 STW [21, Eq.1]
        final_stw = np.eye(4)
        final_stw[:3, :3] = R.from_euler('xyz', res.x[:3]).as_matrix()
        final_stw[:3, 3] = res.x[3:]
        
        return final_stw

    def _fitSTransG(self, gaze, SetVal, g):
        """
        在无SfM（Structure-from-Motion）的情况下，根据校准阶段采集的注视向量与对应屏幕目标点，
        拟合从“相机坐标系”到“屏幕坐标系”的刚体变换矩阵（旋转+平移），并进一步为每个校准点计算
        局部修正的平移向量，用于后续 gaze 映射。

        步骤概览：
        1. 数据准备：将输入的 gaze、SetVal 转为 numpy，并增加维度以便广播。
        2. 定义误差函数：以屏幕平面 Z=0 为约束，建立 gaze 射线与屏幕交点同目标点的残差。
        3. 非线性最小二乘优化：求解最优平移向量 StG = [sx, sy, sz]。
        4. 构造全局变换矩阵 STransG（旋转 SRotG + 平移 StG）。
        5. 为每个校准点计算局部 scale 并生成对应的“辅助变换矩阵”，
           将局部平移向量保存在 self.StG[i] 中，用于后续 gaze 映射时的精细修正。
        6. 将全局变换矩阵保存到实例变量并返回。
        """

        # 1. 数据准备：将 DataFrame 转为 numpy 数组，并增加一维用于后续广播 (n,3)->(n,3,1)
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy()
        # 屏幕坐标系→相机坐标系的旋转：X、Y 反向，Z 同向
        SRotG = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])
        gaze = gaze[:, :, None]          # shape: (N,3,1)

        # 2. 定义残差函数：以屏幕平面 Z=0 为约束，计算 gaze 射线与屏幕交点到目标点的误差
        def alignError(x, *const):
            """
            x: 待优化的平移向量 StG = [sx, sy, sz]
            const: (SRotG, gaze, SetVal)
            返回 flatten 后的误差向量，供 least_squares 使用。
            """
            SRotG, gaze, SetVal = const
            StG = np.array([[x[0]], [x[1]], [x[2]]])  # 3×1
            Gz  = np.array([[0], [0], [1]])            # 屏幕法向量
            # 计算 gaze 射线与平面 Z=0 的交点比例因子 mu
            # mu = (平面原点到达平面的距离) / (gaze 在平面法向的投影)
            mu = (Gz.T @ (-SRotG.T @ StG)) / (Gz.T @ gaze)  # shape: (N,1,1)
            # 将 gaze 映射到屏幕坐标系
            Sg = SRotG @ (mu * gaze) + StG                # shape: (N,3,1)
            # 计算与真实目标点的残差
            error = SetVal[:, :, None] - Sg               # shape: (N,3,1)
            return error.flatten()                          # flatten 成 (N*3,)

        # 3. 非线性最小二乘优化：初值设为屏幕中心 + 一个经验深度
        const = (SRotG, gaze, SetVal)
        x0 = np.array([self.width / 2, self.height / 2, self.width])  # [sx, sy, sz]
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")

        # 4. 构造全局变换矩阵 STransG（4×4 齐次形式）
        StG = np.array([[xopt[0]], [xopt[1]], [xopt[2]]])
        STransG = np.r_[np.c_[SRotG, StG], np.array([[0, 0, 0, 1]])]

        # 5. 为每个校准点计算局部修正的平移向量
        size = len(g)                # 校准点个数
        self.StG = [None] * size
        for i in range(size):
            # 用该点 gaze 的中位数计算 scale，使 gaze 射线与屏幕相交
            scaleGaze = self._getScale(np.median(g[i], axis=0), STransG)
            # 生成该校准点对应的“辅助变换矩阵”
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i],
                                                   np.median(g[i], axis=0), scaleGaze)
            # 仅保存平移部分，供后续映射时局部修正
            self.StG[i] = STransG_[:3, 3, None]

        # 6. 保存到实例变量并返回
        self.STransG = STransG
        return STransG
    
    def _fitSTransG_sfm(self, gaze, SetVal, WTransG, g):
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy() 
        WTransG = WTransG.to_numpy().reshape(-1,4,4)

        WRotG = WTransG[:,:3,:3]
        WtG = WTransG[:,:3,3]
        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])

        gaze = gaze[:,:,None]

        """ Model over camera coordinate system getting gaze from SFM  """
        def alignError(x, *const):
            SRotW, WRotG, gaze, WtG, SetVal = const
            StW = np.array([[x[1]],[x[2]],[0]])
            SRotG = SRotW @ WRotG
            Gz = np.array([[0],[0],[1]])
            mu = (Gz.T @ (-np.transpose(SRotG, axes=(0,2,1)) @ (SRotW @ (x[0]*WtG[:,:,None]) + StW)))/(Gz.T @ gaze)
            Sg = SRotG @ (mu*gaze) + SRotW @  (x[0]*WtG[:,:,None]) + StW
            error = SetVal[:,:,None] - Sg   # (87x3x1)
            return error.flatten()

        const = (SRotW, WRotG, gaze, WtG, SetVal)
        x0 = np.array([1, self.width/2, self.height/2])
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")
        StW = np.array([[xopt[1]],[xopt[2]],[0]])
        self.STransW = np.r_[np.c_[SRotW, StW], np.array([[0,0,0,1]])]
        WTransG = np.concatenate((np.c_[WRotG, xopt[0]*WtG[:,:,None]], np.tile(np.array([[0, 0, 0, 1]]), (WtG.shape[0], 1, 1))), axis=1)
        STransG = self.STransW @ np.median(WTransG, axis=0)
        self.scaleWtG = xopt[0]

        WtG = np.median(WtG[:,:,None], axis=0)

        """ Transformation Matrix to Auxiliary points """
        size = len(g)
        self.StW = [None]*size
        self.StG = [None]*size
        for i in range(size):
            scaleGaze = self._getScale(np.median(g[i],axis=0), STransG)     # compute scale for gaze vector for each calibration point
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i], np.median(g[i],axis=0), scaleGaze)
            self.StG[i] = STransG_[:3,3,None]
            self.StW[i] = STransG_[:3,3,None] - SRotW @ (self.scaleWtG*WtG)

        self.STransG = STransG

        return self.STransW, self.scaleWtG, STransG
        
    def _getCalibValuesOnScreen(self, g, STransG):
        Sg = [None]*len(g)
        SgCalib = [None]*len(g)
        # SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        SRotG = STransG[:3,:3]
        for i in range(len(g)):
            gaze = g[i].to_numpy()
            scaleGaze = self._getScale(gaze, STransG)
            Sg[i] = (STransG @ np.concatenate(( (scaleGaze*gaze[:,:,None]), np.ones((gaze.shape[0],1,1))), axis=1))[:,:3,:]
            STransG_ = np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1])))
            scaleGaze = self._getScale(gaze, STransG_)
            SgCalib[i] = (STransG_ @ np.concatenate(( (scaleGaze*gaze[:,:,None]), np.ones((gaze.shape[0],1,1))), axis=1))[:,:3,:]

        return Sg, SgCalib

    def _getSTransG(self, SRotG, SposA, gazeVector, scaleGaze):
        STransA = np.vstack((np.hstack((np.eye(3), SposA)), np.array([0,0,0,1])))      
        ATransG = np.vstack((np.hstack((SRotG, -SRotG.T @ (scaleGaze*gazeVector[:,None]))), np.array([0,0,0,1])))
        STransG = STransA @ ATransG
        GTransS = np.vstack((np.hstack((STransG[0:3,0:3].T, -STransG[0:3,0:3].T @ STransG[0:3,3].reshape(3,1))), np.array([0,0,0,1])))

        return STransG, GTransS


    
    def _getScale(self, gaze, STransG):
        Gz = np.array([[0],[0],[1]])
        GTransS = util.invHomMatrix(STransG)
        GtS = GTransS[:3,3].reshape(3,1)
        if np.ndim(gaze) == 1:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,None])
        elif np.ndim(gaze) == 2:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,:,None])

        return scaleGaze

    def _ProjectVetorOnPlane(self, Trans, vector):
        """ Translation of homogenous Trans-Matrix must be in same coordinate system as Vector """
        vector = vector.reshape(3,1)
        # VectorNormal2Plane = (Trans @ np.array([[0],[0],[1],[1]]))[0:3]
        VectorNormal2Plane = (Trans[:3,:3] @ np.array([[0],[0],[1]]))
        # Gz = self.GTransB[0:3,2].reshape(3,1) # not sure why this would work for Tobii (was implemented before)
        transVec = Trans[:3,3]
        t = (VectorNormal2Plane.T @ transVec) / (VectorNormal2Plane.T @ vector)
        Vector2Plane = np.vstack((t*vector, 1))
        return Vector2Plane

    def _RemoveOutliers(self):
        """
        根据校准阶段采集的原始数据，按校准点索引分组并去除异常值。
        
        步骤：
        1. 获取最大校准点索引，确定需要处理的组数。
        2. 对每一组数据，分别提取 gaze 向量、设定点坐标、世界变换矩阵及PnP数据。
        3. 对 gaze 向量的 x、y、z 三个维度均调用 _MaskOutliers 进行异常值检测，
           只有三个维度都通过检测的样本才被保留，以提高校准精度。
        4. 对 pnp_tvec 和 pnp_R 也进行异常值检测。
        5. 将过滤后的 gaze、设定点、变换矩阵及 pnp 数据分别存入列表。
        6. 将各组合并，返回统一的 DataFrame 及分组列表，供后续拟合使用。
        
        返回:
            gaze:      过滤后的 gaze 向量 DataFrame（所有组合并）
            SetVal:    过滤后的设定点坐标 DataFrame（所有组合并）
            W_T_G:     过滤后的世界变换矩阵 DataFrame（所有组合并）
            g:         按组存放的 gaze 列表，每个元素为对应组的 DataFrame
            pnp_tvec_filtered: 过滤后的 pnp_tvec DataFrame（所有组合并）
            pnp_R_filtered:    过滤后的 pnp_R DataFrame（所有组合并）
        """
        # 计算总校准点数量（索引从 0 开始，因此最大索引+1）
        idx = int(pd.unique(self.df['idx'])[-1]) + 1  # 若考虑头部转动可改为 -3
        
        # 初始化用于存放各组数据的列表
        g   = [None] * idx   # 存放 gaze 向量
        s   = [None] * idx   # 存放设定点坐标
        WTG = [None] * idx   # 存放世界变换矩阵
        pnp_tvec = [None] * idx   # 存放 pnp_tvec
        pnp_R = [None] * idx      # 存放 pnp_R
        
        # 按校准点索引分组处理
        for i in range(idx):
            # 提取当前组的所有 gaze 向量（x, y, z）
            g_ = self.df[self.df['idx'].values == i].loc[:, 'gaze_x':'gaze_z']
            # 提取当前组的设定点坐标（set_x, set_y, set_z）
            set_val = self.df[self.df['idx'].values == i].loc[:, 'set_x':'set_z']
            # 提取当前组的世界变换矩阵（列名包含 'WTransG'）
            WTG_ = self.df[self.df['idx'].values == i].filter(like='WTransG')
            # 提取当前组的 pnp_tvec（pnp_tvec_x, pnp_tvec_y, pnp_tvec_z）
            pnp_tvec_ = self.df[self.df['idx'].values == i].loc[:, 'pnp_tvec_x':'pnp_tvec_z']
            # 提取当前组的 pnp_R（pnp_R_00 到 pnp_R_22）
            pnp_R_ = self.df[self.df['idx'].values == i].filter(like='pnp_R_')
            
            # 对 gaze 的三个维度分别做异常值检测，取交集保留同时通过检测的样本
            mask = (
                self._MaskOutliers(g_.loc[:, 'gaze_x']) &
                self._MaskOutliers(g_.loc[:, 'gaze_y']) &
                self._MaskOutliers(g_.loc[:, 'gaze_z'])
            )
            
            # 对 pnp_tvec 的三个维度做异常值检测
            if not pnp_tvec_.empty and len(pnp_tvec_) > 0:
                pnp_tvec_mask = (
                    self._MaskOutliers(pnp_tvec_.loc[:, 'pnp_tvec_x']) &
                    self._MaskOutliers(pnp_tvec_.loc[:, 'pnp_tvec_y']) &
                    self._MaskOutliers(pnp_tvec_.loc[:, 'pnp_tvec_z'])
                )
                mask = mask & pnp_tvec_mask
            
            # 应用掩码，保存过滤后的数据
            g[i]   = g_[mask]
            s[i]   = set_val[mask]
            WTG[i] = WTG_[mask]
            pnp_tvec[i] = pnp_tvec_[mask] if not pnp_tvec_.empty else pd.DataFrame()
            pnp_R[i] = pnp_R_[mask] if not pnp_R_.empty else pd.DataFrame()
        
        # 将设定点转换为 numpy 数组并存入实例变量，供后续绘图及评估使用
        self.SetValues = [v.to_numpy()[0][:, None] for v in s]
        
        # 将各组数据合并为整体 DataFrame，便于后续一次性处理
        gaze   = pd.concat(g,   axis=0)
        SetVal = pd.concat(s,   axis=0)
        W_T_G  = pd.concat(WTG, axis=0)
        pnp_tvec_filtered = pd.concat(pnp_tvec, axis=0) if all(v is not None and not v.empty for v in pnp_tvec) else pd.DataFrame()
        pnp_R_filtered = pd.concat(pnp_R, axis=0) if all(v is not None and not v.empty for v in pnp_R) else pd.DataFrame()
        
        # 存储过滤后的 pnp 数据到实例变量
        self.pnp_tvec_filtered = pnp_tvec_filtered
        self.pnp_R_filtered = pnp_R_filtered
        
        return gaze, SetVal, W_T_G, g, pnp_tvec_filtered, pnp_R_filtered

    def _MaskOutliers(self, arr, std_threshold=0.8):
        """
        Removes outliers from a NumPy array using the standard deviation method.
        Parameters:
            arr (numpy.ndarray): The input array.
            std_threshold (float): The number of standard deviations from the mean to use as the threshold for outlier detection.
                                  A smaller value (0.8) provides stricter filtering to improve calibration accuracy.
        Returns:
            numpy.ndarray: The mask to remove outliers.
        """
        # 增加稳健性处理
        if len(arr) < 3:
            return np.ones_like(arr, dtype=bool)  # 如果数据点太少，不过滤
            
        # 使用中位数和绝对中位差(MAD)代替均值和标准差，提高对离群值的鲁棒性
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        
        # 如果MAD为0，回退到标准差方法
        if mad == 0:
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:  # 避免除以零
                return np.ones_like(arr, dtype=bool)
            threshold = std_threshold * std
            mask = np.abs(arr - mean) < threshold
        else:
            # 使用MAD方法
            threshold = std_threshold * mad * 1.4826  # 1.4826是将MAD转换为标准差的常数
            mask = np.abs(arr - median) < threshold
            
        return mask

    def _MaskOutliersPercentile(self, array):
        q75,q25 = np.percentile(array,[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        return (array > min) & (array < max)

    def _WriteStatsInFile(self, STransG):
        """ Write stats in file """
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        with open(os.path.join(self.dir, "results", 'stats.txt'), 'w') as f:
            f.write(f"Transformation matrices: \n")
            # 写入所有6个校准点的转换矩阵
            for i in range(len(self.StG)):
                if self.StG[i] is not None:
                    f.write(f"STransG{i+1}\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.3f}'})}\n")
            
            # 添加校准精度统计信息
            f.write(f"\nCalibration Statistics: \n")
            f.write(f"Number of calibration points: {len(self.StG)}\n")
            
            # 添加校准点数量和总帧数信息
            f.write(f"\nScreen Information: \n")
            f.write(f"Width: {self.width}px, {self.width_mm}mm\n")
            f.write(f"Height: {self.height}px, {self.height_mm}mm\n")
            f.write(f"Webcam Information: \n")
            f.write(f"Width: {self.WC_width}px\n")
            f.write(f"Height: {self.WC_height}px\n")

    def _getARotG(self, p_origin, p_xCoord, p_yCoord):
        """ Rotation Matrix """
        GxA = p_xCoord - p_origin
        GxA = GxA/np.linalg.norm(GxA)
        GyA = p_yCoord - p_origin
        GyA = GyA/np.linalg.norm(GyA)
        GzA = self._cross(GxA, GyA)
        GRotA = np.hstack((GxA.reshape(3,1), GyA.reshape(3,1), GzA.reshape(3,1)))
        ARotG = GRotA.transpose()

        return ARotG

    def _mm2pixel(self, vector_mm):
        vector = vector_mm.copy()
        if vector.ndim == 1 and vector.shape[0] == 2:
            # 处理1维2元素向量（x, y）
            vector[0] = int(vector[0] * self.width/self.width_mm)
            vector[1] = int(vector[1] * self.height/self.height_mm)
        elif vector.ndim == 2 and vector.shape[0] == 3:
            vector[0] = int(vector[0] * self.width/self.width_mm)
            vector[1] = int(vector[1] * self.height/self.height_mm)
            vector[2] = int(vector[2])
        elif vector.ndim == 3 and vector.shape[1] == 3:
            vector[:,0] = (vector[:,0] * self.width/self.width_mm).astype(int)
            vector[:,1] = (vector[:,1] * self.height/self.height_mm).astype(int)
            vector[:,2] = (vector[:,2]).astype(int)
        else:
            raise Exception(f"Vector has wrong shape: {vector.shape}, ndim: {vector.ndim}")

        return vector

    def _pixel2mm(self, vector_px):
        if isinstance(vector_px, list):
            vector_px = np.array(vector_px)
        vector = vector_px.copy()
        if vector.ndim == 1 and vector.shape[0] == 2:
            vector[0] = vector[0] * self.width_mm/self.width
            vector[1] = vector[1] * self.height_mm/self.height
        elif vector.ndim == 2 and vector.shape[1] == 2:
            vector[:,0] = vector[:,0] * self.width_mm/self.width
            vector[:,1] = vector[:,1] * self.height_mm/self.height
        else:
            raise Exception("Vector has wrong shape")

        return vector

    def _PlotGaze2D(self, g, Sg, SgCalib, name="GazeOnScreen"):

        # Sg1 = self._mm2pixel(Sg1)
        # Sg2 = self._mm2pixel(Sg2)
        # Sg3 = self._mm2pixel(Sg3)
        # Sg4 = self._mm2pixel(Sg4)
        # SetBp1 = self._mm2pixel(self.SetValues[0])
        # SetBp2 = self._mm2pixel(self.SetValues[1])
        # SetBp3 = self._mm2pixel(self.SetValues[2])
        # SetBp4 = self._mm2pixel(self.SetValues[3])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

        legend = [None]*len(g)
        for i in range(len(g)):
            """ Axis 0: Raw gaze points """
            gaze = g[i].to_numpy()
            ax[0].scatter(gaze[:,0],gaze[:,1])            
            legend[i] = f"p{i+1} values"
            """ Axis 1: Gaze on screen """
            ax[1].scatter(Sg[i][:,0],Sg[i][:,1])

        for i in range(len(g)):
            gaze = g[i].to_numpy()
            ax[0].plot(np.median(gaze[:,0]),np.median(gaze[:,1]),'r+', linewidth=4,  markersize=12)
            ax[1].plot(np.median(Sg[i][:,0]),np.median(Sg[i][:,1]),'r+', linewidth=4,  markersize=12)
            # ax[1].plot(np.median(SgCalib[i][:,0]),np.median(SgCalib[i][:,1]),'k+', linewidth=4,  markersize=12)
            ax[1].plot(self.SetValues[i][0],self.SetValues[i][1],'y*', linewidth=4, markersize=12)


        # ax[0].legend(legend+["Median gaze point"])
        ax[0].set_title('x-y-corrdinates of raw unit gaze points')
        ax[0].set_xlabel("x-direction (unit length)")
        ax[0].set_ylabel("y-direction (unit length)")
        ax[0].grid()
        # ax[1].legend(legend+["Median gaze point", "Displayed Point"])
        ax[1].set_xlabel("x-direction (mm)")
        ax[1].set_ylabel("y-direction (mm)")
        # ax[1].set_title(f"Gaze on screen with resolution {self.width}x{self.height}")
        ax[1].set_title(f"Gaze on screen with dimensions {self.width_mm}mmx{self.height_mm}mm")
        ax[1].grid()

        plt.savefig(os.path.join(self.dir, "results", name))


if __name__ == '__main__':
    print("Noting called from main")