import scipy.optimize as opt
import cv2
import os
import keyboard
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 从calibration_pygame导入工具函数，不再使用gui_opencv
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
    def __init__(self, directory) -> None:            
        self.dir = directory
        self.width, self.height, self.width_mm, self.height_mm = getScreenSize()
        self.df = pd.DataFrame()
        self.sfm = SFM(directory)
        self.camera_matrix, self.dist_coeffs = ReadCameraCalibrationData(os.path.join(directory, "camera_data"))
        # self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        # Tkinter渲染器（仅用于追踪页面）
        # self.renderer = None
        
        # 调试计数器
        self.debug_counter = 0  # 调试计数器

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
            
            # gazeframe, SetPos = target.DrawTargetGaze(white_frame, self._mm2pixel(FSgaze))
            # gazeframe, SetPos = target.DrawRectangularTargets(white_frame, self._mm2pixel(FSgaze))
            # gazeframe, SetPos = target.DrawSingleTargets(white_frame, self._mm2pixel(FSgaze))
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
            
            # 进一步减少Tkinter窗口更新频率 - OpenCV不需要这个
            # if window_update_counter % 20 == 0:  # 每20帧更新一次
            #     try:
            #         # 简化窗口更新逻辑，减少异常处理开销
            #         if hasattr(self, 'renderer') and self.renderer:
            #             try:
            #                 # 仅尝试基本更新，不进行复杂检查
            #                 self.renderer.update_idletasks()  # 使用update_idletasks代替update，减轻CPU负担
            #             except Exception:
            #                 # 发生任何异常，直接销毁渲染器并继续
            #                 try:
            #                     self.renderer = None
            #                 except:
            #                     pass
            #     except Exception:
            #         # 静默处理所有异常
            #         pass
                
            # 检查键盘事件 - 降低检查频率
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



    def calibrate(self, model, cap, sfm=False):
        """
        校准方法 - 仅使用Pygame界面
        
        Args:
            model: 视线估计模型
            cap: 摄像头捕获对象
            sfm: 是否使用SfM（结构光）
            
        Returns:
            STransG: 校准变换矩阵，如果取消则返回None
        """
        if not PYGAME_AVAILABLE:
            print("错误：Pygame未安装，无法进行校准")
            return None
            
        return self._calibrate_pygame(model, cap, sfm)
    
    def _calibrate_pygame(self, model, cap, sfm=False):
        """Pygame版本的校准方法"""
        print("使用Pygame校准界面...")
        
        # 初始化Pygame（如果尚未初始化）
        if not pygame.get_init():
            pygame.init()
        
        # 创建Pygame校准界面
        calib_ui = PygameCalibrationUI(self.width, self.height)
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
        
        # 模型预热阶段
        print("模型预热中，请等待...")
        warmup_frames = 30
        stable_count = 0
        required_stable_frames = 10
        frame_prev = None
        WTransG1 = np.zeros((4, 4))
        
        for i in range(warmup_frames):
            try:
                ret, frame_cam = cap.read()
                if ret:
                    # 测试模型输出
                    eye_info = model.get_gaze(frame=frame_cam, imshow=False)
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
        
        print("模型预热完成，开始正式校准流程")
        
        # 创建校准目标管理器
        calib_targets = PygameCalibrationTargets(self.width, self.height)
        calib_targets.start_timing()
        
        # 校准主循环
        valid_frames = 0
        calib_time_per_point = 2.0  # 每个校准点停留时间
        
        while cap.isOpened():
            # 获取当前校准点
            idx, SetPos = calib_targets.getTargetCalibration(calib_time_per_point)
            if idx is None:
                print("校准完成！")
                break
            
            # 显示校准点
            calib_ui.show_calibration_point(idx)
            
            try:
                ret, frame_cam = cap.read()
                if not ret:
                    print("视频流结束")
                    break
                
                # SfM处理
                if frame_prev is not None and sfm:
                    WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame_cam)
                
                frame_prev = frame_cam.copy()
                
                # 获取视线信息
                eye_info = model.get_gaze(frame=frame_cam, imshow=False)
                if eye_info is None:
                    print("当前帧未检测到眼睛信息，跳过...")
                    # 处理事件
                    if calib_ui.handle_events() == 'quit':
                        break
                    calib_ui.clock.tick(60)
                    continue
                
                # 增加有效帧计数
                valid_frames += 1
                
                # 只有在can_record为True时才记录校准数据（确保用户有足够时间注视目标）
                if hasattr(calib_targets, 'can_record') and calib_targets.can_record:
                    # 处理eye_info数据
                    arr = np.array([])
                    if eye_info is not None:
                        for i in pd.Series(eye_info).values:
                            arr = np.hstack((arr, i))
                    else:
                        arr = np.zeros(19)
                    
                    timestamp = time.time_ns() / 1000000000
                    SetPos_mm = self._pixel2mm(SetPos)
                    self.df = pd.concat([self.df, pd.DataFrame([np.hstack((timestamp, idx, arr, SetPos_mm, 0, WTransG1.flatten()))])])
                
            except Exception as e:
                pass
            
            # 处理事件
            if calib_ui.handle_events() == 'quit':
                print("用户中断校准")
                break
            
            calib_ui.clock.tick(60)
        
        # 清理Pygame资源
        calib_ui.cleanup()
        
        # 保存校准数据
        self.df.columns = ['Timestamp', 'idx', 'gaze_x', 'gaze_y', 'gaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll', 'HeadBox_xmin', 'HeadBox_ymin', 'RightEyeBox_xmin', 'RightEyeBox_ymin', 'LeftEyeBox_xmin', 'LeftEyeBox_ymin', 'ROpenClose', 'LOpenClose', 'set_x', 'set_y', 'set_z'] + 16 * ['WTransG']
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.dir, "results", "Calibration.csv"))
        
        # 处理数据并计算变换矩阵
        gaze, SetVal, WTransG, g = self._RemoveOutliers()
        
        if sfm:
            STransW, scaleWtG, STransG = self._fitSTransG_sfm(gaze, SetVal, WTransG, g)
        else:
            STransG = self._fitSTransG(gaze, SetVal, g)
        
        Sg, SgCalib = self._getCalibValuesOnScreen(g, STransG)
        
        # 绘制结果
        self._PlotGaze2D(g, Sg, SgCalib, name="GazeOnScreen")
        self._WriteStatsInFile(STransG)
        
        # 保存校准结果
        self._save_calibration_results(STransG, g, SetVal, gaze, sfm, STransW if sfm else None, scaleWtG if sfm else None)
        
        return STransG
    
    def _save_calibration_results(self, STransG, g, SetVal, gaze, sfm=False, STransW=None, scaleWtG=None):
        """
        保存完整的校准结果，包括设备信息和校准点数据
        """
        import json
        import pickle
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
        
        # 保存为JSON文件
        json_file = os.path.join(self.dir, "results", "calibration_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        # 保存为pickle文件（包含完整的Python对象）
        pickle_file = os.path.join(self.dir, "results", "calibration_results.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'STransG': STransG,
                'StG': self.StG if hasattr(self, 'StG') else [],
                'SetValues': self.SetValues if hasattr(self, 'SetValues') else None,
                'g': g,
                'SetVal': SetVal,
                'gaze': gaze,
                'sfm': sfm,
                'STransW': STransW if sfm else None,
                'scaleWtG': scaleWtG if sfm else None,
                'StW': self.StW if hasattr(self, 'StW') and sfm else [],
                'device_info': calibration_data['device_info']
            }, f)
        
        print(f"校准结果已保存到: {json_file}")
        print(f"完整校准数据已保存到: {pickle_file}")

    def load_calibration_results(self, file_path=None):
        """
        从文件加载校准结果
        """
        import pickle
        
        if file_path is None:
            file_path = os.path.join(self.dir, "results", "calibration_results.pkl")
        
        if not os.path.exists(file_path):
            print(f"校准文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                calibration_data = pickle.load(f)
            
            # 恢复校准状态
            self.STransG = calibration_data['STransG']
            self.StG = calibration_data['StG']
            self.SetValues = calibration_data['SetValues']
            
            # 恢复SfM相关数据（如果存在）
            if calibration_data['sfm']:
                self.STransW = calibration_data['STransW']
                self.scaleWtG = calibration_data['scaleWtG']
                self.StW = calibration_data['StW']
            
            print(f"校准结果已从 {file_path} 加载")
            print(f"屏幕尺寸: {calibration_data['device_info']['screen_width']}x{calibration_data['device_info']['screen_height']}")
            print(f"校准点数: {len(calibration_data['g'])}")
            print(f"SfM启用: {calibration_data['sfm']}")
            
            return True
            
        except Exception as e:
            print(f"加载校准结果失败: {e}")
            return False

    def _getGazeOnScreen(self, gaze):
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
        FSgaze = fused gaze vector, overall and for each calibration point
        Sgaze = overall gaze vector, determined over regression in screen coordinate system with head movement
        Sgaze2 = gaze vector from calibration point with head movements
        """
        return FSgaze, Sgaze, Sgaze2
      

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
        2. 对每一组数据，分别提取 gaze 向量、设定点坐标及世界变换矩阵。
        3. 对 gaze 向量的 x、y、z 三个维度均调用 _MaskOutliers 进行异常值检测，
           只有三个维度都通过检测的样本才被保留，以提高校准精度。
        4. 将过滤后的 gaze、设定点及变换矩阵分别存入列表。
        5. 将各组合并，返回统一的 DataFrame 及分组列表，供后续拟合使用。
        
        返回:
            gaze:      过滤后的 gaze 向量 DataFrame（所有组合并）
            SetVal:    过滤后的设定点坐标 DataFrame（所有组合并）
            W_T_G:     过滤后的世界变换矩阵 DataFrame（所有组合并）
            g:         按组存放的 gaze 列表，每个元素为对应组的 DataFrame
        """
        # 计算总校准点数量（索引从 0 开始，因此最大索引+1）
        idx = int(pd.unique(self.df['idx'])[-1]) + 1  # 若考虑头部转动可改为 -3
        
        # 初始化用于存放各组数据的列表
        g   = [None] * idx   # 存放 gaze 向量
        s   = [None] * idx   # 存放设定点坐标
        WTG = [None] * idx   # 存放世界变换矩阵
        
        # 按校准点索引分组处理
        for i in range(idx):
            # 提取当前组的所有 gaze 向量（x, y, z）
            g_ = self.df[self.df['idx'].values == i].loc[:, 'gaze_x':'gaze_z']
            # 提取当前组的设定点坐标（set_x, set_y, set_z）
            set_val = self.df[self.df['idx'].values == i].loc[:, 'set_x':'set_z']
            # 提取当前组的世界变换矩阵（列名包含 'WTransG'）
            WTG_ = self.df[self.df['idx'].values == i].filter(like='WTransG')
            
            # 对 gaze 的三个维度分别做异常值检测，取交集保留同时通过检测的样本
            mask = (
                self._MaskOutliers(g_.loc[:, 'gaze_x']) &
                self._MaskOutliers(g_.loc[:, 'gaze_y']) &
                self._MaskOutliers(g_.loc[:, 'gaze_z'])
            )
            
            # 应用掩码，保存过滤后的数据
            g[i]   = g_[mask]
            s[i]   = set_val[mask]
            WTG[i] = WTG_[mask]
        
        # 将设定点转换为 numpy 数组并存入实例变量，供后续绘图及评估使用
        self.SetValues = [v.to_numpy()[0][:, None] for v in s]
        
        # 将各组数据合并为整体 DataFrame，便于后续一次性处理
        gaze   = pd.concat(g,   axis=0)
        SetVal = pd.concat(s,   axis=0)
        W_T_G  = pd.concat(WTG, axis=0)
        
        return gaze, SetVal, W_T_G, g

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