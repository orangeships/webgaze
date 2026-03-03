import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
import os
import datetime
from gaze_tracking.model import EyeModel
# 从calibration_pygame导入工具函数，不再使用gui_opencv
from gaze_tracking.calibration_pygame import getScreenSize, getWhiteFrame, ReadCameraCalibrationData, get_out_video, PygameCalibrationTargets

# Use Agg backend for canvas

from sfm.estimate_essential_matrix import estimateEssentialMatrix
from sfm.decompose_essential_matrix import decomposeEssentialMatrix
from sfm.disambiguate_relative_pose import disambiguateRelativePose
from sfm.linear_triangulation import linearTriangulation
from sfm.draw_camera import drawCamera
# from sfm.utils import invHomMatrix, fit_plane, rotation_matrix_to_align_plane
import utilities.utils as util
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

class SFM():

    def __init__(self, directory) -> None:            
        self.dir = directory
        self.camera_matrix, self.dist_coeffs = ReadCameraCalibrationData(os.path.join(directory, "camera_data"))
        self.width, self.height, self.width_mm, self.height_mm = getScreenSize()
        self.S_T_W = np.array([[-1,0,0,self.width/2],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        
        # 初始化缓存机制
        # 1. 人脸检测结果缓存
        self.face_detection_cache_prev = None  # 上一帧人脸关键点检测结果
        self.face_detection_cache_curr = None  # 当前帧人脸关键点检测结果
        
        # 2. SfM人脸关键点检测结果缓存
        self.landmark_cache_prev = None  # 上一帧35点人脸关键点检测结果 (去畸变)
        self.landmark_cache_curr = None  # 当前帧35点人脸关键点检测结果 (去畸变)
        
    def RunGazeOnScreen(self, model, cap):

        if cap != None:
            out_video, wc_width, wc_height = get_out_video(cap, os.path.join(self.dir, "results"), file_name = "output_video.mp4", scalewidth=2)

        white_frame = getWhiteFrame(self.width, self.height)
        # 使用PygameCalibrationTargets替代原有的Targets类
        target = PygameCalibrationTargets(self.width, self.height)
        df = pd.DataFrame()
        frame = None
        frame_prev = None
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except StopIteration:
                break
            
            if frame_prev is None:
                frame_prev = frame
                frame = None

            if frame is not None and frame_prev is not None:
                # 预计算人脸特征点
                p1 = model.get_FaceFeatures(frame_prev)
                p2 = model.get_FaceFeatures(frame)
                
                # 更新缓存
                self.update_caches(frame_prev_features=p1, frame_curr_features=p2)
                
                frame_prev = frame

                E = estimateEssentialMatrix(p1, p2, self.camera_matrix, self.camera_matrix)
                # Extract the relative camera positions (R,T) from the essential matrix
                # Obtain extrinsic parameters (R,t) from E
                Rots, u3 = decomposeEssentialMatrix(E)

                # Disambiguate among the four possible configurations
                G_R_Gp, G_t_Gp = disambiguateRelativePose(Rots, u3, p1, p2, self.camera_matrix, self.camera_matrix)

                # Triangulate a point cloud using the final transformation (R,T)
                M1 = self.camera_matrix @ np.eye(3,4)
                M2 = self.camera_matrix @ np.c_[G_R_Gp, G_t_Gp]
                W_P = linearTriangulation(p1, p2, M1, M2)   # Estimated 3D points of face features in world coordinates (previous frame)

                # World is location of previous frame
                W_T_Gp = np.r_[np.c_[np.eye(3), W_P[0:3,0]], np.array([[0,0,0,1]])]
                Gp_T_G = np.r_[np.c_[G_R_Gp.T, -G_R_Gp.T @ G_t_Gp], np.array([[0,0,0,1]])]
                W_T_G = W_T_Gp @ Gp_T_G

                eye_info = model.get_gaze(frame)
                gaze = eye_info['gaze']

                # Project gaze vector on screen
                Ggaze = self._ProjectVetorOnPlane(util.invHomMatrix(W_T_G), gaze)
                Sgaze = self.S_T_W @ W_T_G @ Ggaze

                EyeState = eye_info['EyeState']
                if np.all(np.array(EyeState) == 1):
                    gazeframe, SetPos = target.DrawTargetGaze(white_frame, Sgaze[0:3])
                    if out_video is not None:
                        final_frame = np.concatenate((cv2.flip(cv2.resize(gazeframe, (wc_width, wc_height)), 1), frame), axis=1)
                        out_video.write(final_frame)

                else:
                    # return np.array([-10,-10,-10])
                    pass
            
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                df = pd.concat([ df, pd.DataFrame([np.hstack((timestamp, eye_info['gaze'], Sgaze[0:3].reshape(-1), W_P[0:3,0].reshape(-1), G_t_Gp)) ]) ])

            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        df.columns = ['timestamp(hh:m:s.ms)','gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x', 'Sgaze_y', 'Sgaze_z', 'W_Gpx', 'W_Gpy', 'W_Gpz', 'G_t_Gpx', 'G_t_Gpy', 'G_t_Gpz']
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.dir, "results", "GazeTracking.csv"))

    def getReferenceFrame(self, video_path):
        video_path = os.path.join(self.dir, video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            exit()

        accum_frame = None
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if accum_frame is None:
                # Initialize the accumulator with the first frame
                accum_frame = np.zeros_like(frame, dtype=np.float32)

            accum_frame += frame.astype(np.float32)
            frame_count += 1

        cap.release()

        if frame_count > 0:
            # Compute the average frame
            average_frame = (accum_frame / frame_count).astype(np.uint8)
            self.average_frame = average_frame

            return average_frame
        
        else:
            print("Error: No frames in video")
            return None


    def sfm_video(self, model, video_path):
        """
        对输入视频进行 SfM 处理，生成 3D 点云、相机轨迹动画及可视化视频
        """
        video_path = os.path.join(self.dir, video_path)
        cap = cv2.VideoCapture(video_path)

        # 输出视频
        from gaze_tracking.calibration_pygame import get_out_video
        out_video, _, _ = get_out_video(
            cap,
            os.path.join(self.dir, "results"),
            file_name="eye_features.mp4",
            scalewidth=2
        )

        frame_prev = None
        W_P = None

        # ============================
        # 3D 可视化初始化
        # ============================
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 固定坐标范围（防止抖动）
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0.0, 1.0)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-60)

        # 世界坐标原点
        ax.scatter(0, 0, 0, c='k', s=40)
        ax.text(0, 0, 0, 'W')

        plots = []      # ArtistAnimation 用
        df = pd.DataFrame()
        dfT = pd.DataFrame()

        # ============================
        # 主循环
        # ============================
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_prev is not None:
                face_prev = model.get_FaceFeatures(frame_prev)
                face_curr = model.get_FaceFeatures(frame)

                self.update_caches(
                    frame_prev_features=face_prev,
                    frame_curr_features=face_curr
                )

                W_T_G1, _, W_P = self.get_GazeToWorld(
                    model,
                    keypoints_prev=face_prev,
                    keypoints_curr=face_curr
                )
                sy_WT = W_T_G1.copy()  # 保存原始变换矩阵用于显示
                R_fix = np.array([
                    [1,  0,  0],   # X 不变
                    [0,  0,  1],   # Z → Y
                    [0, -1,  0]    # -Y → Z（让脸朝前）
                ])
                # ========================
                # 保存数据
                # ========================
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                df = pd.concat([
                    df,
                    pd.DataFrame([[ts, *W_P.flatten(), *np.mean(W_P, axis=0)]])
                ])
                dfT = pd.concat([dfT, pd.DataFrame(W_T_G1.reshape(1, -1))])
                W_P = (R_fix @ W_P.T).T

                W_T_G1[:3, :3] = R_fix @ W_T_G1[:3, :3]
                W_T_G1[:3,  3] = R_fix @ W_T_G1[:3,  3]
                # ========================
                # 每一帧重新创建 artist
                # ========================
                frame_artists = []

                # 点云
                sc = ax.scatter(
                    W_P[:, 0],
                    W_P[:, 1],
                    W_P[:, 2],
                    c='b',
                    s=8
                )
                frame_artists.append(sc)

                # 相机坐标轴
                C = W_T_G1[:3, 3]
                R = W_T_G1[:3, :3]
                s = 0.1

                lx, = ax.plot(
                    [C[0], C[0] + s * R[0, 0]],
                    [C[1], C[1] + s * R[1, 0]],
                    [C[2], C[2] + s * R[2, 0]],
                    'r', lw=2
                )
                ly, = ax.plot(
                    [C[0], C[0] + s * R[0, 1]],
                    [C[1], C[1] + s * R[1, 1]],
                    [C[2], C[2] + s * R[2, 1]],
                    'g', lw=2
                )
                lz, = ax.plot(
                    [C[0], C[0] + s * R[0, 2]],
                    [C[1], C[1] + s * R[1, 2]],
                    [C[2], C[2] + s * R[2, 2]],
                    'b', lw=2
                )

                frame_artists.extend([lx, ly, lz])

                # ========================
                # W_T_G1[:3,3] 可视化
                # ========================
                # 获取原始的平移向量（变换之前的）
                tvec_w = W_T_G1[:3, 3]
                
                # 添加小标记点表示平移向量位置
                marker = ax.scatter(
                    [tvec_w[0]], 
                    [tvec_w[1]], 
                    [tvec_w[2]], 
                    c='orange', 
                    s=100, 
                    marker='o',
                    alpha=0.8
                )
                frame_artists.append(marker)

                # 添加文本标签显示数值
                text_label = ax.text(
                    sy_WT[0, 3], sy_WT[1, 3], sy_WT[2, 3] ,
                    f'W_T_G1: [{sy_WT[0, 3]:.3f}, {sy_WT[1, 3]:.3f}, {sy_WT[2, 3]:.3f}]',
                    fontsize=8,
                    color='orange',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
                frame_artists.append(text_label)

                plots.append(frame_artists)

                # ========================
                # 2D 投影可视化
                # ========================
                for p in W_P:
                    I_P = self.camera_matrix @ p.reshape(3, 1)
                    I_P /= I_P[2]
                    cv2.drawMarker(
                        frame,
                        tuple(I_P[:2].astype(int).flatten()),
                        (255, 0, 0),
                        cv2.MARKER_CROSS,
                        2
                    )

            frame_prev = frame.copy()

            # ========================
            # 输出视频
            # ========================
            if out_video is not None:
                draw_frame = frame.copy()
                p1 = model.get_FaceFeatures(draw_frame)
                for i, p in enumerate(p1.T):
                    cv2.drawMarker(
                        draw_frame,
                        tuple(p[:2].astype(int)),
                        (255, 0, 0),
                        cv2.MARKER_CROSS,
                        1
                    )
                    cv2.putText(
                        draw_frame,
                        str(i),
                        tuple(p[:2].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
                out_video.write(cv2.hconcat([draw_frame, frame]))

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        # ============================
        # 生成动画（保证会动）
        # ============================
        ani = animation.ArtistAnimation(
            fig,
            plots,
            interval=33,
            blit=False
        )

        ani.save(
            os.path.join(self.dir, "results", "animation.mp4"),
            writer='ffmpeg',
            fps=30
        )

        plt.show()

        # ============================
        # 保存 CSV
        # ============================
        df.columns = (
            ['timestamp'] +
            ['W_Px', 'W_Py', 'W_Pz'] * W_P.shape[0] +
            ['W_Px_mean', 'W_Py_mean', 'W_Pz_mean']
        )
        df.reset_index(drop=True, inplace=True)

        df.to_csv(os.path.join(self.dir, "results", "W_P.csv"), index=False)
        dfT.to_csv(os.path.join(self.dir, "results", "W_T_G.csv"), index=False)

        return dfT.to_numpy()
    
    def get_GazeToWorld(self, pnp_tvec_curr=None, calibration_tvec=None):
        """
        获取从注视向量到世界坐标系的变换
        
        Args:
            pnp_tvec_curr: 当前帧的pnp平移向量，形状为 (3, 1)
            pnp_tvec_prev: 上一帧的pnp平移向量，形状为 (3, 1) 
            calibration_tvec: 标定时的第一个pnp平移向量，形状为 (3, 1)
            
        Returns:
            W_T_G1: 从标定位置到世界坐标系的变换矩阵
            W_T_G2: 从当前帧到世界坐标系的变换矩阵
            W_P: 3D点云，形状为 (N, 3)（简化版本，返回空点云）

        """
        try:
            
            # 计算当前tvec与标定tvec的差值
            tvec_diff = pnp_tvec_curr - calibration_tvec
            print(f"当前tvec: {pnp_tvec_curr.flatten()}")
            print(f"标定tvec: {calibration_tvec.flatten()}")
            print(f"tvec差值: {tvec_diff.flatten()}")
            
            # 构造W_T_G1（标定位置到世界坐标系）
            W_T_G1 = np.array([
                [1.0, 0.0, 0.0, tvec_diff[0, 0]],    # X轴不变
                [0.0, -1.0, 0.0, tvec_diff[1, 0]], # Y轴翻转并取负
                [0.0, 0.0, -1.0, tvec_diff[2, 0]],  # Z轴翻转
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # 构造W_T_G2（当前帧到世界坐标系）
            # 同样包含坐标轴变换
            W_T_G2 = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # 生成简化的3D点云（返回空点云或默认值）
            W_P = np.zeros((35, 3))  # 返回35个零点的点云
            
            return W_T_G1, W_T_G2, W_P

        except Exception as e:
            print(f"Error in get_GazeToWorld: {e}")
            # 发生异常时返回默认值
            W_T_G1 = np.eye(4)
            W_T_G2 = np.eye(4)
            W_P = np.zeros((35, 3))
            return W_T_G1, W_T_G2, W_P

    def _ProjectVetorOnPlane(self, Trans, vector):
        """ Translation of homogenous Trans-Matrix must be in same coordinate system as Vector """
        vector = vector.reshape(3,1)
        VectorNormal2Plane = (Trans[0:3,0:3] @ np.array([[0],[0],[1]]))
        transVec = Trans[0:3,3]
        t = (VectorNormal2Plane.T @ transVec) / (VectorNormal2Plane.T @ vector)
        Vector2Plane = np.vstack((t*vector, 1))
        return Vector2Plane
        
    def update_caches(self, frame_prev_features=None, frame_curr_features=None):
        """
        更新缓存的人脸检测和关键点检测结果
        
        Args:
            frame_prev_features: 上一帧的人脸特征点
            frame_curr_features: 当前帧的人脸特征点
        """
        # 更新人脸检测结果缓存
        if frame_prev_features is not None:
            self.face_detection_cache_prev = frame_prev_features
        if frame_curr_features is not None:
            self.face_detection_cache_curr = frame_curr_features
            
        # 关键点缓存会在get_GazeToWorld内部自动更新
        
    def clear_caches(self):
        """
        清除所有缓存数据
        """
        self.face_detection_cache_prev = None
        self.face_detection_cache_curr = None

        
    def get_cached_face_features(self, frame_type='prev'):
        """
        获取缓存的人脸特征点
        
        Args:
            frame_type: 'prev' 或 'curr'
            
        Returns:
            缓存的人脸特征点，如果不存在则返回None
        """
        if frame_type == 'prev':
            return self.face_detection_cache_prev
        elif frame_type == 'curr':
            return self.face_detection_cache_curr
        else:
            return None
        


if __name__ == '__main__':
    dir = "C:\\temp\\WebCamGazeEstimation\\"
    model = EyeModel(dir)
    sfm = SFM(dir)

    # video_path = os.path.join(dir, "results", "calibrate.mp4")
    # average_frame = sfm.getReferenceFrame(video_path)
    # p1 = model.get_FaceFeatures(average_frame)
    # for idx, p in enumerate(p1.T):
    #     cv2.drawMarker(average_frame, tuple(p.astype(int)[0:2].flatten()), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
    #     cv2.putText(average_frame, str(idx), tuple(p.astype(int)[0:2].flatten()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the average frame
    # cv2.imshow("Average Frame", average_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    video_path = os.path.join(dir, "results", "output_video.mp4")
    sfm.sfm_video(model, video_path)


    # cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # # cap.set(cv2.CAP_PROP_SETTINGS, 1)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # sfm.RunGazeOnScreen(model, cap)

    # image_path = os.path.join(dir, "results")
    # sfm.sfm_image(model, image_path)