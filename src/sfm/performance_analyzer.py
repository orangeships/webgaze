import time
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

# 添加项目根目录和src目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# 指定测试视频路径
video_path = os.path.join(project_root, 'src', 'WIN_20251111_13_25_39_Pro.mp4')

# 现在可以正确导入模块了
from src.gaze_tracking.model import EyeModel
from src.utilities import utils
from src.sfm.sfm_module import SFM

def measure_performance(num_iterations=100):
    """
    测量SfM模块中各关键环节的执行时间
    """
    print("初始化性能分析工具...")
    
    # 初始化模型和SfM模块
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    eye_model = EyeModel(project_dir)
    sfm = SFM(project_dir)  # SFM类初始化需要提供directory参数
    
    # 打开测试视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    print(f"使用视频文件: {video_path}")
    
    # 等待摄像头稳定
    print("等待摄像头稳定...")
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            cap.release()
            return
    
    # 初始化时间记录
    times = {
        'face_features': [],
        'face_detection': [],  # 新增：人脸检测时间
        'landmark_detection': [],  # 新增：关键点检测时间
        'undistort': [],
        'essential_matrix': [],
        'recover_pose': [],
        'triangulation': [],
        'plane_fitting': [],
        'rotation_matrix': [],
        'total_get_gaze_to_world': []
    }
    
    print(f"开始性能测量，执行{num_iterations}次...")
    
    # 保存原始方法的引用
    original_get_gaze_to_world = sfm.get_GazeToWorld
    
    # 定义一个包装方法来测量各个子环节
    def wrapped_get_gaze_to_world(model, frame_prev, frame, face_features_prev=None, face_features_curr=None):
        try:
            # 测量总时间
            total_start = time.perf_counter()
            
            # 1. 获取人脸特征点（使用外部传入的特征点，避免重复计算）
            if face_features_prev is not None and face_features_curr is not None:
                p1 = face_features_prev[:2,:]
                p2 = face_features_curr[:2,:]
            else:
                # 备用方案：如果没有传入特征点，则内部计算
                p1 = model.get_FaceFeatures(frame_prev)[:2,:]
                p2 = model.get_FaceFeatures(frame)[:2,:]
            
            # 2. 去畸变处理
            undistort_start = time.perf_counter()
            p1_undistorted = cv2.undistortPoints(p1, sfm.camera_matrix, sfm.dist_coeffs, P=sfm.camera_matrix).reshape(-1,2)
            p2_undistorted = cv2.undistortPoints(p2, sfm.camera_matrix, sfm.dist_coeffs, P=sfm.camera_matrix).reshape(-1,2)
            times['undistort'].append(time.perf_counter() - undistort_start)
            
            # 3. 本质矩阵估计
            em_start = time.perf_counter()
            E = cv2.findEssentialMat(p1_undistorted, p2_undistorted, sfm.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
            times['essential_matrix'].append(time.perf_counter() - em_start)
            
            # 4. 相机位姿恢复
            rp_start = time.perf_counter()
            _, G2_R_G1, G2_t_G1, _ = cv2.recoverPose(E, p1_undistorted, p2_undistorted, sfm.camera_matrix)
            times['recover_pose'].append(time.perf_counter() - rp_start)
            
            # 5. 三角化三维点云
            tri_start = time.perf_counter()
            M1 = sfm.camera_matrix @ np.eye(3,4)
            M2 = sfm.camera_matrix @ np.c_[G2_R_G1, G2_t_G1]
            points_4d_homogeneous = cv2.triangulatePoints(M1, M2, p1_undistorted.T, p2_undistorted.T)
            W_P = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T).reshape(-1,3)
            
            # 归一化和方向调整
            W_P = W_P/np.linalg.norm(W_P, axis=1)[:,np.newaxis]
            W_P[W_P[:,2]<0] = W_P[W_P[:,2]<0]*(-1)
            times['triangulation'].append(time.perf_counter() - tri_start)
            
            # 6. 平面拟合
            pf_start = time.perf_counter()
            normal_vector, _ = utils.fit_plane(W_P)
            normal_vector = normal_vector/np.linalg.norm(normal_vector)
            times['plane_fitting'].append(time.perf_counter() - pf_start)
            
            # 7. 旋转矩阵计算
            rm_start = time.perf_counter()
            # 选择特定点计算旋转矩阵
            selected_points = np.array([W_P[0,:], W_P[2,:], W_P[3,:], W_P[18,:]])
            W_R_G1 = utils.rotation_matrix_to_face(normal_vector, selected_points)
            times['rotation_matrix'].append(time.perf_counter() - rm_start)
            
            # 计算总时间
            times['total_get_gaze_to_world'].append(time.perf_counter() - total_start)
            
            # 构造返回值
            WRotG = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            W_T_G1 = np.r_[np.c_[WRotG, np.mean(np.array([W_P[0,:], W_P[2,:]]), axis=0)[:,None]], np.array([[0,0,0,1]])]
            G1_T_G2 = np.r_[np.c_[G2_R_G1.T, -G2_R_G1.T @ G2_t_G1], np.array([[0,0,0,1]])]
            W_T_G2 = W_T_G1 @ G1_T_G2
            
            return W_T_G1, W_T_G2, W_P
        except Exception as e:
            print(f"包装方法中出错: {e}")
            # 返回默认值以避免中断
            return np.eye(4), np.eye(4), np.zeros((35, 3))
    
    # 替换方法进行测量
    sfm.get_GazeToWorld = wrapped_get_gaze_to_world
    
    # 执行多次测量
    successful_iterations = 0
    for i in range(num_iterations):
        ret, frame_prev = cap.read()
        if not ret:
            print(f"第{i+1}次测量时无法读取第一帧")
            continue
        
        ret, frame = cap.read()
        if not ret:
            print(f"第{i+1}次测量时无法读取第二帧")
            continue
        
        print(f"执行测量 {i+1}/{num_iterations}")
        
        # 分别测量人脸检测和关键点检测的时间
        # 第一帧的人脸检测
        face_detection_time_prev = 0
        landmark_detection_time_prev = 0
        face_detection_time_curr = 0
        landmark_detection_time_curr = 0
        
        # 测量第一帧的人脸检测时间
        start_time = time.perf_counter()
        face_boxes_prev = eye_model.face_detection.predict(frame_prev)
        face_detection_time_prev = time.perf_counter() - start_time
        
        # 如果检测到人脸，再测量关键点检测时间
        face_features_prev = np.zeros((3, 36))
        if len(face_boxes_prev) > 0:
            face_box_prev = face_boxes_prev[0]
            face_prev = eye_model.get_crop_image(frame_prev, face_box_prev)
            start_time = time.perf_counter()
            landmarks_prev = eye_model.facial_landmark_35.predict(face_prev)
            landmark_detection_time_prev = time.perf_counter() - start_time
            
            # 计算特征点绝对坐标
            xmin, ymin, xmax, ymax = face_box_prev
            points_prev = np.zeros_like(landmarks_prev)
            for idx, pos in enumerate(landmarks_prev):
                x = pos[0] + xmin
                y = pos[1] + ymin
                points_prev[idx] = [x, y]
            face_features_prev = np.c_[points_prev, np.ones((points_prev.shape[0], 1))].T
        
        # 测量第二帧的人脸检测时间
        start_time = time.perf_counter()
        face_boxes_curr = eye_model.face_detection.predict(frame)
        face_detection_time_curr = time.perf_counter() - start_time
        
        # 如果检测到人脸，再测量关键点检测时间
        face_features_curr = np.zeros((3, 36))
        if len(face_boxes_curr) > 0:
            face_box_curr = face_boxes_curr[0]
            face_curr = eye_model.get_crop_image(frame, face_box_curr)
            start_time = time.perf_counter()
            landmarks_curr = eye_model.facial_landmark_35.predict(face_curr)
            landmark_detection_time_curr = time.perf_counter() - start_time
            
            # 计算特征点绝对坐标
            xmin, ymin, xmax, ymax = face_box_curr
            points_curr = np.zeros_like(landmarks_curr)
            for idx, pos in enumerate(landmarks_curr):
                x = pos[0] + xmin
                y = pos[1] + ymin
                points_curr[idx] = [x, y]
            face_features_curr = np.c_[points_curr, np.ones((points_curr.shape[0], 1))].T
        
        # 计算总时间
        features_time = (face_detection_time_prev + landmark_detection_time_prev + 
                        face_detection_time_curr + landmark_detection_time_curr)
        
        # 记录各阶段时间（合并两帧数据）
        total_face_detection_time = face_detection_time_prev + face_detection_time_curr
        total_landmark_detection_time = landmark_detection_time_prev + landmark_detection_time_curr
        
        # 只记录成功的人脸检测
        if np.any(face_features_prev) and np.any(face_features_curr):  # 检查两个帧都检测到人脸
            times['face_features'].append(features_time)
            times['face_detection'].append(total_face_detection_time)
            times['landmark_detection'].append(total_landmark_detection_time)
            successful_iterations += 1
            
            # 调用SfM计算时传入已计算的特征点，避免重复计算
            try:
                sfm.get_GazeToWorld(eye_model, frame_prev, frame, face_features_prev, face_features_curr)
            except Exception as e:
                print(f"调用SfM计算出错: {e}")
        else:
            print("未检测到人脸，跳过此次测量")
    
    # 恢复原始方法
    sfm.get_GazeToWorld = original_get_gaze_to_world
    
    # 释放资源
    cap.release()
    
    print(f"\n成功完成 {successful_iterations}/{num_iterations} 次测量")
    
    # 如果没有成功的测量，退出
    if successful_iterations == 0:
        print("无法进行有效的性能测量，请确保摄像头能够检测到人脸")
        return
    
    # 释放资源
    cap.release()
    
    # 计算平均时间
    avg_times = {k: np.mean(v) * 1000 for k, v in times.items() if v}  # 转换为毫秒
    
    # 计算SfM内部操作的总时间（不包括人脸特征提取）
    sfm_internal_time = sum(avg_times.get(k, 0) for k in ['undistort', 'essential_matrix', 'recover_pose', 'triangulation', 'plane_fitting', 'rotation_matrix'])
    
    # 获取人脸特征提取时间
    face_features_time = avg_times.get('face_features', 0)
    
    # 计算正确的总时间（人脸特征提取 + SfM内部计算）
    correct_total_time = face_features_time + sfm_internal_time
    
    # 计算百分比 - 每个人脸特征提取和SfM各环节相对于正确总时间的比例
    percentages = {}
    if correct_total_time > 0:
        # 分别计算人脸检测和关键点检测的百分比
        percentages['face_detection'] = (avg_times.get('face_detection', 0) / correct_total_time) * 100
        percentages['landmark_detection'] = (avg_times.get('landmark_detection', 0) / correct_total_time) * 100
        # 其他各环节百分比
        for k in ['undistort', 'essential_matrix', 'recover_pose', 'triangulation', 'plane_fitting', 'rotation_matrix']:
            if k in avg_times:
                percentages[k] = (avg_times[k] / correct_total_time) * 100
    
    # 使用英文标签便于显示
    percentage_names = {
        'face_detection': 'Face Detection',
        'landmark_detection': 'Landmark Detection',
        'undistort': 'Undistortion',
        'essential_matrix': 'Essential Matrix',
        'recover_pose': 'Pose Recovery',
        'triangulation': 'Triangulation',
        'plane_fitting': 'Plane Fitting',
        'rotation_matrix': 'Rotation Matrix'
    }
    percentages = {percentage_names.get(k, k): v for k, v in percentages.items()}
    
    # 输出结果
    print("\n性能测量结果（毫秒）:")
    print("-" * 60)
    print(f"人脸特征点提取（总时间）: {avg_times.get('face_features', 0):.2f} ms")
    print(f"  - 人脸检测时间: {avg_times.get('face_detection', 0):.2f} ms")
    print(f"  - 关键点检测时间: {avg_times.get('landmark_detection', 0):.2f} ms")
    print(f"去畸变处理: {avg_times.get('undistort', 0):.2f} ms")
    print(f"本质矩阵估计: {avg_times.get('essential_matrix', 0):.2f} ms")
    print(f"相机位姿恢复: {avg_times.get('recover_pose', 0):.2f} ms")
    print(f"三维点云三角化: {avg_times.get('triangulation', 0):.2f} ms")
    print(f"平面拟合: {avg_times.get('plane_fitting', 0):.2f} ms")
    print(f"旋转矩阵计算: {avg_times.get('rotation_matrix', 0):.2f} ms")
    print("-" * 60)
    print(f"SfM内部计算总时间: {sfm_internal_time:.2f} ms")
    print(f"总SfM计算时间（含人脸特征）: {correct_total_time:.2f} ms")
    
    # 比较人脸检测和关键点检测的耗时
    face_detection_time = avg_times.get('face_detection', 0)
    landmark_detection_time = avg_times.get('landmark_detection', 0)
    if face_detection_time > landmark_detection_time:
        print(f"\n结论: 人脸检测比关键点检测更耗时，差值为: {face_detection_time - landmark_detection_time:.2f} ms")
    else:
        print(f"\n结论: 关键点检测比人脸检测更耗时，差值为: {landmark_detection_time - face_detection_time:.2f} ms")
    
    print("\n时间占比（相对于总SfM计算时间）:")
    print("-" * 60)
    for k, v in percentages.items():
        print(f"{k:20}: {v:6.2f}%")
    
    # 可视化结果
    results_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'results')
    os.makedirs(results_dir, exist_ok=True)
    text_save_path = os.path.join(results_dir, 'sfm_performance_data.txt')
    if percentages:
        visualize_results(times, percentages, text_save_path)

def visualize_results(times, percentages, text_save_path):
    """
    可视化性能测量结果并保存详细数据到文本文件
    """
    try:
        # 计算平均时间
        avg_times = {k: sum(v) / len(v) * 1000 if v else 0 for k, v in times.items()}
        
        # 保存详细数据到文本文件
        with open(text_save_path, 'w', encoding='utf-8') as f:
            f.write("SfM Module Performance Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入各环节平均执行时间
            f.write("Average Execution Time (ms):\n")
            f.write(f"Face Detection: {avg_times.get('face_detection', 0):.3f} ms\n")
            f.write(f"Landmark Detection: {avg_times.get('landmark_detection', 0):.3f} ms\n")
            f.write(f"Total Face Features: {avg_times.get('face_features', 0):.3f} ms\n")
            f.write(f"Undistortion: {avg_times.get('undistort', 0):.3f} ms\n")
            f.write(f"Essential Matrix: {avg_times.get('essential_matrix', 0):.3f} ms\n")
            f.write(f"Pose Recovery: {avg_times.get('recover_pose', 0):.3f} ms\n")
            f.write(f"Triangulation: {avg_times.get('triangulation', 0):.3f} ms\n")
            f.write(f"Plane Fitting: {avg_times.get('plane_fitting', 0):.3f} ms\n")
            f.write(f"Rotation Matrix: {avg_times.get('rotation_matrix', 0):.3f} ms\n")
            f.write(f"Total SfM Time: {avg_times.get('total_get_gaze_to_world', 0):.3f} ms\n\n")
            
            # 写入人脸检测和关键点检测的比较
            if avg_times.get('face_detection', 0) > avg_times.get('landmark_detection', 0):
                f.write(f"Performance Comparison: Face detection is {avg_times.get('face_detection', 0) - avg_times.get('landmark_detection', 0):.3f} ms faster than landmark detection\n")
            else:
                f.write(f"Performance Comparison: Landmark detection is {avg_times.get('landmark_detection', 0) - avg_times.get('face_detection', 0):.3f} ms faster than face detection\n")
            f.write("\n")
            
            # 写入时间占比
            f.write("Time Percentage (%):\n")
            # 排序后写入，从大到小
            sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
            for k, v in sorted_percentages:
                f.write(f"{k}: {v:.1f}%\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Successful Measurements: {len(times.get('face_features', []))} frames\n")
        
        print(f"\nDetailed performance data saved to '{text_save_path}'")
        
        # 确保中文字体正常显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        plt.figure(figsize=(12, 6))
        
        # 饼图 - 只显示占比大于1%的项，将小项合并为"Other"
        plt.subplot(1, 2, 1)
        
        # 过滤并合并小项
        main_items = {k: v for k, v in percentages.items() if v >= 1.0}
        other_value = sum(v for k, v in percentages.items() if v < 1.0)
        
        if other_value > 0:
            main_items['Other'] = other_value
        
        labels = list(main_items.keys())
        sizes = list(main_items.values())
        explode = (0.1,) + (0,) * (len(sizes) - 1)  # 突出第一个部分
        
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # 保证饼图是圆的
        plt.title('SfM Components Time Distribution')
        
        # 条形图
        plt.subplot(1, 2, 2)
        # 按值排序
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        labels, sizes = zip(*sorted_items)
        
        bars = plt.barh(labels, sizes)
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    ha='left', va='center')
        
        plt.xlabel('Percentage (%)')
        plt.title('SfM Components Time Percentage (Descending)')
        plt.xlim(0, max(sizes) * 1.1)  # Set x-axis range for better label visibility
        
        plt.tight_layout()
        
        # 保存图表到results目录
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'sfm_performance_analysis.png')
        plt.savefig(save_path)
        print(f"Performance analysis chart saved to '{save_path}'")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    print("========================================")
    print("       SfM模块性能分析工具")
    print("========================================")
    print("注意：请确保摄像头可以正常工作并能够检测到人脸")
    print("========================================")
    measure_performance(num_iterations=100)  # 可以调整迭代次数