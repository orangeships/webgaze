import time
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

# 添加项目根目录和src目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# 导入必要的模块
from src.gaze_tracking.model import EyeModel
from src.sfm.sfm_module import SFM

def test_cache_performance(test_frames=300):
    """
    测试缓存机制的性能提升
    
    Args:
        test_frames: 测试的帧数
    """
    print("开始缓存性能测试...")
    
    # 初始化模型和SfM模块
    eye_model = EyeModel(project_root)
    sfm = SFM(project_root)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("摄像头初始化成功")
    
    # 初始化变量
    frame_count = 0
    frame_prev = None
    
    # 性能测量变量
    no_cache_times = []  # 不使用缓存的处理时间
    with_cache_times = []  # 使用缓存的处理时间
    
    # 等待摄像头稳定
    print("等待摄像头稳定...")
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            cap.release()
            return
    
    print(f"开始测试，共测试{test_frames}帧")
    print("第一阶段：不使用缓存（清除缓存）...")
    
    # 第一阶段：不使用缓存
    while frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_prev is not None:
            # 清除缓存
            sfm.clear_caches()
            
            # 计时开始
            start_time = time.perf_counter()
            
            # 不使用缓存的方式：每次都重新计算
            face_features_prev = eye_model.get_FaceFeatures(frame_prev)
            face_features_curr = eye_model.get_FaceFeatures(frame)
            
            # 调用SfM计算
            try:
                W_T_G1, W_T_G2, W_P = sfm.get_GazeToWorld(
                    eye_model, frame_prev, frame,
                    face_features_prev=face_features_prev,
                    face_features_curr=face_features_curr
                )
            except Exception as e:
                print(f"计算出错: {e}")
            
            # 计时结束
            elapsed = time.perf_counter() - start_time
            no_cache_times.append(elapsed * 1000)  # 转换为毫秒
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"已完成 {frame_count}/{test_frames} 帧")
        
        frame_prev = frame.copy()
    
    print("\n第二阶段：使用改进的缓存机制...")
    
    # 重置变量
    frame_count = 0
    frame_prev = None
    cached_face_features_prev = None
    cached_face_features_curr = None
    
    # 第二阶段：使用改进的缓存机制
    while frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_prev is not None:
            # 计时开始
            start_time = time.perf_counter()
            
            # 优化的方式：使用缓存
            face_features_curr = eye_model.get_FaceFeatures(frame)
            
            if cached_face_features_curr is not None:
                face_features_prev = cached_face_features_curr
            else:
                face_features_prev = eye_model.get_FaceFeatures(frame_prev)
            
            # 更新缓存
            cached_face_features_prev = face_features_prev
            cached_face_features_curr = face_features_curr
            
            # 调用SfM计算
            try:
                W_T_G1, W_T_G2, W_P = sfm.get_GazeToWorld(
                    eye_model, frame_prev, frame,
                    face_features_prev=face_features_prev,
                    face_features_curr=face_features_curr
                )
            except Exception as e:
                print(f"计算出错: {e}")
            
            # 计时结束
            elapsed = time.perf_counter() - start_time
            with_cache_times.append(elapsed * 1000)  # 转换为毫秒
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"已完成 {frame_count}/{test_frames} 帧")
        
        frame_prev = frame.copy()
    
    # 释放资源
    cap.release()
    
    # 计算统计数据
    no_cache_avg = np.mean(no_cache_times) if no_cache_times else 0
    with_cache_avg = np.mean(with_cache_times) if with_cache_times else 0
    
    no_cache_fps = 1000 / no_cache_avg if no_cache_avg > 0 else 0
    with_cache_fps = 1000 / with_cache_avg if with_cache_avg > 0 else 0
    
    improvement_percent = ((no_cache_avg - with_cache_avg) / no_cache_avg) * 100 if no_cache_avg > 0 else 0
    fps_improvement_percent = ((with_cache_fps - no_cache_fps) / no_cache_fps) * 100 if no_cache_fps > 0 else 0
    
    # 输出结果
    print("\n========================================")
    print("缓存性能测试结果")
    print("========================================")
    print(f"不使用缓存:")
    print(f"  平均处理时间: {no_cache_avg:.2f} ms")
    print(f"  估计FPS: {no_cache_fps:.1f}")
    print(f"使用缓存:")
    print(f"  平均处理时间: {with_cache_avg:.2f} ms")
    print(f"  估计FPS: {with_cache_fps:.1f}")
    print(f"\n性能提升:")
    print(f"  处理时间减少: {improvement_percent:.2f}%")
    print(f"  FPS提升: {fps_improvement_percent:.2f}%")
    print("========================================")
    
    # 可视化结果
    visualize_results(no_cache_times, with_cache_times)
    
    # 保存详细数据
    save_results(no_cache_times, with_cache_times, no_cache_avg, with_cache_avg, no_cache_fps, with_cache_fps, improvement_percent)

def visualize_results(no_cache_times, with_cache_times):
    """
    可视化性能测试结果
    """
    try:
        # 确保中文字体正常显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(12, 6))
        
        # 1. 处理时间对比条形图
        plt.subplot(1, 2, 1)
        labels = ['无缓存', '有缓存']
        avg_times = [np.mean(no_cache_times), np.mean(with_cache_times)]
        
        bars = plt.bar(labels, avg_times, color=['red', 'green'])
        plt.title('平均处理时间对比')
        plt.ylabel('时间 (毫秒)')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.2f} ms', 
                    ha='center', va='bottom')
        
        # 2. 处理时间分布箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot([no_cache_times, with_cache_times], labels=['无缓存', '有缓存'])
        plt.title('处理时间分布')
        plt.ylabel('时间 (毫秒)')
        
        plt.tight_layout()
        
        # 保存图表
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'cache_performance_comparison.png')
        plt.savefig(save_path)
        print(f"性能对比图表已保存至: {save_path}")
        
    except Exception as e:
        print(f"可视化失败: {e}")

def save_results(no_cache_times, with_cache_times, no_cache_avg, with_cache_avg, no_cache_fps, with_cache_fps, improvement_percent):
    """
    保存详细的测试结果到文本文件
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'cache_performance_results.txt')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("缓存机制性能测试详细结果\n")
        f.write("========================================\n\n")
        
        f.write("处理时间统计 (毫秒):\n")
        f.write(f"不使用缓存 - 平均值: {no_cache_avg:.3f} ms\n")
        f.write(f"不使用缓存 - 最小值: {np.min(no_cache_times):.3f} ms\n")
        f.write(f"不使用缓存 - 最大值: {np.max(no_cache_times):.3f} ms\n")
        f.write(f"不使用缓存 - 标准差: {np.std(no_cache_times):.3f} ms\n\n")
        
        f.write(f"使用缓存 - 平均值: {with_cache_avg:.3f} ms\n")
        f.write(f"使用缓存 - 最小值: {np.min(with_cache_times):.3f} ms\n")
        f.write(f"使用缓存 - 最大值: {np.max(with_cache_times):.3f} ms\n")
        f.write(f"使用缓存 - 标准差: {np.std(with_cache_times):.3f} ms\n\n")
        
        f.write("性能指标:\n")
        f.write(f"不使用缓存 - 估计FPS: {no_cache_fps:.2f}\n")
        f.write(f"使用缓存 - 估计FPS: {with_cache_fps:.2f}\n\n")
        
        f.write("性能提升:\n")
        f.write(f"处理时间减少百分比: {improvement_percent:.2f}%\n")
        f.write(f"FPS提升百分比: {((with_cache_fps - no_cache_fps) / no_cache_fps) * 100:.2f}%\n\n")
        
        f.write("原始数据:\n")
        f.write("无缓存处理时间列表 (毫秒):\n")
        f.write(", ".join([f"{t:.3f}" for t in no_cache_times[:50]]) + "\n")
        f.write("\n有缓存处理时间列表 (毫秒):\n")
        f.write(", ".join([f"{t:.3f}" for t in with_cache_times[:50]]) + "\n")
    
    print(f"详细性能测试结果已保存至: {save_path}")

if __name__ == "__main__":
    print("========================================")
    print("          SfM缓存性能对比测试")
    print("========================================")
    print("此测试将对比启用和禁用缓存机制的性能差异")
    print("========================================")
    test_cache_performance(test_frames=300)