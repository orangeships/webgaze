"""
手部跟踪系统示例脚本

该脚本演示如何使用手部跟踪系统进行实时检测、视频处理和图像处理。
"""

import cv2
import numpy as np
import time
import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from hand_tracking_system import HandTrackingSystem
from gesture_recognizer import GestureType
from hand_utils import HandUtils


def demo_realtime_tracking():
    """演示实时手部跟踪"""
    print("=== 实时手部跟踪演示 ===")
    print("正在启动手部跟踪系统...")
    
    # 创建系统实例
    system = HandTrackingSystem(max_num_hands=2, detection_confidence=0.7)
    
    # 启动实时跟踪
    system.run_realtime(camera_index=0)


def demo_gesture_recognition():
    """演示手势识别"""
    print("=== 手势识别演示 ===")
    
    # 创建手势识别器
    recognizer = GestureRecognizer()
    
    # 创建示例图像（这里用空白图像模拟）
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.fill(255)
    
    # 在图像上绘制一些点来模拟手部关键点
    # 这里只是演示，实际使用时应该从真实检测结果获取
    cv2.putText(image, "手势识别演示", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "请在真实环境中测试手势识别功能", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    print("支持的预定义手势:")
    for gesture in GestureType:
        if gesture != GestureType.UNKNOWN:
            print(f"  - {gesture.value}")
    
    # 显示示例图像
    cv2.imshow("手势识别演示", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_video_processing():
    """演示视频处理"""
    print("=== 视频处理演示 ===")
    
    video_path = input("请输入视频文件路径: ").strip()
    
    if not os.path.exists(video_path):
        print(f"错误：文件 {video_path} 不存在")
        return
    
    # 创建系统实例
    system = HandTrackingSystem()
    
    # 设置输出路径
    output_path = "output_processed_video.mp4"
    
    print(f"处理视频: {video_path}")
    print(f"输出到: {output_path}")
    
    # 处理视频
    system.process_video_file(video_path, output_path)
    
    print("视频处理完成!")


def demo_image_processing():
    """演示图像处理"""
    print("=== 图像处理演示 ===")
    
    image_path = input("请输入图像文件路径: ").strip()
    
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return
    
    # 创建系统实例
    system = HandTrackingSystem()
    
    print(f"处理图像: {image_path}")
    
    # 处理图像
    result = system.process_image(image_path)
    
    if result is not None:
        print("图像处理完成!")


def demo_data_export():
    """演示数据导出功能"""
    print("=== 数据导出演示 ===")
    
    # 创建示例数据
    sample_data = {
        'frames': [
            {
                'timestamp': time.time(),
                'hands': [
                    {
                        'handedness': 'Right',
                        'bbox': (100, 100, 200, 200),
                        'landmarks': [(0.1, 0.2, 0.0), (0.15, 0.25, 0.0)],  # 示例关键点
                        'center': (0.125, 0.225)
                    }
                ]
            }
        ]
    }
    
    # 导出为不同格式
    formats = ['json', 'csv', 'txt']
    
    for fmt in formats:
        filename = f"sample_hand_data_{fmt}"
        filepath = HandUtils.export_landmarks_data(sample_data, filename, fmt)
        print(f"数据已导出为 {fmt} 格式: {filepath}")


def demo_movement_analysis():
    """演示运动分析"""
    print("=== 手部运动分析演示 ===")
    
    # 创建示例运动轨迹数据
    landmarks_sequence = []
    
    # 模拟一个简单的移动轨迹
    for i in range(20):
        landmarks = [(0.3 + i * 0.01, 0.4, 0.0) for _ in range(21)]
        landmarks_sequence.append(landmarks)
    
    # 进行运动分析
    analysis = HandUtils.analyze_hand_movement(landmarks_sequence)
    
    print("运动分析结果:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")


def interactive_demo():
    """交互式演示菜单"""
    while True:
        print("\n" + "="*50)
        print("手部跟踪系统演示菜单")
        print("="*50)
        print("1. 实时手部跟踪")
        print("2. 手势识别演示")
        print("3. 视频处理")
        print("4. 图像处理")
        print("5. 数据导出")
        print("6. 运动分析")
        print("0. 退出")
        print("="*50)
        
        try:
            choice = input("请选择功能 (0-6): ").strip()
            
            if choice == '0':
                print("感谢使用!")
                break
            elif choice == '1':
                demo_realtime_tracking()
            elif choice == '2':
                demo_gesture_recognition()
            elif choice == '3':
                demo_video_processing()
            elif choice == '4':
                demo_image_processing()
            elif choice == '5':
                demo_data_export()
            elif choice == '6':
                demo_movement_analysis()
            else:
                print("无效选择，请重试")
        
        except KeyboardInterrupt:
            print("\n\n用户中断程序")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue


def check_requirements():
    """检查系统要求"""
    print("检查系统要求...")
    
    # 检查必要的库
    required_packages = ['cv2', 'mediapipe', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请使用以下命令安装:")
        print("pip install opencv-python mediapipe numpy")
        return False
    
    print("\n所有要求已满足!")
    return True


def main():
    """主函数"""
    print("手部跟踪系统演示程序")
    print("="*30)
    
    # 检查系统要求
    if not check_requirements():
        return
    
    # 等待用户确认
    input("\n按 Enter 键开始演示...")
    
    # 运行交互式演示
    interactive_demo()


if __name__ == "__main__":
    main()