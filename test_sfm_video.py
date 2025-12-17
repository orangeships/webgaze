#!/usr/bin/env python3
"""
测试SfM视频处理功能
使用testsfm.mp4作为输入进行SfM分析
"""

import os
import sys
import cv2
import numpy as np

# 添加项目路径到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from gaze_tracking.model import EyeModel
from sfm.sfm_module import SFM

def test_sfm_video():
    """测试SfM视频处理功能"""
    
    # 初始化路径
    video_input = "testsfm.mp4"
    video_path = os.path.join(project_root, video_input)
    
    # 检查输入视频是否存在
    if not os.path.exists(video_path):
        print(f"错误：输入视频文件不存在: {video_path}")
        return False
    
    print(f"开始测试SfM视频处理...")
    print(f"输入视频: {video_path}")
    
    try:
        # 创建输出目录
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化模型和SFM模块
        model = EyeModel(project_root)  # 使用项目根目录，模型在intel子目录中
        sfm = SFM(project_root)
        
        print("开始处理视频...")
        # 运行SfM视频处理
        sfm.sfm_video(model, video_input)
        
        print("SfM视频处理完成!")
        
        # 检查输出文件
        output_csv = os.path.join(results_dir, "GazeTracking.csv")
        output_video = os.path.join(results_dir, "eye_features.mp4")
        
        if os.path.exists(output_csv):
            print(f"输出CSV文件已生成: {output_csv}")
        else:
            print("警告：未找到输出CSV文件")
            
        if os.path.exists(output_video):
            print(f"输出视频文件已生成: {output_video}")
        else:
            print("警告：未找到输出视频文件")
        
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sfm_video()
    if success:
        print("测试完成")
    else:
        print("测试失败")
        sys.exit(1)