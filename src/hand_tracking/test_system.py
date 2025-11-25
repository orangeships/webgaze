#!/usr/bin/env python3
"""
简单测试脚本用于验证手部跟踪系统是否正常工作
"""

import cv2
import numpy as np
from hand_tracking_system import HandTrackingSystem
from hand_detector import HandDetector
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer

def test_individual_components():
    """测试各个组件"""
    print("测试手部检测器...")
    detector = HandDetector()
    print("✓ 手部检测器初始化成功")
    
    print("测试手部跟踪器...")
    tracker = HandTracker()
    print("✓ 手部跟踪器初始化成功")
    
    print("测试手势识别器...")
    recognizer = GestureRecognizer()
    print("✓ 手势识别器初始化成功")
    
    return detector, tracker, recognizer

def test_system_initialization():
    """测试系统初始化"""
    print("测试完整系统初始化...")
    system = HandTrackingSystem()
    print("✓ 系统初始化成功")
    
    return system

def test_with_dummy_image():
    """使用虚拟图像测试系统"""
    print("使用虚拟图像测试系统...")
    
    # 创建一个虚拟的RGB图像
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些内容让图像看起来真实
    dummy_image[:] = (50, 100, 150)  # 浅蓝色背景
    cv2.rectangle(dummy_image, (200, 150), (440, 450), (0, 255, 0), 2)
    cv2.putText(dummy_image, "Test Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    system = HandTrackingSystem()
    
    # 测试处理单帧（可能检测不到手部，但应该不会出错）
    try:
        result = system.process_frame(dummy_image.copy())
        print("✓ 虚拟图像处理成功")
        print(f"  输出图像尺寸: {result.shape}")
        return True
    except Exception as e:
        print(f"✗ 虚拟图像处理失败: {e}")
        return False

def test_system_properties():
    """测试系统属性和方法"""
    print("测试系统属性...")
    
    system = HandTrackingSystem()
    
    # 测试属性访问
    try:
        max_hands = system.max_num_hands
        detection_conf = system.detection_confidence
        tracking_conf = system.tracking_confidence
        print(f"✓ 属性访问成功:")
        print(f"  最大手部数: {max_hands}")
        print(f"  检测置信度: {detection_conf}")
        print(f"  跟踪置信度: {tracking_conf}")
        return True
    except Exception as e:
        print(f"✗ 属性访问失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*50)
    print("手部跟踪系统测试")
    print("="*50)
    
    success_count = 0
    total_tests = 4
    
    try:
        # 测试1: 各个组件
        test_individual_components()
        success_count += 1
        print()
        
        # 测试2: 系统初始化
        test_system_initialization()
        success_count += 1
        print()
        
        # 测试3: 虚拟图像处理
        if test_with_dummy_image():
            success_count += 1
        print()
        
        # 测试4: 系统属性
        if test_system_properties():
            success_count += 1
        print()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    print("="*50)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！手部跟踪系统工作正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关组件。")
        return False

if __name__ == "__main__":
    main()