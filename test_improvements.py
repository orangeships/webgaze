import os
import sys
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gaze_tracking.homtransform import HomTransform

def test_bilinear_interpolation():
    """测试双线性插值功能"""
    print("\n测试双线性插值功能...")
    # 创建一个临时的HomTransform实例
    ht = HomTransform(os.path.dirname(__file__))
    
    # 测试数据
    point = np.array([50, 50])  # 查询点
    
    # 四个角点
    points = np.array([
        [0, 0],    # 左上
        [100, 0],  # 右上
        [0, 100],  # 左下
        [100, 100] # 右下
    ])
    
    # 对应的映射值
    values = [
        np.array([[0], [0], [0]]),    # 左上
        np.array([[100], [0], [0]]),  # 右上
        np.array([[0], [100], [0]]),  # 左下
        np.array([[100], [100], [0]]) # 右下
    ]
    
    # 执行插值
    result = ht._bilinear_interpolation(point, points, values)
    
    # 预期结果应该是[50, 50, 0]
    expected = np.array([[50], [50], [0]])
    
    print(f"插值结果: {result.flatten()}")
    print(f"预期结果: {expected.flatten()}")
    print(f"测试{'通过' if np.allclose(result, expected, atol=1e-6) else '失败'}")

def test_get_nearest_points():
    """测试获取最近校准点功能"""
    print("\n测试获取最近校准点功能...")
    ht = HomTransform(os.path.dirname(__file__))
    
    # 设置一些模拟的校准点
    ht.SetValues = np.array([
        [0, 0],     # 左上角
        [100, 0],   # 右上角
        [0, 100],   # 左下角
        [100, 100], # 右下角
        [50, 50]    # 中心点
    ])
    
    # 查询点
    point = np.array([60, 60])
    
    # 获取最近的4个点
    nearest_points, nearest_indices = ht._get_nearest_calibration_points(point, n=4)
    
    print(f"查询点: {point}")
    print(f"最近的4个校准点: {nearest_points}")
    print(f"最近点索引: {nearest_indices}")
    print("测试完成")

if __name__ == "__main__":
    print("开始测试HomTransform改进...")
    
    # 测试双线性插值
    test_bilinear_interpolation()
    
    # 测试获取最近校准点
    test_get_nearest_points()
    
    print("\n所有测试完成！")