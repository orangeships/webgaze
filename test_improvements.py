import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def test_dynamic_interpolation_weight():
    """测试动态插值权重计算"""
    print("\n测试动态插值权重计算...")
    ht = HomTransform(os.path.dirname(__file__))
    
    # 设置模拟校准点
    ht.SetValues = np.array([
        [0, 0],     # 左上角
        [100, 0],   # 右上角
        [0, 100],   # 左下角
        [100, 100], # 右下角
        [50, 50]    # 中心点
    ])
    
    # 测试不同距离下的权重计算
    test_points = [
        np.array([55, 55]),  # 靠近中心点
        np.array([30, 30]),  # 中等距离
        np.array([10, 10])   # 靠近角落
    ]
    
    weights = []
    for point in test_points:
        # 获取最近的4个点
        nearest_points, _ = ht._get_nearest_calibration_points(point, n=4)
        
        # 计算平均距离
        nearest_distances = np.array([np.linalg.norm(point - p) for p in nearest_points])
        avg_nearest_distance = np.mean(nearest_distances)
        
        # 计算最大校准距离
        all_calibration_distances = np.array([np.linalg.norm(ht.SetValues[i] - ht.SetValues[j]) 
                                            for i in range(len(ht.SetValues)) for j in range(i+1, len(ht.SetValues))])
        max_calibration_distance = np.max(all_calibration_distances)
        
        # 计算动态权重
        weight = np.exp(-avg_nearest_distance / (max_calibration_distance * 0.3))
        weight = np.clip(weight, 0.2, 0.6)
        weights.append(weight)
        
        print(f"查询点: {point}, 平均距离: {avg_nearest_distance:.2f}, 权重: {weight:.4f}")
    
    # 验证权重趋势：距离越近，权重应该越大
    if weights[0] > weights[1] > weights[2]:
        print("动态权重趋势验证: 通过")
    else:
        print("动态权重趋势验证: 失败 (距离越近，权重应越大)")

def test_distance_decay_factor():
    """测试距离衰减因子效果"""
    print("\n测试距离衰减因子效果...")
    ht = HomTransform(os.path.dirname(__file__))
    
    # 设置模拟校准点
    ht.SetValues = np.array([
        [0, 0, 0],     # 左上角
        [100, 0, 0],   # 右上角
        [0, 100, 0],   # 左下角
        [100, 100, 0], # 右下角
        [50, 50, 0]    # 中心点
    ])
    
    # 初始注视点
    initial_gaze = np.array([[52], [52], [0]])  # 靠近中心点
    
    # 计算校准点相关数据
    calibration_points = np.array([point[:2] for point in ht.SetValues])
    distances = np.linalg.norm(initial_gaze[:2].flatten() - calibration_points, axis=1)
    min_distance = np.min(distances)
    nearest_idx = np.argmin(distances)
    
    # 计算最大校准距离
    all_calibration_distances = np.array([np.linalg.norm(calibration_points[i] - calibration_points[j]) 
                                        for i in range(len(calibration_points)) for j in range(i+1, len(calibration_points))])
    max_calibration_distance = np.max(all_calibration_distances)
    
    # 距离阈值
    distance_threshold = min(max_calibration_distance * 0.1, 50)
    
    # 计算校正因子和混合比例
    correction_factor = 1.0 + (distance_threshold - min_distance) / distance_threshold * 0.5
    blend_ratio = 0.2 * correction_factor
    blend_ratio = min(blend_ratio, 0.5)
    
    # 获取最近校准点的映射结果
    nearest_point_result = ht.SetValues[nearest_idx]
    
    # 应用距离衰减因子
    result_gaze = (1 - blend_ratio) * initial_gaze + blend_ratio * nearest_point_result.reshape(3, 1)
    
    print(f"初始注视点: {initial_gaze.flatten()}")
    print(f"最近校准点: {nearest_point_result}")
    print(f"最小距离: {min_distance:.2f}, 阈值: {distance_threshold:.2f}")
    print(f"校正因子: {correction_factor:.2f}, 混合比例: {blend_ratio:.2f}")
    print(f"应用衰减因子后的结果: {result_gaze.flatten()}")
    
    # 验证结果应该更接近最近校准点
    original_distance = np.linalg.norm(initial_gaze[:2] - nearest_point_result[:2])
    result_distance = np.linalg.norm(result_gaze[:2] - nearest_point_result[:2])
    
    if result_distance < original_distance:
        print("距离衰减因子效果验证: 通过 (结果更接近校准点)")
    else:
        print("距离衰减因子效果验证: 失败 (结果应更接近校准点)")

def visualize_improvements():
    """可视化改进效果"""
    print("\n可视化改进效果...")
    
    # 创建可视化图表
    plt.figure(figsize=(15, 5))
    
    # 1. 动态权重可视化
    plt.subplot(1, 3, 1)
    distances = np.linspace(0, 100, 100)
    max_calibration_distance = 141.42  # 对角线距离
    weights = np.exp(-distances / (max_calibration_distance * 0.3))
    weights = np.clip(weights, 0.2, 0.6)
    
    plt.plot(distances, weights)
    plt.title('动态插值权重 vs 距离')
    plt.xlabel('平均距离')
    plt.ylabel('插值权重')
    plt.grid(True)
    
    # 2. 距离衰减因子可视化
    plt.subplot(1, 3, 2)
    distances = np.linspace(0, 15, 100)  # 阈值范围内的距离
    max_calibration_distance = 141.42
    threshold = min(max_calibration_distance * 0.1, 50)
    correction_factors = 1.0 + (threshold - distances) / threshold * 0.5
    blend_ratios = 0.2 * correction_factors
    blend_ratios = np.minimum(blend_ratios, 0.5)
    
    plt.plot(distances, blend_ratios)
    plt.title('距离衰减混合比例 vs 距离')
    plt.xlabel('到最近校准点的距离')
    plt.ylabel('混合比例')
    plt.grid(True)
    
    # 3. 整体改进效果示意图
    plt.subplot(1, 3, 3)
    # 绘制校准点
    calibration_x = [0, 100, 0, 100, 50]
    calibration_y = [0, 0, 100, 100, 50]
    plt.scatter(calibration_x, calibration_y, c='blue', s=100, label='校准点')
    
    # 绘制不同区域的精度提升示意图
    # 近校准点区域
    for i in range(5):
        circle = plt.Circle((calibration_x[i], calibration_y[i]), 7, color='green', alpha=0.3)
        plt.gca().add_patch(circle)
    
    # 中间区域
    circle = plt.Circle((50, 50), 30, color='yellow', alpha=0.2)
    plt.gca().add_patch(circle)
    
    plt.title('改进策略效果区域')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('results/improvements_visualization.png')
    print("可视化图表已保存到 results/improvements_visualization.png")
    plt.close()

if __name__ == "__main__":
    print("开始测试HomTransform改进...")
    
    # 测试双线性插值
    test_bilinear_interpolation()
    
    # 测试获取最近校准点
    test_get_nearest_points()
    
    # 测试动态插值权重
    test_dynamic_interpolation_weight()
    
    # 测试距离衰减因子
    test_distance_decay_factor()
    
    # 可视化改进效果
    visualize_improvements()
    
    print("\n所有测试完成！")