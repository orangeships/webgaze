#!/usr/bin/env python3
"""
修复版8x5棋盘格校准程序
解决常见的棋盘格检测问题
"""

import numpy as np
import cv2
import os
from datetime import datetime

def create_8x5_chessboard():
    """生成8x5棋盘格图案"""
    pattern_size = (8, 5)
    square_size = 22.5  # mm
    
    # 创建图案
    dpi = 300
    width, height = 210, 297  # A4 size in mm
    width_px = int(width * dpi / 25.4)
    height_px = int(height * dpi / 25.4)
    
    # 创建白色背景
    img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    
    # 计算方格大小
    square_w = width_px // pattern_size[0]
    square_h = height_px // pattern_size[1]
    square = min(square_w, square_h)
    
    # 居中
    offset_x = (width_px - square * pattern_size[0]) // 2
    offset_y = (height_px - square * pattern_size[1]) // 2
    
    # 绘制棋盘格
    for i in range(pattern_size[1]):
        for j in range(pattern_size[0]):
            if (i + j) % 2 == 0:
                x = offset_x + j * square
                y = offset_y + i * square
                cv2.rectangle(img, (x, y), (x + square, y + square), (0, 0, 0), -1)
    
    # 保存图案
    cv2.imwrite("chessboard_8x5_A4.png", img)
    print("8x5棋盘格图案已生成: chessboard_8x5_A4.png")

def find_working_camera():
    """寻找可用的相机"""
    print("寻找可用的相机...")
    
    # 尝试多个相机索引
    for camera_idx in range(1,3):
        print(f"尝试相机 {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            # 尝试读取一帧
            ret, frame = cap.read()
            if ret:
                print(f"✓ 相机 {camera_idx} 可用 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()
                return camera_idx
            else:
                print(f"✗ 相机 {camera_idx} 无法读取")
        else:
            print(f"✗ 相机 {camera_idx} 无法打开")
        cap.release()
    
    print("没有找到可用的相机")
    return None

def improve_image_quality(gray):
    """改进图像质量以提高棋盘格检测"""
    # 1. 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 2. 高斯模糊减少噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 增强对比度
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    return gray

def calibrate_camera_fixed():
    """修复版相机校准"""
    # 校准参数
    pattern_size = (8, 5)
    square_size = 22.5
    min_samples = 15
    
    # 寻找可用相机
    camera_idx = find_working_camera()
    if camera_idx is None:
        print("无法找到可用的相机，程序退出")
        return False
    
    # 启动相机
    cap = cv2.VideoCapture(camera_idx)
    
    # 设置合理的分辨率
    print("设置相机参数...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"相机实际分辨率: {actual_width}x{actual_height}")
    
    # 创建对象点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 存储点和图像点
    obj_points = []
    img_points = []
    
    # 检测参数
    detection_params = {
        'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    }
    
    count = 0
    last_detection_frame = 0
    
    print(f"\n开始修复版8x5棋盘格校准...")
    print("操作说明:")
    print("  按空格键保存当前视角")
    print("  按't'键显示当前图像信息")
    print("  按'f'键尝试不同的检测参数")
    print("  按'q'键退出并开始校准")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机帧")
            break
            
        # 转换到灰度图并改进质量
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = improve_image_quality(gray)
        
        # 尝试棋盘格检测
        try:
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None, **detection_params)
        except Exception as e:
            print(f"检测时出错: {e}")
            ret = False
        
        if ret:
            # 亚像素精确化
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            last_detection_frame = count
            
            # 绘制角点
            cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
            
            # 计算角点质量
            corners_flat = corners.reshape(-1, 2)
            distances = []
            for i in range(len(corners_flat)-1):
                dist = np.linalg.norm(corners_flat[i+1] - corners_flat[i])
                distances.append(dist)
            avg_distance = np.mean(distances)
            
            # 显示状态信息
            cv2.putText(frame, f"检测成功! 样本: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"角点间距: {avg_distance:.1f}像素", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "按空格键保存", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 颜色编码角点质量
            if avg_distance > 30 and avg_distance < 200:
                cv2.putText(frame, "质量: 优秀", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif avg_distance > 20:
                cv2.putText(frame, "质量: 良好", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "质量: 较差", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 显示失败状态
            cv2.putText(frame, "未检测到棋盘格", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "请确保棋盘格完整可见且有足够光照", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 提供调试信息
            if count > 0 and count - last_detection_frame > 10:
                cv2.putText(frame, "尝试调整距离或角度", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 显示基本信息和进度
        progress_text = f"进度: {count}/{min_samples}+ | 相机: {camera_idx}"
        cv2.putText(frame, progress_text, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "修复版8x5棋盘格校准", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Fixed Chessboard Calibration', frame)
        
        key = cv2.waitKey(60) & 0xFF
        
        if key == ord(' '):  # 空格保存
            if ret:
                count += 1
                obj_points.append(objp.copy())
                img_points.append(corners.copy())
                print(f"✓ 已保存样本 {count}/{min_samples}+")
                
                if count >= min_samples:
                    print(f"达到最小样本数，继续收集更多高质量样本...")
                    # 继续收集更多样本直到用户按q
            else:
                print("✗ 未检测到棋盘格，无法保存")
                
        elif key == ord('t'):  # 显示图像信息
            print(f"\n当前状态:")
            print(f"  样本数: {count}")
            print(f"  图像尺寸: {gray.shape}")
            print(f"  亮度范围: {np.min(gray)} - {np.max(gray)}")
            print(f"  平均亮度: {np.mean(gray):.1f}")
            print(f"  棋盘格检测: {'成功' if ret else '失败'}")
            if ret:
                print(f"  角点数量: {len(corners[0])}")
                
        elif key == ord('f'):  # 切换检测参数
            if 'flags' in detection_params:
                if detection_params['flags'] & cv2.CALIB_CB_ADAPTIVE_THRESH:
                    detection_params['flags'] = cv2.CALIB_CB_NORMALIZE_IMAGE
                    print("切换到标准化图像模式")
                else:
                    detection_params['flags'] = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    print("切换到自适应阈值模式")
            else:
                detection_params['flags'] = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                
        elif key == ord('q'):  # 退出
            if count >= min_samples:
                print(f"收集了 {count} 个样本，开始校准...")
                break
            else:
                print(f"样本数量不足，当前: {count}，至少需要: {min_samples}")
        
        count += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    if len(img_points) < min_samples:
        print(f"样本数量不足 ({len(img_points)})，校准终止")
        return False
    
    # 执行校准
    print(f"\n开始校准，使用 {len(img_points)} 个样本...")
    
    gray_shape = gray.shape[::-1]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray_shape, None, None)
    
    if not ret:
        print("校准失败")
        return False
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(obj_points)):
        img_points_transformed, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], 
                                                     camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_transformed, cv2.NORM_L2) / len(img_points_transformed)
        total_error += error
    
    mean_error = total_error / len(obj_points)
    print(f"平均重投影误差: {mean_error:.4f} 像素")
    
    # 评估校准质量
    if mean_error < 0.5:
        print("✓ 校准质量: 优秀")
    elif mean_error < 1.0:
        print("✓ 校准质量: 良好")
    else:
        print("⚠ 校准质量: 较差，建议重新校准")
    
    # 保存校准数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("camera_data", exist_ok=True)
    
    # 保存为txt格式
    output_file = f"camera_data/calibration_data_fixed_{timestamp}.txt"
    with open(output_file, "w") as f:
        f.write('camera_matrix:\n')
        np.savetxt(f, camera_matrix, fmt='%f')
        
        f.write('dist_coeffs:\n')
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt='%f')
        
        f.write('rvecs:\n')
        for rvec in rvecs:
            np.savetxt(f, rvec.reshape(1, -1), fmt='%f')
        
        f.write('tvecs:\n')
        for tvec in tvecs:
            np.savetxt(f, tvec.reshape(1, -1), fmt='%f')
    
    # 同时更新原始文件
    original_file = "camera_data/calibration_data.txt"
    with open(original_file, "w") as f:
        f.write('camera_matrix:\n')
        np.savetxt(f, camera_matrix, fmt='%f')
        
        f.write('dist_coeffs:\n')
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt='%f')
        
        f.write('rvecs:\n')
        for rvec in rvecs:
            np.savetxt(f, rvec.reshape(1, -1), fmt='%f')
        
        f.write('tvecs:\n')
        for tvec in tvecs:
            np.savetxt(f, tvec.reshape(1, -1), fmt='%f')
    
    print(f"\n=== 校准完成 ===")
    print(f"校准数据已保存到:")
    print(f"  {output_file}")
    print(f"  {original_file}")
    print(f"\n校准统计:")
    print(f"  样本数量: {len(img_points)}")
    print(f"  重投影误差: {mean_error:.4f} 像素")
    print(f"  相机分辨率: {gray_shape[0]}x{gray_shape[1]}")
    print(f"  使用的相机: {camera_idx}")
    
    return True

def main():
    """主函数"""
    print("=== 修复版8x5棋盘格校准程序 ===\n")
    
    # 1. 生成棋盘格图案
    print("1. 生成棋盘格图案...")
    create_8x5_chessboard()
    print("\n请打印并使用 chessboard_8x5_A4.png")
    input("\n准备好后按Enter键开始校准...")
    
    # 2. 执行修复版校准
    print("\n2. 开始修复版校准...")
    success = calibrate_camera_fixed()
    
    if success:
        print("\n✓ 校准成功完成!")
        print("\n接下来您可以:")
        print("  1. 使用生成的calibration_data.txt文件")
        print("  2. 尝试运行其他校准程序验证结果")
        print("  3. 根据需要进行进一步调优")
    else:
        print("\n✗ 校准失败!")
        print("\n建议:")
        print("  1. 检查棋盘格是否打印清晰")
        print("  2. 调整光照条件")
        print("  3. 尝试不同的距离和角度")
        print("  4. 使用调试版本进一步诊断")

if __name__ == "__main__":
    main()