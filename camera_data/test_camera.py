import cv2
import numpy as np
import glob
import os
from datetime import datetime

def camera_calibration():
    # 棋盘格尺寸 (内部角点数量)
    chessboard_size = (7, 4)  # 8x5 棋盘格有 7x4 个内部角点
    
    # 终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ....,(6,3,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # 存储对象点和图像点的数组
    objpoints = []  # 真实世界中的3D点
    imgpoints = []  # 图像中的2D点
    
    # 初始化摄像头
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    print("摄像头校准程序")
    print("请准备一个8x5的棋盘格（7x4内部角点）")
    print("按空格键捕获图像，按'q'键退出")
    print(f"需要捕获 {40} 张有效图像")
    
    captured_count = 0
    window_name = "Camera Calibration - Press SPACE to capture, Q to quit"
    
    while captured_count < 40:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取摄像头画面")
            break
        
        # 复制帧用于显示
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 尝试查找棋盘格角点
        ret_find, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret_find:
            # 精确定位角点
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 在图像上绘制角点
            cv2.drawChessboardCorners(display_frame, chessboard_size, corners_refined, ret_find)
            
            # 显示提示信息
            cv2.putText(display_frame, f"棋盘格已检测到! 按空格键捕获 ({captured_count}/40)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "未检测到棋盘格", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, "按空格键捕获图像，按'q'键退出", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键
            if ret_find:
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                captured_count += 1
                print(f"成功捕获第 {captured_count} 张图像")
                
                # 短暂显示成功信息
                success_frame = frame.copy()
                cv2.putText(success_frame, "图像已保存!", 
                           (frame.shape[1]//2-100, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, success_frame)
                cv2.waitKey(500)
            else:
                print("未检测到棋盘格，请重新尝试")
        
        elif key == ord('q'):  # q键退出
            print("用户中断程序")
            break
    
    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    
    if len(objpoints) < 10:
        print(f"错误: 捕获的有效图像数量不足 ({len(objpoints)})，至少需要10张")
        return
    
    print(f"开始校准... 使用 {len(objpoints)} 张图像")
    
    # 进行摄像头校准
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if ret:
        print("校准成功!")
        save_calibration_results(camera_matrix, dist_coeffs, rvecs, tvecs)
        
        # 计算重投影误差
        mean_error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
        print(f"平均重投影误差: {mean_error:.3f} 像素")
        
        # 显示校准结果
        display_calibration_results(camera_matrix, dist_coeffs)
    else:
        print("校准失败!")

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    total_points = 0
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        total_points += 1
    
    return total_error / total_points

def save_calibration_results(camera_matrix, dist_coeffs, rvecs, tvecs):
    """保存校准结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"camera_calibration_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        # 写入相机矩阵
        f.write("camera_matrix:\n")
        for i in range(3):
            f.write(f"{camera_matrix[i, 0]:.6f} {camera_matrix[i, 1]:.6f} {camera_matrix[i, 2]:.6f}\n")
        
        # 写入畸变系数
        f.write("dist_coeffs:\n")
        dist_str = " ".join(f"{coeff:.6f}" for coeff in dist_coeffs[0])
        f.write(f"{dist_str}\n")
        
        # 写入旋转向量
        f.write("rvecs:\n")
        for rvec in rvecs:
            f.write(f"{rvec[0,0]:.6f} {rvec[1,0]:.6f} {rvec[2,0]:.6f}\n")
        
        # 写入平移向量
        f.write("tvecs:\n")
        for tvec in tvecs:
            f.write(f"{tvec[0,0]:.6f} {tvec[1,0]:.6f} {tvec[2,0]:.6f}\n")
    
    print(f"校准结果已保存到: {filename}")
    
    # 同时保存为YAML格式供OpenCV使用
    yaml_filename = f"camera_calibration_{timestamp}.yml"
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.release()
    print(f"YAML格式配置文件已保存到: {yaml_filename}")

def display_calibration_results(camera_matrix, dist_coeffs):
    """显示校准结果"""
    print("\n" + "="*50)
    print("相机校准结果")
    print("="*50)
    
    print("相机内参矩阵 (camera_matrix):")
    print(f"{camera_matrix[0,0]:.6f} {camera_matrix[0,1]:.6f} {camera_matrix[0,2]:.6f}")
    print(f"{camera_matrix[1,0]:.6f} {camera_matrix[1,1]:.6f} {camera_matrix[1,2]:.6f}")
    print(f"{camera_matrix[2,0]:.6f} {camera_matrix[2,1]:.6f} {camera_matrix[2,2]:.6f}")
    
    print("\n畸变系数 (dist_coeffs):")
    dist_str = " ".join(f"{coeff:.6f}" for coeff in dist_coeffs[0])
    print(dist_str)

def test_undistortion():
    """测试去畸变效果"""
    # 加载校准参数
    calibration_files = glob.glob("camera_calibration_*.yml")
    if not calibration_files:
        print("未找到校准文件")
        return
    
    latest_file = max(calibration_files, key=os.path.getctime)
    
    fs = cv2.FileStorage(latest_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    
    # 测试摄像头
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 去畸变
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # 裁剪图像
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        
        # 调整图像尺寸以匹配高度（取较小的高度）
        frame_height, frame_width = frame.shape[:2]
        undistorted_height, undistorted_width = undistorted.shape[:2]
        
        # 找到最小高度，确保两个图像高度一致
        min_height = min(frame_height, undistorted_height)
        
        # 裁剪到相同高度
        if frame_height > min_height:
            frame_resized = frame[:min_height, :]
        else:
            frame_resized = frame
            
        if undistorted_height > min_height:
            undistorted_resized = undistorted[:min_height, :]
        else:
            undistorted_resized = undistorted
        
        # 并排显示
        combined = np.hstack((frame_resized, undistorted_resized))
        cv2.putText(combined, "原始图像", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "去畸变图像", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("去畸变效果 - 按任意键退出", combined)
        
        if cv2.waitKey(1) & 0xFF != 255:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("选择操作:")
    print("1. 进行摄像头校准")
    print("2. 测试去畸变效果")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        camera_calibration()
        
        # 询问是否测试去畸变效果
        test = input("\n是否测试去畸变效果? (y/n): ").strip().lower()
        if test == 'y':
            test_undistortion()
    elif choice == "2":
        test_undistortion()
    else:
        print("无效选择")