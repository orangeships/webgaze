"""
手部关键点检测系统 - 简化版

只包含手部关键点检测和实时可视化功能。
基于MediaPipe实现，支持实时手部关键点检测和可视化。

主要功能：
- 实时手部检测和关键点提取
- 关键点可视化
- 边界框绘制
- 简单的实时显示
- 食指指尖桌面接触检测

作者: AI Assistant
版本: 2.1.0 (增加食指指尖桌面检测功能)
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import os
from typing import Optional, List, Tuple
import time

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_detector import HandDetector


class SimpleHandDetectionSystem:
    """
    简化版手部关键点检测系统
    
    只包含手部检测和可视化功能，移除了所有其他复杂功能。
    """
    
    def __init__(self, camera_id: int = 0):
        """
        初始化手部检测系统
        
        Args:
            camera_id: 摄像头ID，默认为0
        """
        self.camera_id = camera_id
        self.cap = None
        self.hand_detector = HandDetector()
        
        # 关键点滤波器相关变量
        self.keypoint_history = []
        self.max_history = 3  # 减少历史帧数，让响应更灵敏
        self.smoothing_factor = 0.85  # 增加平滑因子，更信任当前帧
        
        print("手部关键点检测系统初始化完成")
    
    def start_camera(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        """
        启动摄像头
        
        Args:
            camera_index: 摄像头索引
            width: 图像宽度
            height: 图像高度
            
        Returns:
            bool: 启动是否成功
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 {camera_index}")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"摄像头启动成功 (索引: {camera_index}, 分辨率: {width}x{height})")
        return True
    
    def stop_camera(self):
        """停止摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
            print("摄像头已停止")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            np.ndarray: 处理后的帧
        """
        # 检测手部
        hand_results = self.hand_detector.detect_hands(frame)
        
        # 绘制检测结果
        for hand_info in hand_results:
            # 从hand_info字典中提取数据
            landmarks_2d = hand_info['landmarks_2d']
            center = hand_info['center']
            handedness = hand_info.get('handedness', 'Unknown')
            
            # 对关键点进行平滑处理
            smoothed_landmarks = self._smooth_landmarks(landmarks_2d)
            
            # 计算边界框
            if smoothed_landmarks:
                x_coords = [point[0] for point in smoothed_landmarks]
                y_coords = [point[1] for point in smoothed_landmarks]
                x_min, x_max = int(min(x_coords) * frame.shape[1]), int(max(x_coords) * frame.shape[1])
                y_min, y_max = int(min(y_coords) * frame.shape[0]), int(max(y_coords) * frame.shape[0])
                bbox = (x_min, y_min, x_max, y_max)
            else:
                bbox = None
            
            # 绘制边界框
            self._draw_bounding_box(frame, bbox, handedness)
            
            # 检测食指指尖是否接触桌面（使用平滑后的关键点）
            is_touching, debug_info = self._detect_finger_contact(smoothed_landmarks)
            
            # 绘制关键点（使用平滑后的关键点）
            self._draw_landmarks(frame, smoothed_landmarks)
            
            # 绘制调试信息（参考平面和距离）
            self._draw_debug_info(frame, smoothed_landmarks, debug_info)
            
            # 如果检测到接触，绘制特殊效果
            if is_touching:
                self._draw_contact_effect(frame, smoothed_landmarks, handedness)
        
        # 显示检测信息
        self._display_detection_info(frame, len(hand_results))
        
        return frame
    
    def _smooth_landmarks(self, landmarks_2d: List[Tuple]) -> List[Tuple]:
        """
        对关键点进行平滑处理
        
        Args:
            landmarks_2d: 原始关键点列表
            
        Returns:
            List[Tuple]: 平滑后的关键点列表
        """
        if not landmarks_2d:
            return landmarks_2d
        
        # 将当前关键点添加到历史记录
        self.keypoint_history.append(landmarks_2d)
        
        # 保持历史记录在指定长度内
        if len(self.keypoint_history) > self.max_history:
            self.keypoint_history.pop(0)
        
        # 如果历史记录不足，返回原始关键点
        if len(self.keypoint_history) < 2:
            return landmarks_2d
        
        # 使用指数移动平均进行平滑
        smoothed_landmarks = []
        num_landmarks = len(landmarks_2d)
        
        for i in range(num_landmarks):
            # 获取当前和历史的关键点坐标
            current_x, current_y = landmarks_2d[i]
            
            # 计算历史关键点的加权平均
            weighted_x = current_x * self.smoothing_factor
            weighted_y = current_y * self.smoothing_factor
            total_weight = self.smoothing_factor
            
            # 添加历史关键点的权重（越新的权重越大）
            for j, history_landmarks in enumerate(reversed(self.keypoint_history[:-1])):
                if i < len(history_landmarks):
                    hist_weight = (1.0 - self.smoothing_factor) * (0.8 ** j)  # 指数衰减权重
                    weighted_x += history_landmarks[i][0] * hist_weight
                    weighted_y += history_landmarks[i][1] * hist_weight
                    total_weight += hist_weight
            
            # 归一化
            if total_weight > 0:
                smoothed_x = weighted_x / total_weight
                smoothed_y = weighted_y / total_weight
            else:
                smoothed_x = current_x
                smoothed_y = current_y
            
            smoothed_landmarks.append((smoothed_x, smoothed_y))
        
        return smoothed_landmarks
    
    def _detect_finger_contact(self, landmarks_2d: Optional[List[Tuple]]) -> tuple[bool, dict]:
        """
        检测食指指尖是否接触桌面
        
        通过分析大拇指关键点(0,1,2,3,4)构建参考平面，
        判断8号点(食指指尖)是否与该平面共面
        
        Args:
            landmarks_2d: 2D关键点列表
            
        Returns:
            tuple: (bool, dict) - (是否接触, 调试信息)
        """
        if not landmarks_2d or len(landmarks_2d) < 21:
            print(f"[DEBUG] 关键点不足: {len(landmarks_2d) if landmarks_2d else 'None'}")
            return False, {"error": "关键点不足"}
        
        print(f"[DEBUG] 开始检测，关键点数量: {len(landmarks_2d)}")
        
        # 获取大拇指关键点 (0-4号点) - 使用原始坐标，不要放大
        thumb_points = []
        for i in range(5):  # 0, 1, 2, 3, 4
            x = landmarks_2d[i][0]  # 原始归一化坐标
            y = landmarks_2d[i][1]  # 原始归一化坐标
            z = i * 0.01  # 减小z坐标差异，基于点的顺序设置不同的深度
            thumb_points.append([x, y, z])
            print(f"[DEBUG] 大拇指点{i}: ({x:.4f}, {y:.4f}, {z:.4f})")
        
        # 获取食指指尖点 (8号点) - 使用原始坐标
        finger_tip = landmarks_2d[8]
        finger_x = finger_tip[0]  # 原始归一化坐标
        finger_y = finger_tip[1]  # 原始归一化坐标
        finger_z = 0.03  # 稍微调整z坐标
        
        print(f"[DEBUG] 食指指尖: ({finger_x:.4f}, {finger_y:.4f}, {finger_z:.4f})")
        
        # 使用大拇指三个点构建参考平面 (0, 2, 4)
        p1 = np.array(thumb_points[0])  # 手腕
        p2 = np.array(thumb_points[2])  # 大拇指中间
        p3 = np.array(thumb_points[4])  # 大拇指指尖
        
        print(f"[DEBUG] 参考平面点:")
        print(f"[DEBUG]   p1 (手腕): {p1}")
        print(f"[DEBUG]   p2 (拇指中): {p2}")
        print(f"[DEBUG]   p3 (拇指尖): {p3}")
        
        # 计算平面法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        print(f"[DEBUG] 向量 v1: {v1}")
        print(f"[DEBUG] 向量 v2: {v2}")
        print(f"[DEBUG] 法向量: {normal}")
        
        # 如果法向量过小，说明点共线，无法构成平面
        if np.linalg.norm(normal) < 1e-6:
            print(f"[DEBUG] 法向量过小: {np.linalg.norm(normal)}")
            return False, {"error": "参考点共线，无法构成平面"}
        
        # 归一化法向量
        normal = normal / np.linalg.norm(normal)
        
        # 计算食指指尖到平面的距离
        finger_point = np.array([finger_x, finger_y, finger_z])
        plane_point = p1
        
        # 点到平面的距离公式: |n·(p - p0)| / ||n||
        distance = abs(np.dot(normal, finger_point - plane_point)) / np.linalg.norm(normal)
        
        print(f"[DEBUG] 原始距离: {distance:.6f}")
        
        # 将距离转换为像素单位（基于图像尺寸）
        # 假设图像是1280x720，归一化坐标转换为像素
        distance_pixels = distance * 1280  # 使用图像宽度作为参考
        
        print(f"[DEBUG] 转换后距离: {distance_pixels:.6f} 像素")
        
        # 计算夹角：计算从平面中心到食指指尖的向量与平面法向量的夹角
        # 向量从平面中心到指尖
        finger_to_plane_vec = finger_point - plane_point
        
        # 计算指尖向量与平面法向量的夹角（弧度）
        # cos(angle_with_normal) = |n·v| / (||n|| * ||v||)
        cos_angle_with_normal = abs(np.dot(normal, finger_to_plane_vec)) / (np.linalg.norm(normal) * np.linalg.norm(finger_to_plane_vec))
        angle_with_normal_radians = np.arccos(np.clip(cos_angle_with_normal, -1.0, 1.0))
        
        # 指尖与平面的夹角 = 90° - 与法向量的夹角
        angle_with_plane_radians = (np.pi / 2) - angle_with_normal_radians
        angle_degrees = np.degrees(angle_with_plane_radians)
        
        print(f"[DEBUG] 与法向量夹角: {np.degrees(angle_with_normal_radians):.2f} 度")
        print(f"[DEBUG] 与平面夹角: {angle_degrees:.2f} 度")
        
        # 设置接触阈值（以角度为单位）
        contact_threshold_degrees = 10  # 10度夹角作为阈值
        
        # 额外检查：确保大拇指关键点形成的平面是合理的
        # 检查三个参考点的分散程度
        plane_area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        print(f"[DEBUG] 平面面积: {plane_area:.6f}")
        
        if plane_area < 0.0001:  # 降低面积阈值，允许更小的平面
            print(f"[DEBUG] 平面面积过小: {plane_area:.6f}")
            return False, {"error": f"参考平面面积过小: {plane_area:.6f}"}
        
        # 基于夹角判断接触
        is_contacting = angle_degrees < contact_threshold_degrees
        
        print(f"[DEBUG] 接触检测: {is_contacting} (夹角: {angle_degrees:.2f}° < 阈值: {contact_threshold_degrees:.2f}°)")
        
        # 准备调试信息
        debug_info = {
            "is_contacting": is_contacting,
            "angle_degrees": angle_degrees,  # 夹角（度）
            "threshold_degrees": contact_threshold_degrees,
            "distance_pixels": distance_pixels,  # 保留距离信息用于显示
            "plane_area": plane_area * 1000000,  # 放大面积显示
            "thumb_points_2d": [(p[0], p[1]) for p in thumb_points],  # 原始坐标
            "finger_tip_2d": (finger_tip[0], finger_tip[1]),
            "normal_vector": normal.tolist(),
            "p1": p1.tolist(),
            "p2": p2.tolist(), 
            "p3": p3.tolist()
        }
        
        return is_contacting, debug_info
    
    def _draw_contact_effect(self, frame: np.ndarray, landmarks_2d: Optional[List[Tuple]], handedness: str):
        """
        绘制食指指尖接触桌面的视觉效果
        
        Args:
            frame: 图像帧
            landmarks_2d: 2D关键点列表
            handedness: 手部类型 (左手/右手)
        """
        if not landmarks_2d or len(landmarks_2d) < 9:
            return
        
        # 获取8号点（食指指尖）的像素坐标
        finger_tip = landmarks_2d[8]
        finger_x = int(finger_tip[0] * frame.shape[1])
        finger_y = int(finger_tip[1] * frame.shape[0])
        
        # Draw contact effect: red circle and text
        cv2.circle(frame, (finger_x, finger_y), 15, (0, 0, 255), 3)
        cv2.circle(frame, (finger_x, finger_y), 20, (0, 0, 255), 1)
        
        # Display contact text
        cv2.putText(frame, "Touching!", (finger_x + 25, finger_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def _draw_debug_info(self, frame: np.ndarray, landmarks_2d: Optional[List[Tuple]], debug_info: dict):
        """
        绘制调试信息：参考平面和距离
        
        Args:
            frame: 图像帧
            landmarks_2d: 2D关键点列表
            debug_info: 调试信息字典
        """
        if not landmarks_2d or not debug_info or "error" in debug_info:
            return
        
        # 绘制参考平面（大拇指0,2,4号点）
        try:
            thumb_points_2d = debug_info.get("thumb_points_2d", [])
            if len(thumb_points_2d) >= 3:
                # 转换为像素坐标
                p1_px = (int(thumb_points_2d[0][0] * frame.shape[1]), 
                        int(thumb_points_2d[0][1] * frame.shape[0]))
                p2_px = (int(thumb_points_2d[2][0] * frame.shape[1]), 
                        int(thumb_points_2d[2][1] * frame.shape[0]))
                p3_px = (int(thumb_points_2d[4][0] * frame.shape[1]), 
                        int(thumb_points_2d[4][1] * frame.shape[0]))
                
                # 绘制参考平面三角形（半透明绿色）
                pts = np.array([p1_px, p2_px, p3_px], np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0, 50))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # 绘制参考平面边界
                cv2.line(frame, p1_px, p2_px, (0, 255, 0), 2)
                cv2.line(frame, p2_px, p3_px, (0, 255, 0), 2)
                cv2.line(frame, p3_px, p1_px, (0, 255, 0), 2)
                
                # 标记参考点
                for i, point in enumerate([p1_px, p2_px, p3_px]):
                    cv2.circle(frame, point, 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"P{i+1}", (point[0] + 10, point[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制食指指尖到平面的距离线
            finger_tip = debug_info.get("finger_tip_2d", None)
            if finger_tip:
                finger_px = (int(finger_tip[0] * frame.shape[1]), 
                           int(finger_tip[1] * frame.shape[0]))
                
                # 计算在平面上的投影点（简化：使用p1点）
                p1_px = (int(thumb_points_2d[0][0] * frame.shape[1]), 
                        int(thumb_points_2d[0][1] * frame.shape[0]))
                
                # 绘制从食指指尖到参考平面的连线
                cv2.line(frame, finger_px, p1_px, (255, 0, 0), 2)
                
                # 标记食指指尖
                cv2.circle(frame, finger_px, 6, (255, 0, 0), -1)
                cv2.putText(frame, "Finger Tip", (finger_px[0] + 10, finger_px[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Display debug information in top-right corner
            debug_text = [
                 f"Angle: {debug_info.get('angle_degrees', 0):.1f}°",
                 f"Threshold: {debug_info.get('threshold_degrees', 10.0):.1f}°", 
                 f"Status: {'CONTACT' if debug_info.get('is_contacting', False) else 'NO CONTACT'}",
                 f"Plane Area: {debug_info.get('plane_area', 0):.1f}"
             ]
            
            # 创建半透明的背景
            overlay = frame.copy()
            for i, text in enumerate(debug_text):
                y_pos = 60 + i * 25
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 绘制文本背景
                cv2.rectangle(overlay, 
                            (frame.shape[1] - text_size[0] - 20, y_pos - 20),
                            (frame.shape[1] - 10, y_pos + 10), 
                            (0, 0, 0), -1)
                
                # 绘制文本
                cv2.putText(overlay, text, 
                           (frame.shape[1] - text_size[0] - 15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 混合背景到原图
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
        except Exception as e:
            # 如果绘制出错，在角落显示错误信息
            cv2.putText(frame, f"Debug draw error: {str(e)[:30]}", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    def _draw_bounding_box(self, frame: np.ndarray, bbox: Optional[Tuple], handedness: str):
        """绘制边界框"""
        if bbox:
            x1, y1, x2, y2 = bbox
            # 左手用绿色，右手用蓝色
            color = (0, 255, 0) if "Left" in handedness else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, handedness, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks_2d: Optional[List[Tuple]]):
        """绘制手部关键点"""
        if not landmarks_2d:
            return
        
        # MediaPipe手部关键点连接
        # 定义关键点之间的连接关系
        connections = [
            # 大拇指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (5, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (9, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (13, 17), (17, 18), (18, 19), (19, 20),
            # 手掌连接
            (5, 9), (9, 13), (13, 17)
        ]
        
        # 绘制连接线
        for connection in connections:
            if connection[0] < len(landmarks_2d) and connection[1] < len(landmarks_2d):
                point1 = (int(landmarks_2d[connection[0]][0] * frame.shape[1]),
                         int(landmarks_2d[connection[0]][1] * frame.shape[0]))
                point2 = (int(landmarks_2d[connection[1]][0] * frame.shape[1]),
                         int(landmarks_2d[connection[1]][1] * frame.shape[0]))
                cv2.line(frame, point1, point2, (0, 255, 255), 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(landmarks_2d):
            point = (int(x * frame.shape[1]), int(y * frame.shape[0]))
            # 手腕点用红色，其他点用黄色
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.circle(frame, point, 3, color, -1)
            cv2.circle(frame, point, 3, (0, 0, 0), 1)
    
    def _display_detection_info(self, frame: np.ndarray, hand_count: int):
        """显示检测信息"""
        info_text = [
            f"检测到手部: {hand_count}",
            "按 'q' 退出程序"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def run_realtime(self, camera_index: int = 0):
        """运行实时手部关键点检测"""
        if not self.start_camera(camera_index):
            return
        
        self.is_running = True
        
        print("\n=== 手部关键点检测系统 (简化版) ===")
        print("功能:")
        print("- 实时手部关键点检测")
        print("- 关键点和边界框可视化")
        print("控制:")
        print("  'q' - 退出程序")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 显示帧
            cv2.imshow('手部关键点检测系统 (简化版)', processed_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.is_running = False
        self.stop_camera()
        cv2.destroyAllWindows()
        print("系统已清理")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手部关键点检测系统 (简化版)')
    parser.add_argument('--camera', type=int, default=1, help='摄像头索引')
    parser.add_argument('--max_hands', type=int, default=2, help='最大检测手部数')
    parser.add_argument('--detection_conf', type=float, default=0.7, help='检测置信度')
    
    args = parser.parse_args()
    
    # 创建系统实例
    hand_system = SimpleHandDetectionSystem(camera_id=args.camera)
    
    try:
        print("启动实时手部关键点检测模式")
        hand_system.run_realtime(args.camera)
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序错误: {e}")
    finally:
        hand_system.cleanup()


        return smoothed_landmarks


if __name__ == "__main__":
    main()