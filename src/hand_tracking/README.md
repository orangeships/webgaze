# 手部关键点检测系统

基于MediaPipe实现的手部关键点检测、跟踪和手势识别系统。

## 功能特性

### 🎯 主要功能
- **实时手部检测**: 支持单手和双手实时检测
- **关键点跟踪**: 21个手部关键点精确定位
- **手势识别**: 支持多种预定义手势识别
- **运动分析**: 手部轨迹记录和运动模式分析
- **数据导出**: 支持JSON、CSV、TXT格式数据导出
- **多模式支持**: 实时摄像头、视频文件、静态图像处理

### 🤏 支持的手势
- **OK手势** - 拇指和食指形成圆圈
- **竖起大拇指** - 表示赞同
- **胜利手势** - 食指和中指竖起
- **数字手势** - 0到5的数字识别
- **其他手势**: 拳头、手掌、停止等

### 📊 分析功能
- **轨迹追踪**: 记录手部移动轨迹
- **速度分析**: 计算手部移动速度
- **稳定性评估**: 检测手部抖动程度
- **运动模式识别**: 分类移动、静止、快速等状态
- **热力图生成**: 可视化手部活动区域

## 文件结构

```
hand_tracking/
├── __init__.py                   # 模块初始化文件
├── hand_detector.py             # 手部检测器
├── hand_tracker.py              # 手部跟踪器
├── gesture_recognizer.py        # 手势识别器
├── hand_utils.py                # 实用工具函数
├── hand_tracking_system.py      # 主系统入口
├── demo.py                      # 演示脚本
└── README.md                    # 说明文档
```

## 快速开始

### 1. 基本使用

#### 实时检测
```python
from hand_tracking_system import HandTrackingSystem

# 创建系统实例
system = HandTrackingSystem()

# 启动实时跟踪
system.run_realtime()
```

#### 处理视频文件
```python
# 处理视频文件
system.process_video_file("input_video.mp4", "output_video.mp4")
```

#### 处理单张图像
```python
# 处理静态图像
system.process_image("hand_image.jpg")
```

### 2. 高级使用

#### 自定义参数
```python
system = HandTrackingSystem(
    max_num_hands=1,           # 最大检测手部数
    detection_confidence=0.8,  # 检测置信度
    tracking_confidence=0.6    # 跟踪置信度
)
```

#### 单独使用检测器
```python
from hand_detector import HandDetector

detector = HandDetector()
results = detector.detect_hands(frame)

for landmarks, handedness, bbox in results:
    # 处理检测结果
    pass
```

#### 单独使用手势识别
```python
from gesture_recognizer import GestureRecognizer

recognizer = GestureRecognizer()
gesture = recognizer.recognize_gesture(landmarks)
print(f"识别到手势: {gesture.value}")
```

## 演示程序

运行演示程序体验所有功能：

```bash
python demo.py
```

演示程序提供以下功能：
1. 实时手部跟踪
2. 手势识别演示
3. 视频文件处理
4. 图像文件处理
5. 数据导出功能
6. 运动分析

## 命令行使用

### 实时检测模式
```bash
python hand_tracking_system.py --mode realtime
```

### 视频处理模式
```bash
python hand_tracking_system.py --mode video --input video.mp4 --output processed.mp4
```

### 图像处理模式
```bash
python hand_tracking_system.py --mode image --input image.jpg
```

### 参数说明
- `--mode`: 运行模式 (realtime/video/image)
- `--input`: 输入文件路径
- `--output`: 输出文件路径
- `--camera`: 摄像头索引 (默认: 0)
- `--max_hands`: 最大检测手部数 (默认: 2)
- `--detection_conf`: 检测置信度 (默认: 0.7)
- `--tracking_conf`: 跟踪置信度 (默认: 0.5)

## 键盘控制

在实时模式下的键盘操作：
- `q` - 退出程序
- `r` - 重置跟踪
- `s` - 保存数据
- `h` - 显示帮助

## API文档

### HandTrackingSystem类

主要方法：
- `start_camera()` - 启动摄像头
- `process_frame()` - 处理单帧图像
- `run_realtime()` - 运行实时模式
- `process_video_file()` - 处理视频文件
- `process_image()` - 处理图像文件
- `get_statistics()` - 获取统计信息

### HandDetector类

主要方法：
- `detect_hands()` - 检测手部
- `draw_landmarks()` - 绘制关键点
- `get_hand_center()` - 获取手部中心点

### GestureRecognizer类

主要方法：
- `recognize_gesture()` - 识别手势
- `is_finger_extended()` - 检测手指是否伸展
- `calculate_gesture_confidence()` - 计算手势置信度

### HandUtils类

实用工具方法：
- `normalize_coordinates()` - 坐标归一化
- `calculate_distance()` - 计算距离
- `calculate_angle()` - 计算角度
- `export_landmarks_data()` - 导出数据
- `analyze_hand_movement()` - 运动分析

## 数据格式

### 导出的JSON数据格式
```json
{
  "frames": [
    {
      "timestamp": 1234567890.123,
      "hands": [
        {
          "handedness": "Right",
          "bbox": [100, 100, 200, 200],
          "landmarks": [
            [0.1, 0.2, 0.0],
            [0.15, 0.25, 0.0]
          ],
          "center": [0.125, 0.225]
        }
      ]
    }
  ]
}
```

### CSV数据格式
导出包含以下列：
- frame: 帧号
- hand_id: 手部ID
- gesture: 识别的手势
- landmark_i_x/y/z: 关键点坐标

## 性能优化

### 1. 分辨率调整
- 默认处理640x480分辨率
- 可调整摄像头分辨率提高性能
- 预览显示自动缩放减少延迟

### 2. 置信度设置
- 检测置信度: 0.5-0.9 (越高越严格)
- 跟踪置信度: 0.3-0.7 (影响跟踪稳定性)

### 3. 批量处理
- 视频处理支持批量模式
- 多线程处理提高效率

## 常见问题

### Q: 摄像头无法启动？
A: 检查摄像头权限，确认摄像头索引正确（通常为0）

### Q: 检测效果不佳？
A: 调整光线条件，保持手部清晰可见，适当调整置信度参数

### Q: 程序运行缓慢？
A: 降低处理分辨率，减少检测手部数量，关闭不必要的功能

### Q: 手势识别不准确？
A: 确保手势清晰完整，避免背景干扰，提高检测置信度

## 依赖要求

```
opencv-python >= 4.5.0
mediapipe >= 0.8.0
numpy >= 1.19.0
```

安装命令：
```bash
pip install opencv-python mediapipe numpy
```

## 更新日志

### v1.0.0
- 初始版本发布
- 基础手部检测功能
- 手势识别功能
- 数据导出功能
- 演示程序

## 许可证

本项目基于MIT许可证开源。

## 技术支持

如有问题或建议，请联系开发团队。

---

**注意**: 使用前请确保摄像头正常工作且环境光线充足，以获得最佳检测效果。