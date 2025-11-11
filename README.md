# WebCamGazeEstimation 项目文档

## 1. 项目概述

### 项目名称和版本号
**WebCamGazeEstimation** - 基于Webcam的视线追踪系统

### 项目的主要功能和目标
本项目是一个基于普通网络摄像头实现的实时视线追踪系统，主要功能包括：
- 实时检测和追踪人眼视线方向
- 将视线坐标映射到屏幕像素位置
- 支持4点校准系统，提高追踪精度
- 支持历史校准数据的保存和加载
- 集成OpenVINO AI推理引擎，实现高效性能

### 项目的基本架构图
```
摄像头输入 → 图像预处理 → AI模型推理 → 视线估计 → 坐标映射 → 屏幕输出
     ↓              ↓           ↓          ↓          ↓
 相机标定       图像增强      OpenVINO   校准系统   界面显示
```

## 2. 模块详细介绍

### 主要模块及其功能

#### 核心模块
- **`src/main.py`** - 主程序入口，协调各模块运行
- **`src/gaze_tracking/model.py`** - AI模型加载和推理模块
- **`src/gaze_tracking/homtransform.py`** - 视线到屏幕坐标的变换和校准
- **`src/gaze_tracking/gui_opencv.py`** - 用户界面和交互控制

#### 支持模块
- **`camera_data/main_camera_calibration.py`** - 相机内参标定
- **`utilities/utils.py`** - 通用工具函数
- **`sfm/sfm_module.py`** - 结构光运动分析

### 模块输入输出说明

#### model.py
- **输入**: 摄像头帧图像
- **输出**: 视线向量、眼球位置、头部姿态
- **关键参数**: 
  ```python
  gaze = [x, y, z]  # 视线方向向量
  EyeRLCenterPos = [右眼x, 右眼y, 左眼x, 左眼y]  # 眼球中心位置
  HeadPosAnglesYPR = [偏航角, 俯仰角, 滚转角]  # 头部姿态角度
  ```

#### homtransform.py
- **输入**: 视线向量、校准点数据
- **输出**: 屏幕坐标、变换矩阵
- **关键算法**: 齐次坐标变换、最小二乘拟合

#### gui_opencv.py
- **输入**: 用户交互、屏幕尺寸
- **输出**: 可视化界面、校准点位置

### 模块间依赖关系
```
main.py → homtransform.py → model.py
                ↓
           gui_opencv.py
                ↓
           camera_data/
```

### 关键算法实现原理

#### 视线估计
使用OpenVINO优化的深度学习模型，基于面部特征点检测视线方向。模型输入为面部图像，输出为3D视线向量。

#### 坐标映射
通过4点校准系统建立视线向量到屏幕坐标的映射关系：
```python
# 使用齐次坐标变换
STransG = 变换矩阵  # 4x4齐次变换矩阵
screen_coords = STransG @ gaze_vector
```

#### 相机标定
使用棋盘格标定法获取相机内参：
```python
camera_matrix = [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]]  # 相机内参矩阵
dist_coeffs = [k1, k2, p1, p2, k3]  # 畸变系数
```

## 3. 参数解释

### 配置文件参数

#### 校准参数 (`homtransform.py`)
- `total_pts = 4` - 校准点数量（默认4点）
- `time_per_point = 2.0` - 每个校准点停留时间（秒）
- `margin_ratio = 0.10` - 校准点边距比例

#### 模型参数 (`model.py`)
- `device = "AUTO"` - 推理设备（AUTO/GPU/CPU）
- `model_path = "intel/..."` - AI模型文件路径

#### 相机参数
- `frame_width = 1280` - 摄像头帧宽度
- `frame_height = 960` - 摄像头帧高度
- `CAP_PROP_AUTOFOCUS = 0` - 禁用自动对焦

### 可调整参数范围和默认值

| 参数 | 范围 | 默认值 | 影响 |
|------|------|--------|------|
| 校准点数 | 4-9 | 4 | 精度 vs 校准时间 |
| 帧率 | 15-30 fps | 自动 | 性能 vs 流畅度 |
| 图像尺寸 | 640x480 - 1920x1080 | 1280x960 | 精度 vs 性能 |

### 特殊参数注意事项
- **相机标定数据**: 必须预先通过 `main_camera_calibration.py` 生成
- **模型文件**: 需要下载Intel OpenVINO模型到 `intel/` 目录
- **屏幕尺寸**: 自动检测，但需要正确设置显示比例

## 4. 使用说明

### 启动服务
```bash
# 进入项目目录
cd "g:\mattest\Gaze estimation\WebCamGazeEstimation-main"

# 运行主程序
python src/main.py
```

### 操作流程
1. 程序启动后选择"加载历史校准数据"或"进行新校准"
2. 如选择新校准，按照屏幕提示注视4个校准点
3. 校准完成后自动进入视线追踪模式
4. 按ESC键退出程序

### 常见问题解决方法

#### 问题1: 模型加载失败
**解决方法**: 检查 `intel/` 目录下模型文件是否完整

#### 问题2: 摄像头无法打开
**解决方法**: 
- 检查摄像头连接
- 关闭其他占用摄像头的程序
- 尝试降低分辨率设置

#### 问题3: 校准精度差
**解决方法**:
- 确保环境光线充足
- 保持头部相对静止
- 重新运行相机标定程序

## 5. 开发指南

### 代码结构说明
```
WebCamGazeEstimation-main/
├── src/                    # 源代码目录
│   ├── main.py            # 主程序
│   ├── gaze_tracking/     # 视线追踪核心模块
│   └── utilities/         # 工具函数
├── camera_data/           # 相机标定数据和脚本
├── intel/                # AI模型文件
├── results/              # 运行结果输出
└── test/                 # 测试代码
```

### 扩展新模块的方法

#### 添加新的校准算法
1. 在 `homtransform.py` 中继承 `HomTransform` 类
2. 重写 `calibrate` 方法实现新算法
3. 在 `main.py` 中集成新的校准选项

#### 添加新的用户界面
1. 在 `gui_opencv.py` 基础上创建新界面类
2. 实现相应的交互逻辑
3. 在 `homtransform.py` 中更新界面调用

#### 集成新的AI模型
1. 将模型文件放入 `intel/` 目录
2. 在 `model.py` 中扩展 `EyeModel` 类
3. 更新模型输入输出处理逻辑

### 开发注意事项
- 使用OpenVINO进行模型推理以获得最佳性能
- 保持校准数据的向后兼容性
- 注意多线程环境下的资源竞争
- 遵循Python PEP8编码规范

---

**技术栈**: Python, OpenCV, OpenVINO, NumPy, SciPy  
**硬件要求**: 普通网络摄像头, 支持OpenVINO的CPU/GPU  
**适用场景**: 学术研究, 人机交互, 辅助技术
