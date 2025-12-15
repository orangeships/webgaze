请你使用全局鼠标钩子WH_MOUSE_LL拦截双屏交互中触发鼠标跳转的WM_RBUTTONUP事件，注意使用ctypes

GetAsyncKeyState(), WM_CANCELMODE, 模拟鼠标，eventFilter 等等都不够早，会被系统抢先处理。 

请你只要简单实现它就行，千万不要搞的太复杂，最好在80行以内完成，不用什么降级操作，下面是示范：
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
WH_MOUSE_LL = 14
WM_RBUTTONUP = 0x205

mouse_hook = None

def low_level_mouse_proc(nCode, wParam, lParam):
    if nCode == 0:
        if wParam == WM_RBUTTONUP:
            print("右键抬起已被拦截")
            return 1
    return user32.CallNextHookEx(mouse_hook, nCode, wParam, lParam)

LowLevelMouseProc = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)(low_level_mouse_proc)

def install_hook():
    global mouse_hook
    mouse_hook = user32.SetWindowsHookExW(
        WH_MOUSE_LL,
        LowLevelMouseProc,
        None,
        0
    )

def uninstall_hook():
    global mouse_hook
    if mouse_hook:
        user32.UnhookWindowsHookEx(mouse_hook)
        mouse_hook = None

install_hook()

# 必须有一个消息循环，否则钩子不会工作
msg = wintypes.MSG()
while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
    user32.TranslateMessage(ctypes.byref(msg))
    user32.DispatchMessageW(ctypes.byref(msg))

最后在跳转发生时install_hook()，结束后uninstall_hook()就可以







# WebCamGazeEstimation - 智能眼手协同交互系统

## 🎯 项目简介

WebCamGazeEstimation 是一个基于普通网络摄像头的多功能视线追踪与手眼协同交互系统。该系统集成了先进的AI视觉算法，实现了高精度的视线估计、手部检测和智能交互功能。

### ✨ 核心功能

- **🎯 实时视线追踪**：基于OpenVINO AI引擎，支持高精度视线方向检测
- **👋 智能手部交互**：集成手部检测、捏合识别和点击事件检测
- **📱 多模式校准**：支持4点和9点校准系统，确保追踪精度
- **🔄 眼手协同控制**：结合视线和手势的自然交互体验
- **📊 性能分析评估**：完整的精度评估和数据记录功能

## 🏗️ 系统架构

```
摄像头输入 → AI模型推理 → 特征提取 → 坐标变换 → 交互输出
     ↓              ↓           ↓          ↓          ↓
 图像预处理    OpenVINO引擎   视线/手部   屏幕映射   可视化界面
```

### 核心模块结构

```
WebCamGazeEstimation-main/
├── 📁 src/                          # 源代码核心目录
│   ├── 📁 gaze_tracking/           # 视线追踪模块
│   │   ├── model.py                # AI模型加载与推理
│   │   ├── homtransform.py         # 坐标变换与校准
│   │   └── calibration_pygame.py   # 校准界面
│   ├── 📁 hand_tracking/           # 手部检测模块
│   │   ├── hand_tracking_system.py # 手部检测系统核心
│   │   ├── hand_detector.py        # 手部关键点检测
│   │   └── affine_transformer_3d.py # 3D变换处理
│   └── 📁 utilities/               # 工具函数
├── 📁 camera_data/                  # 相机标定数据
├── 📁 intel/                       # OpenVINO模型文件
├── 📁 results/                     # 运行结果输出
└── 📁 analysis_data/              # 性能分析数据
```

## 🚀 快速开始

### 环境要求

- **操作系统**：Windows 10/11, Linux, macOS
- **Python版本**：3.8+
- **硬件要求**：普通网络摄像头，支持OpenVINO的CPU/GPU
- **依赖库**：详见 requirements.txt

### 安装步骤

1. **克隆项目**
```bash
git clone [项目地址]
cd WebCamGazeEstimation-main
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **下载OpenVINO模型**（首次运行需要）
- 模型文件会自动下载到 `intel/` 目录
- 确保网络连接正常

4. **运行主程序**
```bash
# 基础视线追踪
python src/main.py

# 眼手协同交互系统
python eye_hand_interaction.py
```

## 🎯 功能详解

### 1. 视线追踪系统

#### 核心特性
- **实时处理**：30fps稳定帧率，低延迟响应
- **高精度校准**：4点/9点校准系统，支持历史数据保存
- **智能滤波**：卡尔曼滤波平滑视线数据

#### 使用方法
1. 启动程序后选择校准模式
2. 按照屏幕提示完成校准流程
3. 系统自动进入追踪模式
4. 实时显示视线落点位置

### 2. 手部检测系统

#### 检测功能
- **手部关键点**：21个手部关节点精确检测
- **捏合识别**：拇指与食指捏合状态检测
- **点击检测**：快速捏合动作识别为点击事件
- **3D变换**：支持3D空间中的手势识别

#### 交互事件
- **普通捏合**：持续性手势状态
- **快速点击**：瞬时捏合动作（速度阈值可调）
- **多手支持**：同时检测多个手部

### 3. 眼手协同交互

#### 协同模式
- **注视触发**：基于视线稳定性的智能区域检测
- **手势确认**：通过手部动作确认交互意图
- **自然交互**：结合视线和手势的直观操作

#### 触发机制
- **时间窗口**：500ms注视稳定性分析
- **双重验证**：角度离散度 < 3°，像素离散度 < 100px
- **智能显示**：75px半径交互区域可视化

## ⚙️ 配置参数

### 视线追踪参数
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| 校准点数 | 4 | 4-9 | 校准精度与时间的平衡 |
| 帧率 | 30fps | 15-30 | 性能与流畅度的权衡 |
| 图像分辨率 | 1280x960 | 640x480-1920x1080 | 精度与性能的选择 |

### 手部检测参数
| 参数 | 默认值 | 可调范围 | 功能 |
|------|--------|----------|------|
| 捏合阈值 | 手部大小×0.4 | 0.2-0.6 | 普通捏合检测灵敏度 |
| 快速接近阈值 | 手部大小×1.6 | 1.2-2.0 | 快速接近动作识别 |
| 点击速度阈值 | 手部大小×0.9 | 0.6-1.2 | 点击事件触发灵敏度 |

### 交互系统参数
- **注视稳定性时间**：500ms
- **角度离散度阈值**：3°
- **像素离散度阈值**：100px
- **交互区域半径**：75px
- **显示持续时间**：2s

## 📊 性能指标

### 视线追踪精度
- **角度误差**：< 2°（理想条件下）
- **屏幕误差**：< 50px（1920x1080分辨率）
- **响应延迟**：< 100ms

### 手部检测性能
- **检测帧率**：30fps稳定
- **关键点精度**：像素级准确度
- **手势识别率**：> 95%（清晰手势）

### 系统资源占用
- **CPU占用**：15-25%（i7-10700）
- **内存占用**：200-400MB
- **GPU加速**：支持OpenVINO GPU加速

## 🔧 高级配置

### 相机标定
```bash
# 运行相机内参标定
python camera_data/main_camera_calibration.py
```

### 性能分析
```bash
# 运行性能评估
python src/main_compareWithTobii.py
```

### 模型优化
- 支持FP16/FP32精度切换
- 支持CPU/GPU/AUTO设备选择
- 支持批量推理优化

## 🛠️ 开发扩展

### 添加新的手势识别
1. 在 `hand_tracking_system.py` 中扩展检测类
2. 实现新的手势逻辑算法
3. 更新事件处理机制

### 集成新的AI模型
1. 将模型文件放入 `intel/` 目录
2. 在 `model.py` 中配置模型参数
3. 更新输入输出处理逻辑

### 自定义交互界面
1. 基于 `calibration_pygame.py` 扩展UI类
2. 实现自定义交互逻辑
3. 集成到主程序流程

## 🔗 技术栈

### 核心库
- **OpenCV**: 4.8.0 - 计算机视觉处理
- **OpenVINO**: 2022.0 - AI推理加速
- **NumPy**: 1.24.0 - 数值计算
- **Pygame**: 2.5.0 - 界面开发
- **MediaPipe**: 0.10.8 - 手部检测

### 开发工具
- **Python**: 3.8+
- **Git**: 版本控制
- **VS Code**: 开发环境
