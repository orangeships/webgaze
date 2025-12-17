## 实现计划

### 1. 功能概述
在现有眼手交互系统中添加基于UI元素的鼠标跳转功能，通过检测屏幕上的UI控件，根据不同情况将鼠标传送到合适位置，提高交互准确性和效率。

### 2. 核心实现逻辑
**严格按照`目前改进.md`的判断逻辑执行**：
1. **计算注视点中心**：使用当前注视点序列（最多8个点）计算中心点
2. **检查中心点是否在UI控件内**：
   - ✅ **如果是**：跳转到检测出的UI矩形框**中心点**
     - 特殊情况：如果矩形非常大（长或宽≥300px），则跳转到**当前注视位置**
3. **如果中心点不在UI控件内**：
   - 在半径100px范围内随机抽取8个点进行UI检测
   - ✅ **如果找到UI**：跳转到**最近的可控矩形框边界位置**
   - 如果未找到：默认跳转到**凝视点中心**

### 3. 实现步骤

#### 3.1 导入依赖与配置
- 在`hand_eye_coordination.py`中导入`uiautomation`库
- 定义UI检测跳过类型集合（文本、图标等）
- 在`threshold_config`中添加UI跳转相关参数：
  - `ui_based_teleport_enabled`：开关（默认开启）
  - `ui_detection_radius`：检测半径（100px）
  - `ui_sample_count`：采样点数（8个）
  - `large_ui_threshold`：大控件阈值（300px）

#### 3.2 添加UI检测方法
- `_find_non_leaf_control()`：查找实际容器控件（跳过文本/图标）
- `_find_ui_control_at_point()`：检测指定点的UI控件
- `_find_nearest_ui_control_in_radius()`：检测半径内最近UI控件
- `_calculate_sliding_window_center()`：计算注视点序列中心点
- `_get_nearest_point_on_rect()`：计算点到矩形的最近边界点

#### 3.3 更新跳转逻辑
- 修改`_trigger_fade_circle_cursor_move_and_reset()`：添加UI检测逻辑
- 修改`_auto_move_mouse_to_gaze()`：支持右键触发时的UI跳转
- 确保UI跳转可通过配置开关启用/禁用
- 实现完整的UI跳转判断流程

#### 3.4 兼容性处理
- 支持单屏/双屏模式
- 与现有跳转机制无缝集成
- 处理各种异常情况
- 确保系统稳定性

### 4. 技术要点
- 使用绝对坐标进行UI检测
- 采用中位数计算注视点中心，提高鲁棒性
- 准确计算点到矩形的最近边界点
- 添加完善的异常处理
- 支持通过配置调整UI检测参数

### 5. 文件修改
- `hand_eye_coordination.py`：核心实现
- `core_system.py`：添加UI跳转配置参数
- `requirements.txt`：添加`uiautomation`依赖

### 6. 测试要点
- 验证UI跳转开关功能
- 测试注视点在UI控件内时的跳转（中心点）
- 测试大控件内的跳转（当前注视位置）
- 测试注视点外但半径内有UI的跳转（边界点）
- 测试无UI时的跳转（原注视点）
- 验证双屏模式下的表现

### 7. 代码实现细节
- 严格按照`testautoui.py`的UI获取方法
- 使用`uiautomation.ControlFromPoint()`获取UI控件
- 跳过指定类型的UI控件（TextControl、ImageControl等）
- 正确处理控件的BoundingRectangle属性
- 实现高效的点到矩形最近点计算

### 8. 开关控制
- 在`threshold_config`中添加`ui_based_teleport_enabled`参数
- 所有UI跳转逻辑都受此开关控制
- 支持实时调整开关状态

这个实现计划完全符合用户需求，严格遵循`目前改进.md`的判断逻辑，并参考`testautoui.py`的UI获取方法，同时确保功能可通过参数开启或关闭。