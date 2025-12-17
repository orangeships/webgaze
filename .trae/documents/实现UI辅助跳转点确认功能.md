# 实现UI辅助跳转点确认功能（优化版）

## 1. 功能概述
根据用户需求，实现一个UI辅助跳转点确认功能，该功能需支持通过参数进行启用或关闭控制。主要实现位置为 `hand_eye_coordination.py` 文件，功能核心思想是根据获取到的UI位置信息辅助确定最终跳转的目标点，同时不影响正常触发逻辑，并输出UI信息使用的状态。

## 2. 实现逻辑

### 2.1 中心点计算
- 对 `self.sliding_window_gaze_points` 中的所有坐标点进行算术平均，计算出精确的中心点坐标作为当前凝视点。

### 2.2 UI辅助落点优化
- **不影响正常触发逻辑**：只有在确定需要进行跳转时，才会使用UI辅助功能优化落点。
- 检查当前凝视点中心是否位于任何UI控件的矩形框范围内。
- 若检测到有效UI控件：将该UI矩形框的几何中心设定为目标点。
- 控件大小过滤：若检测到的UI控件矩形框宽度或高度大于350像素，则忽略该控件，执行默认传送操作。
- 若当前凝视点中心未检测到任何可控UI控件，启动散布检测。
- 以当前凝视点为圆心，100像素为半径的圆形区域内，随机均匀生成8个检测点。
- 对每个检测点执行UI控件检测，同样排除宽度或高度大于350像素的大型控件。
- 从所有有效检测结果中，选择距离原始凝视点最近的可控UI矩形框。
- 计算并传送到该UI矩形框的边界位置。
- 若上述所有检测均未发现有效UI控件，则将原始计算的凝视点中心点作为最终目标点进行传送。

### 2.3 UI信息使用状态输出
- 在跳转时输出UI信息使用的状态，包括：
  - 命中了UI控件
  - 无效框（控件太大）
  - 随机抽的UI框

## 3. 实现步骤

### 3.1 在 `hand_eye_coordination.py` 文件中添加相关代码
1. 导入 `uiautomation` 库，用于获取UI元素位置信息。
2. 添加 `SKIP_TYPES` 常量，用于跳过文字/图标类型的控件。
3. 添加 `find_non_leaf_control` 函数，用于查找真正的UI容器控件。
4. 添加 `_calculate_gaze_center` 函数，用于计算凝视点中心点。
5. 添加 `_detect_ui_control_at_point` 函数，用于检测指定坐标点的UI控件。
6. 添加 `_generate_scatter_points` 函数，用于生成散布检测点。
7. 添加 `_find_optimal_ui_target` 函数，用于查找最优的UI目标控件，并返回UI信息使用的状态。
8. 修改 `_trigger_fade_circle_cursor_move_and_reset` 函数，集成UI辅助跳转点确认功能，并输出UI信息使用的状态。

### 3.2 在 `core_system.py` 文件中添加相关参数配置
1. 在 `threshold_config` 中添加相关参数，如 `ui_assisted_jump_enabled`、`ui_control_max_size`、`scatter_detection_radius` 等。

### 3.3 添加必要的注释说明
- 添加参数使用方法及功能开关控制逻辑的注释说明。

## 4. 具体实现代码

### 4.1 在 `hand_eye_coordination.py` 文件中

#### 4.1.1 导入必要的库
```python
import uiautomation as auto
```

#### 4.1.2 添加 `SKIP_TYPES` 常量和 `find_non_leaf_control` 函数
```python
# ---------- UI控件检测相关常量和函数 ----------
# 跳过文字/图标类型
SKIP_TYPES = {
    "TextControl",
    "ImageControl",
    "GlyphControl",
    "ListControl"
}

def find_non_leaf_control(ctrl):
    """
    如果鼠标命中了文字、图标，向上爬找到真正的 UI 容器控件
    （例如按钮、面板、菜单项等）
    
    Args:
        ctrl: 原始UI控件
        
    Returns:
        真正的UI容器控件，如果未找到则返回None
    """
    while ctrl:
        if ctrl.ControlTypeName not in SKIP_TYPES:
            return ctrl
        ctrl = ctrl.GetParentControl()
    return None
```

#### 4.1.3 添加 `_calculate_gaze_center` 函数
```python
def _calculate_gaze_center(self):
    """
    计算凝视点中心点
    
    Returns:
        tuple: 凝视点中心点坐标 (x, y) 绝对坐标
    """
    points = np.array([(x, y) for _, x, y in self.sliding_window_gaze_points])
    center = np.mean(points, axis=0)
    return (center[0], center[1])
```

#### 4.1.4 添加 `_detect_ui_control_at_point` 函数
```python
def _detect_ui_control_at_point(self, x, y):
    """
    检测指定坐标点的UI控件
    
    Args:
        x: 检测点X坐标（绝对坐标）
        y: 检测点Y坐标（绝对坐标）
    
    Returns:
        tuple: (UI控件, 边界矩形, 状态)，如果未检测到则返回 (None, None, "")
    """
    try:
        raw_ctrl = auto.ControlFromPoint(x, y)
        if raw_ctrl is None:
            return None, None, ""
        
        ctrl = find_non_leaf_control(raw_ctrl)
        if ctrl is None:
            return None, None, ""
        
        rect = ctrl.BoundingRectangle
        if rect is None:
            return None, None, ""
        
        # 检查控件大小是否超过阈值
        ui_control_max_size = self.threshold_config.get('ui_control_max_size', 350)
        if rect.width > ui_control_max_size or rect.height > ui_control_max_size:
            return ctrl, rect, "无效框（控件太大）"
        
        return ctrl, rect, "命中了UI控件"
    except Exception as e:
        print(f"[DEBUG] UI控件检测异常: {e}")
        return None, None, ""
```

#### 4.1.5 添加 `_generate_scatter_points` 函数
```python
def _generate_scatter_points(self, center_x, center_y, radius, count=8):
    """
    生成散布检测点
    
    Args:
        center_x: 圆心X坐标（绝对坐标）
        center_y: 圆心Y坐标（绝对坐标）
        radius: 圆半径（像素）
        count: 生成的检测点数量
    
    Returns:
        list: 检测点列表，每个元素为 (x, y) 绝对坐标
    """
    points = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((int(x), int(y)))
    return points
```

#### 4.1.6 添加 `_find_optimal_ui_target` 函数
```python
def _find_optimal_ui_target(self, gaze_center):
    """
    查找最优的UI目标控件，并返回UI信息使用的状态
    
    Args:
        gaze_center: 凝视点中心点坐标 (x, y) 绝对坐标
    
    Returns:
        tuple: (最优UI控件的中心坐标, UI信息使用的状态)，如果未找到则返回 (None, "")
    """
    # 1. 初始UI控件检测
    x, y = gaze_center
    ctrl, rect, status = self._detect_ui_control_at_point(x, y)
    if ctrl and rect and status != "无效框（控件太大）":
        # 计算UI控件的几何中心
        center_x = rect.left + rect.width // 2
        center_y = rect.top + rect.height // 2
        return (center_x, center_y), status
    
    # 2. 散布检测机制
    scatter_radius = self.threshold_config.get('scatter_detection_radius', 100)
    scatter_points = self._generate_scatter_points(x, y, scatter_radius)
    
    valid_controls = []
    
    for px, py in scatter_points:
        ctrl, rect, status = self._detect_ui_control_at_point(px, py)
        if ctrl and rect and status != "无效框（控件太大）":
            # 计算UI控件的几何中心
            center_x = rect.left + rect.width // 2
            center_y = rect.top + rect.height // 2
            
            # 计算到凝视点中心点的距离
            distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
            
            valid_controls.append((center_x, center_y, distance, status))
    
    if valid_controls:
        # 选择距离最近的UI控件
        valid_controls.sort(key=lambda x: x[2])
        optimal_target = valid_controls[0][0], valid_controls[0][1]
        return optimal_target, "随机抽的UI框"
    
    # 3. 无效框处理
    if status == "无效框（控件太大）":
        return None, status
    
    return None, ""
```

#### 4.1.7 修改 `_trigger_fade_circle_cursor_move_and_reset` 函数
```python
def _trigger_fade_circle_cursor_move_and_reset(self, fade_circle_x, fade_circle_y):
    """
    触发传送到渐变圆圈边界的光标移动并重置所有相关状态
    
    Args:
        fade_circle_x: 渐变圆圈中心X坐标（绝对坐标）
        fade_circle_y: 渐变圆圈中心Y坐标（绝对坐标）
    """
    try:
        # 保存传送前的鼠标滑动窗口数据，防止传送后窗口数据被污染
        saved_window = list(self.mouse_movement_window)
        
        # 获取当前鼠标位置作为参考点
        current_cursor_pos = win32api.GetCursorPos()
        cursor_x, cursor_y = current_cursor_pos
        
        # 确定注视点所在的屏幕
        gaze_monitor_index = self.screen_manager.get_target_screen(fade_circle_x, fade_circle_y)
        gaze_monitor = self.ui.monitors_info[gaze_monitor_index]
        
        # ===== UI辅助跳转点确认功能 =====
        ui_target_x, ui_target_y = fade_circle_x, fade_circle_y
        ui_status = "未使用UI辅助"
        
        # 检查是否启用了UI辅助跳转功能
        ui_assisted_jump_enabled = self.threshold_config.get('ui_assisted_jump_enabled', True)
        if ui_assisted_jump_enabled and len(self.sliding_window_gaze_points) >= 8:
            # 计算凝视点中心点
            gaze_center = self._calculate_gaze_center()
            
            # 查找最优的UI目标控件
            optimal_target, status = self._find_optimal_ui_target(gaze_center)
            if optimal_target:
                ui_target_x, ui_target_y = optimal_target
                ui_status = status
        
        # 使用滑动窗口第一个和最后一个位置计算移动方向和距离
        if len(self.mouse_movement_window) >= 2:
            # 使用滑动窗口第一个和最后一个位置
            first_pos = self.mouse_movement_window[0]  # (timestamp, x, y)
            last_pos = self.mouse_movement_window[-1]   # (timestamp, x, y)
            move_dx = last_pos[1] - first_pos[1]  # x坐标差值
            move_dy = last_pos[2] - first_pos[2]  # y坐标差值
            move_distance = np.sqrt(move_dx**2 + move_dy**2)  # 滑动窗口总移动距离
            
            # 使用滑动窗口移动方向计算传送目标
            if move_distance > 1e-6:
                # 计算圆周传送目标点
                target_x, target_y = self._calculate_circular_teleport(
                    ui_target_x, ui_target_y, move_dx, move_dy)
            else:
                # 如果移动距离太小，使用固定偏移
                radius = self.threshold_config.get('teleport_circle_radius', 100)
                target_x = ui_target_x - radius
                target_y = ui_target_y - radius
        else:
            # 如果没有足够的鼠标移动数据，直接使用UI目标坐标
            target_x, target_y = ui_target_x, ui_target_y
        
        # 移动鼠标到目标位置（使用绝对坐标）
        win32api.SetCursorPos((int(target_x), int(target_y)))
        
        # 输出UI信息使用的状态
        print(f"[DEBUG] 跳转目标: ({int(target_x)}, {int(target_y)}), UI状态: {ui_status}")
        
        # 确保绿色渐变圆圈的坐标在屏幕内
        # 先找到最合适的显示器（即使注视点在屏幕外，也使用最近的显示器）
        target_monitor_index = 0
        min_distance = float('inf')
        
        for i, monitor in enumerate(self.ui.monitors_info):
            # 计算显示器中心
            monitor_center_x = monitor['x'] + monitor['width'] // 2
            monitor_center_y = monitor['y'] + monitor['height'] // 2
            
            # 计算到注视点的距离
            distance = np.sqrt((fade_circle_x - monitor_center_x)**2 + (fade_circle_y - monitor_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                target_monitor_index = i
        
        # 获取目标显示器信息
        target_monitor = self.ui.monitors_info[target_monitor_index]
        
        # 将渐变圆圈中心限制在目标显示器内
        fade_circle_x_clamped = max(target_monitor['x'], min(fade_circle_x, target_monitor['x'] + target_monitor['width'] - 1))
        fade_circle_y_clamped = max(target_monitor['y'], min(fade_circle_y, target_monitor['y'] + target_monitor['height'] - 1))
        
        # 计算相对于目标显示器的坐标
        relative_x = int(fade_circle_x_clamped - target_monitor['x'])
        relative_y = int(fade_circle_y_clamped - target_monitor['y'])
        
        # 在双屏模式下，使用interaction_overlays
        if self.ui.interaction_overlays and len(self.ui.interaction_overlays) > target_monitor_index:
            self.ui.interaction_overlays[target_monitor_index].add_fade_circle(relative_x, relative_y, radius=100, duration=1500)
        # 兼容单屏模式
        elif hasattr(self.ui, 'current_widget') and self.ui.current_widget:
            self.ui.current_widget.add_fade_circle(int(fade_circle_x_clamped), int(fade_circle_y_clamped), radius=100, duration=1500)
        
        # 应用传送后阻尼效果
        self._apply_post_teleport_damping()
        
    except Exception as e:
        print(f"[DEBUG] 传送执行异常: {e}")
        import traceback
        traceback.print_exc()
```

### 4.2 在 `core_system.py` 文件中

#### 4.2.1 在 `threshold_config` 中添加相关参数
```python
# UI辅助跳转相关配置
'ui_assisted_jump_enabled': True,         # UI辅助跳转功能开关
'ui_control_max_size': 350,               # UI控件最大尺寸（像素），超过则忽略
'scatter_detection_radius': 100,          # 散布检测半径（像素）
```

## 5. 功能测试

1. 运行 `main.py` 启动系统
2. 进入交互模式
3. 测试UI辅助跳转功能是否正常工作
4. 测试通过参数关闭UI辅助跳转功能是否正常
5. 测试不同屏幕分辨率下的功能表现
6. 测试不同UI控件大小下的功能表现
7. 测试UI信息使用状态输出是否正常

## 6. 注意事项

1. 确保 `uiautomation` 库已正确安装
2. 注意UI控件检测的性能问题，避免频繁检测导致系统卡顿
3. 注意跨屏环境下的坐标转换问题
4. 注意UI辅助跳转功能与原有功能的兼容性问题
5. 注意添加必要的错误处理和异常捕获机制
6. 确保UI辅助功能不影响正常触发逻辑，只在判断落点时进行优化
7. 确保跳转时输出UI信息使用的状态

## 7. 后续优化方向

1. 优化UI控件检测算法，提高检测速度和准确性
2. 增加更多的UI控件类型支持
3. 增加UI控件优先级机制，优先选择更重要的UI控件
4. 增加UI辅助跳转功能的自适应调整机制
5. 增加用户自定义配置选项，允许用户调整相关参数
6. 优化散布检测点的生成算法，提高检测覆盖率

通过以上实现步骤，我将完成UI辅助跳转点确认功能的开发，该功能将支持通过参数进行启用或关闭控制，并根据获取到的UI位置信息辅助确定最终跳转的目标点，同时不影响正常触发逻辑，并在跳转时输出UI信息使用的状态。