我需要修改`g:\mattest\Gaze estimation\WebCamGazeEstimation-main\src\gaze_tracking\model.py`文件中的`MediaPipeFace`类的`get_pose`方法，添加保存当前结果作为下一帧初始猜测的功能。

修改内容：
1. 在`get_pose`方法中，当PnP计算成功后，将当前帧的`rvec`和`tvec`保存到实例变量中
2. 这样下一帧调用时，`solvePnPRansac`就会使用上一帧的结果作为初始猜测，提高算法稳定性

具体修改位置：
- 在`model.py`文件的`get_pose`方法中，在返回`tvec`之前添加保存结果的代码
- 对应行号：大约在第458-461行之间

修改后的代码片段：
```python
if success:
    # 保存当前结果作为下一帧的初始猜测
    self.rvec, self.tvec = rvec, tvec
    
    # 只返回tvec平移向量
    return tvec
```

这个修改与`head_pose_estimation.py`中的实现保持一致，能够使连续帧之间的PnP计算更加稳定。