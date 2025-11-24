import numpy as np

class KalmanFilter:
    """
    卡尔曼滤波器类，用于平滑视线追踪数据，减少抖动。
    适用于2D位置数据（x, y坐标）的平滑处理。
    """
    
    def __init__(self, process_noise=0.01, measurement_noise=2.0, error_estimate=1.0):
        """
        初始化卡尔曼滤波器参数
        
        参数:
        process_noise: 过程噪声协方差，表示模型不确定性
        measurement_noise: 测量噪声协方差，表示传感器噪声
        error_estimate: 初始估计误差协方差
        """
        # 状态向量: [x, y, vx, vy]，包含位置和速度
        self.x = np.zeros((4, 1))
        
        # 状态转移矩阵，假设匀速运动模型
        self.F = np.array([
            [1, 0, 1, 0],  # x(t) = x(t-1) + vx(t-1)
            [0, 1, 0, 1],  # y(t) = y(t-1) + vy(t-1)
            [0, 0, 1, 0],  # vx(t) = vx(t-1)
            [0, 0, 0, 1]   # vy(t) = vy(t-1)
        ])
        
        # 观测矩阵，只观测位置，不直接观测速度
        self.H = np.array([
            [1, 0, 0, 0],  # 测量x位置
            [0, 1, 0, 0]   # 测量y位置
        ])
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(4) * process_noise
        
        # 测量噪声协方差矩阵
        self.R = np.eye(2) * measurement_noise
        
        # 估计误差协方差矩阵
        self.P = np.eye(4) * error_estimate
        
        # 是否已初始化
        self.initialized = False
        
        # 自适应参数阈值 - 优化以减少抖动
        self.ACCEL_THRESHOLD = 800.0  # 降低加速度阈值，更早识别眼跳
        self.VELOCITY_THRESHOLD = 8.0   # 降低速度阈值，更好区分注视和眼跳
        self.previous_velocity = 0.0
    
    def predict(self):
        """
        预测下一状态
        """
        if not self.initialized:
            return
        
        # 预测状态
        self.x = np.dot(self.F, self.x)
        
        # 预测误差协方差
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def adaptive_kalman_params(self, velocity, previous_velocity):
        """根据运动特征动态调整噪声参数"""
        
        # 检测速度突变（眼跳特征）
        acceleration = abs(velocity - previous_velocity)
        
        if acceleration > self.ACCEL_THRESHOLD:  # 可能开始眼跳
            # 眼跳期间：信任测量，不信任模型预测
            Q = np.diag([30, 30, 60, 60])  # 减小过程噪声，避免过度响应
            R = np.diag([3, 3])              # 小测量噪声
            
        elif velocity < self.VELOCITY_THRESHOLD:  # 注视状态
            # 注视期间：极强平滑 - 优化参数减少抖动
            Q = np.diag([0.05, 0.05, 0.005, 0.005])    # 更小过程噪声，更强平滑  
            R = np.diag([80, 80])                      # 更大测量噪声，更不信任测量
            
        else:  # 平滑追随
            # 平衡模式 - 优化参数
            Q = np.diag([5, 5, 0.5, 0.5])    # 减小过程噪声
            R = np.diag([15, 15])            # 增大测量噪声
            
        return Q, R

    def update(self, measurement):
        """
        使用新的测量值更新滤波器状态
        
        参数:
        measurement: 包含x和y坐标的测量值
        
        返回:
        滤波后的x和y坐标
        """
        # 首次测量时初始化
        if not self.initialized:
            self.x[0, 0] = measurement[0]  # 初始x位置
            self.x[1, 0] = measurement[1]  # 初始y位置
            self.x[2, 0] = 0  # 初始x速度
            self.x[3, 0] = 0  # 初始y速度
            self.initialized = True
            return measurement
        
        # 计算当前速度
        current_velocity = np.sqrt(self.x[2, 0]**2 + self.x[3, 0]**2)
        
        # 自适应调整参数
        self.Q, self.R = self.adaptive_kalman_params(current_velocity, self.previous_velocity)
        self.previous_velocity = current_velocity
        
        # 预测步骤
        self.predict()
        
        # 更新步骤
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 计算残差
        y = np.array(measurement).reshape(2, 1) - np.dot(self.H, self.x)
        
        # 更新状态估计
        self.x = self.x + np.dot(K, y)
        
        # 更新估计误差协方差
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        # 返回滤波后的位置
        return [self.x[0, 0], self.x[1, 0]]
    
    def reset(self):
        """
        重置滤波器状态
        """
        self.x = np.zeros(4)
        self.P = self.P * 0.1  # 重置误差协方差，但保留一些历史信息
        self.initialized = False