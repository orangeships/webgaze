###只使用pnp的tvec，把pnp用的镜头校准参数也统一加载
还有一点小问题，为啥我的注视点从副屏切换到主屏很顺滑，但是主屏切换到副屏容易注视点被卡在屏幕边缘（这种情况不应该发生，应该及时传送到副屏上）不行，还是有这种现象，我的视线明明都已经到副屏上了，注视点也在副屏上短暂显示了，但是又被拉回到主屏边界上，你是不是有些逻辑写的太复杂导致打架了？
请对文件中的坐标转换函数进行重构，以解决当前实现过于复杂臃肿的问题。修改需遵循以下具体步骤和要求：

1. 坐标转换基础：所有后续分析和计算必须以文件为基础，其中gaze_x_rel_original作为经过变换后的最终相对坐标点。

2. 坐标转换流程：
   a) 将gaze_x_rel_original相对坐标根据当前显示器信息转换为绝对坐标
   b) 对转换后的绝对坐标应用卡尔曼滤波处理
   c) 对滤波后的绝对坐标进行边界判断：当坐标点距离当前屏幕边界的相对距离小于5像素时，触发屏幕切换机制

3. 注视点分析器配置：确保注视点分析器全程采用绝对坐标进行计算和分析（这样的话距离判断也要进行相应修改，确保计算结果正确）

4. 显示处理要求：
   a) 绿色圆圈和注视点在显示前需统一转换为相对坐标，计算可以用绝对坐标
   b) 确保所有视觉元素（包括绿色圆圈和注视点）在转换后显示在屏幕范围内，无溢出或显示不全问题

重构时应注重代码的模块化和可读性，适当拆分复杂逻辑，添加必要注释，并确保修改后的坐标转换功能与系统其他模块兼容。
WT_G1(WTransG)： [[ 1.          0.          0.         -0.04491237]
                  [ 0.         -1.          0.         -0.14170611]
                  [ 0.          0.         -1.          0.98859737]
                  [ 0.          0.          0.          1.        ]]
"STransW": [[ -1.0, 0.0,  0.0, -112.18071287741239  ],
            [  0.0, 1.0, 0.0, 92.14017254237177    ],      
            [  0.0, 0.0, -1.0,  0.0     ],
            [  0.0, 0.0,  0.0,     1.0   ]]

"STransG": [[-1.0,        0.0,        0.0,        -95.90755260770955      ],
            [0.0,        -1.0,        0.0,        76.09702936537636      ],
            [0.0,        0.0,        1.0,        -275.66810640174253      ],
            [0.0,        0.0,        0.0,        1.0                       ]],

当前sfm的真正作用：程序会将sfm当做一个视觉里程计来使用，将其还原成初始标准人脸位置，然后再计算对应的注视向量，最后再加一个屏幕上的平移量就是最终的注视位置，但是这样仍解决不了相机侧视造成的模型输出偏离问题
-------------------------

构造：
pnp所需时间：6ms
实时[]
tvec: [[-55.62721093],[ 21.95714199],[645.32240659]]
W_T_G1[:3,3]: [-0.25872652 -0.05586771  0.9639602 ]

W：世界坐标系（标准人脸坐标系）   G：相机坐标系  S：屏幕坐标系
WTransG[:3,3]: [-0.03928294 -0.14430845  0.98846277]当前人脸距离第一帧的平移向量（归一化向量）
STransW: 标定世界坐标系（wtransg就是算的当前帧和它的平移向量）结合WTransG使用：STransW @ WTransG * scaleWtG，其本质上只编码了世界坐标系 W 的原点在屏幕上的位置（可学习参数），这个世界坐标系的原点就是标准人脸的中心！！！！

STransG: 可优化的世界变换矩阵（核心参数）功能：将相机坐标系的注视向量转换到屏幕坐标系
 
用 PnP 构造一个“SfM 等价平移向量”

✅ 推荐方案（最干净，也最稳）

彻底删除 SfM 的 W 坐标系，引入一个新的 P 坐标系（PnP-face）。

定义：
P 坐标系
原点：当前头部中心
方向：相机坐标系
tvec：P → C 的平移

然后：
PnP 平移 → 屏幕补偿项 → 加到 gaze 投影结果上
你可以直接复用：
def _getGazeOnScreen(self, gaze):
而不是 _getGazeOnScreen_sfm



关于fitStransG:
在这个过程中，模型主要“学”到了（或者说计算出了）两组关键参数。它们分别描述了宏观的物理位置关系和微观的局部修正关系。我们可以把它们拆解为：“全局的大地图” 和 “局部的补丁包”。
1. 全局大地图：STransG (Global Transformation Matrix)这是代码中最核心的产出，是一个 $4 \times 4$ 的刚体变换矩阵。它里面有什么？虽然矩阵很大，但其中大部分是固定死的（旋转部分 SRotG 是硬编码的，假设了坐标轴方向）。模型真正通过优化算法“学”到的，是矩阵右边那一列的平移向量 (Translation Vector)，即代码中的 xopt：$$[s_x, s_y, s_z]$$这三个数代表什么？它们描述了屏幕中心相对于相机中心的物理位置：$s_x$ (水平偏移)： 屏幕中心在相机的左边还是右边？$s_y$ (垂直偏移)： 屏幕中心在相机的上方还是下方？$s_z$ (深度距离)： 这是最重要的参数。它代表屏幕离相机（你的眼睛）有多远。它有什么用？它是通用的“翻译官”。它能把相机坐标系下的“注视方向（Gaze Vector）”，转换到屏幕坐标系下，从而计算出视线落在屏幕平面的哪个位置。
2. 局部补丁包：self.StG (Local Translation Vectors)这是一个列表（List），里面存了一组向量。如果你的校准程序有 9 个点，这里就有 9 个向量。它里面有什么？它存储了针对每一个校准点（比如屏幕左上角、中心、右下角等）的特定位置修正值。为什么要学这个？因为现实世界是不完美的。依靠上面的“全局大地图”算出来的落点，通常会有误差。比如： 在屏幕中心很准，但你看屏幕边缘时，由于摄像头畸变或人眼生理模型的不完美，算出来的点可能总是偏左 1 厘米。这个参数就是记录：“当用户看左上角时，请把计算结果往右修正 1 厘米。”它有什么用？用于插值修正。在后续实时预测 gaze 时，算法会看你盯着哪里，然后根据这组参数，对结果进行微调，大大提高边缘区域的准确度。
总结：这两组参数描述了什么关系？参数名称类型描述的关系 (Relationship)通俗理解STransG全局刚体变换相机坐标系 $\leftrightarrow$ 屏幕坐标系  描述了硬件的相对物理布局。“告诉程序电视机摆在离我 50 厘米远的正前方。”self.StG局部非线性修正注视角度 $\leftrightarrow$ 系统误差  描述了在不同视线角度下的畸变补偿。“告诉程序虽然电视摆正了，但我看右上角时眼神容易飘，记得帮我把坐标拉回来一点。”




首先在校准阶段：
eye_info, landmarks = model.get_gaze(frame=frame_cam, imshow=False)（获得原始gaze）
WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(self, keypoints_prev=landmarks_prev, keypoints_curr=landmarks)  #获得SfM链路的世界变换矩阵
STransW, scaleWtG, STransG = self._fitSTransG_sfm(gaze, SetVal, WTransG, g) #校准完成后计算相关参数

在视线映射阶段：
WTransG1, _, _ = self.homtrans.sfm.get_GazeToWorld(
                            self.model,
                            keypoints_prev=face_features_prev, keypoints_curr=face_features_curr
                        )
FSgaze, Sgaze, Sgaze2 = self.homtrans._getGazeOnScreen_sfm(gaze, WTransG1)  #先获取WTransG1（平移）再应用变换

最后是sfm的相关方法：
def get_GazeToWorld(self, model, keypoints_prev, keypoints_curr):
            # 从三维特征数据中提取2D坐标（x, y）
            p1_original = keypoints_prev[:2,:]
            p2_original = keypoints_curr[:2,:]
            # 对关键点进行去畸变处理
            p1_undistorted = cv2.undistortPoints(p1_original, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape(-1,2)
            p2_undistorted = cv2.undistortPoints(p2_original, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape(-1,2)
            
            # 使用去畸变后的点进行后续计算
            p1, p2 = p1_undistorted, p2_undistorted

            E = cv2.findEssentialMat(p1, p2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
            _, G2_R_G1, G2_t_G1, _= cv2.recoverPose(E, p1, p2, self.camera_matrix)    # G1 cosy is world coordinate system

            # Triangulate a point cloud using the final transformation (R,T)
            M1 = self.camera_matrix @ np.eye(3,4)
            M2 = self.camera_matrix @ np.c_[G2_R_G1, G2_t_G1]
            points_4d_homogeneous = cv2.triangulatePoints(M1, M2, p1.T, p2.T)
            W_P = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T).reshape(-1,3)   # 35x3

            W_P = W_P/np.linalg.norm(W_P, axis=1)[:,np.newaxis]
            W_P[W_P[:,2]<0] = W_P[W_P[:,2]<0]*(-1)      # if z<0, change sign of x,y,z

            # rotation of face in 3d, however provided gaze vector is already rotated back so it aligns with world coordinate system
            normal_vector,_ = util.fit_plane(W_P)
            normal_vector = normal_vector/np.linalg.norm(normal_vector)
            W_R_G1 = util.rotation_matrix_to_face(normal_vector, np.array([W_P[0,:],W_P[2,:],W_P[3,:],W_P[18,:]]) )
            # print(f"W_R_G1\n{np.array2string(W_R_G1, formatter={'float': lambda x: f'{x:.2f}'})}")

            # World is location of previous frame
            WRotG = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            W_T_G1 = np.r_[np.c_[WRotG, np.mean(np.array([W_P[0,:],W_P[2,:]]), axis=0)[:,None]], np.array([[0,0,0,1]])]
            print("WTransG[:3,3]:", W_T_G1[:3,3])
            # W_T_G1 = np.r_[np.c_[W_R_G1, np.mean(np.array([W_P[0,:],W_P[2,:]]), axis=0)[:,None]], np.array([[0,0,0,1]])]  
            G1_T_G2 = np.r_[np.c_[G2_R_G1.T, -G2_R_G1.T @ G2_t_G1], np.array([[0,0,0,1]])]
            W_T_G2 = W_T_G1 @ G1_T_G2       # not really useful
            if W_T_G2[2,3]<0:
                W_T_G2[:3,3] = W_T_G2[:3,3]*(-1)

            # W_T_G1[:2,3] = W_T_G1[:2,3]*(-1)    # flip x,y axis

            return W_T_G1, W_T_G2, W_P

def _getGazeOnScreen_sfm(self, gaze, WTransG):
        WTransG[:3,3] = self.scaleWtG*WTransG[:3,3]
        STransG = self.STransW @ WTransG
        scaleGaze = self._getScale(gaze, STransG)
        Sgaze = (STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """
        for i in range(len(self.StW)):
            STransG_ = np.vstack((np.hstack((SRotW, self.StW[i].reshape(3,1))), np.array([0,0,0,1]))) @ WTransG
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)
        """
        FSgaze = 融合后的注视向量，整体及各校准点均使用
        Sgaze = 在屏幕坐标系下通过回归得到的整体注视向量，已考虑头部运动
        Sgaze2 = 考虑头部运动后，从校准点得到的注视向量
        """
        return FSgaze, Sgaze, Sgaze2

def _fitSTransG_sfm(self, gaze, SetVal, WTransG, g):
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy() 
        WTransG = WTransG.to_numpy().reshape(-1,4,4)

        WRotG = WTransG[:,:3,:3]
        WtG = WTransG[:,:3,3]
        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])

        gaze = gaze[:,:,None]

        """ Model over camera coordinate system getting gaze from SFM  """
        def alignError(x, *const):
            SRotW, WRotG, gaze, WtG, SetVal = const
            StW = np.array([[x[1]],[x[2]],[0]])
            SRotG = SRotW @ WRotG
            Gz = np.array([[0],[0],[1]])
            mu = (Gz.T @ (-np.transpose(SRotG, axes=(0,2,1)) @ (SRotW @ (x[0]*WtG[:,:,None]) + StW)))/(Gz.T @ gaze)
            Sg = SRotG @ (mu*gaze) + SRotW @  (x[0]*WtG[:,:,None]) + StW
            error = SetVal[:,:,None] - Sg   # (87x3x1)
            return error.flatten()

        const = (SRotW, WRotG, gaze, WtG, SetVal)
        x0 = np.array([1, self.width/2, self.height/2])
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")
        StW = np.array([[xopt[1]],[xopt[2]],[0]])
        self.STransW = np.r_[np.c_[SRotW, StW], np.array([[0,0,0,1]])]
        WTransG = np.concatenate((np.c_[WRotG, xopt[0]*WtG[:,:,None]], np.tile(np.array([[0, 0, 0, 1]]), (WtG.shape[0], 1, 1))), axis=1)
        STransG = self.STransW @ np.median(WTransG, axis=0)
        self.scaleWtG = xopt[0]

        WtG = np.median(WtG[:,:,None], axis=0)

        """ Transformation Matrix to Auxiliary points """
        size = len(g)
        self.StW = [None]*size
        self.StG = [None]*size
        for i in range(size):
            scaleGaze = self._getScale(np.median(g[i],axis=0), STransG)     # compute scale for gaze vector for each calibration point
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i], np.median(g[i],axis=0), scaleGaze)
            self.StG[i] = STransG_[:3,3,None]
            self.StW[i] = STransG_[:3,3,None] - SRotW @ (self.scaleWtG*WtG)

        self.STransG = STransG

        return self.STransW, self.scaleWtG, STransG

    def _getScale(self, gaze, STransG):
        Gz = np.array([[0],[0],[1]])
        GTransS = util.invHomMatrix(STransG)
        GtS = GTransS[:3,3].reshape(3,1)
        if np.ndim(gaze) == 1:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,None])
        elif np.ndim(gaze) == 2:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,:,None])

        return scaleGaze

        gai