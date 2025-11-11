# 在 PLGaze 项目中添加新模型的完整指南

## 项目架构概览

PLGaze 项目采用模块化设计，支持多种视线估计模型。项目结构如下：

```
src/plgaze/
├── models/
│   ├── __init__.py          # 模型工厂函数
│   ├── mpiigaze/            # MPIIGaze 模型
│   ├── mpiifacegaze/        # MPIIFaceGaze 模型
│   └── eth-xgaze/           # ETH-XGaze 模型
├── data/configs/            # 模型配置文件
├── gaze_estimator.py        # 主要视线估计类
└── transforms.py            # 数据预处理
```

## 模型注册机制

### 1. 模型工厂模式

在 <mcfile name="__init__.py" path="src/plgaze/models/__init__.py"></mcfile> 中，`create_model` 函数负责根据配置动态创建模型实例：

```python
def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    if mode in ['MPIIGaze', 'MPIIFaceGaze']:
        module = importlib.import_module(
            f'ptgaze.models.{mode.lower()}.{config.model.name}')
        model = module.Model(config)
    elif mode == 'ETH-XGaze':
        model = timm.create_model(config.model.name, num_classes=2)
    else:
        raise ValueError
    # ...
```

### 2. 配置驱动

每个模型对应一个 YAML 配置文件，位于 `data/configs/` 目录：

- `mpiigaze.yaml` - MPIIGaze 模型配置
- `mpiifacegaze.yaml` - MPIIFaceGaze 模型配置  
- `eth-xgaze.yaml` - ETH-XGaze 模型配置

## 添加新模型的完整流程

### 步骤 1: 创建模型实现

#### 1.1 创建模型目录
在 `models/` 目录下创建新模型的文件夹，例如 `mynewmodel/`：

```
models/
├── mynewmodel/
│   ├── __init__.py
│   └── my_model.py
```

#### 1.2 实现模型类
新模型必须继承 `torch.nn.Module` 并实现 `forward` 方法：

```python
# models/mynewmodel/my_model.py
import torch
import torch.nn as nn
from omegaconf import DictConfig

class Model(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        # 定义模型结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.fc = nn.Linear(64 * 14 * 14, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实现前向传播
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

#### 1.3 创建 __init__.py
在模型目录中创建空的 `__init__.py` 文件，使 Python 能识别为包。

### 步骤 2: 注册模型到工厂

修改 <mcfile name="__init__.py" path="src/plgaze/models/__init__.py"></mcfile> 中的 `create_model` 函数：

```python
def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    if mode in ['MPIIGaze', 'MPIIFaceGaze', 'MYNEWMODEL']:  # 添加新模式
        module = importlib.import_module(
            f'ptgaze.models.{mode.lower()}.{config.model.name}')
        model = module.Model(config)
    elif mode == 'ETH-XGaze':
        model = timm.create_model(config.model.name, num_classes=2)
    else:
        raise ValueError
    # ...
```

### 步骤 3: 创建配置文件

在 `data/configs/` 目录下创建新的配置文件 `mynewmodel.yaml`：

```yaml
mode: MYNEWMODEL
device: cpu
model:
  name: my_model
  # 添加模型特定参数
face_detector:
  mode: dlib
  dlib_model_path: ~/.ptgaze/dlib/shape_predictor_68_face_landmarks.dat
gaze_estimator:
  checkpoint: ~/.ptgaze/models/mynewmodel_my_model.pth
  camera_params: ${PACKAGE_ROOT}/data/calib/sample_params.yaml
  normalized_camera_params: ${PACKAGE_ROOT}/data/normalized_camera_params/mpiifacegaze.yaml
  image_size: [224, 224]
```

### 步骤 4: 添加推理方法

在 <mcfile name="gaze_estimator.py" path="src/plgaze/gaze_estimator.py"></mcfile> 中添加新的推理方法：

#### 4.1 在 `estimate_gaze` 方法中添加分支

```python
def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
    # ... 现有代码 ...
    
    if self._config.mode == 'MPIIGaze':
        # ...
    elif self._config.mode == 'MPIIFaceGaze':
        # ...
    elif self._config.mode == 'ETH-XGaze':
        # ...
    elif self._config.mode == 'MYNEWMODEL':  # 添加新分支
        self._head_pose_normalizer.normalize(image, face)
        self._run_mynewmodel_model(face)
    else:
        raise ValueError
```

#### 4.2 实现推理方法

```python
@torch.no_grad()
def _run_mynewmodel_model(self, face: Face) -> None:
    image = self._transform(face.normalized_image).unsqueeze(0)
    
    device = torch.device(self._config.device)
    image = image.to(device)
    prediction = self._gaze_estimation_model(image)
    prediction = prediction.cpu().numpy()
    
    face.normalized_gaze_angles = prediction[0]
    face.angle_to_vector()
    face.denormalize_gaze_vector()
```

### 步骤 5: 添加数据预处理

在 <mcfile name="transforms.py" path="src/plgaze/transforms.py"></mcfile> 中添加预处理逻辑：

```python
def create_transform(config: DictConfig) -> Any:
    if config.mode == 'MPIIGaze':
        return T.ToTensor()
    elif config.mode == 'MPIIFaceGaze':
        return _create_mpiifacegaze_transform(config)
    elif config.mode == 'ETH-XGaze':
        return _create_ethxgaze_transform(config)
    elif config.mode == 'MYNEWMODEL':  # 添加新分支
        return _create_mynewmodel_transform(config)
    else:
        raise ValueError

def _create_mynewmodel_transform(config: DictConfig) -> Any:
    size = tuple(config.gaze_estimator.image_size)
    transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
    ])
    return transform
```

### 步骤 6: 创建归一化相机参数

在 `data/normalized_camera_params/` 目录下创建 `mynewmodel.yaml`：

```yaml
image_width: 224
image_height: 224
camera_matrix:
  rows: 3
  cols: 3
  data: [224, 0, 112, 0, 224, 112, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0, 0, 0, 0, 0]
```

## 模型接口要求

### 输入输出格式

- **输入**: 归一化后的面部图像，尺寸由配置决定
- **输出**: 2D 向量，表示视线角度

### 必需的方法

1. `__init__(self, config: DictConfig)` - 构造函数
2. `forward(self, x: torch.Tensor) -> torch.Tensor` - 前向传播

### 设备兼容性

模型必须支持配置中指定的设备（CPU/GPU），通过 `model.to(device)` 实现。

## 测试新模型

### 1. 加载配置

```python
from omegaconf import OmegaConf

config = OmegaConf.load('src/plgaze/data/configs/mynewmodel.yaml')
```

### 2. 创建视线估计器

```python
from plgaze import GazeEstimator

gaze_estimator = GazeEstimator(config)
```

### 3. 进行推理

```python
# 检测人脸
faces = gaze_estimator.detect_faces(image)

# 估计视线
for face in faces:
    gaze_estimator.estimate_gaze(image, face)
    print(f"Gaze vector: {face.gaze_vector}")
```

## 最佳实践

### 1. 遵循现有模式
- 使用相同的目录结构和命名约定
- 保持配置文件的格式一致性
- 实现标准的模型接口

### 2. 性能优化
- 使用 `@torch.no_grad()` 装饰推理方法
- 确保模型支持批处理
- 优化内存使用

### 3. 错误处理
- 在工厂函数中添加适当的错误检查
- 提供清晰的错误信息
- 验证输入输出格式

## 常见问题

### Q: 模型无法加载怎么办？
A: 检查：
- 模型目录结构是否正确
- `__init__.py` 文件是否存在
- 配置文件中 `mode` 和 `model.name` 是否正确

### Q: 推理结果异常怎么办？
A: 检查：
- 数据预处理是否正确
- 模型输出格式是否符合预期
- 归一化参数是否正确

### Q: 如何添加多输入模型？
A: 参考 MPIIGaze 模型实现，修改 `forward` 方法接受多个输入参数。

通过遵循本指南，您可以成功地将新模型集成到 PLGaze 项目中，并保持与现有架构的一致性。