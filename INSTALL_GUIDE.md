# Eye-Hand Interaction System 安装指南



### 2. 创建虚拟环境 
```bash
# 使用venv创建虚拟环境
python -m venv gaze_env

# 激活虚拟环境
# Windows:
gaze_env\Scripts\activate
# macOS/Linux:
source gaze_env/bin/activate
```

### 3. 升级pip并安装依赖
```bash
python -m pip install --upgrade pip
pip install -r requirements_new.txt
```

### 4. 验证安装
```python
# 测试主要模块导入
python -c "import cv2, numpy, pygame, PyQt5; print('所有主要模块安装成功!')"
```
