import sys
import ctypes
import time
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen
import uiautomation as auto


# ---------- 获取鼠标位置 ----------
def get_mouse_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# ---------- 跳过文字/图标类型 ----------
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
    """
    while ctrl:
        if ctrl.ControlTypeName not in SKIP_TYPES:
            return ctrl
        ctrl = ctrl.GetParentControl()
    return None


# ---------- 高亮层 ----------
class HighlightOverlay(QWidget):
    def __init__(self):
        super().__init__()

        # 透明、置顶、鼠标穿透
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowStaysOnTopHint |
            Qt.WindowDoesNotAcceptFocus |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.rect_to_draw = None
        self.showFullScreen()

    def update_rect(self, rect):
        self.rect_to_draw = rect
        self.update()

    def paintEvent(self, event):
        if not self.rect_to_draw:
            return

        r = self.rect_to_draw
        x = r.left
        y = r.top
        w = r.width()
        h = r.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)

        painter.drawRect(x, y, w, h)


# ---------- 文本信息显示层 ----------
class TextOverlay(QWidget):
    def __init__(self):
        super().__init__()

        # 透明、置顶、鼠标穿透
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowStaysOnTopHint |
            Qt.WindowDoesNotAcceptFocus |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.control_info = None
        self.showFullScreen()

    def update_control_info(self, control_info, detection_time=None):
        """更新控件信息显示"""
        self.control_info = control_info
        self.detection_time = detection_time
        self.update()

    def paintEvent(self, event):
        if not self.control_info:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 设置字体
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)

        # 计算文本显示位置（在控件上方）
        x, y = self.control_info['x'], self.control_info['y']
        w, h = self.control_info['width'], self.control_info['height']
        
        # 文本框位置
        text_x = x
        text_y = max(y - 80, 10)  # 在控件上方80像素，但不小于10
        
        # 绘制背景
        painter.fillRect(text_x - 5, text_y - 25, 220, 58, Qt.black)
        
        # 设置文本颜色
        painter.setPen(Qt.white)
        
        # 绘制文本信息
        lines = [
            f"控件类型: {self.control_info['control_type']}"
        ]
        
        # 如果有检测时间信息，添加到显示中
        if hasattr(self, 'detection_time') and self.detection_time is not None:
            lines.append(f"检测耗时: {self.detection_time:.2f}ms")
        
        for i, line in enumerate(lines):
            painter.drawText(text_x, text_y + i * 18, line)


# ---------- 主逻辑 ----------
class UIHighlighter:
    def __init__(self):
        self.overlay = HighlightOverlay()
        self.text_overlay = TextOverlay()
        self.last_ctrl = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)  # 每 50ms 取一次（20FPS）

    def update_ui(self):
        # 记录检测开始时间
        start_time = time.time()
        
        x, y = get_mouse_pos()
        raw_ctrl = auto.ControlFromPoint(x, y)
        if raw_ctrl is None:
            return

        ctrl = find_non_leaf_control(raw_ctrl)
        if ctrl is None:
            return

        if ctrl != self.last_ctrl:
            rect = ctrl.BoundingRectangle
            if rect:
                # 计算检测耗时（毫秒）
                detection_time = (time.time() - start_time) * 1000
                
                self.overlay.update_rect(rect)
                
                # 获取控件详细信息
                control_info = {
                    'x': rect.left,
                    'y': rect.top,
                    'width': rect.width(),
                    'height': rect.height(),
                    'control_type': getattr(ctrl, 'ControlTypeName', 'Unknown')
                }
                
                # 更新文本信息显示，包含检测时间
                self.text_overlay.update_control_info(control_info, detection_time)

        self.last_ctrl = ctrl


# ---------- 程序入口 ----------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    highlighter = UIHighlighter()
    sys.exit(app.exec_())
