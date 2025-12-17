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


# ---------- 主逻辑 ----------
class UIHighlighter:
    def __init__(self):
        self.overlay = HighlightOverlay()
        self.last_ctrl = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)  # 每 50ms 取一次（20FPS）

    def update_ui(self):
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
                self.overlay.update_rect(rect)

        self.last_ctrl = ctrl


# ---------- 程序入口 ----------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    highlighter = UIHighlighter()
    sys.exit(app.exec_())
