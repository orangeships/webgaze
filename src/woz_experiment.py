import sys
import math
import random
import time
import ctypes
import os
import threading
try:
    import winsound
except Exception:
    winsound = None
from ctypes import wintypes
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QRadioButton,
                             QButtonGroup, QPushButton, QStackedWidget, QMessageBox,
                             QInputDialog)
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, QRect, QObject, QEvent
from PyQt5.QtGui import QPainter, QColor, QCursor, QFont, QPen, QBrush

# ===========================
# 1. 实验配置 (Configuration)
# ===========================
CONFIG = {
    # 屏幕设置 (请根据实际情况调整 SCREEN_B_OFFSET_X)
    # 假设屏幕A在左，屏幕B在右。如果是1920x1080，屏幕B的X偏移量通常是1920
    'SCREEN_B_OFFSET_X': 2390, 
    'SCREEN_A_INDEX': None,
    'SCREEN_B_INDEX': None,
    'SCREEN_A_NAME': None,
    'SCREEN_B_NAME': None,
    'JUMP_TO_A_PROB': 0.3,
    'ENABLE_CALIBRATION': True,
    'CALIBRATION_DISPLAY_INDICES': None,
    'CALIBRATION_POINT_TIME': 2,
    'CALIBRATION_FINISH_MS': 800,
    'CURSOR_DELAY_MS_MIN': 300,
    'CURSOR_DELAY_MS_MAX': 400,
    'CURSOR_JUMP_TRIGGER_PX': 30,
    'TARGET_MARGIN': 70,
    'JUMP_TRIGGER_MODE': 'gaze',  # 'gaze' or 'mouse'
    'GAZE_JUMP_ENABLED': True,
    'GAZE_CAMERA_INDEX': 1,
    'GAZE_JUMP_THRESHOLD': 0.04,
    'GAZE_JUMP_MIN_INTERVAL_MS': 500,
    'GAZE_JUMP_WARMUP_FRAMES': 5,
    'GAZE_JUMP_CONSECUTIVE_FRAMES': 3,
    'GAZE_JUMP_ARM_DELAY_MS': 200,
    'GAZE_JUMP_SMOOTHING_ALPHA': 0.3,
    'GAZE_JUMP_REQUIRE_EYES_OPEN': True,
    'GAZE_JUMP_DEBUG': True,
    'GAZE_JUMP_DEBUG_INTERVAL_MS': 300,
    'PRACTICE_TRIALS': 10,
    
    # 实验条件: 精度半径 (像素)
    'CONDITIONS': {
        'High': 30,
        'Mid': 60,
        'Low': 100
    },
    
    # 实验参数
    'TRIALS_PER_BLOCK': 10,  # 每个条件多少次试验
    'TARGET_SIZE': 60,       # 目标按钮直径
    'FIXATION_TIME': 1900,   # 注视时间 (ms)
    'CLICK_FEEDBACK_MS': 180,
    
    # 视觉风格
    'BG_COLOR': QColor(30, 30, 30),      # 深灰背景
    'TEXT_COLOR': QColor(220, 220, 220), # 浅灰文字
    'ACCENT_COLOR': QColor(70, 130, 180), # 目标颜色 (SteelBlue)
    'SUCCESS_COLOR': QColor(80, 200, 120), # 点击反馈颜色
}

# ===========================
# 2. 数据记录 (Data Logger)
# ===========================
class DataLogger:
    def __init__(self, participant_id=None):
        self.data = []
        output_dir = os.path.join(os.getcwd(), "result")
        os.makedirs(output_dir, exist_ok=True)
        if participant_id is None:
            filename = f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            filename = f"{participant_id}.csv"
        self.filename = os.path.join(
            output_dir,
            filename
        )

    def log_trial(self, condition, trial_id, time_ms, distance_px, jump_offset_px, jump_trigger_time_ms, success):
        self.data.append({
            'type': 'performance',
            'condition': condition,
            'trial_id': trial_id,
            'operation_time_ms': time_ms,
            'mouse_distance_px': distance_px,
            'jump_offset_px': jump_offset_px,
            'jump_trigger_time_ms': jump_trigger_time_ms,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        jump_msg = "NA" if jump_trigger_time_ms is None else f"{int(jump_trigger_time_ms)}"
        print(
            f"Logged Trial: {condition} | Time: {time_ms}ms | Dist: {int(distance_px)}px | "
            f"JumpOffset: {int(jump_offset_px)}px | JumpTime: {jump_msg}ms"
        )

    def log_survey(self, condition, answers):
        # answers 是一个字典 {q1: val, q2: val...}
        entry = {
            'type': 'subjective',
            'condition': condition,
            'timestamp': datetime.now().isoformat()
        }
        entry.update(answers)
        self.data.append(entry)
        print(f"Logged Survey: {condition}")

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")

# ===========================
# 2.1 注视跳变监测 (Gaze Jump Monitor)
# ===========================
class GazeJumpEmitter(QObject):
    gaze_jump = pyqtSignal()


class GazeJumpMonitor:
    def __init__(
        self,
        project_dir,
        camera_index=0,
        threshold=0.35,
        min_interval_ms=500,
        warmup_frames=5,
        consecutive_frames=3,
        arm_delay_ms=200,
        smoothing_alpha=0.3,
        require_eyes_open=True,
    ):
        self.project_dir = project_dir
        self.camera_index = int(camera_index)
        self.threshold = float(threshold)
        self.min_interval = max(0.0, float(min_interval_ms) / 1000.0)
        self.warmup_frames = max(0, int(warmup_frames))
        self.consecutive_frames = max(1, int(consecutive_frames))
        self.arm_delay = max(0.0, float(arm_delay_ms) / 1000.0)
        self.smoothing_alpha = max(0.0, min(1.0, float(smoothing_alpha)))
        self.require_eyes_open = bool(require_eyes_open)
        self.debug_enabled = bool(CONFIG.get('GAZE_JUMP_DEBUG', False))
        self.debug_interval = max(0.0, float(CONFIG.get('GAZE_JUMP_DEBUG_INTERVAL_MS', 300)) / 1000.0)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._armed = False
        self._reset_requested = False
        self._callback = None
        self._prev_gaze = None
        self._warmup_count = 0
        self._last_trigger_ts = 0.0
        self._last_debug_ts = 0.0
        self._arm_time = 0.0
        self._consecutive_hits = 0
        self._ema_gaze = None

    def set_callback(self, callback):
        self._callback = callback

    def reset(self):
        with self._lock:
            self._armed = False
            self._reset_requested = True
            self._prev_gaze = None
            self._ema_gaze = None
            self._warmup_count = 0
            self._consecutive_hits = 0

    def start(self):
        if self._thread.is_alive():
            return
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def arm(self):
        with self._lock:
            self._armed = True
            self._reset_requested = True
            self._arm_time = time.time()
        if self.debug_enabled:
            print("[GazeJump] armed")

    def disarm(self):
        with self._lock:
            self._armed = False
        if self.debug_enabled:
            print("[GazeJump] disarmed")

    def _should_trigger(self, delta, now):
        if delta < self.threshold:
            return False
        if self.min_interval > 0 and (now - self._last_trigger_ts) < self.min_interval:
            return False
        return True

    def _run(self):
        try:
            import cv2
            from gaze_tracking.model import EyeModel
        except Exception as exc:
            print(f"[GazeJump] disabled: {exc}")
            return

        try:
            model = EyeModel(self.project_dir)
        except Exception as exc:
            print(f"[GazeJump] EyeModel init failed: {exc}")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[GazeJump] Failed to open camera {self.camera_index}")
            return

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            try:
                result = model.get_gaze(frame=frame, imshow=False)
            except Exception:
                continue

            if isinstance(result, (tuple, list)):
                eye_info = result[0] if len(result) > 0 else None
            else:
                eye_info = result

            if not eye_info or "gaze" not in eye_info:
                continue

            gaze_vec = eye_info["gaze"]

            with self._lock:
                if self._reset_requested:
                    self._prev_gaze = gaze_vec  # 重置时设为当前 gaze，确保第一帧 delta=0
                    self._ema_gaze = gaze_vec   # EMA 也重置
                    self._warmup_count = 0      # 重新 warmup
                    self._consecutive_hits = 0  # 清空计数器
                    self._reset_requested = False

            if self._ema_gaze is None:
                self._ema_gaze = gaze_vec
                self._prev_gaze = gaze_vec
                self._warmup_count = 1
                continue

            if self._warmup_count < self.warmup_frames:
                self._warmup_count += 1
                continue

            # EMA 平滑处理
            alpha = self.smoothing_alpha
            self._ema_gaze = (
                alpha * gaze_vec + (1.0 - alpha) * self._ema_gaze
            )

            delta = math.hypot(
                self._ema_gaze[0] - self._prev_gaze[0],
                self._ema_gaze[1] - self._prev_gaze[1],
            )
            self._prev_gaze = self._ema_gaze

            with self._lock:
                armed = self._armed

            if not armed:
                continue

            now = time.time()
            if self.arm_delay > 0 and (now - self._arm_time) < self.arm_delay:
                continue
            if self.require_eyes_open:
                eye_state = eye_info.get("EyeState") if isinstance(eye_info, dict) else None
                if isinstance(eye_state, (list, tuple)) and len(eye_state) >= 2:
                    if min(eye_state) < 1:
                        self._consecutive_hits = 0
                        continue
            if self.debug_enabled and (now - self._last_debug_ts) >= self.debug_interval:
                print(
                    f"[GazeJump] delta={delta:.3f} threshold={self.threshold:.3f} "
                    f"hits={self._consecutive_hits}/{self.consecutive_frames} armed={armed}"
                )
                self._last_debug_ts = now
            if delta >= self.threshold:
                self._consecutive_hits += 1
            else:
                self._consecutive_hits = 0
            if self._consecutive_hits >= self.consecutive_frames and self._should_trigger(delta, now):
                with self._lock:
                    if not self._armed:
                        continue
                    self._armed = False
                self._last_trigger_ts = now
                self._consecutive_hits = 0
                if self.debug_enabled:
                    print(f"[GazeJump] TRIGGER delta={delta:.3f}")
                if self._callback:
                    try:
                        self._callback()
                    except Exception:
                        pass

        cap.release()
# ===========================
# 3. 屏幕 A/B: 任务画布 (Task Canvas)
# ===========================
class TaskCanvas(QWidget):
    trial_completed = pyqtSignal(float, float) # time, distance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target_pos = QPoint(0, 0)
        self.target_visible = False
        self.feedback_active = False
        self.mouse_path_length = 0
        self.last_mouse_pos = None
        self.start_time = 0
        self.is_active = False
        self.feedback_timer = QTimer(self)
        self.feedback_timer.setSingleShot(True)
        self.feedback_timer.timeout.connect(self._clear_feedback)
        self.setMouseTracking(True) # 开启鼠标追踪以记录轨迹

    def start_trial(self, target_center):
        self.target_pos = target_center
        self.target_visible = True
        self.feedback_active = False
        self.feedback_timer.stop()
        self.mouse_path_length = 0
        self.last_mouse_pos = QCursor.pos()
        self.start_time = time.time()
        self.is_active = True
        self.update()

    def stop_trial(self):
        self.is_active = False
        self.target_visible = False
        self.feedback_active = False
        self.feedback_timer.stop()
        self.update()

    def paintEvent(self, event):
        if not self.target_visible:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制目标按钮
        color = CONFIG['SUCCESS_COLOR'] if self.feedback_active else CONFIG['ACCENT_COLOR']
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        r = CONFIG['TARGET_SIZE'] // 2
        painter.drawEllipse(self.target_pos, r, r)

    def mouseMoveEvent(self, event):
        if not self.is_active:
            return
        
        current_pos = event.globalPos()
        if self.last_mouse_pos:
            dist = (current_pos - self.last_mouse_pos).manhattanLength() # 简化计算，也可以用欧氏距离
            self.mouse_path_length += dist
        self.last_mouse_pos = current_pos

    def mousePressEvent(self, event):
        if not self.is_active:
            return

        # 检查点击是否在圆内
        click_pos = event.pos()
        dist = (click_pos - self.target_pos).manhattanLength()
        
        if dist <= CONFIG['TARGET_SIZE'] // 2:
            # Success
            self.is_active = False
            self.feedback_active = True
            self.target_visible = True
            self.update()
            duration = (time.time() - self.start_time) * 1000
            self.trial_completed.emit(duration, self.mouse_path_length)
            feedback_ms = max(0, int(CONFIG.get('CLICK_FEEDBACK_MS', 0)))
            if feedback_ms > 0:
                self.feedback_timer.start(feedback_ms)
            else:
                self._clear_feedback()

    def _clear_feedback(self):
        self.feedback_active = False
        self.target_visible = False
        self.update()

# ===========================
# 4. 问卷页面 (Survey Widget)
# ===========================
class SurveyWidget(QWidget):
    survey_completed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"color: {CONFIG['TEXT_COLOR'].name()}; font-size: 14pt;")

        self.questions = [
            ("Q1. 光标的行为通常符合我对系统的预期。 (Predictability)", "predict"),
            ("Q2. 在光标移动之前，我大致能预测它会出现在哪个区域。 (Predictability)", "predict"),
            ("Q3. 在整个任务过程中，我感觉自己能够掌控光标的操作过程。 (Perceived Control)", "control"),
            ("Q4. 即使光标的初始位置存在一定偏差，我也能较快完成后续操作。 (Recoverability)", "recover"),
            ("Q5. 光标移动后，我有时需要花额外时间在视觉上重新找到它。 (Disorientation, reversed)", "disorient"),
            ("Q6. 在这种交互方式下，我愿意继续使用这种操作方式完成任务。 (Acceptance)", "accept"),
        ]


        self.current_index = 0
        self.answers = {}

        self.title = QLabel("以下陈述描述了你在刚才任务中的主观体验。请根据你的实际感受，选择最符合你情况的选项:")
        self.title.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(self.title)

        self.question_label = QLabel("")
        layout.addWidget(self.question_label)

        self.scale_layout = QHBoxLayout()
        self.bg = QButtonGroup(self)
        self.radio_buttons = []

        self.scale_layout.addWidget(QLabel("1 (不同意)"))
        for i in range(1, 8):
            rb = QRadioButton(str(i))
            self.bg.addButton(rb, i)
            self.radio_buttons.append(rb)
            self.scale_layout.addWidget(rb)
        self.scale_layout.addWidget(QLabel("7 (完全同意)"))

        layout.addLayout(self.scale_layout)
        layout.addSpacing(15)

        self.btn = QPushButton("Next")
        self.btn.setStyleSheet(f"background-color: {CONFIG['ACCENT_COLOR'].name()}; color: white; padding: 10px; border-radius: 5px;")
        self.btn.clicked.connect(self.next_question)
        layout.addWidget(self.btn)

        self.setLayout(layout)
        self._load_question()

    def _load_question(self):
        q_text, key = self.questions[self.current_index]
        self.question_label.setText(q_text)
        self.current_key = key
        is_last = (self.current_index == len(self.questions) - 1)
        self.btn.setText("Submit Block Evaluation" if is_last else "Next")
        self.bg.setExclusive(False)
        for btn in self.radio_buttons:
            btn.setChecked(False)
        self.bg.setExclusive(True)

    def next_question(self):
        if self.bg.checkedId() == -1:
            QMessageBox.warning(self, "Warning", "Please answer the question.")
            return
        self.answers[self.current_key] = self.bg.checkedId()
        if self.current_index >= len(self.questions) - 1:
            self.survey_completed.emit(self.answers)
            self.answers = {}
            self.current_index = 0
            self._load_question()
            return
        self.current_index += 1
        self._load_question()

class ContinueWidget(QWidget):
    continue_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"color: {CONFIG['TEXT_COLOR'].name()}; font-size: 16pt;")

        self.title = QLabel("Block completed.")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 18pt; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(self.title)

        self.btn = QPushButton("Start Next Block")
        self.btn.setStyleSheet(f"background-color: {CONFIG['ACCENT_COLOR'].name()}; color: white; padding: 10px 18px; border-radius: 5px;")
        self.btn.clicked.connect(self.continue_clicked)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def set_text(self, title_text, button_text):
        self.title.setText(title_text)
        self.btn.setText(button_text)

class PracticeMenuWidget(QWidget):
    practice_clicked = pyqtSignal()
    experiment_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"color: {CONFIG['TEXT_COLOR'].name()}; font-size: 16pt;")

        self.title = QLabel("校准完成")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 20pt; font-weight: bold; margin-bottom: 16px;")
        layout.addWidget(self.title)

        self.subtitle = QLabel("请选择进入练习或直接开始实验（练习可按 ESC 退出）")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setStyleSheet("font-size: 14pt; margin-bottom: 20px;")
        layout.addWidget(self.subtitle)

        self.practice_btn = QPushButton("开始练习 (10 次)")
        self.practice_btn.setStyleSheet(
            f"background-color: {CONFIG['ACCENT_COLOR'].name()}; color: white; padding: 10px 18px; border-radius: 5px;"
        )
        self.practice_btn.clicked.connect(self.practice_clicked)
        layout.addWidget(self.practice_btn)

        self.start_btn = QPushButton("开始实验")
        self.start_btn.setStyleSheet(
            f"background-color: {CONFIG['SUCCESS_COLOR'].name()}; color: white; padding: 10px 18px; border-radius: 5px;"
        )
        self.start_btn.clicked.connect(self.experiment_clicked)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

# ===========================
# 5. 屏幕 A: 注视/任务/问卷屏 (Fixation/Task/Survey Screen)
# ===========================
class FixationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def _tick(self):
        self.phase += 0.2
        if self.phase > math.tau:
            self.phase -= math.tau
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        center = QPoint(w // 2, h // 2)
        base_len = max(14, min(w, h) // 12)
        pulse = int(5 * (0.5 + 0.5 * math.sin(self.phase)))
        line_len = base_len + pulse

        base_color = QColor(CONFIG['TEXT_COLOR'])
        base_pen = QPen(base_color, 4, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(base_pen)
        painter.drawLine(center.x() - base_len, center.y(), center.x() + base_len, center.y())
        painter.drawLine(center.x(), center.y() - base_len, center.x(), center.y() + base_len)

        accent_color = QColor(CONFIG['ACCENT_COLOR'])
        accent_alpha = int(140 + 90 * (0.5 + 0.5 * math.sin(self.phase)))
        accent_color.setAlpha(accent_alpha)
        accent_pen = QPen(accent_color, 6, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(accent_pen)
        painter.drawLine(center.x() - line_len, center.y(), center.x() + line_len, center.y())
        painter.drawLine(center.x(), center.y() - line_len, center.x(), center.y() + line_len)


def build_fixation_label():
    return FixationWidget()

def get_win32_monitor_rects():
    if sys.platform != "win32":
        return {}
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    monitors = {}

    def _enum_proc(hmonitor, hdc, lprc, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            rect = info.rcMonitor
            monitors[info.szDevice] = QRect(
                rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
            )
        return True

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )
    user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_enum_proc), 0)
    return monitors

def set_cursor_pos_win(x, y):
    if sys.platform != "win32":
        return False
    ctypes.windll.user32.SetCursorPos(int(x), int(y))
    return True

def get_cursor_pos_win():
    if sys.platform != "win32":
        return QCursor.pos()
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return QPoint(point.x, point.y)

def run_fake_calibration(preferred_size=None, display_index=None):
    try:
        import pygame
        from calibration_pygame import PygameCalibrationUI
    except Exception as exc:
        print(f"Calibration skipped (pygame not available): {exc}")
        return True

    pygame.init()
    desktop_sizes = pygame.display.get_desktop_sizes()
    chosen_index = 0
    if isinstance(display_index, int) and 0 <= display_index < len(desktop_sizes):
        chosen_index = display_index
    elif preferred_size and desktop_sizes:
        target_w, target_h = preferred_size
        chosen_index = min(
            range(len(desktop_sizes)),
            key=lambda i: abs(desktop_sizes[i][0] - target_w) + abs(desktop_sizes[i][1] - target_h),
        )
    if desktop_sizes and 0 <= chosen_index < len(desktop_sizes):
        width, height = desktop_sizes[chosen_index]
    else:
        width, height = (1280, 720)

    print(f"Starting calibration on display {chosen_index} size={width}x{height}")
    ui = PygameCalibrationUI(width, height, display_index=chosen_index)
    ui.initialize()

    ui.show_start_screen()
    while True:
        event = ui.handle_events()
        if event == 'quit':
            ui.cleanup()
            pygame.quit()
            return False
        if event in ('start', 'click'):
            break
        ui.clock.tick(60)

    ui.show_precalibration_screen()
    while True:
        event = ui.handle_events()
        if event == 'quit':
            ui.cleanup()
            pygame.quit()
            return False
        if event in ('start', 'click'):
            break
        ui.clock.tick(60)

    ui.start_calibration()
    point_time = float(CONFIG.get('CALIBRATION_POINT_TIME', 1.2))
    if point_time <= 0:
        point_time = 1.2
    while True:
        event = ui.handle_events()
        if event == 'quit':
            ui.cleanup()
            pygame.quit()
            return False
        idx, _ = ui.get_current_calibration_point(time_interval=point_time)
        if idx is None:
            break
        ui.show_calibration_point(idx)
        ui.clock.tick(60)

    ui.screen.fill(ui.WHITE)
    done_text = ui.font_medium.render("校准完成", True, ui.BLACK)
    done_rect = done_text.get_rect(center=(width // 2, height // 2))
    ui.screen.blit(done_text, done_rect)
    pygame.display.flip()
    pygame.time.wait(int(CONFIG.get('CALIBRATION_FINISH_MS', 800)))

    ui.cleanup()
    pygame.quit()
    return True

def resolve_calibration_display_indices(preferred_sizes):
    try:
        import pygame
    except Exception:
        return []

    pygame.display.init()
    sizes = pygame.display.get_desktop_sizes()
    pygame.display.quit()
    if not sizes:
        return []

    indices = []
    used = set()
    for size in preferred_sizes:
        if not size:
            continue
        best = None
        best_score = None
        for i, (w, h) in enumerate(sizes):
            if i in used:
                continue
            score = abs(w - size[0]) + abs(h - size[1])
            if best_score is None or score < best_score:
                best_score = score
                best = i
        if best is not None:
            indices.append(best)
            used.add(best)

    if not indices:
        indices = list(range(min(2, len(sizes))))

    return indices

def play_cue_sound():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        
        import numpy as np
        sample_rate = 44100
        duration = 0.15
        frequency = 880
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
        waveform = (waveform * 32767).astype(np.int16)
        stereo_waveform = np.column_stack((waveform, waveform))
        
        sound = pygame.mixer.Sound(buffer=stereo_waveform)
        sound.set_volume(0.8)
        sound.play()
        print("[Sound] Played via pygame")
        return
    except Exception as e:
        print(f"[Sound] pygame error: {e}")
    
    try:
        import ctypes
        user32 = ctypes.windll.user32
        if user32.MessageBeep(-1):
            print("[Sound] Played MessageBeep")
            return
    except Exception as e:
        print(f"[Sound] MessageBeep error: {e}")
    
    try:
        app = QApplication.instance()
        if app:
            app.beep()
            print("[Sound] Played QApplication.beep")
    except Exception as e:
        print(f"[Sound] QApplication.beep error: {e}")

def resolve_screens():
    screens = QApplication.screens()
    if not screens:
        primary = QApplication.primaryScreen()
        return primary, primary, screens

    name_a = CONFIG.get('SCREEN_A_NAME')
    name_b = CONFIG.get('SCREEN_B_NAME')
    if isinstance(name_a, str) and isinstance(name_b, str):
        screen_by_name = {screen.name(): screen for screen in screens}
        if name_a in screen_by_name and name_b in screen_by_name:
            return screen_by_name[name_a], screen_by_name[name_b], screens

    idx_a = CONFIG.get('SCREEN_A_INDEX')
    idx_b = CONFIG.get('SCREEN_B_INDEX')
    if isinstance(idx_a, int) and isinstance(idx_b, int):
        if 0 <= idx_a < len(screens) and 0 <= idx_b < len(screens):
            return screens[idx_a], screens[idx_b], screens

    if len(screens) == 1:
        return screens[0], screens[0], screens

    screens_sorted = sorted(screens, key=lambda s: s.geometry().x())
    return screens_sorted[0], screens_sorted[-1], screens

class MouseMoveFilter(QObject):
    def __init__(self, on_move, parent=None):
        super().__init__(parent)
        self.on_move = on_move

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            try:
                if event.source() != Qt.MouseEventNotSynthesized:
                    return False
            except Exception:
                pass
            self.on_move(event.globalPos())
        return False

class KeyPressFilter(QObject):
    def __init__(self, on_key, parent=None):
        super().__init__(parent)
        self.on_key = on_key

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            try:
                return bool(self.on_key(event))
            except Exception:
                return False
        return False

class FixationWindow(QMainWindow):
    def __init__(self, assigned_screen=None):
        super().__init__()
        self.assigned_screen = assigned_screen or QApplication.primaryScreen()
        self.setWindowTitle("Screen A: Fixation")
        self.setGeometry(0, 0, 800, 600) # 初始位置，全屏后会覆盖
        self.setStyleSheet(f"background-color: {CONFIG['BG_COLOR'].name()};")
        
        self.stack = QStackedWidget()
        self.fixation = build_fixation_label()
        self.canvas = TaskCanvas()
        self.survey = SurveyWidget()
        self.continue_widget = ContinueWidget()
        self.practice_menu = PracticeMenuWidget()
        self.stack.addWidget(self.fixation) # Index 0
        self.stack.addWidget(self.canvas)   # Index 1
        self.stack.addWidget(self.survey)   # Index 2
        self.stack.addWidget(self.continue_widget) # Index 3
        self.stack.addWidget(self.practice_menu) # Index 4
        self.setCentralWidget(self.stack)

    def _prepare_on_screen(self):
        if self.assigned_screen:
            self.setGeometry(self.assigned_screen.geometry())
            handle = self.windowHandle()
            if handle:
                handle.setScreen(self.assigned_screen)

    def show_fixation(self):
        self.stack.setCurrentIndex(0)
        self._prepare_on_screen()
        self.showFullScreen()

    def show_canvas(self):
        self.stack.setCurrentIndex(1)
        self._prepare_on_screen()
        self.showFullScreen()

    def show_survey(self):
        self.stack.setCurrentIndex(2)
        self._prepare_on_screen()
        self.showFullScreen()

    def show_continue(self, title_text, button_text):
        self.continue_widget.set_text(title_text, button_text)
        self.stack.setCurrentIndex(3)
        self._prepare_on_screen()
        self.showFullScreen()

    def show_practice_menu(self):
        self.stack.setCurrentIndex(4)
        self._prepare_on_screen()
        self.showFullScreen()

# ===========================
# 6. 屏幕 B: 交互与问卷屏 (Task Screen)
# ===========================

# 6.1 屏幕 B 主窗口
class TaskWindow(QMainWindow):
    def __init__(self, assigned_screen=None):
        super().__init__()
        self.setWindowTitle("Screen B: Interaction")
        self.assigned_screen = assigned_screen or QApplication.primaryScreen()
        if self.assigned_screen:
            self.setGeometry(self.assigned_screen.geometry())
        else:
            self.setGeometry(CONFIG['SCREEN_B_OFFSET_X'], 0, 800, 600)
        self.setStyleSheet(f"background-color: {CONFIG['BG_COLOR'].name()};")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.canvas = TaskCanvas()
        self.survey = SurveyWidget()
        self.fixation = build_fixation_label()
        
        self.stack.addWidget(self.canvas) # Index 0
        self.stack.addWidget(self.survey) # Index 1
        self.stack.addWidget(self.fixation) # Index 2

    def _prepare_on_screen(self):
        if self.assigned_screen:
            self.setGeometry(self.assigned_screen.geometry())
            handle = self.windowHandle()
            if handle:
                handle.setScreen(self.assigned_screen)

    def show_canvas(self):
        self.stack.setCurrentIndex(0)
        self._prepare_on_screen()
        self.showFullScreen()

    def show_survey(self):
        self.stack.setCurrentIndex(1)
        self._prepare_on_screen()
        self.showFullScreen()

    def show_fixation(self):
        self.stack.setCurrentIndex(2)
        self._prepare_on_screen()
        self.showFullScreen()

# ===========================
# 5. 实验主控制器 (Controller)
# ===========================
class ExperimentController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        # Initialize jump state before installing mouse event filter.
        self.waiting_for_jump = False
        self.jump_start_pos = None
        self.jump_armed = False
        self.jump_trigger_px = int(CONFIG.get('CURSOR_JUMP_TRIGGER_PX', 30))
        self.pending_cursor_pos = None
        self.pending_active_screen = None
        self.pending_active_canvas = None
        self.mouse_move_filter = MouseMoveFilter(self._on_global_mouse_move)
        self.app.installEventFilter(self.mouse_move_filter)
        self.keypress_filter = KeyPressFilter(self._on_key_press)
        self.app.installEventFilter(self.keypress_filter)
        self.app.aboutToQuit.connect(self._stop_gaze_monitor)
        self.gaze_jump_emitter = GazeJumpEmitter()
        self.gaze_jump_emitter.gaze_jump.connect(self._on_gaze_jump)
        self.fixation_timer = QTimer(self.app)
        self.fixation_timer.setSingleShot(True)
        self.fixation_timer.timeout.connect(self.execute_jump)

        self.participant_id = self._ask_participant_id()
        self.logger = DataLogger(self.participant_id)
        self.screen_a, self.screen_b, self.screens = resolve_screens()
        self.win_monitor_rects = get_win32_monitor_rects()
        self.screen_mappings = self._build_screen_mappings()
        self._log_screens()
        self.win_a = FixationWindow(self.screen_a)
        self.win_b = TaskWindow(self.screen_b)
        
        # 实验状态
        self.blocks = self._build_latin_square_blocks()
        self.current_block_idx = 0
        self.current_trial_idx = 0
        self.current_condition = ""
        self.current_radius = 0
        self.current_jump_offset = 0.0
        self.waiting_for_continue = False
        self.waiting_for_start = False
        self.waiting_for_gaze_jump = False
        self._trial_transitioning = False  # 防止 trial 切换时的竞赛条件
        self.gaze_monitor = None
        self.waiting_for_menu = False
        self.practice_active = False
        self.practice_trial_idx = 0
        self.practice_total_trials = int(CONFIG.get('PRACTICE_TRIALS', 10))
        self.practice_conditions = list(CONFIG.get('CONDITIONS', {}).keys())
        self.trial_active = False
        self.jump_trigger_start_ts = None
        self.current_jump_trigger_ms = None
        
        # 信号连接
        self.win_a.canvas.trial_completed.connect(self.on_trial_finish)
        self.win_b.canvas.trial_completed.connect(self.on_trial_finish)
        self.win_a.survey.survey_completed.connect(self.on_survey_finish)
        self.win_a.continue_widget.continue_clicked.connect(self.on_continue_clicked)
        self.win_a.practice_menu.practice_clicked.connect(self.on_practice_clicked)
        self.win_a.practice_menu.experiment_clicked.connect(self.on_experiment_clicked)
        self._start_gaze_monitor()

    def _ask_participant_id(self):
        participant_id, ok = QInputDialog.getInt(
            None,
            "Participant ID",
            "请输入受试者编号：",
            value=1,
            min=1,
        )
        if not ok:
            print("Participant ID input canceled.")
            sys.exit(0)
        return participant_id

    def _on_key_press(self, event):
        if event.key() != Qt.Key_Escape:
            return False
        if self.practice_active:
            self._end_practice(early=True)
            return True
        return False

    def _start_gaze_monitor(self):
        if self.gaze_monitor is not None:
            return
        if not CONFIG.get('GAZE_JUMP_ENABLED', True):
            return
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.gaze_monitor = GazeJumpMonitor(
            project_dir,
            camera_index=CONFIG.get('GAZE_CAMERA_INDEX', 0),
            threshold=CONFIG.get('GAZE_JUMP_THRESHOLD', 0.35),
            min_interval_ms=CONFIG.get('GAZE_JUMP_MIN_INTERVAL_MS', 500),
            warmup_frames=CONFIG.get('GAZE_JUMP_WARMUP_FRAMES', 5),
            consecutive_frames=CONFIG.get('GAZE_JUMP_CONSECUTIVE_FRAMES', 3),
            arm_delay_ms=CONFIG.get('GAZE_JUMP_ARM_DELAY_MS', 200),
            smoothing_alpha=CONFIG.get('GAZE_JUMP_SMOOTHING_ALPHA', 0.3),
            require_eyes_open=CONFIG.get('GAZE_JUMP_REQUIRE_EYES_OPEN', True),
        )
        self.gaze_monitor.set_callback(self.gaze_jump_emitter.gaze_jump.emit)
        self.gaze_monitor.start()

    def _stop_gaze_monitor(self):
        if self.gaze_monitor is None:
            return
        self.gaze_monitor.stop()
        self.gaze_monitor = None

    def _arm_gaze_jump(self):
        if self.gaze_monitor is None:
            self.waiting_for_gaze_jump = False
            return
        self.waiting_for_gaze_jump = True
        self.gaze_monitor.arm()

    def _disarm_gaze_jump(self):
        self.waiting_for_gaze_jump = False
        if self.gaze_monitor is not None:
            self.gaze_monitor.disarm()

    def _on_gaze_jump(self):
        if not self.waiting_for_gaze_jump or self._trial_transitioning:
            return
        self.waiting_for_gaze_jump = False
        if self.fixation_timer.isActive():
            self.fixation_timer.stop()
        if CONFIG.get('GAZE_JUMP_DEBUG', False):
            print("[GazeJump] controller received trigger, executing jump")
        self._do_jump_action(triggered_by_gaze=True)

    def on_practice_clicked(self):
        if not self.waiting_for_menu:
            return
        self.waiting_for_menu = False
        self.practice_active = True
        self.practice_trial_idx = 0
        self._trial_transitioning = False
        self.win_a.show_canvas()
        self.win_b.show_fixation()
        self.start_trial()

    def on_experiment_clicked(self):
        if not self.waiting_for_menu:
            return
        self.waiting_for_menu = False
        self.practice_active = False
        self.start_block()

    def _end_practice(self, early=False):
        self.practice_active = False
        self.practice_trial_idx = 0
        self._disarm_gaze_jump()
        if self.fixation_timer.isActive():
            self.fixation_timer.stop()
        self.win_a.canvas.stop_trial()
        self.win_b.canvas.stop_trial()
        self.jump_trigger_start_ts = None
        self.current_jump_trigger_ms = None
        self.win_a.show_practice_menu()
        self.win_b.show_fixation()
        self.waiting_for_menu = True
        self.trial_active = False
        self._trial_transitioning = False
        if early:
            print("[Practice] ended by ESC")

    def _build_latin_square_blocks(self):
        base_blocks = list(CONFIG['CONDITIONS'].keys())
        if not base_blocks:
            return []
        shift = (self.participant_id - 1) % len(base_blocks)
        return base_blocks[shift:] + base_blocks[:shift]

    def _start_jump_watch(self, active_screen, cursor_pos, active_canvas):
        self.pending_active_screen = active_screen
        self.pending_cursor_pos = cursor_pos
        self.pending_active_canvas = active_canvas
        # Use Qt coordinates to match QMouseEvent.globalPos() (avoids HiDPI mismatch).
        self.jump_start_pos = QCursor.pos()
        self.jump_armed = False
        self.waiting_for_jump = True

    def _cancel_jump_watch(self):
        self.waiting_for_jump = False
        self.jump_start_pos = None
        self.jump_armed = False
        self.pending_cursor_pos = None
        self.pending_active_screen = None
        self.pending_active_canvas = None

    def _prepare_cursor_jump(self, active_screen, cursor_pos, active_canvas):
        self.pending_active_screen = active_screen
        self.pending_cursor_pos = cursor_pos
        self.pending_active_canvas = active_canvas

    def _perform_cursor_jump(self):
        if self.pending_active_canvas is None:
            return
        if self.jump_trigger_start_ts is not None and self.current_jump_trigger_ms is None:
            self.current_jump_trigger_ms = (time.time() - self.jump_trigger_start_ts) * 1000.0
        self._set_cursor(self.pending_active_screen, self.pending_cursor_pos, self.pending_active_canvas)
        self.pending_active_screen = None
        self.pending_cursor_pos = None
        self.pending_active_canvas = None

    def _on_global_mouse_move(self, pos):
        if not self.waiting_for_jump:
            return
        if self.jump_start_pos is None:
            self.jump_start_pos = pos
            self.jump_armed = False
            return
        if not self.jump_armed:
            self.jump_armed = True
            return
        dx = pos.x() - self.jump_start_pos.x()
        dy = pos.y() - self.jump_start_pos.y()
        if math.hypot(dx, dy) >= self.jump_trigger_px:
            self.waiting_for_jump = False
            if self.pending_active_canvas is not None:
                if self.jump_trigger_start_ts is not None and self.current_jump_trigger_ms is None:
                    self.current_jump_trigger_ms = (time.time() - self.jump_trigger_start_ts) * 1000.0
                self._set_cursor(self.pending_active_screen, self.pending_cursor_pos, self.pending_active_canvas)
            self.pending_active_screen = None
            self.pending_cursor_pos = None
            self.pending_active_canvas = None
            self.jump_start_pos = None
            self.jump_armed = False

    def _build_screen_mappings(self):
        mappings = {}
        for screen in self.screens:
            qt_geo = screen.geometry()
            win_geo = self.win_monitor_rects.get(screen.name())
            if not win_geo or qt_geo.width() <= 0 or qt_geo.height() <= 0:
                continue
            scale_x = win_geo.width() / qt_geo.width()
            scale_y = win_geo.height() / qt_geo.height()
            mappings[screen.name()] = (win_geo, scale_x, scale_y)
        return mappings

    def _log_screens(self):
        primary = QApplication.primaryScreen()
        for i, screen in enumerate(self.screens):
            geo = screen.geometry()
            tag = "primary" if screen == primary else "secondary"
            win_geo = self.win_monitor_rects.get(screen.name())
            if win_geo:
                scale = self.screen_mappings.get(screen.name())
                if scale:
                    _, scale_x, scale_y = scale
                    print(f"[Screen {i}] {screen.name()} {tag} geo=({geo.x()},{geo.y()},{geo.width()}x{geo.height()}) win=({win_geo.x()},{win_geo.y()},{win_geo.width()}x{win_geo.height()}) scale=({scale_x:.2f},{scale_y:.2f})")
                else:
                    print(f"[Screen {i}] {screen.name()} {tag} geo=({geo.x()},{geo.y()},{geo.width()}x{geo.height()}) win=({win_geo.x()},{win_geo.y()},{win_geo.width()}x{win_geo.height()})")
            else:
                print(f"[Screen {i}] {screen.name()} {tag} geo=({geo.x()},{geo.y()},{geo.width()}x{geo.height()})")
        if self.screen_a and self.screen_b:
            print(f"Screen A = {self.screen_a.name()}, Screen B = {self.screen_b.name()}")

    def _choose_target_pos(self, canvas_rect):
        min_margin = CONFIG['TARGET_SIZE'] // 2 + 10
        margin_x = max(CONFIG.get('TARGET_MARGIN', 0), min_margin, int(canvas_rect.width() * 0.1))
        margin_y = max(CONFIG.get('TARGET_MARGIN', 0), min_margin, int(canvas_rect.height() * 0.1))
        if canvas_rect.width() <= margin_x * 2:
            x = canvas_rect.width() // 2
        else:
            x = random.randint(margin_x, canvas_rect.width() - margin_x)
        if canvas_rect.height() <= margin_y * 2:
            y = canvas_rect.height() // 2
        else:
            y = random.randint(margin_y, canvas_rect.height() - margin_y)
        return QPoint(x, y)

    def _set_cursor(self, active_screen, cursor_pos, active_canvas):
        if active_screen:
            mapping = self.screen_mappings.get(active_screen.name())
            if mapping:
                win_geo, scale_x, scale_y = mapping
                cursor_global_x = win_geo.x() + int(cursor_pos.x() * scale_x)
                cursor_global_y = win_geo.y() + int(cursor_pos.y() * scale_y)
                if not set_cursor_pos_win(cursor_global_x, cursor_global_y):
                    QCursor.setPos(cursor_global_x, cursor_global_y)
                return
        QCursor.setPos(active_canvas.mapToGlobal(cursor_pos))

    def start(self):
        if CONFIG.get('ENABLE_CALIBRATION', False):
            preferred_sizes = []
            for screen in (self.screen_a, self.screen_b):
                if not screen:
                    continue
                win_geo = self.win_monitor_rects.get(screen.name())
                if win_geo:
                    preferred_sizes.append((win_geo.width(), win_geo.height()))
                else:
                    geo = screen.geometry()
                    preferred_sizes.append((geo.width(), geo.height()))

            display_indices = CONFIG.get('CALIBRATION_DISPLAY_INDICES')
            if isinstance(display_indices, int):
                display_indices = [display_indices]
            elif isinstance(display_indices, (list, tuple)):
                display_indices = [i for i in display_indices if isinstance(i, int)]
            else:
                display_indices = resolve_calibration_display_indices(preferred_sizes)

            if not display_indices:
                display_indices = [0]

        for i, display_index in enumerate(display_indices):
            preferred_size = preferred_sizes[i] if i < len(preferred_sizes) else None
            if run_fake_calibration(preferred_size, display_index) is False:
                self._stop_gaze_monitor()
                return
        if CONFIG.get('ENABLE_CALIBRATION', False):
            self.win_a.show_practice_menu()
            self.win_b.show_fixation()
            self.waiting_for_menu = True
        else:
            self.win_a.showFullScreen()
            self.win_b.showFullScreen()
            self.start_block()
        sys.exit(self.app.exec_())

    def start_block(self):
        if self.current_block_idx >= len(self.blocks):
            self.end_experiment()
            return

        self.current_condition = self.blocks[self.current_block_idx]
        self.current_radius = CONFIG['CONDITIONS'][self.current_condition]
        self.current_trial_idx = 0
        self._trial_transitioning = False
        
        print(f"--- Starting Block: {self.current_condition} (Radius: {self.current_radius}) ---")
        self.win_b.show_canvas()
        self.start_trial()

    def start_trial(self):
        if self.practice_active:
            if self.practice_trial_idx >= self.practice_total_trials:
                self._end_practice()
                return
            self._cancel_jump_watch()
            if self.gaze_monitor is not None:
                self.gaze_monitor.reset()
            self.win_a.canvas.stop_trial()
            self.win_b.canvas.stop_trial()
            self.jump_trigger_start_ts = None
            self.current_jump_trigger_ms = None
            if self.practice_conditions:
                self.current_condition = random.choice(self.practice_conditions)
                self.current_radius = CONFIG['CONDITIONS'].get(self.current_condition, self.current_radius)
            self.target_on_a = (random.random() < CONFIG['JUMP_TO_A_PROB'])
            if self.target_on_a:
                self.win_a.show_canvas()
                self.win_b.show_fixation()
            else:
                self.win_b.show_canvas()
                self.win_a.show_fixation()
            self.fixation_timer.start(CONFIG['FIXATION_TIME'])
            self.trial_active = True
            self._trial_transitioning = False
            if CONFIG.get('GAZE_JUMP_DEBUG', False):
                print(f"[Practice] trial {self.practice_trial_idx + 1}/{self.practice_total_trials}")
            return
        if self.current_trial_idx >= CONFIG['TRIALS_PER_BLOCK']:
            # Block 结束，进入问卷
            self._disarm_gaze_jump()
            if self.fixation_timer.isActive():
                self.fixation_timer.stop()
            self.win_a.show_survey()
            self.win_b.show_fixation()
            return

        self._cancel_jump_watch()
        if self.gaze_monitor is not None:
            self.gaze_monitor.reset()
        self.win_a.canvas.stop_trial()
        self.win_b.canvas.stop_trial()
        self.jump_trigger_start_ts = None
        self.current_jump_trigger_ms = None

        # 1. 随机选择目标屏，对应屏显示注视点
        self.target_on_a = (random.random() < CONFIG['JUMP_TO_A_PROB'])
        if self.target_on_a:
            self.win_a.show_canvas()
            self.win_b.show_fixation()
        else:
            self.win_b.show_canvas()
            self.win_a.show_fixation()
        
        # 2. 注视结束后显示目标，再开始检测视线跳变（不使用超时回退）
        self.fixation_timer.start(CONFIG['FIXATION_TIME'])
        self.trial_active = True
        self._trial_transitioning = False
        if CONFIG.get('GAZE_JUMP_DEBUG', False):
            print(f"[GazeJump] fixation started ({CONFIG['FIXATION_TIME']}ms)")

    def execute_jump(self):
        # 隐藏注视点，显示画布
        if self.target_on_a:
            self.win_b.show_canvas()
        else:
            self.win_a.show_canvas()
        
        # 计算目标位置 (屏幕中心附近的随机位置，防止被试肌肉记忆)
        active_canvas = self.win_a.canvas if self.target_on_a else self.win_b.canvas
        canvas_rect = active_canvas.rect()
        target_pos = self._choose_target_pos(canvas_rect)
        
        # WoZ 核心逻辑：计算模拟视线落点 (Target + Error)
        # 在圆内随机生成一个偏移量
        # 使用 sqrt(random) 保证在圆内分布均匀
        target_radius = CONFIG['TARGET_SIZE'] // 2
        if self.current_condition == 'High':
            r_inner = 0
            r_outer = target_radius
        elif self.current_condition == 'Mid':
            r_inner = target_radius
            r_outer = target_radius + 30
        else:
            r_inner = 0
            r_outer = self.current_radius
        angle = random.uniform(0, 2 * math.pi)
        if r_inner <= 0:
            dist = math.sqrt(random.uniform(0, 1)) * r_outer
        else:
            dist = math.sqrt(random.uniform(r_inner * r_inner, r_outer * r_outer))
        
        offset_x = int(dist * math.cos(angle))
        offset_y = int(dist * math.sin(angle))
        self.current_jump_offset = math.hypot(offset_x, offset_y)
        
        self.cursor_pos = QPoint(target_pos.x() + offset_x, target_pos.y() + offset_y)
        
        # 先显示目标，并在目标出现瞬间播放提示音
        active_canvas.start_trial(target_pos)
        play_cue_sound()
        self.jump_trigger_start_ts = time.time()
        self.current_jump_trigger_ms = None
        
        # 执行跳转并启动眼跳检测
        self._do_jump_action()

    def _do_jump_action(self, triggered_by_gaze=False):
        # 执行鼠标跳转
        active_screen = self.win_a.assigned_screen if self.target_on_a else self.win_b.assigned_screen
        active_canvas = self.win_a.canvas if self.target_on_a else self.win_b.canvas
        if CONFIG.get('JUMP_TRIGGER_MODE', 'gaze') == 'mouse':
            if not triggered_by_gaze:
                self._start_jump_watch(active_screen, self.cursor_pos, active_canvas)
            return

        if triggered_by_gaze:
            self._perform_cursor_jump()
            return

        self._prepare_cursor_jump(active_screen, self.cursor_pos, active_canvas)
        self._arm_gaze_jump()
        if CONFIG.get('GAZE_JUMP_DEBUG', False):
            print("[GazeJump] target shown, waiting for gaze trigger")

    def on_trial_finish(self, duration, distance):
        self._cancel_jump_watch()
        self._disarm_gaze_jump()
        self.trial_active = False
        if self.practice_active:
            self.practice_trial_idx += 1
            if self.practice_trial_idx >= self.practice_total_trials:
                self._end_practice()
            else:
                self._trial_transitioning = True
                QTimer.singleShot(1000, self.start_trial)
            return
        self.logger.log_trial(
            self.current_condition,
            self.current_trial_idx,
            duration,
            distance,
            self.current_jump_offset,
            self.current_jump_trigger_ms,
            True,
        )
        self.current_trial_idx += 1
        
        # 标记进入 trial 切换状态，防止排队的 gaze_jump 信号被处理
        self._trial_transitioning = True
        QTimer.singleShot(1000, self.start_trial)

    def on_survey_finish(self, answers):
        self.logger.log_survey(self.current_condition, answers)
        self.logger.save() # 每个 block 保存一次以防崩溃
        
        self.current_block_idx += 1
        self.waiting_for_continue = True
        if self.current_block_idx >= len(self.blocks):
            self.win_a.show_continue("All blocks completed. Click to finish.", "Finish")
        else:
            self.win_a.show_continue("Block completed. Click to start next block.", "Start Next Block")
        self.win_b.show_fixation()

    def on_continue_clicked(self):
        if self.waiting_for_start:
            self.waiting_for_start = False
            self.win_a.show_canvas()
            self.win_b.show_canvas()
            self.start_block()
            return
        if not self.waiting_for_continue:
            return
        self.waiting_for_continue = False
        if self.current_block_idx >= len(self.blocks):
            self.end_experiment()
        else:
            self.start_block()

    def end_experiment(self):
        self._stop_gaze_monitor()
        self.logger.save()
        msg = QMessageBox(self.win_a)
        msg.setWindowTitle("Finished")
        msg.setText("Experiment Completed! Thank you.")
        msg.setIcon(QMessageBox.Information)
        msg.setStyleSheet(
            "QMessageBox { background-color: #1e1e1e; color: #f0f0f0; }"
            "QLabel { color: #f0f0f0; }"
            "QPushButton { background-color: #3a3a3a; color: #f0f0f0; padding: 6px 12px; }"
        )
        msg.exec_()
        self.app.quit()

if __name__ == "__main__":
    # 确保 high DPI 缩放正常
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    except:
        pass
        
    ctrl = ExperimentController()
    ctrl.start()
