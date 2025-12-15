import ctypes
import time
import threading
from pynput import keyboard
import math

user32 = ctypes.windll.user32
SPI_SETMOUSESPEED = 0x0071
SPI_GETMOUSESPEED = 0x0070
SPIF_SENDCHANGE = 0x0002

def set_mouse_speed(speed):
    speed = int(max(1, min(20, speed)))
    user32.SystemParametersInfoW(SPI_SETMOUSESPEED, 0, speed, SPIF_SENDCHANGE)

def get_mouse_speed():
    v = ctypes.c_int()
    user32.SystemParametersInfoW(SPI_GETMOUSESPEED, 0, ctypes.byref(v), 0)
    return v.value

# --- 参数 ---
TARGET_LOW = 2
RESTORE_TARGET = 10
RESTORE_TIME = 0.6
RESTORE_STEP_DELAY = 0.01

EASING_POWER = 3   # 越大 → 前面越快，后面越慢（可调）

restoring = False

def restore_speed_ease_out(start_speed):
    global restoring
    restoring = True

    steps = int(RESTORE_TIME / RESTORE_STEP_DELAY)

    for i in range(steps + 1):
        if not restoring:
            return

        t = i / steps                     # 0 → 1
        eased = 1 - (1 - t) ** EASING_POWER

        new_speed = start_speed + (RESTORE_TARGET - start_speed) * eased

        set_mouse_speed(new_speed)
        time.sleep(RESTORE_STEP_DELAY)

    restoring = False
    set_mouse_speed(RESTORE_TARGET)


def on_press(key):
    global restoring
    if key == keyboard.Key.ctrl_l:
        restoring = False  # 终止恢复
        set_mouse_speed(TARGET_LOW)
        print("阻尼模式 -> DPI = 2")


def on_release(key):
    global restoring

    if key == keyboard.Key.ctrl_l:
        start = get_mouse_speed()
        restoring = False

        thread = threading.Thread(
            target=restore_speed_ease_out,
            args=(start,),
            daemon=True
        )
        thread.start()

        print("非线性恢复开始...")


print("按住 Ctrl = 阻尼\n松开 Ctrl = 缓出式非线性恢复\n")

set_mouse_speed(RESTORE_TARGET)

with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
    l.join()
