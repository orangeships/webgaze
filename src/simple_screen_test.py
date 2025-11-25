#!/usr/bin/env python
"""
简单屏幕检测测试
"""

import sys
import os
import ctypes
from ctypes import wintypes

print("=== 屏幕检测方法测试 ===")

# 1. 测试pygame方法
print("\n1. 测试pygame方法:")
try:
    import pygame
    pygame.init()
    info = pygame.display.Info()
    print(f"  主屏幕: {info.current_w}x{info.current_h}")
    modes = pygame.display.list_modes()
    print(f"  可用模式数: {len(modes) if modes else 0}")
    display_count = pygame.display.get_num_displays()
    print(f"  检测到显示器数量: {display_count}")
    
    for i in range(display_count):
        try:
            modes_i = pygame.display.list_modes(i)
            if modes_i:
                print(f"  显示器{i}: {modes_i[0]} (最大)")
            else:
                print(f"  显示器{i}: 无模式")
        except Exception as e:
            print(f"  显示器{i}: 错误 - {e}")
            
except Exception as e:
    print(f"  pygame方法失败: {e}")

# 2. 测试win32gui方法
print("\n2. 测试win32gui方法:")
try:
    import win32gui
    
    def enum_callback(hmonitor, hdc, lprect, lparam):
        print(f"    发现显示器: {hmonitor}")
        return True
    
    monitors = win32gui.EnumDisplayMonitors(None, None, enum_callback, 0)
    print(f"  win32gui检测到 {len(monitors)} 个显示器")
    
except Exception as e:
    print(f"  win32gui方法失败: {e}")

# 3. 测试ctypes方法
print("\n3. 测试ctypes方法:")
try:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    
    # Windows API 常量
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    SM_CMONITORS = 80
    
    virtual_screen_left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    virtual_screen_top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    virtual_screen_width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    virtual_screen_height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    monitor_count = user32.GetSystemMetrics(SM_CMONITORS)
    
    print(f"  虚拟屏幕区域: ({virtual_screen_left}, {virtual_screen_top}) - {virtual_screen_width}x{virtual_screen_height}")
    print(f"  物理显示器数量: {monitor_count}")
    
except Exception as e:
    print(f"  ctypes方法失败: {e}")

print("\n=== 测试完成 ===")
print("按任意键退出...")
try:
    input()
except:
    pass