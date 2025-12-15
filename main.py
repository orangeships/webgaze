#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼手交互系统主入口

该文件是眼手交互系统的主入口，负责初始化系统、显示菜单和启动交互模式。

主要功能：
1. 系统初始化和资源准备
2. 主菜单显示和交互模式选择
3. 校准流程启动
4. 交互模式运行
5. 系统资源释放和退出

使用流程：
1. 运行main.py启动系统
2. 选择交互模式（单屏/多屏）
3. 执行校准（如果需要）
4. 进入交互模式
5. 使用鼠标右键触发手眼协调机制  
6. 按ESC键退出系统

关键依赖：
- PyQt5: 应用程序框架
- core_system: 系统核心协调器

"""
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PyQt5.QtWidgets import QApplication
from core_system import EyeHandInteractionSystem

def main():
    # 获取项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建Qt应用程序实例
    app = QApplication(sys.argv) 
    
    # 创建眼手交互系统实例
    system = EyeHandInteractionSystem(project_dir)
    
    # 初始化系统
    if not system.initialize():
        print("系统初始化失败")
        sys.exit(1)
    
    # 显示主菜单
    interaction_mode, choice = system.show_menu()
    
    if choice == 'calibrate':
        # 设置双屏模式标志
        system.is_dual_screen_mode = (interaction_mode == 'multi')
        
        # 运行校准
        calibration_success = system.run_calibration(interaction_mode)
        
        if calibration_success:
            print("校准成功，进入交互模式")
            # 运行交互模式
            system.run_interaction_mode()
        else:
            print("校准失败或被用户取消")
    
    # 释放资源
    if system.cap:
        system.cap.release()
    
    # 退出应用程序，不再等待Qt事件循环
    sys.exit(0)

if __name__ == "__main__":
    main()