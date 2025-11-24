#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI控件位置可视化
将检测到的UI控件按其位置绘制在坐标图上
"""

import time
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import matplotlib.font_manager as fm

# 检查pywinauto是否可用
PYWINAUTO_AVAILABLE = True
try:
    import pywinauto
    from pywinauto import Application
    import pywinauto.findwindows as findwindows
except ImportError:
    PYWINAUTO_AVAILABLE = False
    print("❌ pywinauto库不可用，无法进行控件检测")

def get_controls_with_positions():
    """获取控件位置信息"""
    start_time = time.time()
    
    if not PYWINAUTO_AVAILABLE:
        return {
            'error': 'pywinauto库不可用',
            'elapsed_time': time.time() - start_time,
            'controls': [],
            'window_info': None
        }
    
    try:
        # 1. 获取活动窗口
        windows = findwindows.find_windows(active_only=True, enabled_only=True)
        if not windows:
            return {
                'error': '没有找到活动窗口',
                'elapsed_time': time.time() - start_time,
                'controls': [],
                'window_info': None
            }
        
        top_window_handle = windows[0]
        
        # 2. 连接窗口
        backend_used = "uia"
        try:
            app = Application(backend="uia").connect(handle=top_window_handle)
        except:
            app = Application(backend="win32").connect(handle=top_window_handle)
            backend_used = "win32"
        
        top_window = app.window(handle=top_window_handle)
        window_text = top_window.window_text()
        
        # 3. 获取窗口位置信息
        try:
            window_rect = top_window.rectangle()
            window_left, window_top = window_rect.left, window_rect.top
            window_width = window_rect.width()
            window_height = window_rect.height()
        except:
            window_left, window_top = 0, 0
            window_width, window_height = 1920, 1080
        
        # 4. 获取所有控件
        all_controls = top_window.descendants()
        
        # 5. 定义要查找的控件类型
        target_types = [
            'Button', 'Edit', 'ComboBox', 'CheckBox', 'RadioButton', 'Tab',
            'ListBox', 'TreeView', 'ListView', 'MenuItem', 'Hyperlink'
        ]
        
        controls = []
        for control in all_controls:
            try:
                # 检查可见性
                if not control.is_visible():
                    continue
                
                # 获取控件类型
                try:
                    control_type = control.element_info.control_type
                except:
                    control_type = 'Unknown'
                
                # 检查是否为目标类型
                if control_type in target_types:
                    # 获取名称
                    try:
                        name = control.window_text()
                        if not name:
                            name = control.element_info.name
                    except:
                        name = '无名称'
                    
                    # 获取位置和尺寸
                    try:
                        rect = control.rectangle()
                        left = rect.left - window_left
                        top = rect.top - window_top
                        width = rect.width()
                        height = rect.height()
                        
                        # 过滤掉无效位置的控件
                        if width <= 0 or height <= 0:
                            continue
                        if left < 0 or top < 0:
                            continue
                        
                        controls.append({
                            'name': name or '无名称',
                            'type': control_type,
                            'position': (left, top),
                            'size': (width, height),
                            'center': (left + width/2, top + height/2)
                        })
                    except:
                        continue
                        
            except:
                continue
        
        elapsed_time = time.time() - start_time
        
        return {
            'window_info': {
                'title': window_text,
                'backend': backend_used,
                'rect': (window_left, window_top, window_width, window_height)
            },
            'controls': controls,
            'statistics': {
                'total_found': len(all_controls),
                'interactive_count': len(controls),
                'elapsed_time': elapsed_time,
                'backend_used': backend_used
            }
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'elapsed_time': time.time() - start_time,
            'controls': [],
            'window_info': None
        }

def visualize_controls(window_info, controls, save_path='controls_visualization.png'):
    """可视化控件位置"""
    
    # 设置字体（处理中文显示）
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    if not controls:
        print("❌ 没有可交互控件，无法进行可视化")
        return False
    
    # 获取窗口信息
    window_rect = window_info['rect']
    window_title = window_info['title']
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # 设置坐标轴范围（以窗口的左上角为原点）
    ax.set_xlim(0, window_rect[2])
    ax.set_ylim(window_rect[3], 0)  # 翻转y轴，让原点在左上角
    
    # 定义颜色映射
    color_map = {
        'Button': '#FF6B6B',      # 红色
        'Edit': '#4ECDC4',        # 青色
        'ComboBox': '#45B7D1',    # 蓝色
        'CheckBox': '#96CEB4',    # 绿色
        'RadioButton': '#FFEAA7', # 黄色
        'Tab': '#DDA0DD',         # 紫色
        'ListBox': '#98D8C8',     # 薄荷绿
        'TreeView': '#F7DC6F',    # 浅黄
        'ListView': '#BB8FCE',    # 淡紫
        'MenuItem': '#85C1E9',    # 淡蓝
        'Hyperlink': '#F8C471',   # 橙色
        'Unknown': '#D5DBDB'      # 灰色
    }
    
    # 绘制每个控件
    for i, control in enumerate(controls):
        pos = control['position']
        size = control['size']
        ctrl_type = control['type']
        name = control['name']
        
        color = color_map.get(ctrl_type, '#D5DBDB')
        
        # 绘制矩形
        rect = patches.Rectangle(pos, size[0], size[1], 
                               linewidth=1, edgecolor='black', 
                               facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        
        # 添加控件名称标签
        center_x = pos[0] + size[0] / 2
        center_y = pos[1] + size[1] / 2
        
        # 如果控件太小，缩小字体
        font_size = 8 if min(size) > 30 else 6
        
        ax.text(center_x, center_y, f"{i+1}\n{name[:8]}", 
               ha='center', va='center', fontsize=font_size, 
               weight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
    
    # 设置标题和标签
    ax.set_title(f'UI控件位置可视化 - {window_title}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X坐标 (像素)', fontsize=12)
    ax.set_ylabel('Y坐标 (像素)', fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 创建图例
    legend_elements = []
    for ctrl_type, color in color_map.items():
        if any(ctrl['type'] == ctrl_type for ctrl in controls):
            legend_elements.append(patches.Patch(color=color, label=f'{ctrl_type}'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 添加统计信息
    type_counts = defaultdict(int)
    for control in controls:
        type_counts[control['type']] += 1
    
    stats_text = f"总控件数: {len(controls)}\n"
    for ctrl_type, count in sorted(type_counts.items()):
        stats_text += f"{ctrl_type}: {count}个\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存
    
    print(f"✅ 控件可视化图已保存到: {save_path}")
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("📊 UI控件位置可视化")
    print("   将检测到的控件按位置绘制在坐标图上")
    print("=" * 60)
    
    print(f"\n📋 开始获取控件信息...")
    result = get_controls_with_positions()
    
    if 'error' in result:
        print(f"❌ 检测失败: {result['error']}")
        return
    
    controls = result['controls']
    window_info = result['window_info']
    stats = result['statistics']
    
    print(f"\n🏠 窗口: {window_info['title']}")
    print(f"📊 检测结果:")
    print(f"  - 可交互控件: {len(controls)} 个")
    print(f"  - 耗时: {stats['elapsed_time']:.3f} 秒")
    print(f"  - 后端: {stats['backend_used']}")
    
    if controls:
        print(f"\n📋 控件列表:")
        for i, control in enumerate(controls, 1):
            pos = control['position']
            size = control['size']
            print(f"  {i:2d}. {control['name']:<20} | {control['type']:<10} | 位置: ({pos[0]:4d},{pos[1]:4d}) 尺寸: {size[0]:3d}x{size[1]:3d}")
        
        print(f"\n🎨 开始生成可视化图...")
        if visualize_controls(window_info, controls, 'UI控件位置可视化图.png'):
            print(f"✅ 可视化完成!")
    else:
        print(f"\n⚠️ 没有检测到可交互控件，无法生成可视化图")
        print("   请确保在有可交互控件的窗口中运行此脚本")

if __name__ == "__main__":
    main()