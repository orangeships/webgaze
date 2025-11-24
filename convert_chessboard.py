#!/usr/bin/env python
"""
将SVG棋盘格转换为PNG格式，方便打印
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

def create_chessboard_8x5(output_path="chessboard_8x5.png"):
    """
    创建8×5的棋盘格标定板
    
    Args:
        output_path: 输出PNG文件路径
    """
    # 棋盘格参数
    rows = 5
    cols = 8
    square_size = 25  # 每个方格25mm
    border = 20  # 边框宽度
    
    # 计算图像尺寸（转换为像素，假设300 DPI）
    dpi = 300
    px_per_mm = dpi / 25.4  # 每毫米对应多少像素
    
    width_mm = cols * square_size + 2 * border
    height_mm = rows * square_size + 2 * border
    
    width_px = int(width_mm * px_per_mm)
    height_px = int(height_mm * px_per_mm)
    
    # 创建白色背景
    img = Image.new('RGB', (width_px, height_px), 'white')
    draw = ImageDraw.Draw(img)
    
    # 计算每个方格的像素大小
    square_px = int(square_size * px_per_mm)
    border_px = int(border * px_per_mm)
    
    # 绘制棋盘格
    for row in range(rows):
        for col in range(cols):
            # 计算方格位置
            x = border_px + col * square_px
            y = border_px + row * square_px
            
            # 交替填充黑白
            if (row + col) % 2 == 0:
                draw.rectangle([x, y, x + square_px, y + square_px], fill='black')
            else:
                draw.rectangle([x, y, x + square_px, y + square_px], fill='white')
    
    # 保存图片
    img.save(output_path, 'PNG', dpi=(dpi, dpi))
    print(f"棋盘格已保存到: {output_path}")
    
    # 打印尺寸信息
    print(f"图像尺寸: {width_px}x{height_px} 像素")
    print(f"物理尺寸: {width_mm}x{height_mm} mm")
    print(f"方格尺寸: {square_size}x{square_size} mm")
    print(f"DPI: {dpi}")
    
    return output_path

if __name__ == "__main__":
    # 生成棋盘格
    output_file = create_chessboard_8x5("g:\\mattest\\Gaze estimation\\WebCamGazeEstimation-main\\camera_data\\chessboard_8x5_A4.png")