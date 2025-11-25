"""
基于MediaPipe的手部关键点检测模块

该模块提供了基于MediaPipe的手部检测和关键点识别功能，
包括实时手部跟踪、手势识别和手部关键点可视化。
"""

from .hand_detector import HandDetector
from .hand_tracker import HandTracker
from .gesture_recognizer import GestureRecognizer
from .hand_utils import HandUtils

__version__ = "1.0.0"
__author__ = "MediaPipe Hand Tracking"

__all__ = [
    'HandDetector',
    'HandTracker', 
    'GestureRecognizer',
    'HandUtils'
]