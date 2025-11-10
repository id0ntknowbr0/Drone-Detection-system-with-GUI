import threading
import os
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap

from detector.drone_detector import DroneDetector
from detector.fullscreen_viewer import FullscreenViewer
from detector.clickable_label import ClickableLabel

MODEL_PATH = "assets/best.pt"
ALARM_PATH = "assets/beep.mp3"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Detection")
        self.setStyleSheet("background-color: #2e2e2e; color: white;")
        # (Everything else unchanged — paste your original MainWindow code here)
