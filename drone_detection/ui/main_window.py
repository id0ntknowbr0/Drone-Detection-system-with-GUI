import sys
import cv2
import numpy as np
import threading
import pygame
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLineEdit, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import deque

from detector.drone_detector import DroneDetector
from detector.fullscreen_viewer import FullscreenViewer
from detector.clickable_label import ClickableLabel

# Hide pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

MODEL_PATH = "assets/best.pt"
ALARM_PATH = "assets/beep.mp3"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Detection")
        self.setStyleSheet("background-color: #2e2e2e; color: white;")

        self.labels = []
        self.detectors = []
        self.threads = []
        self.rtsp_fields = []

        self.is_recording = False
        self.video_writer = None
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.record_frame)

        layout = QVBoxLayout()
        title_layout = QHBoxLayout()

        logo_label = QLabel()
        logo_path = "assets/images.jpg"
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaledToHeight(24, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setFixedSize(logo_pixmap.size())
        else:
            logo_label.setText("Logo Not Found")
            logo_label.setStyleSheet("font-size: 12px; color: red;")
        title_layout.addWidget(logo_label)

        title = QLabel("Drone Detection")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        url_bar = QHBoxLayout()
        for i in range(4):
            field = QLineEdit()
            field.setPlaceholderText(f"Camera {i+1} RTSP URL")
            field.setStyleSheet("background-color: #444; color: white;")
            self.rtsp_fields.append(field)
            url_bar.addWidget(field)
        layout.addLayout(url_bar)

        grid_container = QWidget()
        grid_container.setStyleSheet("background-color: white;")

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        for i in range(4):
            label = ClickableLabel("No Stream")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #000000; color: white;")
            label.setScaledContents(True)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setMinimumSize(200, 150)
            label.doubleClicked.connect(lambda l=label: self.open_fullscreen(l))
            grid.addWidget(label, i // 2, i % 2)
            self.labels.append(label)

        grid_container.setLayout(grid)
        layout.addWidget(grid_container)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Detection")
        self.stop_btn = QPushButton("Stop Detection")
        self.screenshot_btn = QPushButton("Screenshot")
        self.record_btn = QPushButton("Start Screen Recording")

        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        self.record_btn.clicked.connect(self.toggle_recording)

        for btn in [self.start_btn, self.stop_btn, self.screenshot_btn, self.record_btn]:
            btn.setFixedHeight(40)
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def start_detection(self):
        self.detectors = []
        self.threads = []
        rtsp_urls = [field.text() for field in self.rtsp_fields]

        for i in range(4):
            if not rtsp_urls[i]:
                continue
            detector = DroneDetector(rtsp_urls[i], self.labels[i], ALARM_PATH, MODEL_PATH)
            t = threading.Thread(target=detector.run, daemon=True)
            t.start()
            self.detectors.append(detector)
            self.threads.append(t)

    def stop_detection(self):
        for detector in self.detectors:
            detector.running = False

    def take_screenshot(self):
        screenshot = self.grab()
        screenshot_image = screenshot.toImage()
        width, height = screenshot_image.width(), screenshot_image.height()
        ptr = screenshot_image.bits()
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        arr = arr[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(script_dir, f"screenshot_{timestamp}.png")

        if not cv2.imwrite(filename, arr):
            QMessageBox.critical(self, "Error", f"Failed to save screenshot to {filename}")
        else:
            QMessageBox.information(self, "Screenshot Saved", f"Screenshot saved as {filename}")

    def toggle_recording(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp4_filename = os.path.join(script_dir, f"recording_{timestamp}.mp4")

        if not self.is_recording:
            self.is_recording = True
            self.record_btn.setText("Stop Screen Recording")

            screenshot = self.grab()
            width, height = screenshot.width(), screenshot.height()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(mp4_filename, fourcc, 10.0, (width, height))
            self.record_timer.start(100)

            QMessageBox.information(self, "Recording Started", f"Recording will be saved as {mp4_filename}")
        else:
            self.is_recording = False
            self.record_btn.setText("Start Screen Recording")
            self.record_timer.stop()
            self.video_writer.release()
            self.video_writer = None
            QMessageBox.information(self, "Recording Stopped", f"Recording saved as {mp4_filename}")

    def record_frame(self):
        if self.is_recording and self.video_writer is not None:
            screenshot = self.grab()
            screenshot_image = screenshot.toImage()
            width, height = screenshot_image.width(), screenshot_image.height()
            ptr = screenshot_image.bits()
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            arr = arr[:, :, :3]
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            self.video_writer.write(arr)

    def open_fullscreen(self, label):
        self.fullscreen_viewer = FullscreenViewer(label)
        self.fullscreen_viewer.show()



