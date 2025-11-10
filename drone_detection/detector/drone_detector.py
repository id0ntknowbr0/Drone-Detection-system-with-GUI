import cv2
import numpy as np
import pygame
import os
from collections import deque
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from PySide6.QtGui import QImage, QPixmap

class DroneDetector:
    def __init__(self, rtsp_url, label, alarm_path, model_path):
        self.rtsp_url = rtsp_url
        self.label = label
        self.model_path = model_path
        self.alarm_path = alarm_path

        self.model = YOLO(self.model_path)
        self.running = True

        pygame.mixer.init()

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 1000
        self.kf.R = np.array([[5, 0],
                              [0, 5]])
        self.kf.Q = np.eye(4) * 0.01
        self.kf.x = np.array([[0], [0], [0], [0]])

        self.tracked_drones = deque(maxlen=200)
        self.drone_count = 0

    def play_alarm(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(self.alarm_path)
            pygame.mixer.music.play(-1)

    def stop_alarm(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    def run(self):
        cap = cv2.VideoCapture(0 if self.rtsp_url == "0" else self.rtsp_url)
        if not cap.isOpened():
            self.label.setText(f"Could not open {self.rtsp_url}")
            return

        frames = 0
        skipped = 5

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (320, 240))

                frames += 1
                if frames % skipped != 0:
                    continue

                results = self.model.predict(source=frame, show=False, conf=0.35)
                boxes = []
                confidences = []

                for box in results[0].boxes:
                    conf = box.conf.item()
                    if conf >= 0.35:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        confidences.append(conf)

                if boxes:
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.4)
                    boxes = [boxes[i] for i in indices.flatten()]

                self.play_alarm() if boxes else self.stop_alarm()

                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    x2, y2 = x + w, y + h
                    z = np.array([[x + w // 2], [y + h // 2]])
                    self.kf.predict()
                    self.kf.update(z)
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Drone {confidences[i]:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(image))

        finally:
            cap.release()
            self.stop_alarm()
            pygame.mixer.quit()
