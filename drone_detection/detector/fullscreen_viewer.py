from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QTimer

class FullscreenViewer(QWidget):
    def __init__(self, source_label):
        super().__init__()
        self.setWindowFlags(Qt.Window)
        self.setWindowState(Qt.WindowFullScreen)

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)
        self.source_label = source_label
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)

    def update_image(self):
        pixmap = self.source_label.pixmap()
        if pixmap:
            self.label.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
