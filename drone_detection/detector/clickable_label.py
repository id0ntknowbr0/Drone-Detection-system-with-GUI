from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal

class ClickableLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()