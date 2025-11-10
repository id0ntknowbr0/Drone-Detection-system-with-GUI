import sys
import os
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

