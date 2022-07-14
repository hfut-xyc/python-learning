import cv2
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

from Ui_mainwindow import Ui_MainWindow

"""
video player based on cv.waitkey
"""
class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.btn_open.clicked.connect(self.on_open_clicked)

    def on_open_clicked(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(self, filter="*.mp4")
        if fileName == "":
            return
        capture = cv2.VideoCapture(fileName)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            height, width, _ = frame.shape
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            image = QImage(frame.data, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.screen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.screen.setPixmap(pixmap)
            cv2.waitKey(40)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())

