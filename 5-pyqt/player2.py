import cv2
import sys
import os
import time

from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

from Ui_mainwindow import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.btn_play.clicked.connect(self.on_play_clicked)
        self.btn_open.clicked.connect(self.on_open_clicked)

    def on_timeout(self) -> None:
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            height, width, _ = frame.shape
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
            image = QImage(frame.data, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.screen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.screen.setPixmap(pixmap)
            cv2.waitKey(40)
            # time.sleep(0.04)

    def on_play_clicked(self) -> None:
        dir = "./test/video"
        play_list = os.listdir(dir)
        for file in play_list:
            print(file)
            self.capture = cv2.VideoCapture(os.path.join(dir, file))
            self.on_timeout()

    def on_open_clicked(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(self, caption="选择本地文件", directory="./test/video", filter="*.mp4")
        if fileName == "":
            return
        self.capture = cv2.VideoCapture(fileName)
        self.on_timeout()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())

