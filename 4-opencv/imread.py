import numpy as np
import cv2
import os


img = cv2.imread("dog.jpg", cv2.IMREAD_COLOR)
# print(os.getcwd())
print(img.shape)


# cv2.imshow("test", img)
# cv2.waitKey(0)
