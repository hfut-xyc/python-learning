import numpy as np
import cv2

'''
read image
'''
img = cv2.imread("./res/test.jpg")

cv2.imshow("frame1", img)
cv2.waitKey(0)
cv2.imwrite("test1.jpg", img)

cv2.imshow("frame2", img[:, :, ::-1])
cv2.waitKey(0)
cv2.imwrite("test2.jpg", img[:, :, ::-1])

cv2.imshow("frame3", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.imwrite("test3.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

'''
read video
'''
# capture = cv2.VideoCapture("d:/xyc/test.mp4")
# cv2.namedWindow("frame")
# while True:
#     ret, frame = capture.read()

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == 27:
#         break
