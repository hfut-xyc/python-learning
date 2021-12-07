import cv2
import numpy as np


def onChange(x):
    pass


img = np.zeros((200, 300, 3), np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, onChange)
cv2.createTrackbar('G', 'image', 0, 255, onChange)
cv2.createTrackbar('B', 'image', 0, 255, onChange)

while (True):
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    img[:] = [b, g, r]

    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break