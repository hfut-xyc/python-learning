import cv2

backSub = cv2.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()

capture = cv2.VideoCapture('D:/workspace/video/test.mov')
if not capture.isOpened():
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 27:
        break