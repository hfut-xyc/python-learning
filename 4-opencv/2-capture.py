import cv2

capture = cv2.VideoCapture("d:/xyc/test.mp4")
cv2.namedWindow("frame")
while True:
    ret, frame = capture.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break
