
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image():
    img = cv2.imread("res/test.jpg")
    img = img / 255
    cv2.imshow("frame1", img)
    cv2.waitKey(0)
    cv2.imwrite("test1.jpg", img)

    cv2.imshow("frame2", img[:, :, ::-1])
    cv2.waitKey(0)
    cv2.imwrite("test2.jpg", img[:, :, ::-1])

    cv2.imshow("frame3", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.imwrite("test3.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def histogram():

    # img = cv2.imread('dog.jpg')
    # color = ('b','g','r')

    # for i, item in enumerate(color):
    #     hist = cv2.calcHist([img], [i], None, [256], [0,256])
    #     plt.plot(hist, color = item)
    #     plt.xlim([0,256])
    # plt.show()

    img = cv2.imread('res/test.jpg', 0)

    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221)
    plt.imshow(img, 'gray')

    plt.subplot(222)
    plt.imshow(mask, 'gray')

    plt.subplot(223)
    plt.imshow(masked_img, 'gray')

    plt.subplot(224)
    plt.plot(hist_full, color='red')
    plt.plot(hist_mask, color='blue')
    plt.xlim([0, 256])

    plt.show()


def read_video():
    capture = cv2.VideoCapture("res/test.mp4")
    cv2.namedWindow("frame")
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(30) == 27:
            break

def bg_substract():
    # backSub = cv2.createBackgroundSubtractorMOG2()
    backSub = cv2.createBackgroundSubtractorKNN()

    capture = cv2.VideoCapture('res/test.mp4')

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame = backSub.apply(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) == 27:
            break


if __name__ == '__main__':
    histogram()