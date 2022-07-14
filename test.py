import cv2
import os
import numpy as np


def normalize(imgs):
    imgs = imgs / 255
    return imgs

def play_video(path_video: str):
    files = os.listdir(path_video)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    
    video = [cv2.imread(os.path.join(path_video, f)) for f in files]
    # video = [img[:, :, ::-1] for img in video] # BGR -> RGB
    video = np.stack(video, axis=0)

    for i in range(video.shape[0]):
        frame = video[i]
        cv2.imshow('frame', frame)
        cv2.waitKey(100)


if __name__ == '__main__':
    play_video('D:/xyc/workspace/dataset/GRID/lip/swbz4p')
