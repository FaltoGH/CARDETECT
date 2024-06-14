import os
import cv2
from cv2.typing import MatLike
import numpy as np

DIRNAME = os.path.dirname(__file__)
IMAGE_FILENAME = os.path.join(DIRNAME, "images", "0.jpg")

def imshow(im:MatLike) -> int:
    cv2.imshow("im", im)
    return cv2.waitKey()

def preprocess(src:MatLike) -> MatLike:
    srcc = src.copy()
    image_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=non%20color%20pixels.-,Full%20code%20%3A,-1

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv2.inRange(image_hsv, lower1, upper1)
    upper_mask = cv2.inRange(image_hsv, lower2, upper2)
    
    full_mask = lower_mask + upper_mask
    
    mask_where = np.where(full_mask)
    srcc[mask_where] = (0,0,255)
    return srcc

def main():
    assert os.path.isfile(IMAGE_FILENAME)
    src = cv2.imread(IMAGE_FILENAME)
    pre = preprocess(src)
    imshow(pre)

if __name__ == "__main__":
    main()
