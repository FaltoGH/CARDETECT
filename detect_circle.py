# license: unlicense
# author: Falto
# created: June 13th, 2024

# incomplete

import numpy as np
import cv2
import os
import math
import sys
from cv2.typing import MatLike

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "crop_images")
assets_list = os.listdir(assets)
Q = ord('q')

def imshow(im:MatLike) -> int:
    cv2.imshow("imshow", im)
    return cv2.waitKey(0)

def preprocess(im:MatLike) -> MatLike:
    # Convert it to gray
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Reduce the noise to avoid false circle detection
    blur = cv2.medianBlur(gray, 5)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

def get_circles(pre:MatLike) -> np.ndarray | None:
    lon = max(pre.shape[0], pre.shape[1])
    circles = cv2.HoughCircles(pre, cv2.HOUGH_GRADIENT, 1, lon, param1=200, param2=14, minRadius=1, maxRadius=0)
    return circles

def draw_circles(src:MatLike, circles:np.ndarray | None) -> int:
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])

            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)

            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)
        
        return 0
    
    return 1

def main():
    for asset in assets_list:
        abspath = os.path.join(assets, asset)

        im = cv2.imread(abspath)

        # https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html

        pre = preprocess(im)
        if imshow(pre) == Q:
            break

        circles = get_circles(pre)
        print("circles=", circles)
        draw_circles(im, circles)

        if imshow(im) == Q:
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
