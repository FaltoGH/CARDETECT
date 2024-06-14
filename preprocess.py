# June 15th, 2024

import os
import cv2
import util

DIRNAME = os.path.dirname(__file__)
IMAGE_FILENAME = os.path.join(DIRNAME, "images", "0.jpg")

def main():
    assert os.path.isfile(IMAGE_FILENAME)
    src = cv2.imread(IMAGE_FILENAME)
    pre = util.mask_red(src)
    util.imshow(pre)

if __name__ == "__main__":
    main()
