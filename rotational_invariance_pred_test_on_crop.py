# obsolete

import os
import math
import sys

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
import cv2
from cv2.typing import MatLike

import util

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "crop_images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")

def main():
    yolo = YOLO(model_path)

    for asset in assets_list:
        abspath = os.path.join(assets, asset)

        im = cv2.imread(abspath)
        
        result = util.predict_v2(yolo, im)

        util.show_result(result)

        boxes = result.boxes
        print("boxes.shape=", boxes.shape)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
