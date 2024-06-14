import os
import math
import sys

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
import cv2
from cv2.typing import MatLike

from rotational_invariance_pred import pred
import confine
import util

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")
crop_images = os.path.join(dirname, "crop_images")

def main():
    yolo = YOLO(model_path)

    abspath = os.path.join(assets, "0.jpg")

    im = cv2.imread(abspath)
    
    result = pred(yolo, im)


    boxes = result.boxes

    datac = boxes.data.clone()
    boxidx = 0

    for box in boxes:
        name = util.get_cls_name(result, box)
        if "7" in name:
            crop_img = util.get_crop_img(result, box)
            is_confined = confine.is_there_any_confined_space(crop_img)
            if is_confined:

                # Make box class name contain "9"
                # Set class
                datac[boxidx, -1] = int(datac[boxidx, -1]) + 8
        
        boxidx += 1
    
    result.boxes = Boxes(datac, result.orig_shape)
    
    util.show_result(result)
    result.save(os.path.join(assets, "0.jpg_rc_pred.jpg"))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
