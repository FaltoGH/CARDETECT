# license: unlicense
# author: Falto
# created: June 13th, 2024

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

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")
crop_images = os.path.join(dirname, "crop_images")

def get_xyxy(box:Boxes) -> list:
    ret = [*map(float, box.xyxy[0])]
    
    assert len(ret) == 4
    for i in ret:
        assert i >= 0

    return ret

def show_result(result:Results) -> None:
    """
    Show result. If the user press q key, return 1.
    Otherwise, return 0.
    """
    plot = result.plot()
    plot:np.ndarray

    # Resize plot if it is too big to display
    while plot.shape[0] > 1000 or plot.shape[1] > 1900:
        plot = cv2.resize(plot, (plot.shape[1]//2, plot.shape[0]//2))
    
    
    key = imshow(plot) & 0xFF

    if key == ord("q"):
        return 1

def imshow(im:MatLike) -> int:
    cv2.imshow("imshow", im)
    return cv2.waitKey(0)

def get_cls_name(result:Results, box:Boxes)->str:
    return result.names[int(box.cls)]

def get_crop_img(result:Results, box:Boxes)->MatLike:
    xyxy = get_xyxy(box)
    crop_img = result.orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
    return crop_img


def main():
    yolo = YOLO(model_path)

    abspath = os.path.join(assets, "0.jpg")

    im = cv2.imread(abspath)
    
    result = pred(yolo, im)


    boxes = result.boxes

    datac = boxes.data.clone()
    boxidx = 0

    for box in boxes:
        name = get_cls_name(result, box)
        if "7" in name:
            crop_img = get_crop_img(result, box)
            is_confined = confine.is_there_any_confined_space(crop_img)
            if is_confined:

                # Make box class name contain "9"
                # Set class
                datac[boxidx, -1] = int(datac[boxidx, -1]) + 8
        
        boxidx += 1
    
    result.boxes = Boxes(datac, result.orig_shape)
    
    show_result(result)
    result.save(os.path.join(assets, "0.jpg_rc_pred.jpg"))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
