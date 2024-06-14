import math
from ultralytics.engine.results import Results, Boxes
import numpy as np
import cv2
from cv2.typing import MatLike

def get_box_xyxy(box:Boxes) -> list:
    """
    Get box's xyxy. It is guaranteed that all coordinates are non-negative.
    """
    ret = [*map(float, box.xyxy[0])]
    
    assert len(ret) == 4
    for i in ret:
        assert i >= 0

    return ret

def imshow(im:MatLike) -> int:
    """
    Show image. Return key waited.
    """
    cv2.imshow("imshow", im)
    return cv2.waitKey()

def show_result(result:Results) -> None:
    """
    Show YOLOv8 result by opencv.
    If the user press q key, return 1; otherwise, 0.
    """
    plot = result.plot()
    plot:np.ndarray

    # Resize plot if it is too big to display
    while plot.shape[0] > 1000 or plot.shape[1] > 1900:
        plot = cv2.resize(plot, (plot.shape[1]//2, plot.shape[0]//2))
    
    key = imshow(plot) & 0xFF

    if key == ord("q"):
        return 1

def get_cls_name(result:Results, box:Boxes)->str:
    return result.names[int(box.cls)]

def get_crop_img(result:Results, box:Boxes)->MatLike:
    xyxy = get_box_xyxy(box)
    crop_img = result.orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
    return crop_img
