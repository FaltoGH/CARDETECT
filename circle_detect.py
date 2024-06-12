# license: unlicense
# author: falto
# created: June 13rd, 2024

# incomplete

from rotational_invariance_pred import pred
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
import cv2
import os
import math

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")

def get_xywh(box:Boxes) -> list:
    ret = [*map(float, box.xywh[0])]
    
    assert len(ret) == 4
    for i in ret:
        assert i >= 0

    return ret

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
    
    cv2.imshow("plot", plot)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        return 1
    
if __name__ == "__main__":
    yolo = YOLO(model_path)

    for asset in assets_list:
        if "pred" in asset: continue
        if asset != "0.jpg": continue

        abspath = os.path.join(assets, asset)

        im = cv2.imread(abspath)
        
        result = pred(yolo, im)

        show_result(result)

        boxes = result.boxes
        orig_img = result.orig_img

        cv2.destroyAllWindows()

        for box in boxes:
            xyxy = get_xyxy(box)
            
            crop_img = orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
            cv2.imshow("crop_img", crop_img)
            cv2.waitKey(0)
        
    cv2.destroyAllWindows()
