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
import sys
from cv2.typing import MatLike

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")

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

def main(argc:int, argv:list[str]):
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

        while True:

            line = sys.stdin.readline()
            param2 = int(line)

            for box in boxes:
                xyxy = get_xyxy(box)
                crop_img = orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]

                #https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
                # Convert it to gray
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

                # Reduce the noise to avoid false circle detection
                gray = cv2.medianBlur(gray, 5)

                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=param2, minRadius=1, maxRadius=0)
                print("circles=", circles)

                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for i in circles[0, :]:
                        center = (i[0], i[1])

                        # circle center
                        cv2.circle(crop_img, center, 1, (0, 100, 100), 3)

                        # circle outline
                        radius = i[2]
                        cv2.circle(crop_img, center, radius, (255, 0, 255), 3)

                imshow(crop_img)
        
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    ret=main(len(sys.argv), sys.argv)
    if ret:raise Exception(ret)
