import os

import cv2
from ultralytics import YOLO
import util

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")

def main():
    yolo = YOLO(model_path)
    abspath = os.path.join(assets, "0.jpg")
    im = cv2.imread(abspath)
    result = util.predict_v2(yolo, im)
    boxes = result.boxes
    print("boxes.shape=", boxes.shape)
    result.save(os.path.join(assets, "0.jpg_r_pred.jpg"))
    util.show_result(result)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
