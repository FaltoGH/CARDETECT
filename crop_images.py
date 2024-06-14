# June 13th, 2024

from ultralytics import YOLO
import numpy as np
import cv2
import os
import math
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

    util.show_result(result)

    boxes = result.boxes
    print("boxes.shape=", boxes.shape)

    orig_img = result.orig_img

    writedir = os.path.join(dirname, "crop_images")
    if not os.path.isdir(writedir):
        os.mkdir(writedir)

    i = 0

    for box in boxes:
        xyxy = util.get_box_xyxy(box)
        crop_img = orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
        cv2.imwrite(os.path.join(writedir, "%d.jpg"%i), crop_img)
        util.imshow(crop_img)
        i += 1
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
