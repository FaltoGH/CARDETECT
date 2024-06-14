# June 15th, 2024

import os
import sys
import math
from pathlib import Path
from typing import Union

import skimage
import skimage.measure
import torch
from torch import Tensor
import ultralytics
import ultralytics.engine
import ultralytics.engine.results
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from ultralytics.utils.plotting import Annotator
import numpy as np
from numpy import ndarray
import cv2
from cv2.typing import MatLike
import util


MODEL_FILENAME = os.path.join(util.DIRNAME, "yolov8s_playing_cards.pt")
if not os.path.isfile(MODEL_FILENAME):
    MODEL_FILENAME = os.path.join(util.DIRNAME, "yolov8_playing_card_detect", "yolov8s_playing_cards.pt")
yolo = YOLO(MODEL_FILENAME)

def test_predict_images() -> None:
    d=os.path.join(util.DIRNAME, "images")
    for i in ["0.jpg", "640x360.png", "854x480.png", "1280x720.png", "1280x720_2.png"]:
        util.predict_v3a(os.path.join(d, i)).show()

def test_predict_webcam() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            result = util.predict_v3a(frame)

            # Visualize the results on the frame
            annotated_frame = result.plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            wkey = cv2.waitKey(444) & 0xFF

            # Break the loop if 'q' is pressed
            if wkey == util.ORD_Q:
                break
            
            # Pause the loop if 'p' is pressed
            if wkey == ord("p"):
                cv2.waitKey(0)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def main():
    test_predict_webcam()

if __name__ == "__main__":
    main()
