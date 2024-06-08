import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)

smodel=os.path.join(dirname, "runs", "detect", "train7", "weights", "best.pt")

# if useyolo is False, only preprocessed image is shown.
useyolo = True

test=os.path.join(dirname, "images")

if useyolo and not os.path.isfile(smodel):
    raise FileNotFoundError(smodel)

if not os.path.isdir(test):
    raise FileNotFoundError(test)

import cv2
import numpy as np

def preprocess(src):
    # incomplete
    return src

def preprocess_ims():
    for x in os.listdir(test):
        if x.endswith(".jpg"):
            abspath = os.path.join(test, x)
            im = cv2.imread(abspath)
            yield preprocess(im)

if useyolo:
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO(smodel)

def predict(x):
    if not useyolo:
        cv2.imshow("Preprocess Result", x)
        cv2.waitKey(0)
        return

    # Run YOLOv8 inference on the frame
    results = model(x)

    for result in results:
        # Visualize the results on the frame
        annotated_frame = result.plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        cv2.waitKey(0)

if __name__ == "__main__":

    for i in preprocess_ims():
        predict(i)

    # Close the display window
    cv2.destroyAllWindows()
