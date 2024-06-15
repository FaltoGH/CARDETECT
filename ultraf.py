import math
import os
from typing import Callable

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from cv2.typing import MatLike

import plainf
import cv2f
import boxesf
import resultsf

DIRNAME = os.path.dirname(__file__)

def get_cls_name(result:Results, box:Boxes)->str:
    return result.names[boxesf.get_cls(box)]

def get_crop_img(result:Results, box:Boxes)->MatLike:
    xyxy = boxesf.get_box_xyxy(box)
    crop_img = result.orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
    return crop_img

def predict(yolo:YOLO, im:MatLike) -> Results:
    results = yolo(im)
    assert len(results) == 1
    result = results[0]
    return result

def predict4results(yolo:YOLO, im:MatLike) -> list:
    """
    Returns four results for each angle, 0, 90, 180, 270,
    but their image is aligned as 0 clockwise.
    """

    # Predict for four angles respectively.
    results = [0]*4

    for i in range(4):
        result = predict(yolo, im)
        results[i] = result

        # Do not rotate at last epoch.
        if i < 3:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    # Rotate results to make them 0 clockwise.
    # Do not rotate the first result.
    for i in range(1, 4):
        for j in range(4 - i):
            rotate_result(results[i])

    return results

def predict_merge(yolo:YOLO, im:MatLike) -> Results:
    results = predict4results(yolo, im)
    ret = resultsf.merge_results(results)
    return ret

def rotate_result(result:Results):
    """
    Rotate the result 90-clockwise inplace.
    """
    result.orig_shape = plainf.rotate_shape(result.orig_shape)
    result.orig_img = cv2f.rotate(result.orig_img)
    result.boxes = boxesf.rotate_boxes(result.boxes)

def predictbest(yolo:YOLO, im:MatLike) -> Results:
    result = predict_merge(yolo, im)
    parent = boxesf.union_boxes(result.boxes)
    result.boxes = boxesf.extract_best(parent, result.boxes)
    return result

def predictconfine(yolo:YOLO, source) -> Results:
    if isinstance(source, str):
        source = cv2.imread(source)

    result = predictbest(yolo, source)
    boxes = result.boxes
    datac = boxes.data.clone()
    boxidx = 0
    for box in boxes:
        name = get_cls_name(result, box)
        if "7" in name:
            crop_img = get_crop_img(result, box)
            is_confined = cv2f.is_there_any_confined_space(crop_img)
            if is_confined:

                # Make box class name contain "9"
                # Set class
                datac[boxidx, -1] = int(datac[boxidx, -1]) + 8
        
        boxidx += 1
    result.boxes = Boxes(datac, result.orig_shape)
    return result

def new_yolo(model=None) -> YOLO:
    if model == None:
        model = os.path.join(DIRNAME, "yolov8s_playing_cards.pt")
    
        if not os.path.isfile(model):
            model = os.path.join(DIRNAME, "yolov8_playing_card_detect", "yolov8s_playing_cards.pt")

            if not os.path.isfile(model):
                raise FileNotFoundError(model)

    yolo = YOLO(model)
    return yolo

def predictred(yolo:YOLO, mat:MatLike):
    mat = cv2f.redmask(mat)
    return predictconfine(yolo, mat)

def do_yolo(yolo:YOLO, predict:Callable[[YOLO, MatLike], Results]):
    mat = cv2f.imread()
    result = predict(yolo, mat)
    show_result(result)
    do_cam_yolo(yolo, predict)

def do_cam_yolo(yolo:YOLO, predict:Callable[[YOLO, MatLike], Results]) -> None:
    """
    YOLO yolo; // YOLOv8 model instance.

    Results predict(YOLO, MatLike); // A function that returns Results.
    """
    def f(frame):
        # Run YOLOv8 inference on the frame
        result = predict(yolo, frame)

        # Visualize the results on the frame
        annotated_frame = result.plot()

        return annotated_frame

    cv2f.do_cam2(f)