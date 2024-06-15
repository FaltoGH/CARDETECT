import math
import os
from typing import Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from cv2.typing import MatLike

import plainf
import cv2f

DIRNAME = os.path.dirname(__file__)

def get_box_xyxy(box:Boxes) -> list:
    """
    Get box's xyxy. It is guaranteed that all coordinates are non-negative.
    """
    ret = [*map(float, box.xyxy[0])]
    
    assert len(ret) == 4
    for i in ret:
        assert i >= 0

    return ret

def get_cls(box:Boxes)->int:
    return int(box.cls)

def get_conf(box:Boxes)->float:
    """
    Return confidence (0, 1].
    """
    return float(box.conf)

def extract_best(parent:list, boxes:Boxes) -> Boxes:
    """
    Returns the best boxes.
    """

    # Use two pointer algorithm to do work on the boxes in the same set
    begin = 0
    end = 0
    n = boxes.shape[0]

    if n < 2:
        return boxes
    
    assert len(parent) == n

    for i in range(n):
        parent[i] = plainf.find(parent, i)

    boxarr = [*boxes]
    for i in range(n):
        boxarr[i].parent = parent[i]


    boxarr.sort(key = lambda x:x.conf, reverse = True)

    # The sort is in-place (i.e. the list itself is modified) and stable (i.e. the order of two equal elements is maintained).
    boxarr.sort(key = lambda x:x.parent)

    while begin < n and end < n:

        while end + 1 < n and boxarr[end].parent == boxarr[end + 1].parent:
            end += 1
        
        # begin=Beginning inclusive index of the same set
        # end=Ending inclusive index of the same set

        # d[cls] = (x,y)
        # x is a number of boxes for that class
        # y is sum of confidence.
        d = dict()

        for i in range(begin, end+1):
            box = boxarr[i]
            box:Boxes
            cls = get_cls(box)
            conf = get_conf(box)

            if cls not in d:
                d[cls] = [0,0]
            
            d[cls][0] += 1
            d[cls][1] += conf
        
        bestcls = plainf.argmax_dict(d)
        for i in range(begin, end+1):
            box = boxarr[i]
            cls = get_cls(box)
            if bestcls == cls:
                box.best = True

                # Only one box can be the best box,
                # which has the greatest confidence.
                # (It was previously sorted by confidence by stable)
                bestcls = -1

            else:
                box.best = False

        assert bestcls == -1

        begin = end + 1
        end = end + 1
    
    ret = None

    for box in boxarr:
        
        if ret == None and box.best:
            ret = box

            del box.best
            del box.parent

            continue

        if box.best:
            del box.best
            del box.parent

            ret = cat_boxes(ret, box)
    
    return ret

def cat_boxes(boxes0:Boxes, boxes1:Boxes) -> Boxes:
    """
    Concatenates two boxes into one boxes and returns it.
    Two boxes must have the same orig_shape.
    """
    assert boxes0.orig_shape == boxes1.orig_shape
    return Boxes(torch.cat((boxes0.data, boxes1.data)), boxes0.orig_shape)


def get_cls_name(result:Results, box:Boxes)->str:
    return result.names[get_cls(box)]

def get_crop_img(result:Results, box:Boxes)->MatLike:
    xyxy = get_box_xyxy(box)
    crop_img = result.orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
    return crop_img

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
    
    key = cv2f.imshow(plot) & 0xFF

    if key == ord("q"):
        return 1

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

def rotate_data(data:torch.Tensor, orig_shape:tuple) -> torch.Tensor:
    """
    Rotate data 90 clockwise.
    """

    ret = data.clone()
    shape = ret.shape
    nrow = shape[0]

    for i in range(nrow):
        row = ret[i]
        rotate_row(row, orig_shape)

    return ret

def rotate_row(row:torch.Tensor, orig_shape:tuple) -> None:
    """
    Rotate a tensor row.
    """

    for j in {0,2}:
        x = float(row[j])
        y = float(row[j+1])
        rret = plainf.rotate(x, y, orig_shape)
        row[j] = rret[0]
        row[j+1] = rret[1]
    
    a,b,c,d = float(row[2]), float(row[1]), float(row[0]), float(row[3])
    row[0] = a
    row[1] = b
    row[2] = c
    row[3] = d

    assert a <= c
    assert b <= d

def rotate_boxes(boxes:Boxes) -> Boxes:
    """
    Rotate boxes of result.
    """
    orig_shape = boxes.orig_shape
    data = boxes.data
    data = rotate_data(data, orig_shape)
    orig_shape = orig_shape[::-1]
    ret = Boxes(data, orig_shape)
    return ret

def merge_results(results:list) -> Results:
    """
    Merge multiple results into one.
    """
    ret = results[0]
    ret:Results

    for result in results[1:]:
        result:Results

        for key in ret.speed:
            ret.speed[key] += result.speed[key]

        ret.boxes = cat_boxes(ret.boxes, result.boxes)

    return ret

def union_boxes(boxes:Boxes, standard:float=0.45) -> list:
    """
    Union boxes whose IOU is greater than standard.
    Returns parent array.

    Union operation can be performed only once per each box.
    That means, chained boxes should not be considered as one object,
    like the situation (  [ )  { ]  < }  >
    """

    if standard >= 1:
        return

    n = boxes.shape[0]
    parent = [*range(n)]

    unioned = [False] * n

    xyxys = tuple(map(get_box_xyxy, boxes))

    for i in range(n-1):
        for j in range(i+1, n):

            a = xyxys[i]
            b = xyxys[j]

            iou = plainf.get_xyxy_IOU(a,b)
            if iou > standard:

                # if i has not been unioned, it can be unioned with another box.
                if not unioned[i]:

                    plainf.union(parent, i, j)
                    unioned[i] = True
    
    for i in range(n):
        parent[i] = plainf.find(parent, i)

    return parent

def predict_merge(yolo:YOLO, im:MatLike) -> Results:
    results = predict4results(yolo, im)
    ret = merge_results(results)
    return ret

def rotate_result(result:Results):
    """
    Rotate the result 90-clockwise inplace.
    """
    result.orig_shape = plainf.rotate_shape(result.orig_shape)
    result.orig_img = cv2f.rotate(result.orig_img)
    result.boxes = rotate_boxes(result.boxes)

def predict2(yolo:YOLO, im:MatLike) -> Results:
    result = predict_merge(yolo, im)
    parent = union_boxes(result.boxes)
    result.boxes = extract_best(parent, result.boxes)
    return result

def predict3(yolo:YOLO, source) -> Results:
    if isinstance(source, str):
        source = cv2.imread(source)

    result = predict2(yolo, source)
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

def predict4(yolo:YOLO, mat:MatLike):
    mat = cv2f.redmask(mat)
    return predict3(yolo, mat)

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

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        wkey = cv2.waitKey(444) & 0xFF

        # Break the loop if 'q' is pressed
        if wkey == ord('q'):
            return 1
        
        # Pause the loop if 'p' is pressed
        if wkey == ord("p"):
            cv2.waitKey()
        return 0

    cv2f.do_cam(f)