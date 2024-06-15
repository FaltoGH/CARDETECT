# June 15th, 2024

import math
import os
from typing import Union, Callable

import skimage
from torch import Tensor
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import numpy as np
import cv2
from cv2.typing import MatLike

DIRNAME = os.path.dirname(__file__)
BLACK = 0
WHITE = 255
GRAY = 180
ESC = 40
DX=(0,0,1,-1)
DY=(1,-1,0,0)
ORD_Q = ord('q')

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
    while im.shape[1] < 800 and im.shape[0] < 400:
        im = cv2.resize(im, (im.shape[1]<<1, im.shape[0]<<1))
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

def get_cls(box:Boxes)->int:
    return int(box.cls)

def get_cls_name(result:Results, box:Boxes)->str:
    return result.names[get_cls(box)]

def get_conf(box:Boxes)->float:
    """
    Return confidence (0, 1].
    """
    return float(box.conf)

def get_crop_img(result:Results, box:Boxes)->MatLike:
    xyxy = get_box_xyxy(box)
    crop_img = result.orig_img[int(xyxy[1]):math.ceil(xyxy[3]), int(xyxy[0]):math.ceil(xyxy[2])]
    return crop_img

def predict_v1(yolo:YOLO, im:MatLike) -> Results:
    results = yolo(im)
    assert len(results) == 1
    result = results[0]
    return result

def find(parent:list, x:int) -> int:
    """
    Find root node index of the tree.
    """
    if parent[x] == x:
        return x
    
    ret = find(parent, parent[x])
    parent[x] = ret
    return ret

def union(parent:list, x:int, y:int) -> int:
    """
    Union two nodes.
    """
    xp = find(parent, x)
    yp = find(parent, y)
    if yp < xp:
        parent[xp] = yp
    else:
        parent[yp] = xp
    
    return xp

def get_intersect_length(a:float,b:float,c:float,d:float)->float:
    if a > b:
        b, a = a, b
    
    if c > d:
        d, c = c, d
    
    if a > c:
        a, b, c, d = c, d, a, b
    
    if a != 0:
        a, b, c, d = 0, b-a, c-a, d-a

    # a == 0

    if c == 0:
        ret = min(b, d)

    elif b <= c:
        ret = 0
    
    elif d <= b:
        ret = d - c
    
    else:
        ret = max(0, b - c)
    
    assert ret >= 0

    return ret

def get_intersection_area(xyxy0:list, xyxy1:list) -> float:
    assert len(xyxy0)==4
    assert len(xyxy1)==4
    xl = get_intersect_length(xyxy0[0], xyxy0[2], xyxy1[0], xyxy1[2])
    yl = get_intersect_length(xyxy0[1], xyxy0[3], xyxy1[1], xyxy1[3])
    return xl * yl

def get_area(xyxy:list)->float:
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    assert w > 0
    assert h > 0
    ret = w * h
    assert ret > 0
    return ret

def get_xyxy_IOU(xyxy0:list, xyxy1:list) -> float:
    """
    Returns real number that belongs to interval [0, 1].
    """

    ia = get_intersection_area(xyxy0, xyxy1)
    ua = get_area(xyxy0) + get_area(xyxy1) - ia

    assert ua > 0

    ret =  ia / ua

    if ret < 0 or ret > 1:
        raise AssertionError("ret=%f"%ret)

    return ret

def argmax_dict(d:dict) -> int:
    """
    Return class which has the greatest number of boxes.
    If there are multiple, higher sum of confidences is chosen.
    """
    first = True
    ret = -1

    for key in d:
        if first:
            first = False
            ret = key
            continue

        # number of box compare
        if d[key][0] > d[ret][0]:
            ret = key

        elif d[key][0] == d[ret][0]:

            # sum of confidence compare, if number of box are equal
            if d[key][1] > d[ret][1]:
                ret = key
    
    assert ret != -1
    return ret


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
        parent[i] = find(parent, i)

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
        
        bestcls = argmax_dict(d)
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


def get_four_results(yolo:YOLO, im:MatLike) -> list:
    """
    Returns four results for each angle, 0, 90, 180, 270,
    but their image is aligned as 0 clockwise.
    """

    # Predict for four angles respectively.
    results = [0]*4

    for i in range(4):
        result = predict_v1(yolo, im)
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


def cat_boxes(boxes0:Boxes, boxes1:Boxes) -> Boxes:
    """
    Concatenates two boxes into one boxes and returns it.
    Two boxes must have the same orig_shape.
    """
    assert boxes0.orig_shape == boxes1.orig_shape
    return Boxes(torch.cat((boxes0.data, boxes1.data)), boxes0.orig_shape)

def rotate_xy(x:float,y:float,orig_shape:tuple)->tuple:
    """
    Rotate 90 clockwise.
    """
    
    assert len(orig_shape) == 2
    h = orig_shape[0]
    return (h - y, x)

def rotate_row(row:Tensor, orig_shape:tuple) -> None:
    """
    Rotate a tensor row.
    """

    for j in {0,2}:
        x = float(row[j])
        y = float(row[j+1])
        rret = rotate_xy(x, y, orig_shape)
        row[j] = rret[0]
        row[j+1] = rret[1]
    
    a,b,c,d = float(row[2]), float(row[1]), float(row[0]), float(row[3])
    row[0] = a
    row[1] = b
    row[2] = c
    row[3] = d

    assert a <= c
    assert b <= d


def rotate_data(data:Tensor, orig_shape:tuple) -> Tensor:
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

def get_merged_result(yolo:YOLO, im:MatLike) -> Results:
    results = get_four_results(yolo, im)
    ret = merge_results(results)
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

            iou = get_xyxy_IOU(a,b)
            if iou > standard:

                # if i has not been unioned, it can be unioned with another box.
                if not unioned[i]:

                    union(parent, i, j)
                    unioned[i] = True
    
    for i in range(n):
        parent[i] = find(parent, i)

    return parent

def rotate_orig_shape(orig_shape:tuple) -> tuple:
    """
    Returns rotated shape.
    """
    return orig_shape[::-1]

def rotate_orig_img(orig_img:MatLike) -> MatLike:
    """
    Returns rotated img.
    """
    return cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)

def rotate_result(result:Results) -> int:
    """
    Rotate the result 90 clockwise inplace.
    """
    result.orig_shape = rotate_orig_shape(result.orig_shape)
    result.orig_img = rotate_orig_img(result.orig_img)
    result.boxes = rotate_boxes(result.boxes)
    return 0

def predict_v2(yolo:YOLO, im:MatLike) -> Results:
    result = get_merged_result(yolo, im)
    parent = union_boxes(result.boxes)
    result.boxes = extract_best(parent, result.boxes)
    return result


def thresh(im:MatLike) -> np.ndarray:
    """
    Preprocess an image.
    Letter and symbol part is denoted as 255 (white).
    The other, such as background, is denoted as 0 (black).
    """

    # Convert it to gray
    ret = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Reduce the noise to avoid false circle detection
    ret = cv2.medianBlur(ret, 5)

    ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    ret = skimage.measure.block_reduce(ret, (2,2), np.max)

    return ret

def shift(start:list, shape:tuple) -> int:
    """
    Shift a start point to a next point.
    If a start point gets outside, return 1; otherwise, 0.
    """

    if start[1] + 1 < shape[1]:
        start[1] += 1
        return 0
    
    if start[0] + 1 < shape[0]:
        start[0] += 1
        start[1] = 0
        return 0
    
    start[0] += 1
    start[1] = 0
    return 1

def mov_start(mat:MatLike, start:list) -> int:
    shift(start, mat.shape)

    while (start[0] < mat.shape[0]) and (mat[start[0],start[1]] != BLACK):
        shift(start, mat.shape)
    
    if start[0] >= mat.shape[0]:
        return 1
    
    return 0

def is_inside(point:tuple, shape:tuple) -> bool:
    """
    Is point inside shape?
    """
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]

def do_start(mat:MatLike, start:list) -> tuple[int, MatLike]:
    """
    DFS.
    If a confined space is not found, return (0, mat).
    If a confined space is found, return (the number of visit, mat).
    """

    stack = []
    stack.append(start)
    nvisit = 0

    while stack:
        top = stack.pop()

        color = mat[top[0], top[1]]
        if color != BLACK:
            continue
        
        nvisit += 1
        mat[top[0], top[1]] = GRAY

        for i in range(4):
            nextpoint = (top[0]+DY[i], top[1]+DX[i])

            # if nextpoint is out of bound
            if not is_inside(nextpoint, mat.shape):
                # Escaping out of the box is possible!
                # So it is not confined.
                # Mark the path escapable.
                # Replace GRAY to ESC.
                mat = np.where(mat == GRAY, ESC, mat)
                return (0, mat)
            
            nextcolor = mat[nextpoint[0], nextpoint[1]]

            # if escape possible
            if nextcolor == ESC:
                mat = np.where(mat == GRAY, ESC, mat)
                return (0, mat)
        
            # if already visited
            if nextcolor == GRAY:
                continue

            # if it is wall
            if nextcolor == WHITE:
                continue

            assert nextcolor == BLACK
            stack.append(nextpoint)

    return (nvisit, mat)


def confine(mat:MatLike) -> tuple[int, MatLike]:
    start = [0,0]

    while start[0] < mat.shape[0]:
        nvisit, mat = do_start(mat, start)
        if nvisit > 0:
            return (nvisit, mat)
        mov_start(mat, start)
    
    return (0, mat)

def is_there_any_confined_space(mat:MatLike) -> bool:
    pre = thresh(mat)
    area, mat = confine(pre)
    return area > 0

def predict_v3(yolo:YOLO, im:MatLike) -> Results:
    result = predict_v2(yolo, im)
    boxes = result.boxes
    datac = boxes.data.clone()
    boxidx = 0
    for box in boxes:
        name = get_cls_name(result, box)
        if "7" in name:
            crop_img = get_crop_img(result, box)
            is_confined = is_there_any_confined_space(crop_img)
            if is_confined:

                # Make box class name contain "9"
                # Set class
                datac[boxidx, -1] = int(datac[boxidx, -1]) + 8
        
        boxidx += 1
    result.boxes = Boxes(datac, result.orig_shape)
    return result

def predict_v3a(yolo:YOLO, source:Union[str, MatLike]) -> Results:
    if isinstance(source, str):
        source = cv2.imread(source)
    return predict_v3(yolo, source)

def mask_red(src:MatLike) -> MatLike:
    srcc = src.copy()
    image_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=non%20color%20pixels.-,Full%20code%20%3A,-1

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv2.inRange(image_hsv, lower1, upper1)
    upper_mask = cv2.inRange(image_hsv, lower2, upper2)
    
    full_mask = lower_mask + upper_mask
    
    mask_where = np.where(full_mask)
    srcc[mask_where] = (0,0,255)
    return srcc

def do_cam(f:Callable[[MatLike], int], index:int=0, apiPreference:int=cv2.CAP_DSHOW) -> None:
    cap = cv2.VideoCapture(index, apiPreference)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            if f(frame): break

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def new_yolo(model=None) -> YOLO:
    if model == None:
        model = os.path.join(DIRNAME, "yolov8s_playing_cards.pt")
    
        if not os.path.isfile(model):
            model = os.path.join(DIRNAME, "yolov8_playing_card_detect", "yolov8s_playing_cards.pt")

            if not os.path.isfile(model):
                model = None

    yolo = YOLO(model)
    return yolo

def predict_v4(yolo:YOLO, mat:MatLike):
    mat = mask_red(mat)
    return predict_v3a(yolo, mat)

def test_red_mask_cam() -> None:
    def f(frame):
        result = mask_red(frame)

        # Display the annotated frame
        cv2.imshow("test_red_mask_cam", result)

        wkey = cv2.waitKey(444) & 0xFF

        # Break the loop if 'q' is pressed
        if wkey == ORD_Q:
            return 1
        
        # Pause the loop if 'p' is pressed
        if wkey == ord("p"):
            cv2.waitKey()
        
        return 0
    
    do_cam(f)

def test_yolo_cam(yolo:YOLO, predict:Callable[[YOLO, MatLike], Results]) -> None:
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
        if wkey == ORD_Q:
            return 1
        
        # Pause the loop if 'p' is pressed
        if wkey == ord("p"):
            cv2.waitKey()
        return 0

    do_cam(f)

def test_predict4(yolo:YOLO):
    test_yolo_im(yolo, predict_v4)
    test_yolo_cam(yolo, predict_v4)

def test_yolo_im(yolo:YOLO, predict:Callable[[YOLO, MatLike], Results], source:Union[str, MatLike]=None) -> int:
    if source == None:
        source = os.path.join(DIRNAME, "images", "0.jpg")
        
        if not os.path.isfile(source):
            source = None
    
    if isinstance(source, str):
        source = cv2.imread(source)
    
    result = predict(yolo, source)
    return show_result(result)

if __name__ == "__main__":
    # Insert your test code below

    yolo = new_yolo()
    test_predict4(yolo)
