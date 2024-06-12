from cv2.typing import MatLike
import numpy as np
from torch import Tensor
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import os
import cv2
import time
from functools import wraps

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "images")
assets_list = os.listdir(assets)
model_path = os.path.join(dirname, "yolov8s_playing_cards.pt")

def classic_predict(yolo:YOLO, im:MatLike) -> Results:
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

def get_intersection_length(a:float,b:float,c:float,d:float)->float:
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
    xl = get_intersection_length(xyxy0[0], xyxy0[2], xyxy1[0], xyxy1[2])
    yl = get_intersection_length(xyxy0[1], xyxy0[3], xyxy1[1], xyxy1[3])
    return xl * yl

def get_xyxy(box:Boxes) -> list:
    ret = [*map(float, box.xyxy[0])]
    
    assert len(ret) == 4
    for i in ret:
        assert i >= 0

    return ret

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

def getcls(box:Boxes)->int:
    return int(box.cls)

def getconf(box:Boxes)->float:
    """
    Confidence (0, 1]
    """
    return float(box.conf)

def argmax(d:dict) -> int:
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
            cls = getcls(box)
            conf = getconf(box)

            if cls not in d:
                d[cls] = [0,0]
            
            d[cls][0] += 1
            d[cls][1] += conf
        
        bestcls = argmax(d)
        for i in range(begin, end+1):
            box = boxarr[i]
            cls = getcls(box)
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
        result = classic_predict(yolo, im)
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

def union_boxes(boxes:Boxes) -> list:
    """
    Union boxes whose IOU is greater than 0.4.
    Returns parent array.
    """
    n = boxes.shape[0]
    parent = [*range(n)]
    xyxys = tuple(map(get_xyxy, boxes))

    for i in range(n-1):
        for j in range(i+1, n):

            a = xyxys[i]
            b = xyxys[j]

            iou = get_xyxy_IOU(a,b)
            if iou > 0.4:
                union(parent, i, j)
    
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

def pred(yolo:YOLO, im:MatLike) -> Results:
    result = get_merged_result(yolo, im)
    parent = union_boxes(result.boxes)
    result.boxes = extract_best(parent, result.boxes)
    return result

if __name__ == "__main__":
    print("main start")

    torch.set_printoptions(sci_mode=False)
    print("set_printoptions done")

    yolo = YOLO(model_path)
    print("initialize YOLO done")

    for asset in assets_list:

        if "pred" in asset: continue
        if asset != "0.jpg": continue

        abspath = os.path.join(assets, asset)

        im = cv2.imread(abspath)
        print("cv2.imread done")
        
        result = pred(yolo, im)
        result.save(os.path.join(assets, asset+"_r_pred.jpg"))

        if show_result(result) != 0:
            break
        
    cv2.destroyAllWindows()