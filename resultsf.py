from ultralytics.engine.results import Results
import numpy as np

import cv2f
import boxesf

def show_result(result:Results) -> int:
    """
    Show YOLOv8 result by opencv.
    If the user press q key, return 1; otherwise, 0.
    """
    plot = result.plot()
    plot:np.ndarray

    key = cv2f.imshow(plot) & 0xFF

    if key == ord("q"):
        return 1
    
    return 0

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

        ret.boxes = boxesf.cat_boxes(ret.boxes, result.boxes)

    return ret
