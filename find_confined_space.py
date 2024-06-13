# license: unlicense
# author: Falto
# created: June 13th, 2024

# incomplete

import numpy as np
import cv2
import os
import math
import sys
from cv2.typing import MatLike
from numpy import ndarray

dx=(0,0,1,-1)
dy=(1,-1,0,0)

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "crop_images")
assets_list = os.listdir(assets)
Q = ord('q')

def imshow(im:MatLike) -> int:
    cv2.imshow("imshow", im)
    return cv2.waitKey(0)

def preprocess(im:MatLike) -> ndarray:
    # Convert it to gray
    ret = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Reduce the noise to avoid false circle detection
    ret = cv2.medianBlur(ret, 5)

    ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    ret = np.where(ret != 0, 1, 0)

    return ret

def shift(start:list, shape) -> int:
    """
    Shift a start point to a next point.
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

def is_visitable(point:list, shape, mat:MatLike, visit:np.ndarray) -> bool:
    if visit[point[0]][point[1]]:
        return False
    
    if mat[point[0]][point[1]] != 0:
        return False
    
    if point[0] < 0 or point[1] < 0:
        return False
    
    if point[0] >= shape[0] or point[1] >= shape[1]:
        return False
    
    return True

def mov_to_visitable(start:list, shape, mat:MatLike, visit:np.ndarray) -> int:
    """
    If moving is unavailable, return 1.
    Otherwise, return 0.
    """

    while not is_visitable(start, shape, mat, visit):
        if shift(start, shape) != 0:
            return 1
    
    return 0

def inside(point:tuple, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]

def is_confined(start:list, shape, mat:MatLike, visit:np.ndarray) -> bool:
    
    stack = []
    stack.append(tuple(start))

    nvisit = 0

    while stack:
        top = stack.pop()

        visit[top[0]][top[1]] = True
        nvisit += 1

        for i in range(4):
            nextpoint = (top[0]+dy[i], top[1]+dx[i])

            if not inside(nextpoint, shape):
                # Escaping out of the box is possible!
                # So it is not confined.
                return False
        
            # if it is wall
            if mat[nextpoint[0]][nextpoint[1]] != 0:
                continue

            # if it is already visited
            if visit[nextpoint[0]][nextpoint[1]]:
                continue

            stack.append(nextpoint)

    return True


def is_there_a_confined_space(mat:MatLike) -> bool:
    shape = mat.shape
    # height = shape[0]
    # width = shape[1]
    visit = np.zeros(shape, bool)
    start = [0, 0]

    while True:
        if mov_to_visitable(start, shape, mat, visit) != 0:
            break
        
        if is_confined(start, shape, mat, visit):
            return True

    return False

def main():
    for asset in assets_list:
        abspath = os.path.join(assets, asset)

        im = cv2.imread(abspath)

        # https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html

        pre = preprocess(im)

        result = is_there_a_confined_space(pre)
        print(result)

        if imshow(pre) == Q:
            print("Interrupted by pressed key Q")
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
