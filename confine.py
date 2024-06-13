# license: unlicense
# author: Falto
# created: June 13th, 2024


import numpy as np
import cv2
import os
import math
import sys
from cv2.typing import MatLike
from numpy import ndarray
import skimage.measure

dx=(0,0,1,-1)
dy=(1,-1,0,0)
dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "crop_images")
assets_list = os.listdir(assets)
ord_q = ord('q')

BLACK = 0
WHITE = 255
GRAY = 180
ESC = 40


def imshow(im:MatLike) -> int:
    while im.shape[1] < 800 and im.shape[0] < 400:
        im = cv2.resize(im, (im.shape[1]<<1, im.shape[0]<<1))
    cv2.imshow("imshow", im)
    return cv2.waitKey(0)

def preprocess(im:MatLike) -> ndarray:
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
            nextpoint = (top[0]+dy[i], top[1]+dx[i])

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

def main():
    for asset in assets_list:

        abspath = os.path.join(assets, asset)

        org_image = cv2.imread(abspath)

        pre = preprocess(org_image)

        area, mat = confine(pre)
        print(area, area>0)

        if imshow(mat) == ord_q:
            print("Interrupted by pressed key Q")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
