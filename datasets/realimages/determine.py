"""
Created: 2024-06-08
Author: falto

This program determines and fixes a file name of image and label.
Since labelme2yolo generates random file name,
this program is needed for deterministic.

First execute result and second execute result and after all results are guaranteed as equal.
That is, there is no problem when you execute this program more than once.
"""

import os

def sum_file(filename):
    with open(filename, "rb") as f:
        bytess = f.read()
        return sum(bytess)

dirname = os.path.dirname(__file__)
os.chdir(dirname)

dataset = os.path.join(dirname, "YOLODataset")
images = os.path.join(dataset, "images")
labels = os.path.join(dataset, "labels")
modes = ["train", "val"]

if __name__ == "__main__":
    for mode in modes:
        imdir = os.path.join(images, mode)
        lbdir = os.path.join(labels, mode)

        imlist = os.listdir(imdir)
        for im in imlist:
            abspath = os.path.join(imdir, im)
            labelpath = os.path.join(lbdir, im[:-3]+"txt")
            filesum = sum_file(abspath)
            os.rename(abspath, os.path.join(imdir, "%d.png"%filesum))
            os.rename(labelpath, os.path.join(lbdir, "%d.txt"%filesum))
