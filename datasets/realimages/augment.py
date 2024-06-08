import os
from PIL import Image

def rotate(x:float, y:float, w:float, h:float, angle:int) -> tuple:
    """
    Rotate xywh for clockwise.
    x, y, w, h are real numbers which belong to interval [0, 1] which is relative to image size.
    """

    xcand = [x, 1-y, 1-x, y]
    ycand = [y, x, 1-y, 1-x]
    wcand = [w, h, w, h]
    hcand = [h, w, h, w]

    assert angle % 90 == 0
    p = angle//90

    return (xcand[p], ycand[p], wcand[p], hcand[p])

class Label:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, "r") as f:
            self.lines = f.readlines()
        
    def save(self, angle:int):
        newlines = []
        for line in self.lines:

            tokens = line.split()
            ntok = len(tokens)
            if ntok == 0:
                continue

            assert ntok == 5
            
            xywh = tuple(map(float, tokens[1:]))

            newxywh = rotate(xywh[0], xywh[1], xywh[2], xywh[3], angle)
            snewxywh = [*map(str, newxywh)]

            newlines.append(tokens[0] + " " + (" ".join(snewxywh)) + "\n")

        with open(self.filename[:-3] + ("%d.aug.txt"%angle), "w") as f:
            f.writelines(newlines)

dirname = os.path.dirname(__file__)
os.chdir(dirname)

dataset = os.path.join(dirname, "YOLODataset")
images = os.path.join(dataset, "images")
labels = os.path.join(dataset, "labels")

modes = ["train", "val"]
angles = [90, 180, 270]

if __name__ == "__main__":
    for mode in modes:
        imdir = os.path.join(images, mode)
        lbdir = os.path.join(labels, mode)

        imlist = os.listdir(imdir)
        for im in imlist:

            if im.endswith(".aug.png"):
                continue

            abspath = os.path.join(imdir, im)
            number_plus_dot = im[:-3]
            labelpath = os.path.join(lbdir, number_plus_dot+"txt")

            org_image = Image.open(abspath)

            # Only square matrix is expected.
            if org_image.width != org_image.height:
                continue

            label = Label(labelpath)

            for angle in angles:
                im1 = org_image.rotate(angle)
                im1.save(os.path.join(imdir, number_plus_dot + ("%d.aug.png" % angle)))
                label.save(angle)
