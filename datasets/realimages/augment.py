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

            for angle in angles:
                im1 = org_image.rotate(angle)
                im1.save(os.path.join(imdir, number_plus_dot + ("%d.aug.png" % angle)))

