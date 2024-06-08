import os
from PIL import Image

dirname = os.path.dirname(__file__)
os.chdir(dirname)

dataset = os.path.join(dirname, "YOLODataset")
images = os.path.join(dataset, "images")
labels = os.path.join(dataset, "labels")
modes = ["train", "val"]
angles = [90,180,270]

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

            for angle in angles:
                im1 = org_image.rotate(angle)
                im1.save(os.path.join(imdir, number_plus_dot + ("%d.aug.png" % angle)))
