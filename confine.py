# June 15th, 2024

import os

import cv2
import util

dirname = os.path.dirname(__file__)
assets = os.path.join(dirname, "crop_images")
assets_list = os.listdir(assets)

def main():
    for asset in assets_list:

        abspath = os.path.join(assets, asset)

        org_image = cv2.imread(abspath)

        pre = util.thresh(org_image)

        area, mat = util.confine(pre)
        print(area, area>0)

        if util.imshow(mat) == util.ORD_Q:
            print("Interrupted by pressed key Q")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
