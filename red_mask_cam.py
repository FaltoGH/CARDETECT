import util

yolo = util.new_yolo()

def f(y, m):
    m = util.mask_red(m)
    return util.predict_v3a(y, m)

util.test_im(yolo, f)
