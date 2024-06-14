import util

yolo = util.new_yolo()
util.test_im(yolo, util.predict_v4)
util.test_cam(yolo, util.predict_v4)
