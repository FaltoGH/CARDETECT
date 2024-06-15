import cv2
import cv2f
import ultraf

if __name__ == "__main__":
    # Insert your test code below
    yolo = ultraf.new_yolo()
    def f(y, m):
        m = cv2f.thresh(m, 0, False)
        m = cv2.bitwise_not(m)
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        result = ultraf.predictconfine(y, m)
        return result
    
    ultraf.do_yolo(yolo, f)
