import cv2
import numpy as np
import os

# 0:green, 1:gray, 2:orange, 3:yellow, 4:purple (lower, upper, bgr)
color = [
    [[ 35,  43,  46], [ 77, 255, 255], (  0, 230,   0)],
    [[ 90,  12, 100], [115,  60, 210], (220, 220, 220)],
    [[ 11, 220, 170], [ 16, 255, 255], (  0, 165, 255)],
    [[ 18, 130, 160], [ 32, 210, 210], (  0, 255, 255)],
    [[130,  43, 100], [170, 120, 170], (255,   0, 255)]
]

def hsv_detect():
    for file in os.listdir():
        if file.endswith('.JPG'):
            img_bgr = cv2.imread(file)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) # COLOR_BGR2HSV_FULL
            
            for i in range(len(color)):
                lower = np.array(color[i][0])
                upper = np.array(color[i][1])
                mask = cv2.inRange(img_hsv, lower, upper) # using hsv for color detecting
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)) # structure initialization
                mask = cv2.dilate(mask, kernel) # dilate for surely connected area

                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # RETR_TREE

                # find the largest rect-area, max 2
                maxarea = 0
                max2area = 0
                maxindex = 0
                max2index = 0
                for index in range(len(contours)):
                    x, y, w, h = cv2.boundingRect(contours[index])
                    if w * h > max2area:
                        max2index = index
                        max2area = w * h
                        if w * h > maxarea:
                            max2index = maxindex
                            max2area = maxarea
                            maxindex = index
                            maxarea = w * h

                # viewable using boundingRect
                x, y, w, h = cv2.boundingRect(contours[maxindex])
                x2, y2, w2, h2 = cv2.boundingRect(contours[max2index])
                if w * h > 600000: # lower threshold, omit the candies
                    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color[i][2], 5)
                if w2 * h2 > 600000:
                    cv2.rectangle(img_bgr, (x2, y2), (x2 + w2, y2 + h2), color[i][2], 5)

            cv2.imwrite('out_' + file, img_bgr)

if __name__ == '__main__':
    hsv_detect()