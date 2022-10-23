import cv2
import os
import numpy as np
import matplotlib.pylab  as plt
from numpy.core.fromnumeric import diagonal

def crop(img): # default cropped into 6 pieces
    height = int(img.shape[0] / 2)
    width = int(img.shape[1] / 3)
    print(img.shape)
    print('height = ' + str(height))
    print('width = ' + str(width))

    for i in range(6):
        if i == 1:
            img_crop = img[0 : height, 0 : width]
            cv2.imwrite(str(i) + '.png', img_crop)
        # if i == 2:
        #     img_crop = img[0 : height, width : 2 * width]
        #     cv2.imwrite(str(i) + '.png', img_crop)
        # if i == 3:
        #     img_crop = img[0 : height, 0 : height]
        #     cv2.imwrite(str(i) + '.png', img_crop)
        # if i == 4:
        #     img_crop = img[0 : height, 0 : height]
        #     cv2.imwrite(str(i) + '.png', img_crop)
        # if i == 5:
        #     img_crop = img[0 : height, 0 : height]
        #     cv2.imwrite(str(i) + '.png', img_crop)
        # if i == 6:
        #     img_crop = img[0 : height, 0 : height]
        #     cv2.imwrite(str(i) + '.png', img_crop)



def test():
    for filename in os.listdir():
        if filename.endswith('0.png'): # carocr024744d4-77c5-4dc0-8078-8442b07a19e8.png
            img = cv2.imread(filename, 0) # 0 for directly grayscale
            # print(img.shape)

            # sharpen
            # kernel = np.array([[0, -1, 0],
            #                 [-1, 5, -1],
            #                 [0, -1, 0]])
            # dst = cv2.filter2D(img, -1, kernel) # can be complemented on grayscale
            
            # convert to binary
            # gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

            # threshold to binary
            ret, img_dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # otsu on sharpen
            # ret, img_dst = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

            cv2.imshow('test', img_dst)
            cv2.waitKey(0)
            # cv2.imwrite('tri_sharpen.png', dst)
            # cv2.imwrite('sharpen.png', dst)
            # cv2.imwrite('ostu_sharpen2.png', img_dst)



def blob(): # 斑块法不能检测到打算去掉的瑕疵
    img = cv2.imread('binary.png', 0)
    # img = cv2.imread('carocr024744d4-77c5-4dc0-8078-8442b07a19e8.png')
    # print(img.shape)

    # params
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.maxArea = 80 # params = 20


    detector = cv2.SimpleBlobDetector_create(params) # with params
    # detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(img)
    # print(keypoints[0])
    with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", with_keypoints)
    cv2.waitKey(0)
    cv2.imwrite('2_2.png', with_keypoints)

    # hough
    # img_bk = cv2.imread('binary.png')
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=10, minRadius=0, maxRadius=20) # params2: 25 or 30
    # for i in circles[0, :]:  # [[pos_x, pos_y], radius], color, thickness
    #     pos_x = i[0]
    #     pos_y = i[1]
    #     radius = i[2]

    #     cv2.circle(img_bk, (pos_x, pos_y), int(radius), (0, 255, 0), 2) # outer round(green), needing radius
    #     cv2.circle(img_bk, (pos_x, pos_y), 2, (0, 0, 255), 3) # heart(red), no need for radius

    # cv2.imshow('show', img_bk)
    # cv2.waitKey(0)




def scan_mask(): # 暴力法效果很差
    img = cv2.imread('binary.png', 0)
    height = img.shape[0]
    width = img.shape[1]
    # print(height)
    # print(width)

    bl_pix = 0
    # row = []

    for i in range(height):
        for j in range(width):
            # print(img[i][j])
            if img[i][j] == 0:
                bl_pix = bl_pix + 1
        # print(bl_pix)
        if bl_pix < 40: # 30 as params
            for x in range(width):
                img[i][x] = 255
            # row.append(i)
        bl_pix = 0

    # cv2.imshow('output', img)
    # cv2.waitKey(0)
    cv2.imwrite('1_3_40.png', img)
    # mask = np.zeros(height, width)
    # for each_row in row:
    #     for x in range(width):



def contours():
    img = cv2.imread('4_5.png', 0)
    img_op = 255 - img
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL cv2.RETR_LIST

    # dispose noises
    disposed = []

    for mono in contours:
        # print(len(mono))
        if len(mono) < 33: # 30
            # contours.remove(mono)
            disposed.append(mono)
    
    for single_dis in disposed:
        contours.remove(single_dis)


    # mask = np.ones(img.shape, np.uint8) * 255
    # cv2.drawContours(mask, contours, -1, 0, 1) # -1 = all writing, 0 = color,  thickness = -1

    cv2.drawContours(img, disposed, -1, 255, -1)


    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.imwrite('dst1.png', img)
    


def structured():
    img = cv2.imread('dst1.png', 0)

    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dst = cv2.erode(img, kernel)

    # for i in range(3):
    #     img_out = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    #     cv2.imshow('output', img_out)
    #     cv2.waitKey(0)
    #     img = img_out



    # img1 = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # cv2.imwrite('4_5.png', img1)
    # cv2.imshow('output', img1)
    # cv2.waitKey(0)

    # img2 = cv2.dilate(img1, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # cv2.imwrite('4_6.png', img2)
    # cv2.imshow('output', img2)
    # cv2.waitKey(0)


    img_out = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cv2.imwrite('dst3.png', img_out)

    cv2.imshow('output', img_out)
    cv2.waitKey(0)



def sliding_windows():
    img = cv2.imread('binary.png', 0)
    mask = np.ones(img.shape, np.uint8) * 255
    # size of the windows
    windows_height = 20
    windows_width = 10
    step_height = 5
    step_width = 5
    start_h = 0
    start_w = 0
    block_num = 0

    loop_flag = True

    print(img.shape[0])
    print(img.shape[1])

    while(loop_flag):
        pos_pixel = 0

        if (start_h + windows_height) < img.shape[0]:
            if (start_w + windows_width) < img.shape[1]:

                for i in range(start_h, start_h + windows_height, 1):
                    for j in range(start_w, start_w + windows_width, 1):
                        # print(str(i) + '  ' + str(j))
                        if img[i][j] == 0:
                            pos_pixel = pos_pixel + 1

                # print('==============')
                # print('BLOCK: ' + str(block_num))
                # print(pos_pixel)
                if pos_pixel > 30:
                    # print('over and coresponding')
                    for m in range(start_h, start_h + windows_height, 1):
                        for n in range(start_w, start_w + windows_width, 1):
                            mask[m][n] = 0


        start_w = start_w + step_width
        start_h = start_h + step_height
        block_num = block_num + 1    # cv2.imshowng', img2)
    # cv2.imshow('output', img2)
    # cv2.waitKey(0)

if __name__ == '__main__':
    # test()
    # img = cv2.imread('carocr024744d4-77c5-4dc0-8078-8442b07a19e8.png')
    # crop(img)
    # blob()
    # scan_mask()
    # contours()
    structured()
    # sliding_windows()




# divided into different bloccks, use otsu or histogram-matching separately


# drawing histogram
# plt.hist(img.ravel(), 256)
# plt.savefig('hist.jpg')


# peng shaung: dividing into blocks and adapt otsu's method to separate at once
# maybe sharpen is useful? make a try

# my works: lowering noises, angle adapting & minimum rectangles for masks