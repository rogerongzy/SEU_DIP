import cv2
import numpy as np

def myenhance(name):
    img_raw = cv2.imread(name)
    gray = True
    # print(img_raw.shape)

    # gray image confirmation
    for m in range(img_raw.shape[0]):
        for n in range(img_raw.shape[1]):
            if img_raw[m][n][0] != img_raw[m][n][1] or img_raw[m][n][0] != img_raw[m][n][2]:
                gray = False
    if gray:
        print('GRAY image.')
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY) # RGB & BGR
    # print(img_gray.shape)

    digits = np.zeros((256,1),dtype=np.int)
    for m in range(img_gray.shape[0]):
        for n in range(img_gray.shape[1]):

            # find different digits numbers
            # for i in range(256):
            #     if img_gray[m][n] == i:
            #         digits[i] += 1

            # enhancement process
            if img_gray[m][n] < 127:
                img_gray[m][n] = img_gray[m][n] * 20 #
            else:
                if img_gray[m][n] > 227:
                    img_gray[m][n] = img_gray[m][n] + 25 #
                elif img_gray[m][n] < 227:
                    img_gray[m][n] = img_gray[m][n] - 25 #
    
    # for i in range(256):
    #     print(str(i) + ':  ' + str(digits[i]))
    # cv2.imshow('show', img_gray)
    # cv2.waitKey(0)
    cv2.imwrite('output_' + name, img_gray)

def main():
    name = 'Fig0326(a)(embedded_square_noisy_512).tif'
    myenhance(name)
    

if __name__ == '__main__':
    main()
