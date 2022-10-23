import cv2
import numpy as np
import matplotlib.pylab  as plt

static_method = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
static_method_name = ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
# THRESH_BINARY: directly compared to threshold value, if larger make it 255, else if smaller make it 0
# THRESH_BINARY_INV: opposite to the THRESH_BINARY, exchange black(0) and white(255)
# THRESH_TRUNC: if larger than threshold value make it 255, else if smaller keep the former value
# THRESH_TOZERO: if smaller than threshold value make it 0, else if larger keep the former value
# THRESH_TOZERO_INV: if larger than threshold value make it 0, else if smaller keep the former value

def static_threshold(img, trs_val): # derived from cv2
    for index, ev_method in enumerate(static_method):
        ret, img_dst = cv2.threshold(img, trs_val, 255, ev_method)
        cv2.imwrite(static_method_name[index] + '.jpg', img_dst)

adaptive_method = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
adaptive_method_name = ['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C']
# ADAPTIVE_THRESH_MEAN_C: take average as threshold value within the nearing block
# ADAPTIVE_THRESH_GAUSSIAN_C: assign weights using Gaussian blocks

def adaptive_threshold(img):
    for index, ev_method in enumerate(adaptive_method):
        img_dst= cv2.adaptiveThreshold(img, 255, ev_method, cv2.THRESH_BINARY, 11, 2) # src, maxValue, adaptiveMethod, thresholdType, blocksize, constant
        cv2.imwrite(adaptive_method_name[index] + '.jpg', img_dst)


def otsu_threshold(img):
    ret1, img_dst1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('otsu.jpg', img_dst1)
    
    # using Gaussian filtering to blur
    blur = cv2.GaussianBlur(img, (5,5) ,0) # structure (5,5)
    ret2, img_dst2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('otsu_after_blur.jpg', img_dst2)
    
    # plt.hist(img.ravel(), 256)
    # plt.savefig('hist.jpg')
    # plt.show()

def otsu(img): # self-write
    h = img.shape[0]
    w = img.shape[1]
    img_dst = np.zeros((h,w),np.uint8)
    threshold_max = threshold = 0 # for temporary and final threshold
    histogram = np.zeros(256, np.int32)
    probability = np.zeros(256, np.float32)
    for i in range (h):
        for j in range (w):
            s = img[i,j]
            histogram[s] += 1 # each pixel
    for k in range (256):
        probability[k] = histogram[k] / (h * w) # proportion of each
    for i in range (255):
        w0 = w1 = 0
        fgs = bgs = 0
        for j in range (256):
            if j <= i: # i as threshold_temp
                w0 += probability[j]
                fgs += j * probability[j]
            else:
                w1 += probability[j]
                bgs += j * probability[j]
        u0 = fgs / w0 # average gray-scale of foreground image
        u1 = bgs / w1 # average gray-scale of background image
        g = w0 * w1 * (u0-u1)**2 # maximal variance between-class
        if g >= threshold_max:
            threshold_max = g
            threshold = i
    # print(threshold)
    for i in range (h):
        for j in range (w):
            if img[i,j] > threshold:
                img_dst[i,j] = 255
            else:
                img_dst[i,j] = 0
    cv2.imwrite('otsu_sf.jpg', img_dst)
    
    # return img_dst
    


if __name__ == '__main__':
    img_gray = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE) # replaced by 0
    # static_threshold(img_gray, 127) # static threshold value
    # adaptive_threshold(img_gray)
    # otsu_threshold(img_gray)
    otsu(img_gray)





