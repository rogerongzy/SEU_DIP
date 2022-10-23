import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_split(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    b, g, r = cv2.split(img)
    return b, g, r

def img_merge(b, g, r):
    img_m = cv2.merge([b, g, r])
    return img_m

def print_hist(y, name):
    plt.figure()
    x = np.arange(0, 256)
    plt.bar(x, y, width=1)
    # plt.show()
    plt.savefig('hist_' + name)

# histogram of frequency
def get_hist_mono(img):
    gray = np.zeros(256)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            gray[img[h][w]] += 1
    gray /= (img.shape[0] * img.shape[1]) # frequency of digitsValue
    return gray

# accumulation histogram of the frequency
def get_hist_acmu(gray):
    hist_acmu = []
    acmu_val = 0.
    for i in gray:
        acmu_val += i
        hist_acmu.append(acmu_val)
    return hist_acmu

# fill pixels according to the hist_acmu
def repixels(img):
    des_img = np.zeros((img.shape[0], img.shape[1]), int)
    hist = get_hist_mono(img)
    hist_acmu = get_hist_acmu(hist)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            des_img[h][w] = int(hist_acmu[img[h][w]] * 255.0 + 0.5)  
    return des_img


# def run_histogram_equalization(file_path):
#     img = loadimg(file_path)
#     hist = get_hist_mono(img)
#     hist_acmu = get_hist_acmu(hist)
#     des_img = repixels(img)
#     # print_hist(hist, "before")
#     # print_hist(hist_acmu, "acmu")
#     new_hist = get_hist_mono(des_img)
#     # print_hist(new_hist, "after")
#     # cv2.imshow('After Equalization', des_img)


def run_histogram_match_mono(img_mono_input, img_mono_template):

    hist1 = get_hist_mono(img_mono_input)
    hist2 = get_hist_mono(img_mono_template)

    # print_hist(hist1, "input")
    # print_hist(hist2, "template")

    hist_acmu1 = get_hist_acmu(hist1)
    hist_acmu2 = get_hist_acmu(hist2)

    new_array = []
    for acmu_val in hist_acmu1:
        # get the difference between histograms of input and template
        diff = list(abs(np.array(hist_acmu2 - acmu_val))) 
        closest_index = diff.index(min(diff))
        new_array.append(closest_index)

    new_img = np.zeros((img_mono_input.shape[0], img_mono_input.shape[1]), int)
    for h in range(img_mono_input.shape[0]):
        for w in range(img_mono_input.shape[1]):
            new_img[h][w] = new_array[img_mono_input[h][w]]

    new_hist = get_hist_mono(new_img) 
    # print_hist(new_hist, "output")
    return new_img, new_hist


def run_histogram_match_bgr(file_path_input, file_path_temp):
    b_in, g_in, r_in = img_split(file_path_input)
    b_tp, g_tp, r_tp = img_split(file_path_temp)

    nimg_b, nhist_b = run_histogram_match_mono(b_in, b_tp)
    nimg_g, nhist_g = run_histogram_match_mono(g_in, g_tp)
    nimg_r, nhist_r = run_histogram_match_mono(r_in, r_tp)

    img_final = img_merge(nimg_b, nimg_g, nimg_r)

    return img_final


if __name__ == '__main__':
    file_path_input = 'input.jpg'
    file_path_temp = 'template.jpg'  
    img_final = run_histogram_match_bgr(file_path_input, file_path_temp)
    # cv2.imshow('final', img_final)
    # cv2.waitKey(0)
    cv2.imwrite('output.jpg', img_final)