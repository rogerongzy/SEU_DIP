import cv2
import numpy as np

def sw16to8(image_path):
    img16 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    min16 = np.min(img16)
    max16 = np.max(img16)
    img8 = np.array(np.rint(255 * ((img16 - min16) / (max16 - min16))), dtype=np.uint8)
    cv2.imwrite('task 1 test_8-bit.jpg', img8) # jpg

sw16to8('./task 1 test_16-bit.tif')