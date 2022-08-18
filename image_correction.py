import numpy as np
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

img_original = cv2.imread('SMA B2 olddataset/A.bmp', 0)
img_original = cv2.resize(img_original, (400, 400))
img_inverted = cv2.bitwise_not(img_original)
cv2.imshow("inverted", img_inverted)


def gamma_correction(src, gamma):
    inv_gamma = 1 / gamma

    table = [((i / 255) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def clahe_correction(src):
    lab_img = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    clahe_img = clahe.apply(1)
    CLAHE_img = cv2.merge((clahe_img, a, b))
    corrected_img = cv2.cvtColor(CLAHE_img, cv2.COLOR_LAB2BGR)
    return corrected_img


def rolling_ball_correction(src):
    radius = 50
    final_img, background = subtract_background_rolling_ball(src, radius, light_background=True,
                                                             use_paraboloid=False, do_presmooth=True)
    return final_img, background


img_rb, img_background = rolling_ball_correction(img_inverted)
img_gamma = gamma_correction(img_rb, 0.3)

clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_rb)

img_denoised = cv2.fastNlMeansDenoising(img_clahe, None, 40, 9, 9)
x = img_denoised

detector = cv2.SimpleBlobDetector_create()

keypoint_info = detector.detect(x)

blobs = cv2.drawKeypoints(x, keypoint_info, np.array([]), (255, 0, 0), flags=4)

cv2.imshow("original", img_original)

cv2.imshow("background", img_background)
cv2.imshow("without background", img_rb)
cv2.imshow("clahe corrected", img_clahe)
cv2.imshow("Displaying blobs", blobs)


cv2.waitKey(0)
cv2.destroyAllWindows()
