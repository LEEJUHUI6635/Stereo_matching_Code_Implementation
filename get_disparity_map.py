import cv2 as cv
import matplotlib.pyplot as plt

def get_disparity_map(left_image, right_image):
    stereo = cv.StereoSGBM_create(numDisparities=128, blockSize=15)
    disparity = stereo.compute(left_image, right_image)
    plt.imshow(disparity, 'gray')
    plt.show()