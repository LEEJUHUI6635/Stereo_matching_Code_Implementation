import cv2 as cv

def rectify_image(left_image, right_image, H1, H2, height, width):
    left_rectified = cv.warpPerspective(left_image, H1, (width, height))
    right_rectified = cv.warpPerspective(right_image, H2, (width, height))
    
    return left_rectified, right_rectified