import cv2 as cv

def estimate_Rrect(src_pts, dst_pts, F, height, width):
    _, H1, H2 = cv.stereoRectifyUncalibrated(src_pts, dst_pts, F, imgSize=(width, height))
    
    return H1, H2