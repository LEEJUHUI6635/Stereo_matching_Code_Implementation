import cv2 as cv

def get_F_matrix(src_pts, dst_pts):
    F, mask = cv.findFundamentalMat(src_pts, dst_pts, cv.FM_LMEDS)
    
    return F, mask