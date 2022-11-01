import numpy as np
import cv2 as cv

def undistort_image(file_name, distorted_image, intrinsic_parameters, distortion_parameters, lr):
    height = distorted_image.shape[0]
    width = distorted_image.shape[1]
    clean_image = np.zeros_like(distorted_image)

    fx = intrinsic_parameters['fx']
    cx = intrinsic_parameters['cx']
    fy = intrinsic_parameters['fy']
    cy = intrinsic_parameters['cy']
    
    k1 = distortion_parameters['k1']
    k2 = distortion_parameters['k2']
    k3 = distortion_parameters['k3']
    p1 = distortion_parameters['p1']
    p2 = distortion_parameters['p2']
    
    for y in range(height):
        for x in range(width):
            y_nu = (y-cy)/fy
            x_nu = (x-cx)/fx

            ru2 = x_nu*x_nu + y_nu*y_nu
            radial_d = 1 + k1*ru2 + k2*ru2*ru2 + k3*ru2*ru2*ru2
            
            x_nd = radial_d*x_nu + 2*p1*x_nu*y_nu + p2*(ru2 + 2*x_nu*x_nu)
            y_nd = radial_d*y_nu + p1*(ru2 + 2*y_nu*y_nu) + 2*p2*x_nu*y_nu

            x_pd = fx*x_nd + cx
            y_pd = fy*y_nd + cy

            x_pd = int(x_pd)
            y_pd = int(y_pd)
            clean_image[y, x] = distorted_image[y_pd, x_pd]
            
    if lr.lower() == 'left':
        cv.imwrite('./left_results/{}'.format(file_name), clean_image)
    elif lr.lower() == 'right':
        cv.imwrite('./right_results/{}'.format(file_name), clean_image)