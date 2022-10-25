import cv2 as cv
import numpy as np
import os

# main.py에서 가져다 쓸 수 있도록 함수로 만들기

if not os.path.exists('./left_results'):
    os.makedirs('./left_results')
if not os.path.exists('./right_results'):
    os.makedirs('./right_results')

left_folder = 'KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_00/data'
right_folder = 'KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_01/data'

# left image에 대하여, 
# 왜곡된 영상 -> I_d / 보정된 I_u, I_u의 각 픽셀값을 해당 픽셀 좌표를 왜곡시켰을 때의 I_d에 대응되는 픽셀값으로 채우는 것
# 모든 image에 대하여,
distorted_files = sorted([file for file in os.listdir(left_folder)])
# print(distorted_files)

distorted_images = [cv.imread(os.path.join(left_folder, file)) for file in distorted_files]
# print(len(distorted_images)) # 28

height = distorted_images[0].shape[0] 
width = distorted_images[0].shape[1]

# print(height, width) # 512, 1392

# left intrinsic parameter
fx = 9.842439e+02
cx = 6.900000e+02
fy = 9.808141e+02
cy = 2.331966e+02

# left distortion parameter
k1 = -3.728755e-01
k2 = 2.037299e-01
k3 = 2.219027e-03
p1 = 1.383707e-03
p2 = -7.233722e-02

clean_image = np.zeros_like(distorted_images[0])
for idx, distorted_image in enumerate(distorted_images):
    for y in range(height): # [0, 512]
        for x in range(width): # [0, 1392]
            y_nu = (y-cy)/fy
            x_nu = (x-cx)/fx
            
            # 왜곡 계수
            ru2 = x_nu*x_nu + y_nu*y_nu
            radial_d = 1 + k1*ru2 + k2*ru2*ru2 + k3*ru2*ru2*ru2
            
            x_nd = radial_d*x_nu + 2*p1*x_nu*y_nu + p2*(ru2 + 2*x_nu*x_nu)
            y_nd = radial_d*y_nu + p1*(ru2 + 2*y_nu*y_nu) + 2*p2*x_nu*y_nu

            # 최종적으로 아래의 값들이 이미지의 픽셀값이 되어야 한다.
            x_pd = fx*x_nd + cx
            y_pd = fy*y_nd + cy
            # print(x_pd, y_pd)
            x_pd = int(x_pd)
            y_pd = int(y_pd)
            clean_image[y, x] = distorted_image[y_pd, x_pd]
    cv.imwrite('./left_results/{0:010d}.png'.format(idx), clean_image)
    
# # right image에 대하여,

# distorted_files = sorted([file for file in os.listdir(right_folder)])
# print(distorted_files)

# distorted_images = [cv.imread(os.path.join(right_folder, file)) for file in distorted_files]
# # print(len(distorted_images)) # 28

# height = distorted_images[0].shape[0] 
# width = distorted_images[0].shape[1]

# # print(height, width) # 512, 1392

# # right intrinsic parameter
# # K_01: 9.895267e+02 0.000000e+00 7.020000e+02 0.000000e+00 9.878386e+02 2.455590e+02 0.000000e+00 0.000000e+00 1.000000e+00
# fx = 9.895267e+02
# cx = 7.020000e+02
# fy = 9.878386e+02
# cy = 2.455590e+02

# # D_01: -3.644661e-01 1.790019e-01 1.148107e-03 -6.298563e-04 -5.314062e-02
# # right distortion parameter
# k1 = -3.644661e-01
# k2 = 1.790019e-01
# k3 = 1.148107e-03
# p1 = -6.298563e-04
# p2 = -5.314062e-02

# clean_image = np.zeros_like(distorted_images[0])
# for idx, distorted_image in enumerate(distorted_images):
#     for y in range(height): # [0, 512]
#         for x in range(width): # [0, 1392]
#             y_nu = (y-cy)/fy
#             x_nu = (x-cx)/fx
            
#             # 왜곡 계수
#             ru2 = x_nu*x_nu + y_nu*y_nu
#             radial_d = 1 + k1*ru2 + k2*ru2*ru2 + k3*ru2*ru2*ru2
            
#             x_nd = radial_d*x_nu + 2*p1*x_nu*y_nu + p2*(ru2 + 2*x_nu*x_nu)
#             y_nd = radial_d*y_nu + p1*(ru2 + 2*y_nu*y_nu) + 2*p2*x_nu*y_nu

#             # 최종적으로 아래의 값들이 이미지의 픽셀값이 되어야 한다.
#             x_pd = fx*x_nd + cx
#             y_pd = fy*y_nd + cy
#             # print(x_pd, y_pd)
#             x_pd = int(x_pd)
#             y_pd = int(y_pd)
#             clean_image[y, x] = distorted_image[y_pd, x_pd]
#     cv.imwrite('./right_results/{0:010d}.png'.format(idx), clean_image)
    
# # 하나의 image에 대하여,
# distorted_image = cv.imread('KITTI-Dataset/2011_09_26_drive_0048/unsync_unrect/image_00/data/0000000000.png')
# # print(distorted_image.shape) # [512, 1392, 3] = [height, width, channel]
# # 하나의 image에 대하여 모든 픽셀 좌표를 불러온다.
# height = distorted_image.shape[0]
# width = distorted_image.shape[1]
# # intrinsic parameter
# # 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00
# fx = 9.842439e+02
# cx = 6.900000e+02
# fy = 9.808141e+02
# cy = 2.331966e+02

# # distortion parameter
# # D_00: -3.728755e-01 2.037299e-01 2.219027e-03 1.383707e-03 -7.233722e-02
# k1 = -3.728755e-01
# k2 = 2.037299e-01
# k3 = 2.219027e-03
# p1 = 1.383707e-03
# p2 = -7.233722e-02

# clean_image = np.zeros_like(distorted_image)
# for y in range(height): # [0, 512]
#     for x in range(width): # [0, 1392]
#         y_nu = (y-cy)/fy
#         x_nu = (x-cx)/fx
        
#         # 왜곡 계수
#         ru2 = x_nu*x_nu + y_nu*y_nu
#         radial_d = 1 + k1*ru2 + k2*ru2*ru2 + k3*ru2*ru2*ru2
        
#         x_nd = radial_d*x_nu + 2*p1*x_nu*y_nu + p2*(ru2 + 2*x_nu*x_nu)
#         y_nd = radial_d*y_nu + p1*(ru2 + 2*y_nu*y_nu) + 2*p2*x_nu*y_nu

#         # 최종적으로 아래의 값들이 이미지의 픽셀값이 되어야 한다.
#         x_pd = fx*x_nd + cx
#         y_pd = fy*y_nd + cy
#         # print(x_pd, y_pd)
#         x_pd = int(x_pd)
#         y_pd = int(y_pd)
#         clean_image[y, x] = distorted_image[y_pd, x_pd]

# if not os.path.exists('./results'):
#     os.makedirs('./results')

# cv.imwrite('./results/first_image.png', clean_image)