import math
import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=15, win_size=5) -> (np.ndarray, np.ndarray):
    if win_size % 2 == 0:
        raise Exception("Window size must be odd")
    if im1.shape != im2.shape:
        raise Exception("Images' shapes must be equal")
    # The lists that will be returned
    u_v_list = []
    change_points = []
    # The der kernels
    kernel_x = np.array([[1, 0, -1]])
    kernel_y = kernel_x.transpose()
    fx = cv2.filter2D(im1, -1, kernel_x)
    fy = cv2.filter2D(im1, -1, kernel_y)
    ft = im1 - im2
    # For the loop
    half = win_size//2
    # Loop over a window size by every step size
    for i in range(win_size+step_size, im1.shape[0] - win_size, step_size):
        for j in range(win_size+step_size, im1.shape[1] - win_size, step_size):
            change_points.append([j, i])
            # Create a matrix by the window size and flatten it
            Ix = fx[i - half:i + half + 1, j - half: j + half+1].flatten()
            Iy = fy[i - half:i + half + 1, j - half: j + half+1].flatten()
            It = ft[i - half:i + half + 1, j - half: j + half+1].flatten()
            # Multiplying
            a1 = np.sum(np.matmul(Ix, Ix))
            a2 = np.sum(np.matmul(Ix, Iy))
            a3 = np.sum(np.matmul(Iy, Ix))
            a4 = np.sum(np.matmul(Iy, Iy))
            A_t_A = np.array([[a1, a2], [a3, a4]])
            # Find the direction of u and v
            direction = np.linalg.pinv(A_t_A)
            self_values = np.linalg.eigvals(A_t_A)
            min_val = np.min(self_values)
            max_val = np.max(self_values)
            if (max_val >= min_val > 1): # The other condition (max_val/min_val) < 100 gives an error
                # Using the formula of the tirgul and of the project's file
                d1 = np.sum(np.matmul(Ix, It))
                d2 = np.sum(np.matmul(Iy, It))

                A_t_b = np.array([[-d1], [-d2]])
                u_v = np.matmul(direction, A_t_b)

                u_v_list.append([u_v[0][0], u_v[1][0]])

    return np.array(change_points), np.array(u_v_list)
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """



def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    gauss_list = gaussianPyr(img, levels)
    lap_list = []
    # Create a gaussian kernel
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)
    # Add the original
    lap_list.append(gauss_list[levels - 1])
    # Original - Expand
    for i in reversed(range(1, levels)):
        lap_list.insert(0, gauss_list[i-1] - gaussExpand(gauss_list[i], kernel))
    return lap_list

    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    # Create a gaussian kernel
    k_size = 5
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(k_size, sigma)
    kernel = kernel.dot(kernel.T)
    original_img = np.ndarray([])
    # Expand the original
    temp = gaussExpand(lap_pyr[len(lap_pyr)-1], kernel)
    for i in reversed(range(len(lap_pyr))):
        # According to the fact that original - expand = lap
        original_img = temp + lap_pyr[i-1]
        temp = gaussExpand(original_img, kernel)
        if i == 1:
            return original_img
    return original_img
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    pyramid_list = []
    # Resize
    row = len(img) % pow(2, levels)
    col = len(img[0]) % pow(2, levels)
    if row != 0:
        img = img[:-row, :]
    if col != 0:
        img = img[:, :-col]
    # Add the original
    pyramid_list.append(img)
    for i in range(1, levels + 1):
        # Blur
        temp = cv2.GaussianBlur(pyramid_list[i - 1], (5, 5), 0)
        # Takes only half from the rows and the cols
        temp = temp[::2, ::2]
        pyramid_list.append(temp)
    return pyramid_list
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    row_size = len(img)*2
    col_size = len(img[0])*2
    dim_size = img.ndim
    # If gray
    if dim_size == 2:
        expand_img = np.zeros(shape=(row_size, col_size))
        for x in range(0, row_size, 2):
            for y in range(0, col_size, 2):
                expand_img[x, y] = img[int(x/2), int(y/2)]
    else:
        expand_img = np.zeros(shape=(row_size, col_size, dim_size))
        for i in range(dim_size):
            for x in range(0, row_size, 2):
                for y in range(0, col_size, 2):
                    expand_img[x, y, i] = img[int(x/2), int(y/2), i]
    # We want to "cover" the zeros by more power of the kernel
    expand_img = cv2.filter2D(expand_img, -1, gs_k*4)
    return expand_img
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """



def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    lap_img1 = laplaceianReduce(img_1, levels)
    lap_img2 = laplaceianReduce(img_2, levels)
    gaus_mask = gaussianPyr(mask, levels)

    # More accurate algo - using pyramids
    list = []
    for i in range(levels):
        list.append(lap_img1[i]*gaus_mask[i] + lap_img2[i]*(1 - gaus_mask[i]))
    img = laplaceianExpand(list)

    # naive algo
    merge = img_1*mask + img_2*(1 - mask)

    return merge, img
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """

