# -*- coding: utf-8 -*-
"""
Created on Tue Mar 9 16:50:21 2021

@author: cheng liu
Mar. 17 2021    Haoyu Feng modified it to a function to be used in the pole_avoidance
"""

import os
from os.path import join
import cv2
import numpy as np


# TODO: Add threshold (minimum) of the number of green pixels to decide whether to turn left/right
'''
# folder_path = "green_test_graphs/"
folder_path = os.getcwd() + '/Datasets/cyberzoo_poles/20190121-135009/'
filenames = os.listdir(folder_path)

print("files for green pixel counting test:", filenames)

# YUV range for green. TODO: This range is not right. by: Haoyu
GREEN_MIN = np.array([0, 0, 0], dtype="uint8")
GREEN_MAX = np.array([179, 32, 95], dtype="uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
# Count the number of pixels that satisfies the green range
for _file in filenames:
    if _file.split('.')[-1] == 'jpg':
        image = cv2.imread(folder_path+_file)

        # The original size of the image is [520*240], only the middle bottom
        # of the image is relevant. Crop the image such that only 
        # [130:390 in width, 0:200 in height] will be analyzed.
        # img = image[130:390, 0:200]
        img = image[0:260, 0:200]
        cv2.imshow('1', img)
        cv2.waitKey(0)
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # convert to YUV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.inRange(img, GREEN_MIN, GREEN_MAX) # output 0 for pixels without green
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imshow('2', dst)

        num_green = cv2.countNonZero(dst) # count pixels with green
        cv2.waitKey(0)
        print("{} has {} pixels with green".format(_file, num_green))

# We can count the number of pixels with green and tell the UAV to change
# direction when cnt is smaller than the threshold.
'''


def count_green_pixels(img):
    # TODO: because the YUV range given above is not right, use HSV for now.
    # img.shape = 260*200
    num_green = cv2.countNonZero(img)
    return num_green