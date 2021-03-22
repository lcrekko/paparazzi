# python=3.7.10  OpenCV=4.5.1.48
# Floor detection added to avoid out-of-bounds	March 15 2021
# When the pole is in the middle, the drone turn to the side with more green pixels   March 17 2021
import numpy as np
import os
import re
import cv2
from count_green_pixels import count_green_pixels
from loadCamCoefficients import load_coefficients

# Change the datasets directory locally
image_dir_name = os.getcwd() + '/Datasets/cyberzoo_poles/20190121-135009/'
# image_dir_name = os.getcwd() + '/Datasets/cyberzoo_poles_panels/20190121-140205/'
# image_dir_name = os.getcwd() + '/Datasets/cyberzoo_poles_panels_mats/20190121-142935/'
image_type = 'jpg'
image_names = []
# Sort image names
for file in os.listdir(image_dir_name):
    if file.endswith(image_type):
        image_names.append(image_dir_name + file)
image_names.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(image_names[88])
random_img = cv2.imread(image_names[0])
# Load camera calibration coefficients
mtx, dist = load_coefficients(os.getcwd() + '/camera_coefficients.yaml')
h, w = random_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
# Read images
prev_image = cv2.imread(image_names[285])
image = cv2.imread(image_names[289])
# Undistort images
prev_image = cv2.undistort(prev_image, mtx, dist, None, newcameramtx)
image = cv2.undistort(image, mtx, dist, None, newcameramtx)

# Color filters parameters
lower_pole = np.array([0, 104, 122], dtype="uint8")
upper_pole = np.array([179, 255, 255], dtype="uint8")
lower_floor = np.array([0, 0, 0], dtype="uint8")
upper_floor = np.array([179, 32, 95], dtype="uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Min floor area for 'Out of bounds" detection
# TODO: Tune this parameter
MIN_FLOOR_AREA = 2500

for i, img_file in enumerate(image_names):
    image_raw = cv2.imread(img_file)
    # Undistort images
    image_undist = cv2.undistort(image_raw, mtx, dist, None, newcameramtx)
    image = cv2.cvtColor(image_undist, cv2.COLOR_BGR2HSV)
    image_pole = cv2.inRange(image, lower_pole, upper_pole)
    image_floor = cv2.inRange(image, lower_floor, upper_floor)
    # Morphological Transformations to remove noise
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    image_pole = cv2.morphologyEx(image_pole, cv2.MORPH_OPEN, kernel, iterations=1)
    image_floor = cv2.morphologyEx(image_floor, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Sometimes noise due to racks cannot be removed, thus setting the right 1/3 of the image to black
    image_floor[:, int(2*w/3):w-1] = 0
    # Find contours
    cnts_pole = cv2.findContours(image_pole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_pole = cnts_pole[0] if len(cnts_pole) == 2 else cnts_pole[1]


    # The following section is for visualization. TODO: Comment out the following in real application
    cnts_floor = cv2.findContours(image_floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_floor = cnts_floor[0] if len(cnts_floor) == 2 else cnts_floor[1]
    floor_area = 0
    for c in cnts_floor:
        floor_area += cv2.contourArea(c)
        cv2.drawContours(image_undist, [c], 0, (90, 255, 0), 2)
    # TODO: Comment out the above section in real application


    # Use the method by Cheng Liu instead to count the floor area. Much faster
    floor_area = count_green_pixels(image_floor)
    # Count the floor area on the left and right half of the camera view
    left_image_floor = image_floor[0:int(h/2), :]
    right_image_floor = image_floor[int(h/2):h-1, :]
    # Use the method proposed by Cheng Liu in count_green_pixels.py
    floor_area_left = count_green_pixels(left_image_floor)
    floor_area_right = count_green_pixels(right_image_floor)
    # If "Within the bounds"
    if floor_area > MIN_FLOOR_AREA:
        max_area = 0
        how_many_poles = 0
        index_max_area = 10
        for j, c in enumerate(cnts_pole):
            contour_area = cv2.contourArea(c)
            if contour_area > 1000:
                how_many_poles = how_many_poles + 1
                if contour_area > max_area:
                    # The one with the largest area is the closest
                    max_area = contour_area
                    index_max_area = j
                cv2.drawContours(image_undist, [c], 0, (255, 0, 0), 2)

        # ************** Strategy *****************
        # This only covers the pole avoidance, floor detection to avoid out-of-bounds should be added
        # The image is evenly divided into 5 segments vertically, from top to bottom: seg 1 2 3 4 5
        # In real environment, "vertically" here stands for "horizontally", from left to right.
        # If the center of the closest pole lies inside seg 1 or 5, go straight
        # If the center of the closest pole lies inside seg 2, turn right
        # If the center of the closest pole lies inside seg 4, turn left
        # If the center of the closest pole lies inside seg 3 which is the center of the image, turn left(right)
        heading = 'Go straight'
        if index_max_area != 10:
            cloest_pole = cnts_pole[index_max_area]
            # Mark the closest pole with red
            cv2.drawContours(image_undist, [cloest_pole], 0, (0, 0, 255), 2)
            min_y = min(cloest_pole[:, 0, 1])
            max_y = max(cloest_pole[:, 0, 1])
            center = 0.5 * (min_y + max_y)
            if 0 < center <= h / 5 or 4 * h / 5 < center < h:
                heading = "Go straight"
            elif h / 5 < center <= 2 * h / 5:
                heading = "Turn right"
            elif 3 * h / 5 < center <= 4 * h / 5:
                heading = "Turn left"
            else:
                # Turn to the side with more green pixels
                if floor_area_left >= floor_area_right:
                    heading = "Turn left"
                else:
                    heading = "Turn right"
        text = str(how_many_poles) + ' poles ' + heading
    # If "Reaching the bounds"
    else:
        text = 'Closing to bounds'

    cv2.putText(image_undist, text, (0, 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 240, 10))
    cv2.imshow('frame', image_undist)
    cv2.waitKey(80)
