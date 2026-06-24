import cv2
import numpy as np

from config import ROBOT_BASE_WORLD, WORLD_POINTS


def find_workspace_homography(image_points):
    image_points = np.array(image_points, dtype=np.float32)
    homography, status = cv2.findHomography(image_points, WORLD_POINTS)
    return homography, status


def pixel_to_world(homography, u, v):
    pixel_point = np.array([[[u, v]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(pixel_point, homography)
    xw = world_point[0][0][0]
    yw = world_point[0][0][1]
    return xw, yw


def world_to_robot(xw, yw, zw=0):
    xr = xw - ROBOT_BASE_WORLD[0]
    yr = yw - ROBOT_BASE_WORLD[1]
    zr = zw - ROBOT_BASE_WORLD[2]
    return xr, yr, zr


def is_in_detection_zone(division_line_world_points, xw, yw):
    if len(division_line_world_points) != 2:
        return False

    (x1, y1), (x2, y2) = division_line_world_points
    side = (x2 - x1) * (yw - y1) - (y2 - y1) * (xw - x1)

    # If blocks are detected on the wrong side of the division line,
    # flip this condition to: return side <= 0
    return side >= 0

