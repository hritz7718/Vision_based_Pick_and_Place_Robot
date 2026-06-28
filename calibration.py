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


def world_to_pixel(homography, xw, yw):
    inverse_homography = np.linalg.inv(homography)
    world_point = np.array([[[xw, yw]]], dtype=np.float32)
    pixel_point = cv2.perspectiveTransform(world_point, inverse_homography)
    u = pixel_point[0][0][0]
    v = pixel_point[0][0][1]
    return u, v


def world_to_robot(xw, yw, zw=0):
    xr = xw - ROBOT_BASE_WORLD[0]
    yr = yw - ROBOT_BASE_WORLD[1]
    zr = zw - ROBOT_BASE_WORLD[2]
    return xr, yr, zr


def is_inside_workspace(workspace_world_points, xw, yw):
    if len(workspace_world_points) < 3:
        return False

    polygon = np.array(workspace_world_points, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, (float(xw), float(yw)), False) >= 0


def is_in_detection_zone(division_line_world_points, xw, yw):
    if len(division_line_world_points) != 2:
        return False

    return get_zone_side(division_line_world_points, xw, yw) >= 0


def is_in_placement_zone(division_line_world_points, xw, yw):
    if len(division_line_world_points) != 2:
        return False

    return get_zone_side(division_line_world_points, xw, yw) < 0


def get_zone_side(division_line_world_points, xw, yw):
    (x1, y1), (x2, y2) = division_line_world_points
    return (x2 - x1) * (yw - y1) - (y2 - y1) * (xw - x1)
