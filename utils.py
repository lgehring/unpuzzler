import glob as gl
import itertools
import os

import cv2
import numpy as np


def load_puzzle_pieces(puzzle_dir: str) -> list:
    """
    Loads the puzzle piece images from a directory with cv2. PNG Format is recommended
    :param puzzle_dir: directory where the puzzle pieces is located
    :return: list of cv2 images of the puzzle pieces
    """
    pieces_path = gl.glob(os.path.join(puzzle_dir, '*.png'))
    return [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in pieces_path]


def detect_puzzle_corners(corners):
    """
    Detect the true puzzle corners from many detected corners in a puzzle image.
    First it generates all possible combination of sets of four corners and sorts each set counter-clockwise, so
    it can calculate the area of the polygon. The polygon with the biggest area consist of corners corresponding
    on puzzle corners.
    :param corners: numpy array with numbers of detected corner coordinates (x, y) with shape (num corners, 2)
    :return: Set of four corners which maximizes the area of the polygon
    """
    num_points = len(corners)
    combinations = list(itertools.combinations(range(num_points), 4))

    p_area = []
    ordered_combinations = []
    for comb in combinations:
        # Get set of corners
        sub_corners = corners[list(comb)]

        # Sort set of corners counter-clockwise
        corner_order = sort_points_cw(sub_corners)

        # Add ordered corner set to list
        ordered_combinations.append(corner_order)

        # Add calculated area to list
        p_area.append(polygon_area(corner_order))

    # Choose index with highes area
    index_max_area = np.argmax(np.array(p_area))
    return ordered_combinations[index_max_area]


def sort_points_cw(pts):
    """
    Sort points counter-clock-wise
    :param pts: numpy array of shape (num_points, 2)
    :return: sorted numpy array
    """
    x = pts[:, 0]
    y = pts[:, 1]
    x0 = np.mean(pts[:, 0])
    y0 = np.mean(pts[:, 1])

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    angles = np.where((y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    return np.column_stack((x_sorted, y_sorted))


def polygon_area(pts):
    """
    https://www.geeksforgeeks.org/area-of-a-polygon-with-given-n-ordered-vertices/
    Calculates the area of a polygon
    :param pts: ordered list of points
    :return: area of the polygon
    """

    X = pts[:, 0]
    Y = pts[:, 1]
    n = len(X)

    # Initialize area
    area = 0.0

    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i  # j is previous vertex to i

    # Return absolute value
    return int(abs(area / 2.0))
