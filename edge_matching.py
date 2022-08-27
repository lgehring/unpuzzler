import glob as gl
import itertools
import math
import operator
import os
from functools import reduce

import cv2
import numpy as np
import scipy
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks, savgol_filter


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        transp_mask = img[:, :, 3] == 0
        img[transp_mask] = (0, 0, 0, 0)  # replace transparency with black
    else:
        # Add alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def binarize_img(img, median_k=3):
    blurred_img = cv2.medianBlur(img, ksize=median_k)
    bin_img = cv2.threshold(blurred_img, 1, 255, cv2.THRESH_BINARY)[1]
    gray_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
    return gray_img


def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def get_potential_corners_harris(img, block_size=9, ksize=5, k=0.04):
    harris = cv2.cornerHarris(img, block_size, ksize, k)
    filt_harris = np.where(harris > harris.max() * 0.25, harris, 0)
    max_area = maximum_filter(filt_harris, size=5)  # sets area to max of local neighborhood
    max_center = np.where(max_area == filt_harris, filt_harris, 0)  # sets all but original max to 0
    corners = np.fliplr(np.argwhere(max_center > 0))
    return corners


def get_potential_corners(contours, bin_img, dist_trehsh=5, block_size=9, ksize=5, k=0.04):
    def cart2pol(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    # Get contour points in polar coordinates
    rhos = []
    (center_x, center_y), _ = cv2.minEnclosingCircle(contours)
    for i in range(len(contours)):
        x, y = contours[i][0]
        rhos.append(cart2pol(x - center_x, y - center_y))

    # Extend borders to avoid edge effects
    rhos_ext = np.concatenate((rhos, rhos[0:10]))
    contours_ext = np.concatenate((contours, contours[0:10]))

    # Smooth the rhos function and find peaks
    rhos_smooth = np.array(savgol_filter(rhos_ext, 7, 2))
    peaks = scipy.signal.find_peaks(rhos_smooth, prominence=0.1, distance=10)[0]

    # Plot
    # plt.plot(rhos_ext, color="red")
    # plt.plot(rhos_smooth, color="blue")
    # for peak in peaks:
    #     plt.axvline(x=peak, color='y')
    # plt.show()

    # Find cartesian coordinates of peaks
    polar_corners = []
    for peak in peaks:
        polar_corners.append(contours_ext[peak][0])

    # For every polar corner, check if there is a corresponding point in harris_corns
    harris_corns = get_potential_corners_harris(bin_img, block_size=block_size, ksize=ksize, k=k)
    corners = []
    for polar_corner in polar_corners:
        closest_harris = closest_point_idx(polar_corner, harris_corns)
        dist_polar_harris = np.linalg.norm(polar_corner - harris_corns[closest_harris])
        if dist_polar_harris < dist_trehsh:
            corners.append(harris_corns[closest_harris])
    return corners


# Adapted from https://stackoverflow.com/a/51075698
def sort_vertices(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)


# Adapted from https://stackoverflow.com/a/30408825
def calc_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# Adapted from https://stackoverflow.com/a/20253693
def calc_angle_dev(coords):
    total_dev = 0
    for i in range(len(coords)):
        p1 = coords[i]
        ref = coords[i - 1]
        p2 = coords[i - 2]
        x1, y1 = p1[0] - ref[0], p1[1] - ref[1]
        x2, y2 = p2[0] - ref[0], p2[1] - ref[1]
        divisor = math.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2))
        if divisor == 0:
            divisor = 0.000001
        angle = math.degrees(math.acos((x1 * x2 + y1 * y2) / divisor))
        total_dev += abs(90 - angle)
    return total_dev


def select_best_corners(corners, area_mod=0.018, new_are_mod=0.85):
    """
    For all combinations of 4 corners, find the subset of 4 that maximizes
    - the area between them
    - the "closeness" of the angles between them to 90 degrees
    :param corners: all potential corners
    :param area_mod: [0, 1] regulates the areas relative influence on the score
    :param new_are_mod: [0, 1] regulates the threshold for the proposed area relative the current best area
    :return:
    """
    corners = sort_vertices(corners)
    best_corners = []
    if len(corners) > 4:
        max_area = 0
        best_coefficient = -np.inf

        num_points = len(corners)
        corners = np.asarray(corners)
        for comb in list(itertools.combinations(range(num_points), 4)):
            # For all combinations of 4 corners
            curr_corners = corners[list(comb)]
            area = calc_area(curr_corners[:, 0], curr_corners[:, 1])  # x,y
            angle_deviation = calc_angle_dev(curr_corners)

            coefficient = area * area_mod - angle_deviation  # maximizes area, minimizes angle deviation
            if coefficient > best_coefficient and area >= new_are_mod * max_area:
                # print("Area: {}, Angle deviation: {}, Coefficient: {}".format(area * area_mod, angle_deviation, coefficient))
                max_area = area
                best_coefficient = coefficient
                best_corners = curr_corners
    else:
        best_corners = corners
    return best_corners


def segment_contours(corners, contours):
    """
    Segment a contour of a puzzle piece into 4 segments which corresponds to each side of a puzzle piece.
    It takes the coordinates of a corner detection and finds the nearest point in the contour which denotes the
    start/end of a segment.
    :param corners: List of corner coordinates (y, x). Should be 4 corners
    :param contours: List of 2d coordinates which forms a contour of the puzzle piece
    :return: list of segments which consists of list of 2d coordinates and list of corner coordinates
    corresponding to the contours
    """
    contours_reshape = contours.reshape((contours.shape[0], 2))
    closest_points = [closest_point_idx(corner, contours_reshape) for corner in corners]
    closest_points.sort()
    segments = [
        contours[closest_points[0]:closest_points[1]],
        contours[closest_points[1]:closest_points[2]],
        contours[closest_points[2]:closest_points[3]],
        np.roll(contours, -closest_points[3], axis=0)[0:contours.shape[0] - closest_points[3] + closest_points[0]]
    ]
    reshaped_contours = contours_reshape[closest_points]
    avg_x_coords = []
    avg_y_coords = []
    sorted_segments_ind = []
    for segment in segments:
        avg_x_coords.append(np.mean(segment[:, 0, 0]))
        avg_y_coords.append(np.mean(segment[:, 0, 1]))
    # Get segment order: left, bottom, right, top
    sorted_segments_ind.append(avg_x_coords.index(min(avg_x_coords)))
    sorted_segments_ind.append(avg_y_coords.index(max(avg_y_coords)))
    sorted_segments_ind.append(avg_x_coords.index(max(avg_x_coords)))
    sorted_segments_ind.append(avg_y_coords.index(min(avg_y_coords)))
    # Sort segments according to the order
    sorted_segments = [segments[i] for i in sorted_segments_ind]
    sorted_reshaped_contours = [reshaped_contours[i] for i in sorted_segments_ind]

    return sorted_segments, sorted_reshaped_contours


def closest_point_idx(point, points):
    """
    Determine index the closest point from a list of points
    :param point: Reference point
    :param points: list of points where to calculate the nearest point
    :return: index referring to the closest point
    """
    return np.argmin(np.linalg.norm(points - point, axis=1))


def get_edge_type(edge, c, threshold=5):
    e1 = edge[0]
    e2 = edge[-1]

    for e in edge[::int(len(edge) / 20)]:
        #  If distance between point of edge and line between head and tail of edge
        if np.abs(np.linalg.norm(np.cross(e2 - e1, e1 - e))) / np.linalg.norm(e2 - e1) > threshold:
            # Barycentre of segment edge in the middle "half" of the edge
            b = np.mean(edge[int(len(edge) / 4):int(-len(edge) / 4)], axis=0).astype(int)

            # Calculate distance of puzzle center and edge
            ce = np.abs(np.linalg.norm(np.cross(e2 - e1, e1 - c))) / np.linalg.norm(e2 - e1)

            # Otherwise it has either a bulge or notch.
            cb = np.linalg.norm(b - c)
            edge_type = "i" if cb < ce else "o"

            return edge_type

    return "e"


def get_piece_type(edge_types, piece_type_dict):
    for key in piece_type_dict:
        match_str = key + key
        if match_str.find("".join(edge_types)) > -1:
            return piece_type_dict[key]
    return None


def puzzle_characterization(pz_img, block_size=9, ksize=5, k=0.04, area_mod=0.018, new_are_mod=0.85, inward_offset=0):
    piece_type_dict = {
        "iiee": "r0",
        "ioee": "r1",
        "oiee": "r2",
        "ooee": "r3",
        "iiie": "e0",
        "ioie": "e1",
        "iioe": "e2",
        "oiie": "e3",
        "iooe": "e4",
        "ooie": "e5",
        "oioe": "e6",
        "oooe": "e7",
        "iiii": "i0",
        "oiii": "i1",
        "oioi": "i2",
        "iooi": "i3",
        "oooi": "i4",
        "oooo": "i5"
    }

    contours = get_contours(binarize_img(pz_img))[0]
    corns_potential = get_potential_corners(contours, binarize_img(pz_img), block_size=block_size, ksize=ksize, k=k)
    corns = np.asarray(select_best_corners(corns_potential, area_mod=area_mod, new_are_mod=new_are_mod))
    segments, closest_corners = segment_contours(corns, contours)
    col_segments = get_contour_color_vector(pz_img, closest_corners, inward_offset=inward_offset)

    center = np.mean(contours, axis=0).astype(int)

    edge_types = []
    for i, segment in enumerate(segments):
        edge_types.append(get_edge_type(segment, center))

    piece_type = get_piece_type(edge_types, piece_type_dict)

    return contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type


def plot_puzzle_segments(img, segments, corners, edge_types, show=False, resize=False, circle_size=2):
    # Dict matching edge type to color
    edge_type_color = {
        "e": (255, 0, 0),
        "i": (0, 255, 0),
        "o": (0, 0, 255)
    }

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for i, segment in enumerate(segments):
        cv2.drawContours(img_corners, (segments[i]), -1, edge_type_color[edge_types[i]], 1)

    for corner in corners:
        cv2.circle(img_corners, corner, circle_size, (0, 255, 255), 1)

    # Resize image
    resized = img_corners
    if resize:
        scale_percent = 500  # percent of original size
        width = int(img_corners.shape[1] * scale_percent / 100)
        height = int(img_corners.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_corners, dim)

    if show:
        cv2.imshow('corners', resized)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return resized


def plot_puzzle_contours(img, contours, corners):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_corners, contours, -1, (0, 0, 255), 1)

    for corner in corners:
        cv2.circle(img_corners, corner, 2, (0, 255, 255), 1)

    # Resize image
    scale_percent = 500  # percent of original size
    width = int(img_corners.shape[1] * scale_percent / 100)
    height = int(img_corners.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img_corners, dim)

    cv2.imshow('corners', resized)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def get_contour_color_vector(img, corns, inward_offset=1, show=False):
    # Get all white pixels in the image
    bin_img = binarize_img(img)
    # Add a black frame around the binary image to prevent errors when checking for neighbors
    bin_img = np.pad(bin_img, pad_width=inward_offset, mode='constant', constant_values=0)
    bin2 = bin_img.copy()

    for i in range(inward_offset):
        white_pixels = np.where(bin_img == 255)  # piece
        for pixel in zip(white_pixels[0], white_pixels[1]):
            if bin_img[pixel[0] - 1, pixel[1]] == 0 or bin_img[pixel[0] + 1, pixel[1]] == 0 or \
                    bin_img[pixel[0], pixel[1] - 1] == 0 or bin_img[pixel[0], pixel[1] + 1] == 0:
                bin_img[pixel[0], pixel[1]] = 1
        gray_pixels = np.where(bin_img == 1)
        for pixel in zip(gray_pixels[0], gray_pixels[1]):
            bin_img[pixel[0], pixel[1]] = 0

    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Get pixel values of contours
    contour_points = contours[0].flatten().reshape(contours[0].shape[0], 2)

    color_contours = []
    for point in contour_points:
        j = point[0]
        i = point[1]
        color_contours.append([img[i, j], [i, j]])

    # PLot contours
    if show:
        img_corners = cv2.cvtColor(bin2, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_corners, contours, -1, (0, 0, 255), 1)
        # Resize image
        scale_percent = 500  # percent of original size
        width = int(img_corners.shape[1] * scale_percent / 100)
        height = int(img_corners.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_corners, dim)
        cv2.imshow('corners', resized)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # Get segment split points
    corner_indices = []
    for corner in corns:
        # Find the closest point in contours to corner
        closest_point_ind = np.argmin(np.linalg.norm(contour_points - corner, axis=1))
        corner_indices.append(closest_point_ind)
    # Split contours into 4 segments
    ranges = zip(corner_indices, np.roll(corner_indices, -1))
    color_segments = []
    for range_ in ranges:
        if range_[0] < range_[1]:
            color_segments.append(color_contours[range_[0]:range_[1]])
        else:
            color_segments.append(color_contours[range_[0]:] + color_contours[:range_[1]])

    return color_segments


def plot_contour_pixels(img, segment_pixels):
    # Draw white image with same size as contour imagee
    img_col_segments = np.zeros(img.shape, np.uint8)
    img_col_segments[:] = (255, 255, 255, 255)
    for segment in segment_pixels:
        for pixel in segment:
            img_col_segments[pixel[1][0], pixel[1][1]] = pixel[0]

        # Resize image
        scale_percent = 500  # percent of original size
        width = int(img_col_segments.shape[1] * scale_percent / 100)
        height = int(img_col_segments.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_col_segments, dim)

        cv2.imshow('corners', resized)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()


def plot_corners(img, corners):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for corner in corners:
        cv2.circle(img_corners, corner, 2, (0, 255, 255), 1)

    # Resize image
    scale_percent = 500  # percent of original size
    width = int(img_corners.shape[1] * scale_percent / 100)
    height = int(img_corners.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img_corners, dim)

    cv2.imshow('corners', resized)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def generate_debug_images(puzzle_dir, block_size=9, ksize=5, k=0.04, inward_offset=0):
    pieces_paths = gl.glob(os.path.join(puzzle_dir, '*.png'))

    for pz in pieces_paths:
        image = read_img(pz)
        print(pz)

        contours, corns, closest_corners, segments, col_segments, center, edge_types, piece_type = puzzle_characterization(
            image, block_size, ksize, k, inward_offset=inward_offset)

        for corner in closest_corners:
            cv2.circle(binarize_img(image), corner, 2, (255, 0, 255), 1)
        # plot_puzzle_contours(image, contours, closest_corners)
        # plot_puzzle_segments(image, segments, closest_corners, edge_types, True)
        plot_contour_pixels(image, col_segments)


# Show one by one
if __name__ == "__main__":
    generate_debug_images("puzzle_data/photo_puzzles/real/01", block_size=9, ksize=5, k=0.04, inward_offset=1)
