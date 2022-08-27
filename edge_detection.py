import math

import cv2
import numpy as np

from edge_matching import binarize_img, get_contours, select_best_corners, get_potential_corners


def kmeans_masking(img, k=3, attempts=10):
    """
    Execute kmeans algorithm on image
    :param img: Source image in numpy format
    :param k: Determines number of clusters that the kmeans algorithm has to search
    :param attempts: number of times the algorithm tries to attempt
    :return: resulted image after kmeans, ret, label, center
    """

    img_2d = img.reshape((-1, 3))
    img_2d = np.float32(img_2d)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(img_2d, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    result_image = cv2.medianBlur(result_image, 5)

    return result_image, ret, label, center


def black_masking(img, label, center, bg, mask_color=(0, 0, 0)):
    """
    Black Masking selected backgrounds of a kmeans processed image
    :param img: kmeans processed image
    :param label: label corresponding to kmeans cluster id
    :param center: color code of the kmeans background
    :param bg: list of background indexes of center to black masked
    :return: black masked image
    """
    temp_img = np.copy(img)
    temp_label = np.copy(label)
    temp_center = np.copy(center)
    temp_center_showcase = np.copy(center)
    for i in range(len(temp_center)):
        if i in bg:
            temp_center[i] = np.asarray((0, 0, 0))
            temp_center_showcase[i] = np.asarray(mask_color)

    res = temp_center[temp_label.flatten()]
    result_image = res.reshape((temp_img.shape))
    result_image = cv2.medianBlur(result_image, 5)

    res_showcase = temp_center_showcase[temp_label.flatten()]
    result_image_showcase = res_showcase.reshape((temp_img.shape))
    result_image_showcase = cv2.medianBlur(result_image_showcase, 5)

    gray = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
    transparent_mask = np.reshape(gray, (gray.shape[0], gray.shape[1], 1))

    return result_image_showcase, transparent_mask


def color_masking(img, l_val, h_val):
    """
    Execute color masking by choosing a color range
    :param img: Source image
    :param l_val: lower color value
    :param h_val: higher color value
    :return: result image, mask
    """
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    l_color = np.array([l_val, 0, 0])
    h_color = np.array([h_val, 255, 255])

    mask = cv2.inRange(hsv_img, l_color, h_color)
    mask = cv2.medianBlur(mask, 5)

    result = cv2.bitwise_and(img, img, mask=mask)

    return result, mask


def color_transparent_masking(mask):
    """
    convert color mask to transparent mask
    :param mask: color mask
    :return: transparent mask
    """
    transparent_mask = (np.ones(mask.shape, dtype=mask.dtype) * 255) - mask
    transparent_mask = np.reshape(transparent_mask, (transparent_mask.shape[0], transparent_mask.shape[1], 1))

    return transparent_mask


def create_png_img(img, transparent_mask):
    png_img = np.concatenate([img, transparent_mask], axis=2)
    return png_img


def extract_img_from_contour(img, contour, blocksize=10, ksize=5, k=0.04, area_mod=0.018, new_are_mod=0.85):
    new_img = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(new_img, [contour], (255, 255, 255))
    res = cv2.bitwise_and(img, new_img)

    center, size, angle = cv2.minAreaRect(contour)

    center, size = tuple(map(int, center)), tuple(map(int, (size[0] * 1.2, size[1] * 1.2)))

    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(res, m, (res.shape[1], res.shape[0]))
    out = cv2.getRectSubPix(rotated_image, size, center)

    bin_img = binarize_img(out)
    contours = get_contours(bin_img)
    corns = select_best_corners(get_potential_corners(contours[0], bin_img, block_size=blocksize, ksize=ksize, k=k),
                                area_mod=area_mod, new_are_mod=new_are_mod)

    return align_rect(out, np.asarray(corns))


def rotate_image(mat, angle):
    """
    https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def align_rect(img, corns):
    # plot_corners(img, corns)
    ind = np.argsort(corns[:, 0])

    vec = corns[ind[1]] - corns[ind[0]]
    radian = math.atan2(vec[1], vec[0])
    deg = radian * (180 / np.pi)
    rot_img = rotate_image(img, deg)

    bin_img2 = binarize_img(rot_img)
    contours2 = get_contours(bin_img2)
    center, radius = cv2.minEnclosingCircle(contours2[0])
    center, size = tuple(map(int, center)), tuple(map(int, (radius * 2.2, radius * 2.2)))

    out = cv2.getRectSubPix(rot_img, size, center)
    #
    # cv2.imshow("asdf", out)
    # cv2.waitKey()

    return out


def plot_img(img):
    # Resize image
    scale_percent = 500  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim)

    cv2.imshow('corners', resized)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    puzzle_dir = "puzzle_data/photo_puzzles/real/01.jpg"
    img = cv2.imread(puzzle_dir, cv2.IMREAD_UNCHANGED)

    plot_img(img)

    k_img, ret, label, center = kmeans_masking(img, 6, 10)
    result, transparent_mask = black_masking(k_img, label, center, bg=[0, 3])

    plot_img(result)

    # result, mask = color_masking(img, 20, 110)
    # transparent_mask = transparent_masking(mask)

    contours, _ = cv2.findContours(transparent_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        puzzle_img = extract_img_from_contour(img, contour, blocksize=10, ksize=5, k=0.04, area_mod=0.018,
                                              new_are_mod=0.85)
        plot_img(puzzle_img)
