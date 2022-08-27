import os

import cv2

import edge_detection as ed


def extract_real_pieces(image_path, method=None, blocksize=10, ksize=5, k=0.04, area_mod=0.018, new_are_mod=0.85):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # %% Segment puzzles from image using kmeans
    if method == "kmeans":
        k_img, ret, label, center = ed.kmeans_masking(img, 6, 10)
        result, transparent_mask = ed.black_masking(k_img, label, center, bg=[0, 3])
    else:
        # color masking
        result, mask = ed.color_masking(img, 20, 110)
        transparent_mask = ed.color_transparent_masking(mask)

    # %% Extract Puzzles from image using contours
    contours, _ = cv2.findContours(transparent_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    puzzle_images = []
    for contour in contours:
        puzzle_img = ed.extract_img_from_contour(img, contour, blocksize=blocksize, ksize=ksize, k=k, area_mod=area_mod,
                                                 new_are_mod=new_are_mod)
        # Plot  contour
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
        puzzle_images.append(puzzle_img)

    # Create directory for piece type if it doesn't exist
    dir_path = image_path.split('.')[0]
    if not os.path.exists(dir_path):
        # Create dir in same directory as image and name it after image name
        os.makedirs(dir_path)
    for ind, puzzle_image in enumerate(puzzle_images):
        cv2.imwrite(os.path.join(dir_path, str(ind) + ".png"), puzzle_image)

    return puzzle_images


extract_real_pieces("puzzle_data/photo_puzzles/real/01.jpg", method="color", blocksize=10, ksize=5, k=0.04,
                    area_mod=0.018, new_are_mod=0.85)
