import glob as gl
import os
import random

import cv2
import cvzone
import numpy as np


def generate_points_with_min_distance(n, shape, min_dist):
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(0., shape[1] - 1, num_x, dtype=np.float32)
    y = np.linspace(0., shape[0] - 1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1, 2)

    # compute spacing
    init_dist = np.min((x[1] - x[0], y[1] - y[0]))

    # perturb points
    max_movement = (init_dist - min_dist) / 2
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise

    return coords


puzzle_dir = "puzzle_data/digital_puzzles/100_pieces"
save_dir = "puzzle_data/photo_puzzles/digital/100_pieces"
puzzles = os.listdir(puzzle_dir)

for pz in puzzles:
    files = gl.glob(os.path.join(puzzle_dir, pz, '*.png'))

    background_color = (200, 200, 200)

    data = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]

    width, height, channel = data[0].shape
    num_pz = len(data)

    max_len = min(width, height)
    padding = 100
    bg_width = 1920
    bg_height = 1080

    print("Point")
    points = generate_points_with_min_distance(num_pz, (bg_width, bg_height), max_len)
    points[:, 0] += np.absolute(np.min(points[:, 0])) + padding
    points[:, 1] += np.absolute(np.min(points[:, 1])) + padding
    points = points.astype(int)
    np.random.shuffle(points)
    print("Point 2")

    y_max = np.max(points[:, 0])
    x_max = np.max(points[:, 1])

    blank_image = np.zeros((y_max + padding + max_len, x_max + padding + max_len, 3), np.uint8)
    blank_image[:] = background_color

    for i, piece in enumerate(data):
        rotated_img = cvzone.rotateImage(piece, random.randint(0, 360), 1)
        blank_image = cvzone.overlayPNG(blank_image, rotated_img, [points[i, 1], points[i, 0]])

    cv2.imwrite(os.path.join(save_dir, pz + ".png"), blank_image)
