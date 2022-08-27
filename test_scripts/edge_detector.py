#!/usr/bin/env python

from pathlib import Path
from PIL import Image, ImageFilter
import sys


def make_bw_image(img: Image, deviation: int = 0) -> Image:
    """Paint image black and white.

    All pixels which do not match the color of the background +- a deviation are
    painted black, the background is painted white.
    The background color is defined by the value in the top-left corner.

    :param img:        Input image
    :param deviation:  Deviation which values will be interpreted as background
    :return:           Black and white image
    """
    grayscale_image = img.convert("L")

    # Get background value
    pixels = grayscale_image.load()
    value = pixels[0, 0]

    # Create initial black and white image
    bw_image = grayscale_image.point(
        lambda x: 255 if abs(x - value) <= deviation else 0
    )

    # Smoothen image to get rid of single white pixels in pieces
    smooth_image = bw_image.filter(ImageFilter.SMOOTH)

    # Create final black and white image
    bw_image = smooth_image.point(lambda x: 0 if x < 255 else 255)

    return bw_image


def detect_edges(img: Image) -> Image:
    """Detect edges of an image.

    :param img: Input image
    :return:    Image with detected edges
    """
    grayscale_image = img.convert("L")

    # Smoothen image to get nicer results
    smooth_image = grayscale_image.filter(ImageFilter.SMOOTH)

    # Apply edge filter
    edge_image = smooth_image.filter(ImageFilter.FIND_EDGES)

    return edge_image


if __name__ == "__main__":

    base_path = Path("./data/photo_puzzles/digital/30_pieces")

    image = Image.open(base_path / "p12503218415.png")
    image = make_bw_image(image)
    image = detect_edges(image)
    image.save("test.png")
