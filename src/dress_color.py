import cv2
import numpy as np


def get_dress_color(image):
    """
    Estimate dominant clothing colour from the lower part
    of the image.

    This is an approximate colour estimator, not a trained
    clothing segmentation model.
    """

    if image is None:
        return "Unknown"

    height, width = image.shape[:2]

    # Use lower 45% of image as approximate clothing region
    y_start = int(height * 0.55)

    clothing = image[y_start:height, :]

    if clothing.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(clothing, cv2.COLOR_BGR2HSV)

    # Remove very dark pixels and very bright background pixels
    pixels = hsv.reshape(-1, 3)

    # Keep pixels with reasonable saturation
    filtered = pixels[
        (pixels[:, 1] > 35) &
        (pixels[:, 2] > 30)
    ]

    if len(filtered) == 0:
        return "Black / Dark"

    # Median HSV
    h = np.median(filtered[:, 0])
    s = np.median(filtered[:, 1])
    v = np.median(filtered[:, 2])

    # Colour classification
    if v < 50:
        return "Black"

    if s < 40 and v > 180:
        return "White"

    if s < 50:
        if v < 120:
            return "Gray"
        return "White / Gray"

    # OpenCV Hue range = 0-179
    if h < 10 or h >= 170:
        return "Red"

    if h < 25:
        return "Orange"

    if h < 35:
        return "Yellow"

    if h < 85:
        return "Green"

    if h < 105:
        return "Cyan / Blue"

    if h < 135:
        return "Blue"

    if h < 160:
        return "Purple"

    return "Pink / Red"