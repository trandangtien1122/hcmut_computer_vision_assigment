import os
import copy

import numpy as np
import cv2


def rgb_to_gray(image_path, is_save=True):
    img = cv2.imread(image_path)
    red = np.array(img[:, :, 0])
    green = np.array(img[:, :, 1])
    blue = np.array(img[:, :, 2])

    red = red * .299
    green = green * .587
    blue = blue * .114

    avg = red + green + blue
    gray_image = img.copy()

    for i in range(3):
        gray_image[:, :, i] = avg
    if is_save:
        name_paths = os.path.basename(image_path).split(".")
        input_dir = os.path.dirname(image_path)
        gray_filepath = os.path.join(input_dir, f"{name_paths[0]}_gray.{name_paths[1]}")
        cv2.imwrite(gray_filepath, gray_image)
    else:

        cv2.imshow("Gray Image", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def gray_to_color(gray_image_path, is_save=True):
    gray_image = cv2.imread(gray_image_path)
    color_image = copy.deepcopy(gray_image)

    # Change RGB channels by multiply each channel with a factor
    red = gray_image[:, :, 0] * 1.1  # Red channel
    green = gray_image[:, :, 0] * 1.  # Green channel
    blue = gray_image[:, :, 0] * 1.5  # Blue channel

    color_image[:, :, 0] = red
    color_image[:, :, 1] = green
    color_image[:, :, 2] = blue

    if is_save:
        name_paths = os.path.basename(gray_image_path).split(".")
        input_dir = os.path.dirname(gray_image_path)
        gray_filepath = os.path.join(input_dir, f"{name_paths[0]}_colorized.{name_paths[1]}")
        cv2.imwrite(gray_filepath, color_image)
    else:

        cv2.imshow("Colorized Image", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"D:\Learning\Computer_Vision\testing.jpg"
    rgb_to_gray(image_path, is_save=True)

    image_path = r"D:\Learning\Computer_Vision\testing_gray.jpg"
    gray_to_color(image_path, is_save=True)


