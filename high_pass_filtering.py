import typing as tp

import cv2
import numpy as np


def sobel_filter(image: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    # Apply Sobel kernels to compute gradients
    gradient_x = cv2.filter2D(image, -1, kernel_x)
    gradient_y = cv2.filter2D(image, -1, kernel_y)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Compute gradient direction (in degrees)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    return gradient_magnitude, gradient_direction


# Read an image
image = cv2.imread("D:\Learning\Computer_Vision\cat_1.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter
gradient_magnitude, gradient_direction = sobel_filter(image)
cv2.imwrite('D:\Learning\Computer_Vision\cat_1_gradient.jpg', gradient_direction.astype(np.uint8))

# Display the gradient magnitude and direction (optional)
cv2.imshow('Gradient Magnitude', gradient_magnitude.astype(np.uint8))
cv2.imshow('Gradient Direction', gradient_direction.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
