import numpy as np
import cv2


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


def gaussian_filter(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


# Read an image
image = cv2.imread("D:\Learning\Computer_Vision\cat_1.jpg")

# Apply Gaussian filter manually
filtered_image = gaussian_filter(image, kernel_size=5, sigma=1.0)

cv2.imwrite('D:\Learning\Computer_Vision\cat_1_gaussian_1.jpg', filtered_image)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
