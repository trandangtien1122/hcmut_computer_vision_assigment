import abc

import cv2
import numpy as np


class Transformation:
    def __init__(self, image_filepath: str):
        self.image = self.read_image(image_filepath)
        self.rows, self.cols = self.image.shape[:2]

    @staticmethod
    def read_image(image_filepath: str):
        original_image = cv2.imread(image_filepath)
        return original_image

    @abc.abstractmethod
    def transformation(self, matrix):
        pass

    def apply_transformation(self, x: float, y: float) -> np.ndarray:
        matrix = np.array([[1, 0, x],
                           [0, 1, y],
                           [0, 0, 1]])
        return self.transformation(matrix)

    def apply_reflect(self, axis: str) -> np.ndarray:
        original_image = cv2.imread(image_path)
        w, h, _ = original_image.shape
        if axis == 'x':
            matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 1, -h], [0, 0, 1]])
        elif axis == 'y':
            matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ np.array([[1, 0, -w], [0, 1, 0], [0, 0, 1]])
        else:
            raise TypeError('Only support reflect along x and y')
        return self.transformation(matrix)

    def apply_scale(self, s_x: float, s_y: float) -> np.ndarray:
        matrix = np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]])
        return self.transformation(matrix)

    def apply_rotation(self, angle: float) -> np.ndarray:
        """
        Rotate the image by angle (in degree) counter-clockwise
        """
        original_image = cv2.imread(image_path)
        w, h, _ = original_image.shape
        theta = np.radians(angle)
        matrix = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]]) @ np.array(
            [[np.cos(theta), np.sin(theta), 0], [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]]) @ np.array(
            [[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
        return self.transformation(matrix)

    def apply_shear(self, shear_factor: float):
        matrix = np.array([[1, shear_factor, 0], [shear_factor, 1, 0], [0, 0, 1]])
        return self.transformation(matrix)


class AffineTransformation(Transformation):
    def __init__(self, image_filepath: str):
        super().__init__(image_filepath)

    def transformation(self, matrix):
        matrix = matrix.astype(np.float32)[0:2]
        # Apply the affine transformation to the image
        return cv2.warpAffine(self.image, matrix, (self.cols, self.rows))


class ProjectiveTransformation(Transformation):
    def __init__(self, image_filepath: str):
        super().__init__(image_filepath)

    def transformation(self, matrix):
        matrix = matrix.astype(np.float32)
        # Apply the affine transformation to the image
        return cv2.warpPerspective(self.image, matrix, (self.cols, self.rows))


def stick_image(background_path: str, stick_image_path: str, plane_corners) -> np.ndarray:
    # Load the background image (plane)
    background = cv2.imread(background_path)
    # Load the image to stick on the plane
    image_to_stick = cv2.imread(stick_image_path)

    # Resize the image to stick to match the region on the background
    resize_width = 300  # Adjust as needed
    resize_height = 200  # Adjust as needed
    image_to_stick = cv2.resize(image_to_stick, (resize_width, resize_height))
    image_corners = np.array([[0, 0], [resize_width, 0], [resize_width, resize_height], [0, resize_height]],
                             dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(image_corners, plane_corners)

    # Apply the perspective transformation to stick the image onto the plane
    result = cv2.warpPerspective(image_to_stick, matrix, (background.shape[1], background.shape[0]))

    # Create a mask for the image to stick
    mask = np.zeros_like(background)
    cv2.fillConvexPoly(mask, plane_corners.astype(int), (255, 255, 255))

    # Inverse the mask to get the region of the plane not covered by the image
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the background image to create the final result
    masked_background = cv2.bitwise_and(background, mask_inv)
    final_result = cv2.add(masked_background, result)
    return final_result


# stick image
# background_path = r"D:\Learning\Computer_Vision\transformation\20240409_094733.jpg"
# image_path = r"D:\Learning\Computer_Vision\transformation\koko.jpg"
# plane_corners = np.array([[1220, 837], [1959, 1110], [1897, 2527], [1066, 2431]], dtype=np.float32)
# stick_image(background_path, image_path, plane_corners)

# Basic transformations
image_path = r"D:\Learning\Computer_Vision\transformation\cat_1.jpg.jpg"
affine_transformation = AffineTransformation(image_path)
projective_transformation = ProjectiveTransformation(image_path)
transformed_image = affine_transformation.apply_transformation(20, 100)
p_transformed_image = projective_transformation.apply_transformation(20, 100)

# transformed_image = affine_transformation.apply_reflect('x')
# p_transformed_image = projective_transformation.apply_reflect('x')
#
# transformed_image = affine_transformation.apply_rotation(15)
# p_transformed_image = projective_transformation.apply_rotation(15)
#
# transformed_image = affine_transformation.apply_scale(0.5, 0.25)
# p_transformed_image = projective_transformation.apply_scale(0.5, 0.25)
#
# transformed_image = affine_transformation.apply_shear(0.5)
# p_transformed_image = projective_transformation.apply_shear(0.5)

cv2.imshow('Affine transformation', transformed_image)
cv2.imshow('Projective transformation', p_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
