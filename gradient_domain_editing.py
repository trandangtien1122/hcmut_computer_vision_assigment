import typing as tp

import cv2
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def resize_object_image_and_generate_mask(filepath, mask_x_location, mask_y_location,
                                          source_filepath, x_location, y_location):
    image = cv2.imread(filepath)
    mask_height = image.shape[0]
    mask_width = image.shape[1]
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Define the center and radius of the circle
    circle_center = (int(mask_width * 0.8 + mask_x_location) // 2, int(mask_height * 0.8 + mask_y_location) // 2)
    circle_radius = 100

    # Draw a filled white circle on the mask
    cv2.circle(mask, circle_center, circle_radius, 255, -1)  # -1 for filled circle

    source = cv2.imread(source_filepath)
    resize_mask = np.zeros((source.shape[0], source.shape[1]))
    resize_image = np.zeros(source.shape)
    resize_mask[x_location:mask.shape[0] + x_location, y_location:mask.shape[1] + y_location] = mask
    resize_image[x_location:image.shape[0] + x_location, y_location:image.shape[1] + y_location, :] = image
    # Save the mask image
    cv2.imwrite('mask_image_circle.jpg', resize_mask)
    cv2.imwrite('image.jpg', resize_image)


def get_image(img_path: str, mask: bool = False, scale: bool = True) -> np.array:
    """
    Gets image in appriopiate format

    Parameters:
    img_path (str): Image path
    mask (bool): True if read mask image
    scale (bool): True if read and scale image to 0-1

    Returns:
    np.array: Image in numpy array
    """
    if mask:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return np.where(binary_mask == 255, 1, 0)

    if scale:
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('double') / 255.0

    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def neighbours(i: int, j: int, max_i: int, max_j: int) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns 4-connected neighbours for given pixel point.
    :param i: i-th index position
    :param j: j-th index position
    :param max_i: max possible i-th index position
    :param max_j: max possible j-th index position
    """
    pairs = []

    for n in [-1, 1]:
        if 0 <= i + n <= max_i:
            pairs.append((i + n, j))
        if 0 <= j + n <= max_j:
            pairs.append((i, j + n))

    return pairs


def poisson_blend(img_s: np.ndarray, mask: np.ndarray, img_t: np.ndarray) -> np.ndarray:
    """
    Returns a Poisson blended image with masked img_s over the img_t.
    :param img_s: the image containing the foreground object
    :param mask: the mask of the foreground object in object_img
    :param img_t: the background image
    """
    img_s_h, img_s_w = img_s.shape

    nnz = (mask > 0).sum()
    im2var = -np.ones(mask.shape, dtype='int32')
    im2var[mask > 0] = np.arange(nnz)

    ys, xs = np.where(mask == 1)

    A = sp.sparse.lil_matrix((4 * nnz, nnz))
    b = np.zeros(4 * nnz)

    e = 0
    for n in range(nnz):
        y, x = ys[n], xs[n]

        for n_y, n_x in neighbours(y, x, img_s_h - 1, img_s_w - 1):

            A[e, im2var[y][x]] = 1
            b[e] = img_s[y][x] - img_s[n_y][n_x]
            if im2var[n_y][n_x] != -1:
                A[e, im2var[n_y][n_x]] = -1
            else:

                b[e] += img_t[n_y][n_x]
            e += 1

    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]

    img_t_out = img_t.copy()

    for n in range(nnz):
        y, x = ys[n], xs[n]
        img_t_out[y][x] = v[im2var[y][x]]

    return np.clip(img_t_out, 0, 1)


source_img_filepath = "D:\Learning\Computer_Vision\gradient_domain_editing\Image-6.jpg"
obj_img_filepath = "D:\Learning\Computer_Vision\gradient_domain_editing\image.jpg"
mask_img_filepath = "D:\Learning\Computer_Vision\gradient_domain_editing\mask_image_circle.jpg"
blend_img_file_path = "D:\Learning\Computer_Vision\gradient_domain_editing/blended_image.jpg"
bg_img = get_image(source_img_filepath)
obj_img = get_image(obj_img_filepath)
mask_img = get_image(mask_img_filepath, mask=True)

blend_img = np.zeros(bg_img.shape)
for b in np.arange(3):
    blend_img[:,:,b] = poisson_blend(obj_img[:,:,b], mask_img, bg_img[:,:,b].copy())
plt.imshow(blend_img)
plt.axis('off')
plt.savefig(blend_img_file_path, bbox_inches='tight', pad_inches=0)