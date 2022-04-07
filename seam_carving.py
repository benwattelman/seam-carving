from copy import deepcopy
from typing import Dict, Any, List, Tuple

import numpy as np

from utils import get_gradients, to_grayscale

NDArray = np.ndarray

def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """

    def calculate_cost_matrix() -> NDArray:
        pixel_energy_matrix

    def find_optimal_seam() -> Tuple[NDArray, NDArray]:
        current_seam = np.array(cost_matrix.shape[0])
        original_index_seam = np.array(cost_matrix.shape[0])
        row = cost_matrix.shape[0] - 1
        col = np.argmin(cost_matrix[row])
        original_index_seam[row] = index_mapping_matrix[row, col]
        current_seam[row] = col
        while row > 0:
            if col > 0 and cost_matrix[row, col] == pixel_energy_matrix[row, col] + \
                    cost_matrix[row - 1, col - 1] + c_l_matrix[row, col]:
                col -= 1
            elif col < current_image.shape[1] - 1 and cost_matrix[row, col] == pixel_energy_matrix[
                row, index_mapping_matrix[row, col]] + cost_matrix[row - 1, col + 1] + c_r_matrix[row, col]:
                col += 1
            row -= 1
            original_index_seam[row] = index_mapping_matrix[row, col]
            current_seam[row] = col
        return original_index_seam, current_seam

    def shift_matrix(image, seam):
        mask = create_mask(seam, dimensions)

    def create_mask()


    pixel_energy_matrix = get_gradients(deepcopy(image))
    vertical_seams_to_find = abs(out_width - image.shape[1])
    horizontal_seams_to_find = abs(out_height - image.shape[0])
    vertical_seams = np.array((vertical_seams_to_find, image.shape[0]))
    horizontal_seams = np.array((horizontal_seams_to_find, image.shape[1]))
    # initialize_pixel_energy_matrix()
    index_mapping_matrix = np.indices((image.shape[0], image.shape[1]))[1]
    current_image = to_grayscale(image)

    for i in range(vertical_seams_to_find):
        cost_matrix, c_v_matrix, c_l_matrix, c_r_matrix = calculate_cost_matrix()
        original_index_seam, current_seam = find_optimal_seam()
        vertical_seams[i] = original_index_seam
        shift_matrix(current_image, current_seam)
        shift_matrix(index_mapping_matrix, current_seam)
        shift_matrix(pixel_energy_matrix, current_seam)

    vertical_seam_image = paint_seams_in_image(np.copy(image), vertical_seams, mask, color)
    if out_width < image.shape[1]:
        rgb_image_without_vertical_seams = remove_seams(np.copy(image), vertical_seams)
    else:

        #todo: change to resized image




    current_image = np.rot90(current_image, k=1, axes=(0,1))
    index_mapping_matrix = np.rot90(index_mapping_matrix, k=1, axes=(0,1))
    pixel_energy_matrix = np.rot90(pixel_energy_matrix, k=1, axes=(0,1))

    for i in range(horizontal_seams_to_find):
        cost_matrix, c_v_matrix, c_l_matrix, c_r_matrix = calculate_cost_matrix()
        original_index_seam, current_seam = find_optimal_seam()
        horizontal_seams[i] = original_index_seam
        shift_matrix(current_image, current_seam)
        shift_matrix(index_mapping_matrix, current_seam)
        shift_matrix(pixel_energy_matrix, current_seam)

    rotated_rgb_image = np.rot90(rgb_image_without_vertical_seams, k=1, axes=(0, 1))
    rotated_horizontal_seam_image = paint_seams_in_image(rotated_rgb_image, mask, color)
    horizontal_seam_image = np.rot90(rgb_image_without_vertical_seams, k=-1, axes=(0, 1))



    # raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


