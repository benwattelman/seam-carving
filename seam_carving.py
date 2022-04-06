from copy import deepcopy
from typing import Dict, Any, List, Tuple

import numpy as np

from utils import get_gradients, to_grayscale

NDArray = Any


class Seam:
    def __init__(self, pixels: NDArray):
        self.pixels = pixels



class Pixel:
    def __init__(self, i: int, j: int):
        self.indices = np.array([i,j])


# pixel_energy_matrix = None
# index_mapping_matrix = None
# vertical_seams = None
# horizontal_seams = None
# vertical_seams_to_find = -1
# horizontal_seams_to_find = -1

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

def find_optimal_seam(cost_matrix: NDArray) -> Seam:
    pixels = []
    last_column_index = cost_matrix.shape[0] - 1
    row_index = np.argmin(cost_matrix[last_column_index])
    pixels.append((last_column_index, row_index))
    for i in range(last_column_index):
        column_index = last_column_index - 1
        if cost_matrix[column_index, row_index] == pixel_energy_matrix[column_index, row_index] + cost_matrix[
            column_index - 1, row_index] + c_left(column_index, row_index)

    return Seam(pixels)

    pixel_energy_matrix = get_gradients(deepcopy(image))
    vertical_seams_to_find = abs(out_width - image.shape[1])
    horizontal_seams_to_find = abs(out_width - image.shape[0])
    # initialize_pixel_energy_matrix()
    # initialize_index_mapping_matrix(dimensions)
    current_image = to_grayscale(image)

    for i in range(vertical_seams_to_find):
        cost_matrix = calculate_cost_matrix()
        seam = find_optimal_seam(cost_matrix)
        remove_seam(seam, current_image) # remove from image, add seam to list, shift index matrix

    # raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}





