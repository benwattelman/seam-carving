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
        working_cost_matrix = np.zeros_like(current_image)
        rows = working_cost_matrix.shape[0]
        cols = working_cost_matrix.shape[1]

        left = np.roll(current_image, shift=1, axis=1)
        right = np.roll(current_image, shift=-1, axis=1)
        above = np.roll(current_image, shift=1, axis=0)

        # calculate Cv for entire matrix
        cv_matrix = np.abs(right - left)
        cv_matrix[:, 0] = 255.0  # Cv(i,0) is undefined - set to 255.0 according to guidelines
        cv_matrix[:, cols-1] = 255.0  # Cv(i,cols-1) is undefined

        # calculate Cl for entire matrix
        cl_matrix = cv_matrix + np.abs(above - left)
        cl_matrix[0, :] = 255  # Cl(0,j) is undefined
        cl_matrix[:, 0] = 255  # Cl(i,0) is undefined
        cl_matrix[cl_matrix > 255] = 255

        # calculate Cr for entire matrix
        cr_matrix = cv_matrix + np.abs(above - right)
        cr_matrix[0, :] = 255  # Cr(0,j) is undefined
        cr_matrix[:, cols-1] = 255.0  # Cr(i,cols-1) is undefined


        return working_cost_matrix

    def find_optimal_seam(cost_matrix: NDArray) -> Seam:
        pixels = []
        last_column_index = cost_matrix.shape[0] - 1
        row_index = np.argmin(cost_matrix[last_column_index])
        pixels.append((last_column_index, row_index))
        for i in range(last_column_index):
            column_index = last_column_index - 1
            # if cost_matrix[column_index, row_index] == pixel_energy_matrix[column_index, row_index] + cost_matrix[
            #     column_index - 1, row_index] + c_left(column_index, row_index)

        return Seam(pixels)

    index_mapping_matrix = None #todo: implement this matrix
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





