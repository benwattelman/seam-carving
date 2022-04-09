from typing import Dict, Tuple

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
        working_cost_matrix = np.zeros_like(current_image)
        rows = working_cost_matrix.shape[0]
        cols = working_cost_matrix.shape[1]
        # calculate first row of cost matrix
        working_cost_matrix[0] = np.copy(pixel_energy_matrix[0])
        #  calculate remaining rest
        for row in range(1, rows):
            cost_above = working_cost_matrix[row - 1]
            cost_above_left = np.roll(cost_above, shift=1, axis=0)
            cost_above_left[0] = 255.0
            cost_above_right = np.roll(cost_above, shift=-1, axis=0)
            cost_above_right[cols - 1] = 255.0
            working_cost_matrix[row, 0] = pixel_energy_matrix[row, 0] + min(cost_above[0], cost_above_right[0])
            working_cost_matrix[row, cols - 1] = pixel_energy_matrix[row, cols - 1] + min(
                cost_above[cols - 1],
                cost_above_left[cols - 1])
            minimal_cost_above = np.minimum(np.minimum(cost_above[1:cols - 1], cost_above_left[1:cols - 1]),
                                            cost_above_right[1:cols - 1])
            working_cost_matrix[row, 1:cols - 1] = pixel_energy_matrix[row, 1:cols - 1] + minimal_cost_above

        return working_cost_matrix

    def calculate_forward_looking_cost_matrix() -> (NDArray, NDArray, NDArray, NDArray):
        working_forward_looking_cost_matrix = np.zeros_like(current_image)
        rows = working_forward_looking_cost_matrix.shape[0]
        cols = working_forward_looking_cost_matrix.shape[1]

        left = np.roll(current_image, shift=1, axis=1)
        right = np.roll(current_image, shift=-1, axis=1)
        above = np.roll(current_image, shift=1, axis=0)

        # calculate Cv for entire matrix
        cv_matrix = np.abs(right - left)
        cv_matrix[:, 0] = 255.0  # Cv(i,0) is undefined - set to 255.0 according to guidelines
        cv_matrix[:, cols - 1] = 255.0  # Cv(i,cols-1) is undefined

        # calculate Cl for entire matrix
        cl_matrix = cv_matrix + np.abs(above - left)
        cl_matrix[0, :] = 255  # Cl(0,j) is undefined
        cl_matrix[:, 0] = 255  # Cl(i,0) is undefined
        cl_matrix[cl_matrix > 255] = 255

        # calculate Cr for entire matrix
        cr_matrix = cv_matrix + np.abs(above - right)
        cr_matrix[0, :] = 255  # Cr(0,j) is undefined
        cr_matrix[:, cols - 1] = 255.0  # Cr(i,cols-1) is undefined

        # calculate first row of cost matrix
        working_forward_looking_cost_matrix[0] = np.copy(pixel_energy_matrix[0])
        #  calculate remaining rest
        for row in range(1, rows):
            cost_above = working_forward_looking_cost_matrix[row - 1]
            cost_above_left = np.roll(cost_above, shift=1, axis=0)
            cost_above_left[0] = 255.0
            cost_above_right = np.roll(cost_above, shift=-1, axis=0)
            cost_above_right[cols - 1] = 255.0
            working_forward_looking_cost_matrix[row, 0] = pixel_energy_matrix[row, 0] + min(
                (cost_above[0] + cv_matrix[row, 0]), (cost_above_right[0] + cr_matrix[row, 0]))
            working_forward_looking_cost_matrix[row, cols - 1] = pixel_energy_matrix[row, cols - 1] + min(
                (cost_above[cols - 1] + cv_matrix[row, cols - 1]),
                (cost_above_left[cols - 1] + cl_matrix[row, cols - 1]))
            minimal_cost_above = np.minimum(
                np.minimum((cost_above[1:cols - 1] + cv_matrix[row, 1:cols - 1]),
                           (cost_above_left[1:cols - 1] + cl_matrix[row, 1:cols - 1])),
                (cost_above_right[1:cols - 1] + cr_matrix[row, 1:cols - 1]))
            working_forward_looking_cost_matrix[row, 1:cols - 1] = pixel_energy_matrix[row,
                                                                   1:cols - 1] + minimal_cost_above

        return working_forward_looking_cost_matrix, cl_matrix, cv_matrix, cr_matrix

    def find_optimal_seam() -> Tuple[NDArray, NDArray]:
        current_optimal_seam = np.zeros(cost_matrix.shape[0])
        current_original_index_seam = np.zeros(cost_matrix.shape[0])
        row = cost_matrix.shape[0] - 1
        col = np.argmin(cost_matrix[row])
        current_original_index_seam[row] = index_mapping_matrix[row, col]
        current_optimal_seam[row] = col
        if forward_implementation:
            while row > 0:
                if col > 0 and cost_matrix[row, col] == pixel_energy_matrix[row, col] + \
                        cost_matrix[row - 1, col - 1] + c_l_matrix[row, col]:
                    col -= 1
                elif col < current_image.shape[1] - 1 and \
                        cost_matrix[row, col] == pixel_energy_matrix[row, index_mapping_matrix[row, col]] + cost_matrix[
                    row - 1, col + 1] + c_r_matrix[row, col]:
                    col += 1
                row -= 1
                current_original_index_seam[row] = index_mapping_matrix[row, col]
                current_optimal_seam[row] = col
        else:
            while row > 0:
                if col > 0 and cost_matrix[row, col] == pixel_energy_matrix[row, col] + \
                        cost_matrix[row - 1, col - 1]:
                    col -= 1
                elif col < current_image.shape[1] - 1 and cost_matrix[row, col] == pixel_energy_matrix[
                    row, index_mapping_matrix[row, col]] + cost_matrix[row - 1, col + 1]:
                    col += 1
                row -= 1
                current_original_index_seam[row] = index_mapping_matrix[row, col]
                current_optimal_seam[row] = col
        return current_original_index_seam, current_optimal_seam

    def shift_matrix(matrix: NDArray):
        for row, col in enumerate(current_seam):
            matrix[row][col:-1] = matrix[row][col + 1:]

    def shift_matrix_with_mask(matrix: NDArray, seams: NDArray) -> NDArray:
        mask = get_mask_for_matrix_and_seams(matrix, seams)
        return matrix[mask].reshape(matrix.shape[0], matrix.shape[1] - 1)

    def get_mask_for_matrix_and_seams(matrix: NDArray, seams: NDArray) -> NDArray:
        mask = np.ones_like(matrix, dtype=bool)
        for seam in seams:
            for row, col in enumerate(seam):
                mask[row, col] = False
        return mask

    def duplicate_seams_in_image(image_to_enlarge: NDArray, mask: NDArray, output_width: int) -> NDArray:
        enlarged_image = np.zeros((image_to_enlarge.shape[0], output_width))  # initialize empty NDArray
        for row in range(enlarged_image.shape[0]):
            original_col_index = 0
            output_col_index = 0
            while original_col_index < image_to_enlarge.shape[1]:
                enlarged_image[row, output_col_index] = image_to_enlarge[
                    row, original_col_index]  # current pixel keeps its original color, regardless of whether it's duplicated or not
                if not mask[
                    row, original_col_index]:  # mask[row,original_col_index] == False indicates this is a pixel to duplicate
                    enlarged_image[row, output_col_index + 1] = image_to_enlarge[row, original_col_index]
                    output_col_index += 1  # skip next pixel since we already colored it
                original_col_index += 1
                output_col_index += 1

        return enlarged_image

    def paint_seams_in_image(image_to_paint: NDArray, mask: NDArray, color: str) -> NDArray:
        color_to_use = [255, 0, 0] if color == "red" else [0, 0, 0, ]  # [255, 0, 0] is red, [0, 0, 0,] is black
        image_to_paint[mask == False] = color_to_use

        return image_to_paint

    ### Main flow ###
    pixel_energy_matrix = get_gradients(np.copy(image))
    vertical_seams_to_find = abs(out_width - image.shape[1])
    horizontal_seams_to_find = abs(out_height - image.shape[0])
    vertical_seams = np.zeros((vertical_seams_to_find, image.shape[0]))
    horizontal_seams = np.zeros((horizontal_seams_to_find, out_width))
    index_mapping_matrix = np.indices((image.shape[0], image.shape[1]))[1]
    current_image = to_grayscale(np.copy(image))
    images_dict = dict.fromkeys(['resized', 'vertical_seams', 'horizontal_seams'])

    for i in range(vertical_seams_to_find):
        if forward_implementation:
            cost_matrix, c_v_matrix, c_l_matrix, c_r_matrix = calculate_forward_looking_cost_matrix()
        else:
            cost_matrix = calculate_cost_matrix()
        original_index_seam, current_seam = find_optimal_seam()
        vertical_seams[i] = original_index_seam
        shift_matrix(
            current_image)  # optional - replace with: current_image = shift_matrix_with_mask(current_image, [current_optimal_seam])
        shift_matrix(index_mapping_matrix)  # as above
        shift_matrix(pixel_energy_matrix)  # as above

    mask_for_vertical_seam_image = get_mask_for_matrix_and_seams(image, vertical_seams)
    vertical_seam_image = paint_seams_in_image(np.copy(image), mask_for_vertical_seam_image, "red")
    images_dict['vertical_seams'] = vertical_seam_image
    rgb_image_without_vertical_seams = shift_matrix_with_mask(np.copy(image), vertical_seams) if out_width < \
                                                                                                 image.shape[1] \
        else duplicate_seams_in_image(np.copy(image), mask_for_vertical_seam_image, out_width)

    current_image = np.rot90(current_image, k=1, axes=(0, 1))
    index_mapping_matrix = np.rot90(index_mapping_matrix, k=1, axes=(0, 1))
    pixel_energy_matrix = np.rot90(pixel_energy_matrix, k=1, axes=(0, 1))

    for i in range(horizontal_seams_to_find):
        cost_matrix, c_v_matrix, c_l_matrix, c_r_matrix = calculate_cost_matrix()
        original_index_seam, current_seam = find_optimal_seam()
        horizontal_seams[i] = original_index_seam
        shift_matrix(current_image)
        shift_matrix(index_mapping_matrix)
        shift_matrix(pixel_energy_matrix)

    mask_for_horizontal_seam_image = np.ones_like(horizontal_seams, dtype=bool)
    rotated_rgb_image = np.rot90(rgb_image_without_vertical_seams, k=1, axes=(0, 1))
    rotated_horizontal_seam_image = paint_seams_in_image(rotated_rgb_image, mask_for_horizontal_seam_image, "black")
    horizontal_seam_image = np.rot90(rgb_image_without_vertical_seams, k=-1, axes=(0, 1))
    images_dict['horizontal_seams'] = horizontal_seam_image

    return images_dict
    # raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}
