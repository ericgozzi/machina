import numpy as np

from machina.colors import BLACK, WHITE

from ..picture import Picture
from .filter import PictureFilter


class BayerDither(PictureFilter):
    """
    Applies Bayer dithering to an image, reducing it to a 1-bit
    (black and white) color depth.

    This filter uses a predefined Bayer matrix to create a patterned,
    ordered dither effect, which gives the illusion of more shades of
    gray than are actually present.

    Args:
        threshold (int): The size of the Bayer matrix to use (2, 4, or 8).
                        A larger matrix produces a more detailed dither pattern.
                        Defaults to 4.
    """

    BAYER_MATRICES = {
        2: np.array([[0, 2], [3, 1]]),
        4: np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]),
        8: np.array(
            [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ]
        ),
    }

    def __init__(self, threshold: int = 4):
        self.threshold = threshold
        self.matrix = generate_bayer_matrix(threshold)
        # Normalize the matrix to a 0-1 range
        self.normalized_matrix = self.matrix / (threshold**2) - 0.5

    def apply(self, picture: Picture) -> Picture:
        """
        Applies the Bayer dithering effect to the picture.

        Args:
            picture: The Picture object to process.

        Returns:
            A new Picture object with the dither effect applied.
        """
        # Work on a copy to avoid modifying the original
        dithered_pic = picture.copy()
        dithered_pic.to_grayscale()

        width = dithered_pic.width
        height = dithered_pic.height
        pixels = dithered_pic.pixels

        # Get the size of the Bayer matrix
        matrix_size = self.threshold

        # Define black and white in the required format (0-255)
        black_pixel = [int(BLACK.r * 255), int(BLACK.g * 255), int(BLACK.b * 255)]
        white_pixel = [int(WHITE.r * 255), int(WHITE.g * 255), int(WHITE.b * 255)]

        for y in range(height):
            for x in range(width):
                # Get the grayscale value (0-255) of the pixel
                # We only need the first channel since it's grayscale
                old_pixel_value = pixels[y, x][0]

                # Get the corresponding threshold from the Bayer matrix
                threshold_value = self.normalized_matrix[
                    y % matrix_size, x % matrix_size
                ]

                # Compare the normalized pixel value with the threshold
                # Normalizing the pixel value to a -0.5 to 0.5 range
                if (old_pixel_value / 255.0 - 0.5) > threshold_value:
                    pixels[y, x] = white_pixel
                else:
                    pixels[y, x] = black_pixel

        return dithered_pic


def generate_bayer_matrix(size: int):
    """
    Generates a bayer matrix of a given size usign a recursive algorhtim.

    The size must be a power of two (2, 3, 4, 16, ...)

    Returns:
        A NumPy array represeing the Bayer matrix.

    """
    # power check
    is_power_of_two = (size > 0) and (size & (size - 1) == 0)
    if not is_power_of_two:
        raise ValueError("Size must be a power of two.")

    # the small matrix
    if size == 2:
        return np.array([[0, 2], [3, 1]])

    # recursive step, generate a smaller matrix
    smaller_matrix = generate_bayer_matrix(size // 2)

    # construct the larger matrix using the recursive formula
    # M_2n = | 4*M_n + 0   4*M_n + 2 |
    #        | 4*M_n + 3   4*M_n + 1 |
    top_left = 4 * smaller_matrix
    top_right = 4 * smaller_matrix + 2
    bottom_left = 4 * smaller_matrix + 3
    bottom_right = 4 * smaller_matrix + 1

    return np.block([[top_left, top_right], [bottom_left, bottom_right]])
