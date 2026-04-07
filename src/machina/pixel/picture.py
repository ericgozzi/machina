from __future__ import annotations

from typing import Optional

import imageio.v2 as imageio
import numpy as np

from machina.colors import WHITE, Color


class Picture:
    """
    Represents a BitMap image.

    Args:
        pixels: A 3D list or NumPy array of shape (height, width, 3) where each innermost list/array contains RGB values (0-255).

    Attributes:
        pixels: The pixel data as a NumPy array.
        height: The height of the image in pixels.
        width: The width of the image in pixels.


    """

    def __init__(self, pixels: list[list[list[int]]] | np.ndarray):
        # dtype is uint8 for RGB values (0-255)
        self._pixels = np.array(pixels, dtype=np.uint8)

        if self._pixels.ndim != 3 or self._pixels.shape[2] != 3:
            raise ValueError("Pixels must be a 3D array with shape (height, width, 3)")

    def __repr__(self) -> str:
        return f"Picture(width={self.width}, height={self.height})"

    @property
    def data(self) -> dict:
        return {
            "type": "picture",
            "pixels": self._pixels.tolist(),
        }

    @property
    def pixels(self) -> np.ndarray:
        return self._pixels

    @property
    def height(self) -> int:
        return self._pixels.shape[0]

    @property
    def width(self) -> int:
        return self._pixels.shape[1]

    @classmethod
    def from_data(cls, data: dict) -> Picture:
        if data["type"] != "picture":
            raise ValueError("Data type must be 'picture'")
        pixels = np.array(data["pixels"], dtype=np.uint8)
        return cls(pixels)

    @classmethod
    def from_file(cls, file_path: str) -> Picture:
        """
        Creates a Picture instance by loading an image file.

        Args:
            file_path: The path to the image file to load.

        Returns:
            A Picture instance containing the loaded image data.

        """
        pixels = imageio.imread(file_path)
        return cls(pixels)

    @classmethod
    def from_blank(
        cls, width: int, height: int, color: Optional[Color] = None
    ) -> Picture:
        """
        Creates a blank picture of the specified width and height, filled with the given color.
        Args:
            width: The width of the blank picture in pixels.
            height: The height of the blank picture in pixels.
            color: An optional Color instance to fill the picture with (default is white).
        """
        if color is None:
            color = WHITE
        r = int(color.r * 255)
        g = int(color.g * 255)
        b = int(color.b * 255)
        pixels = np.full((height, width, 3), (r, g, b), dtype=np.uint8)
        return cls(pixels)

    def copy(self) -> Picture:
        """Creates a deep copy of the current Picture instance."""
        return Picture(self._pixels.copy())

    def save(self, file_path: str) -> None:
        """Saves the current picture to an image file."""
        imageio.imsave(file_path, self._pixels)

    def show(self):
        """Displays the current picture using the default image viewer."""
        import os
        import subprocess
        import sys
        import tempfile

        # 1. Create a temporary file path
        # suffix=".png" helps the OS identify the file type
        # delete=False ensures the file stays on disk long enough for the viewer to load it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        # 2. Save the current pixels to that temporary path
        self.save(temp_path)

        # 3. Open the file based on the operating system
        try:
            if sys.platform.startswith("darwin"):  # macOS
                subprocess.call(("open", temp_path))
            elif os.name == "nt":  # Windows
                os.startfile(temp_path)
            elif os.name == "posix":  # Linux/Unix
                subprocess.call(("xdg-open", temp_path))
        except Exception as e:
            pass

    def pixel(self, x, y) -> Color:
        """Returns the RGB color of the pixel at (x, y) as a tuple.

        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.

        Returns:
            A tuple (r, g, b) representing the color of the pixel at (x
        """
        color = self._pixels[y, x]
        r, g, b = color
        color = Color.from_rgb(r, g, b)
        return color

    def set_pixel(self, x, y, color: tuple[int, int, int]):
        """Sets the RGB color of the pixel at (x, y) in place."""
        self._pixels[y, x] = color

    def flip_horizontal(self) -> Picture:
        """Flips the current picture left-to-right in place."""
        self._pixels = np.flip(self._pixels, axis=1)
        return self

    def flip_vertical(self) -> Picture:
        """Flips the current picture top-to-bottom in place."""
        self._pixels = np.flip(self._pixels, axis=0)
        return self

    def rotate_90(self, clockwise: bool = True) -> None:
        """Rotates the current picture 90 degrees in place."""
        k_val = 3 if clockwise else 1
        self._pixels = np.rot90(self._pixels, k=k_val)

    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """Crops the image to the specified rectangle in place."""
        # Ensure we stay within bounds to avoid empty arrays
        x_end = min(x + width, self.width)
        y_end = min(y + height, self.height)

        # NumPy uses [rows, columns] -> [y, x]
        self._pixels = self._pixels[y:y_end, x:x_end].copy()

    def adjust_brightness(self, factor: float) -> None:
        """Adjusts the brightness of the current picture in place.
        factor > 1.0 brightens, factor < 1.0 darkens."""
        result = self._pixels.astype(float) * factor
        result = np.clip(result, 0, 255)
        self._pixels = result.astype(np.uint8)

    def adjust_contrast(self, factor: float) -> None:
        """
        Adjusts contrast in place.
        factor > 1.0 increases contrast, factor < 1.0 decreases it.
        """
        # Formula: (pixel - 128) * factor + 128
        res = (self._pixels.astype(float) - 128) * factor + 128
        self._pixels = np.clip(res, 0, 255).astype(np.uint8)

    def to_grayscale(self) -> None:
        """Converts the image to grayscale in place (retains 3 channels)."""
        # Standard luminosity weights
        weights = np.array([0.299, 0.587, 0.114])
        # Dot product reduces (H, W, 3) to (H, W)
        gray = np.dot(self._pixels[..., :3], weights)
        # To keep our 3-channel (RGB) structure, we broadcast the gray back
        # This stacks the same 2D gray array 3 times
        self._pixels = np.stack([gray] * 3, axis=-1).astype(np.uint8)

    def paste_on(self, other: Picture, x: int, y: int) -> Picture:
        """Pastes the current picture onto another picture at the specified (x, y) position in place."""
        # Calculate the region of the other picture to paste onto
        x_end = min(x + self.width, other.width)
        y_end = min(y + self.height, other.height)

        # Calculate the corresponding region of the current picture to paste
        paste_width = x_end - x
        paste_height = y_end - y

        if paste_width <= 0 or paste_height <= 0:
            return

        # Paste the pixels from the current picture onto the other picture
        other._pixels[y:y_end, x:x_end] = self._pixels[0:paste_height, 0:paste_width]
        return other


if __name__ == "__main__":
    pic = Picture.from_file("checker.jpg")

    pic2 = pic.copy()
    pic2.crop(0, 0, 100, 100)
    pic = pic2.paste_on(pic, 1000, 50)

    print(
        pic.pixel(50, 100)
    )  # Should print the color of the pixel at (1000, 50) after pasting

    pic.show()
