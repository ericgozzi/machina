from __future__ import annotations

from typing import TYPE_CHECKING

from machina.colors import WHITE
from machina.pixel.filters.filter import PictureFilter

if TYPE_CHECKING:
    from machina.pixel.picture import Picture


class ShatterBox(PictureFilter):
    """
    A filter that divides an image into a grid and rotates alternating tiles
    by 180 degrees, creating a "shattered" mosaic effect.

    Args:
        divisions (int): The number of divisions to split the image into
                         along both the width and height. Defaults to 8.
    """

    def __init__(self, divisions: int = 8):
        if not isinstance(divisions, int) or divisions <= 0:
            raise ValueError("Divisions must be a positive integer.")
        self.divisions = divisions

    def apply(self, picture: Picture) -> Picture:
        """
        Applies the ShatterBox filter to the given picture.

        Args:
            picture: The Picture object to process.

        Returns:
            A new Picture object with the ShatterBox effect applied.
        """
        width = picture.width
        height = picture.height

        # Calculate the width and height of each grid tile
        tile_w = width // self.divisions
        tile_h = height // self.divisions

        # Create a new blank picture to build the mosaic on
        mosaic = picture.from_blank(width, height, WHITE)

        for x in range(self.divisions):
            for y in range(self.divisions):
                # Create a copy of the original to work with
                quad = picture.copy()

                # Crop the copy to the current tile's dimensions
                quad.crop(x * tile_w, y * tile_h, tile_w, tile_h)

                # Rotate the tile 180 degrees if it's in an "even" position
                if (x + y) % 2 == 0:
                    quad.rotate_90()
                    quad.rotate_90()

                # Paste the processed tile onto the mosaic
                quad.paste_on(mosaic, x * tile_w, y * tile_h)

        return mosaic
