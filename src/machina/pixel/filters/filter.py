from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from machina.pixel.picture import Picture


class PictureFilter(ABC):
    """
    An abstract base class for creating picture filters.

    This class defines the interface that all picture filters must implement.
    Subclasses should override the `apply` method to define their specific
    filtering logic.
    """

    @abstractmethod
    def apply(self, picture: Picture) -> Picture:
        """
        Applies the filter to a given Picture instance.

        This method must be implemented by all concrete filter subclasses.

        Args:
            picture: The Picture object to which the filter will be applied.

        Returns:
            A Picture object with the filter applied. This can be either
            a new Picture instance or a modified version of the original.
        """
        pass

    def __call__(self, picture: Picture) -> Picture:
        """
        A convenience method to allow the filter to be called like a function.
        """
        return self.apply(picture)
