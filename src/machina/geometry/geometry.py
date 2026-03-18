from abc import ABC, abstractmethod


class Geometry(ABC):
    def __init__(self, **kwargs):
        self.attributes = {}
        self.attributes.update(kwargs)

    @property
    @abstractmethod
    def data(self) -> dict:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_data(cls, data):
        raise NotImplementedError

    def copy(self):
        data = self.data
        return type(self).from_data(data)
