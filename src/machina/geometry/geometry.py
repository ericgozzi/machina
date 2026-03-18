from abc import ABC, abstractmethod


class Geometry(ABC):
    def __init__(self, **kwargs):
        self.attributes = {}
        self.attributes.update(kwargs)

    @property
    def data(self) -> dict:
        data = {}
        for key, value in self.attributes.items():
            data[key] = value
        return data

    @classmethod
    @abstractmethod
    def from_data(cls, data):
        raise NotImplementedError

    def copy(self):
        data = self.data
        return type(self).from_data(data)
