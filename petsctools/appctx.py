import itertools


class AppContext:
    def __init__(self):
        self._count = itertools.count()
        self._data = {}
        self._missing_key = next(self._count)

    @property
    def missing_key(self):
        return self._missing_key

    def insert(self, val):
        key = next(self._count)
        self._data[key] = val
        return key

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        if key == self.missing_key:
            return default
        return self._data.get(key, default=default)

    def values(self):
        return self._data.values()
