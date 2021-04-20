"""
Routines to load example data locally or from web
"""

from functools import partial
from pathlib import Path

import pandas as pd


def _attribute_getter(self, name):
    return self[name]


class LoadExampleData(object):

    _names = [
        "O2_ABand_Drouin_2017_linelist",
        "O2_ABand_Long_2010_linelist",
        "O2_SingletDelta_HITRAN",
        "O2_SingletDelta_Mendonca",
        "CO2_30012_NIST",
    ]

    _prefix_web = (
        "https://raw.githubusercontent.com/usnistgov/MATS/master/MATS/Linelists/"
    )
    _prefix_local = Path(__file__).parent / "Linelists"

    def __init__(self, names=None):

        if names is not None:
            self._names = names
        self._cache = {}

        # create attributes for names
        for k in self.names:
            kk = k.replace(" ", "_")
            setattr(LoadExampleData, kk, property(partial(_attribute_getter, name=k)))

    def _get_file(self, name):
        if name not in self.names:
            raise ValueError("file name must be in {}".format(self.names))

        name = name + ".csv"

        # try local
        path = self._prefix_local / name
        if not path.exists():
            # fallback to url
            path = self._prefix_web + name
        if name not in self._cache:
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]

    @property
    def names(self):
        return self._names

    def __getitem__(self, index):
        if isinstance(index, int):
            name = self.names[index]
        elif isinstance(index, str):
            name = index
        else:
            raise ValueError("bad index {}".format(index))
        return self._get_file(name)


global_loader = LoadExampleData()
