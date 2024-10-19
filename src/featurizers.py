# module to contain all featurizers used for ML analysis
from functools import reduce
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from read_to_sql import StructureProperty, SubstituentProperty

class Featurizer (ABC):

    def __init__(self, feature_names, navalue=None):
        self.feature_names = feature_names
        self.navalue = navalue

    @abstractmethod
    def _featurize(self, session, structure_ids) -> np.array:
        pass

    def featurize(self, session, structure_ids) -> pd.DataFrame:
        vecs = self._featurize(session, structure_ids)
        df = pd.DataFrame(vecs, index=structure_ids, columns=self.feature_names)
        if self.navalue is not None:
            df = df.fillna(self.navalue)
        return df

    def __add__(self, other):
        if not isinstance(other, Featurizer):
            raise ValueError("Cannot add 'Featurizer' to {}".format(type(other)))
        return ComboFeaturizer([self, other])


class ComboFeaturizer (Featurizer):

    def __init__(self, featurizers: List[Featurizer]):
        self.featurizers = featurizers
        names = []
        for x in featurizers:
            names.extend(x.feature_names)
        super().__init__(names)

    def _featurize(self, session, structure_ids) -> np.array:
        vecs = tuple([feat.featurize(session, structure_ids).to_numpy() for feat in self.featurizers])
        return np.hstack(vecs)

        
class StructurePropertyFeaturizer (Featurizer):

    def __init__(self, property_names, property_units, navalue):
        super().__init__(property_names, navalue)
        self.property_names = property_names
        self.property_units = property_units

    def _featurize(self, session, structure_ids) -> np.array:
        res = []
        for sid in structure_ids:
            vec = []
            for pname, punits in zip(self.property_names, self.property_units):
                vec.append(self.structure_property(session, sid, pname, punits))
            res.append(vec)
        return np.array(res)

    @staticmethod
    def structure_property(session, sid: int, property: str, units: str):
        q = session.query(StructureProperty.value).filter(StructureProperty.structure == sid).filter(StructureProperty.property == property)
        if units is not None:
            q = q.filter(StructureProperty.units == units)
        v = q.all()
        if len(v) == 0:
            return None
        else:
            return v[0][0]


class SubstituentPropertyFeaturizer (Featurizer):

    def __init__(self, property_name, property_units, positions, navalue):
        super().__init__(positions, navalue)
        self.property_name = property_name
        self.property_units = property_units
        self.positions = positions

    def _featurize(self, session, structure_ids) -> np.array:
        res = []
        for sid in structure_ids:
            vec = self.structure_property(session, sid, self.property_name, self.property_units)
            res.append(vec)
        df = pd.DataFrame(res)
        df = df[self.positions]
        return df.values

    def structure_property(self, session, sid: int, prop: str, units: str):
        q = session.query(SubstituentProperty.position, SubstituentProperty.position_index, SubstituentProperty.value).filter(SubstituentProperty.structure == sid).filter(SubstituentProperty.property == prop).order_by(SubstituentProperty.position, SubstituentProperty.position_index)
        if units is not None:
            q = q.filter(StructureProperty.units == units)
        rows = q.all()
        if len(rows) == 0:
            raise ValueError("The property {} does not exists".format(prop))
        else:
            ajr = {v[0] + str(v[1]): v[2] for v in rows}
            for p in self.positions:
                if not p in ajr:
                    ajr[p] = self.navalue
            return ajr


class FunctionFeaturizer (Featurizer):

    def __init__(self, name: str, func, navalue):
        if type(name) is str:
            super().__init__([name], navalue)
        else:
            super().__init__(name, navalue)
        self.func = func

    def _featurize(self, session, structure_ids) -> np.array:
        res = np.array([self.func(session, sid) for sid in structure_ids])
        return res
        return res.reshape((-1, 1))