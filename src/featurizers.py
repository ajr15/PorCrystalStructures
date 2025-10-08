# module to contain all featurizers used for ML analysis
from functools import reduce
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.sqlmodels import StructureProperty, SubstituentProperty

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

    def __init__(self, property_names, property_units, navalue, property_source=None):
        super().__init__(property_names, navalue)
        self.property_names = property_names
        self.property_units = property_units
        self.property_source = property_source

    def _featurize(self, session, structure_ids) -> np.array:
        dfs = []
        for pname, punits in zip(self.property_names, self.property_units):
            dfs.append(self.query_to_df(session, pname, punits, self.property_source))
        ajr = pd.concat(dfs, axis=1, join="outer")
        # filter to only requested ids - and fill in empty values
        ajr = ajr.reindex(structure_ids)
        return ajr.values

    def query_to_df(self, session, pname: str, units: str, source: str) -> pd.DataFrame:
        # build query
        q = session.query(StructureProperty.structure, StructureProperty.value).\
            filter(StructureProperty.property == pname)
        if units is not None:
            q = q.filter(StructureProperty.units == units)
        if source is not None:
            q = q.filter(StructureProperty.source.like(f"%{source}%"))
        # run & format
        ajr = pd.DataFrame(q.all(), columns =["sid", pname])
        ajr = ajr.set_index('sid')
        return ajr


class SubstituentPropertyFeaturizer (Featurizer):

    def __init__(self, property_name, positions, navalue):
        super().__init__(positions, navalue)
        self.property_name = property_name
        self.positions = positions

    def _featurize(self, session, structure_ids) -> np.array:
        res = []
        for sid in structure_ids:
            vec = self.structure_property(session, sid, self.property_name)
            res.append(vec)
        df = pd.DataFrame(res)
        df = df[self.positions]
        return df.values

    def structure_property(self, session, sid: int, prop: str):
        q = session.query(SubstituentProperty.position, SubstituentProperty.position_index, SubstituentProperty.value).filter(SubstituentProperty.structure == sid).filter(SubstituentProperty.property == prop).order_by(SubstituentProperty.position, SubstituentProperty.position_index)
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


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///{}".format("../main.db"))
    session = sessionmaker(bind=engine)()
    feat = StructurePropertyFeaturizer(["inner_circuit homa", "inner_circuit en"], [None, None], navalue=None, property_source="dft")
    df = feat.featurize(session, ["ADIQAI", "AKOTUQ", "ALITEU"])
    print(df)