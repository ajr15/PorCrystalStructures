# module to contain all featurizers used for ML analysis
from sqlalchemy.orm import Session
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.sqlmodels import StructureProperty, SubstituentProperty, StructureSubstituents

class Featurizer (ABC):

    def __init__(self, feature_names, navalue=None, prefix: str=""):
        self.feature_names = feature_names
        self.navalue = navalue
        self.prefix = prefix

    @abstractmethod
    def _featurize(self, session, structure_ids) -> np.array:
        pass

    def featurize(self, session, structure_ids) -> pd.DataFrame:
        vecs = self._featurize(session, structure_ids)
        cols = [self.prefix + c for c in self.feature_names]
        df = pd.DataFrame(vecs, index=structure_ids, columns=cols)
        if self.navalue is not None:
            df = df.fillna(self.navalue)
        return df

    def transform(self, transform_dict: dict, take_features=None) -> 'Featurizer':
        return TransformedFeaturizer(self, transform_dict, take_features)
    
    def __add__(self, other) -> 'Featurizer':
        if not isinstance(other, Featurizer):
            raise ValueError("Cannot add 'Featurizer' to {}".format(type(other)))
        return ComboFeaturizer([self, other])

class TransformedFeaturizer (Featurizer):

    def __init__(self, featurizer: Featurizer, transform_dict: dict, take_features=None):
        tfeats = list(transform_dict.keys())
        if take_features == "all":
            self.og_featuress = featurizer.feature_names
        elif take_features is None:
            self.og_featuress = []
        elif type(take_features) is list:
            self.og_featuress = take_features
        else:
            raise ValueError(f"Illegal option for take_features ({take_features})")
        super().__init__(self.og_featuress + tfeats, featurizer.navalue, featurizer.prefix)
        self.featurizer = featurizer
        self.transform_dict = transform_dict

    def _featurize(self, session, structure_ids) -> np.array:
        df = self.featurizer.featurize(session, structure_ids)
        for c, func in self.transform_dict.items():
            df[c] = func(df)
        return df[self.feature_names].values

class ComboFeaturizer (Featurizer):

    def __init__(self, featurizers: List[Featurizer]):
        self.featurizers = featurizers
        names = []
        for x in featurizers:
            names.extend([x.prefix + c for c in x.feature_names])
        super().__init__(names)

    def _featurize(self, session, structure_ids) -> np.array:
        vecs = tuple([feat.featurize(session, structure_ids).to_numpy() for feat in self.featurizers])
        return np.hstack(vecs)

        
class StructurePropertyFeaturizer (Featurizer):

    def __init__(self, property_names, property_units, navalue, property_source=None, prefix: str=""):
        super().__init__(property_names, navalue, prefix)
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

    def __init__(self, property_name: str, positions: List[str], navalue, source: str=None, prefix: str=""):
        super().__init__(positions, navalue, prefix)
        self.property_name = property_name
        self.positions = positions
        self.source = source


    def _featurize(self, session, structure_ids) -> np.array:
        res = []
        for sid in structure_ids:
            vec = self.structure_property(session, sid, self.property_name)
            res.append(vec)
        df = pd.DataFrame(res)
        df = df[self.positions]
        return df.values
    
    def fetch_substituent_properties(self, session: Session):
        rows = session.query(SubstituentProperty.substituent, SubstituentProperty.value).filter(SubstituentProperty.property == self.property_name).filter(SubstituentProperty.structure.is_(None)).filter(SubstituentProperty.source == self.source).all()
        return {r[0]: r[1] for r in rows}

    def structure_property(self, session, sid: int, prop: str):
        # check if property exists
        exists = session.query(SubstituentProperty).filter(SubstituentProperty.property == self.property_name).count()
        if exists == 0:
            raise ValueError("The property {} does not exists".format(prop))
        # try to see if the proeprty is given for a specific structure
        q = session.query(SubstituentProperty.position, SubstituentProperty.position_index, SubstituentProperty.value).filter(SubstituentProperty.structure == sid).filter(SubstituentProperty.property == prop).order_by(SubstituentProperty.position, SubstituentProperty.position_index)
        rows = q.all()
        if len(rows) > 0:
            ajr = {v[0] + str(v[1]): v[2] for v in rows}
        else:
            substituents = session.query(StructureSubstituents).filter(StructureSubstituents.structure == sid).all()
            values = self.fetch_substituent_properties(session)
            ajr = {s.position + str(s.position_index): values.get(s.substituent, None) for s in substituents}
        for p in self.positions:
            if not p in ajr:
                ajr[p] = self.navalue
        return ajr

SYMMETRY_OPS = {
    "T1": {
        "meso": {
            1: 3
        }, 
        "beta": {
            1: 4,
            2: 3,
            5: 8,
            6: 7,
        }
    },
    "T2": {
        "meso": {
            2: 4
        }, 
        "beta": {
            1: 8,
            2: 7,
            5: 4,
            6: 3,
        }
    },
    "T12": {
        "meso": {
            1: 4,
            2: 3
        }, 
        "beta": {
            1: 6,
            2: 5,
            8: 7,
            3: 4,
        }
    },
    "T21": {
        "meso": {
            1: 2,
            3: 4
        }, 
        "beta": {
            1: 2,
            6: 5,
            8: 3,
            7: 4,
        }
    },
    "R1": {"meso": {x: (x) % 4 + 1 for x in range(1, 5)}, "beta": {x: (x + 1) % 8 + 1 for x in range(1, 9)}},
    "R2": {"meso": {x: (x + 1) % 4 + 1 for x in range(1, 5)}, "beta": {x: (x + 3) % 8 + 1 for x in range(1, 9)}},
    "R3": {"meso": {x: (x + 2) % 4 + 1 for x in range(1, 5)}, "beta": {x: (x + 5) % 8 + 1 for x in range(1, 9)}},
    "axial": {"axial": {1: 2}}
}

class SymmetryAwareFeaturizer (Featurizer):

    default_symmetries = {
        "all": list(SYMMETRY_OPS.keys()),
        "macrocycle": [s for s in SYMMETRY_OPS.keys() if not s == "axial"],
        "meso": [s for s in SYMMETRY_OPS.keys() if not s == "axial" and not "R" in s],
        "axial": ["axial"]
    }

    def __init__(self, substituent_featurizer: SubstituentPropertyFeaturizer, symmetries: List[str]="all", add_sum: bool=True, custom_funcs: dict=None, prefix: str=""):
        self.substituent_featurizer = substituent_featurizer
        if type(symmetries) is list and any([s not in SYMMETRY_OPS for s in symmetries]):
            raise ValueError(f"Unknown symmetry in {symmetries}. allowed symmetries are {', '.join(SYMMETRY_OPS.keys())}")
        elif type(symmetries) is str and symmetries in self.default_symmetries:
            self.symmetries = self.default_symmetries[symmetries]
        elif type(symmetries) is list:
            self.symmetries = symmetries
        else:
            raise ValueError(f"Unknown symmetry option {symmetries}. allowed options are {', '.join(self.default_symmetries.keys())}")
        self.custom_funcs = {} if custom_funcs is None else custom_funcs
        self.add_sum = add_sum
        feature_names = self.symmetries + list(self.custom_funcs.keys())
        if self.add_sum:
            feature_names += ["sum"]
        super().__init__(feature_names, substituent_featurizer.navalue, prefix)

    @staticmethod
    def calc_symmetry(symmetry: str, df: pd.DataFrame):
        cols = {}
        ops = SYMMETRY_OPS[symmetry]
        for position, d in ops.items():
            # add the inverse to each dict
            d.update({v: k for k, v in d.items()})
            for c in df.columns:
                idx = int(c[-1])
                if position in c:
                    cols[c] = position + str(d.get(idx, idx))
        return np.sum((df.values - df[[cols.get(c, c) for c in df.columns]].values)**2, axis=1)


    def _featurize(self, session, structure_ids) -> np.array:
        df = self.substituent_featurizer.featurize(session, structure_ids=structure_ids)
        ajr = {}
        for symmetry in self.symmetries:
            ajr[symmetry] = self.calc_symmetry(symmetry, df)
        for name, custom_func in self.custom_funcs.items():
            ajr[name] = custom_func(df)
        if self.add_sum:
            ajr["sum"] = np.sum([ajr[s] for s in self.symmetries], axis=0)
        df = pd.DataFrame(ajr, index=structure_ids)
        return df.values


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
    from src import config
    import os
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///{}".format(os.path.join(config.PROJECT_SRC_DIR, "main.db")))
    session = sessionmaker(bind=engine)()
    all_sids = [x[0] for x in session.execute(text("SELECT id FROM structures")).all()]
    positions = [f"meso{i + 1}" for i in range(4)]
    subsfeat = SubstituentPropertyFeaturizer("hydrogen_mulliken", positions, navalue=None, source="orca_substituent_property/with_h")
    feat = SymmetryAwareFeaturizer(subsfeat, symmetries="macrocycle", custom_funcs={"total_beta": lambda df: df[[c for c in df.columns if "beta" in c]].sum(axis=1), "total_meso": lambda df: df[[c for c in df.columns if "meso" in c]].sum(axis=1)}, add_sum=True)
    df = feat.featurize(session, all_sids)
    print(df.dropna())