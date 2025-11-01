from multiprocessing import Pool
from tqdm import tqdm
import json
import random
import os
from itertools import product
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from openbabel import openbabel as ob
from sqlalchemy.orm import Session
from src.featurizers import SubstituentPropertyFeaturizer, FunctionFeaturizer, SymmetryAwareFeaturizer
from src.sqlmodels import Substituent, StructureSubstituents, SubstituentProperty
from src import utils


MESO_POSITIONS = [f"meso{i + 1}" for i in range(4)]
BETA_POSITIONS = [f"beta{i + 1}" for i in range(8)]
AXIAL_POSITIONS = [f"axial{i + 1}" for i in range(2)]

def substituent_featurizer(positions, property_name: str, navalue, with_h: bool, symmetry: str=None, prefix: str=""):
    source = "orca_substituent_property/with_h" if with_h else "orca_substituent_property/no_h"
    feat = SubstituentPropertyFeaturizer(property_name, positions, navalue=navalue, source=source, prefix=prefix)
    if symmetry is not None:
        feat.prefix = ""
        feat = SymmetryAwareFeaturizer(feat, symmetry, prefix=prefix, add_sum=True)
    return feat

def featurize_metal(session: Session, sid: str):
    """Get the VDW radius of the metal center"""
    # Get the substituent id for the metal position
    metal_sub = session.query(StructureSubstituents.substituent).filter(
        StructureSubstituents.structure == sid,
        StructureSubstituents.position == "metal"
    ).scalar()
    # Get the SMILES string for the metal substituent
    metal_smiles = session.query(Substituent.smiles).filter(Substituent.id == metal_sub).scalar()
    metal = utils.mol_from_smiles(metal_smiles).GetAtom(1)
    metal_charge = session.query(SubstituentProperty.value).filter(SubstituentProperty.structure == sid).filter(SubstituentProperty.property == "charge").first()[0]
    d_population = session.query(SubstituentProperty.value).filter(SubstituentProperty.structure == sid).filter(SubstituentProperty.property == "d_population").first()[0]
    # get the coordination number
    coord = 6 - session.query(StructureSubstituents).filter(StructureSubstituents.structure == sid, StructureSubstituents.position == "axial").count()
    return [coord, metal.GetAtomicNum(), metal_charge, d_population]

def target_featurizer(session: Session, sid: str):
    props_df = pd.read_sql(f"SELECT property,source,value FROM structure_properties WHERE structure='{sid}' AND source LIKE 'parser%'", session.connection())
    props_df = props_df.pivot_table(index=["property", "source"], values="value", aggfunc="first").unstack("source")
    props_df.columns = ["{}".format(col[1].split("/")[-1]) for col in props_df.columns]
    if any([col not in props_df.columns for col in ["S1", "S3", "S5"]]):
        return [None, None, None, None]
    props_df = props_df[["S1", "S3", "S5"]]
    spin_state = props_df.loc["final_energy", :].idxmin()
    energies = props_df.loc["final_energy", :].sort_values(ascending=True)
    spin_shift = (energies.values[1] - energies.values[0]) * 27.2114 # convert to eV from Ha
    return [
        props_df.loc["HOMO-0_energy", spin_state], 
        props_df.loc["LUMO+0_energy", spin_state], 
        props_df.loc["HOMO-LUMO_gap", spin_state],
        spin_shift
    ]

TARGET_FEATURIZER = FunctionFeaturizer(["HOMO", "LUMO", "gap", "spin_shift"], target_featurizer, navalue=None)
BASE_METAL_FEATURIZER = FunctionFeaturizer(["coordination", "z", "charge", "d_population"], featurize_metal, navalue=None)

METAL_FEATURIZERS = {
    "base": BASE_METAL_FEATURIZER
}

AXIAL_FEATURIZERS = {
    pname: substituent_featurizer(AXIAL_POSITIONS, pname, navalue=0, with_h=False, symmetry="axial", prefix="axial_")
    for pname in ["connected_atom_mulliken", "connected_atom_loewdin"]
}
AXIAL_FEATURIZERS["homo_lumo"] = substituent_featurizer(AXIAL_POSITIONS, "HOMO", navalue=0, with_h=False, symmetry="axial", prefix="axial_homo_") +\
    substituent_featurizer(AXIAL_POSITIONS, "LUMO", navalue=0, with_h=False, symmetry="axial", prefix="axial_lumo_")

MACROCYCLE_FEATURIZERS = {
    pname: substituent_featurizer(MESO_POSITIONS, pname, navalue=None, with_h=True, symmetry="meso", prefix="meso_") +\
        substituent_featurizer(BETA_POSITIONS, pname, navalue=None, with_h=True, symmetry="macrocycle", prefix="beta_")
        for pname in ["connected_atom_mulliken", "connected_atom_loewdin", "hydrogen_mulliken", "hydrogen_loewdin"]
}

MACROCYCLE_FEATURIZERS.update({
    "raw_" + pname: substituent_featurizer(MESO_POSITIONS, pname, navalue=None, with_h=True, symmetry=None) +\
        substituent_featurizer(BETA_POSITIONS, pname, navalue=None, with_h=True, symmetry=None)
        for pname in ["connected_atom_mulliken", "connected_atom_loewdin", "hydrogen_mulliken", "hydrogen_loewdin"]
})

def make_data(features: pd.DataFrame, target: pd.DataFrame, test_size: int=30):
    X = features.copy().dropna()
    y = target.copy().dropna()
    joined_index = X.index.intersection(y.index)
    X = X.loc[joined_index]
    y = y.loc[joined_index]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)
    return xtrain, xtest, ytrain, ytest


def run_bootstrap(features: pd.DataFrame, target: pd.DataFrame, test_size: int, bootstrap_id: int, cv: int, save_dir: str):
    # fix the seed to make everything the same
    seed = bootstrap_id + 100
    np.random.seed(seed)
    random.seed(seed)
    # now run
    xtrain, xtest, ytrain, ytest = make_data(features, target, test_size)
    # get sids
    train_sids = xtrain.index
    # now convert to np
    xtrain = xtrain.values
    xtest = xtest.values
    ytrain = ytrain.values.ravel()
    ytest = ytest.values.ravel()
    model = RandomForestRegressor(n_estimators=1000)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_metrics = None
    # run CV
    for train_idx, val_idx in kf.split(xtrain):
        xtr, xval = xtrain[train_idx], xtrain[val_idx]
        ytr, yval = ytrain[train_idx], ytrain[val_idx]
        model.fit(xtr, ytr)
        val_metrics = utils.estimate_regression_fit(model.predict(xval), yval, "val_")
        if cv_metrics is None:
            cv_metrics = {k: [v] for k, v in val_metrics.items()}
        else:
            for k, v in val_metrics.items():
                cv_metrics[k].append(v)

    # Fit the model on the full training set
    model.fit(xtrain, ytrain)
    # Calculate metrics
    metrics = utils.estimate_regression_fit(model.predict(xtrain), ytrain, "train_")
    metrics.update(utils.estimate_regression_fit(model.predict(xtest), ytest, "test_"))
    for k, v in cv_metrics.items():
        metrics["avg_" + k] = np.mean(v)
    # calclating SHAP values
    explainer = shap.TreeExplainer(model)
    shap_df = pd.DataFrame(explainer.shap_values(xtrain), columns=features.columns, index=train_sids)
    # Save metrics as JSON
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    # Save SHAP values as CSV
    shap_path = os.path.join(save_dir, "shap.csv")
    shap_df.to_csv(shap_path)

def _run_bootstrap(args):
    return run_bootstrap(*args)


def main(session, models_dir: str, cv: int, nbootstraps: int, nworkers: int):
    sids = utils.sids_by_type(session)
    print("Building target properties...")
    targets = TARGET_FEATURIZER.featurize(session, sids)
    base_features = {"metal": {}, "macrocycle": {}, "axial": {}}
    for metal, macro, axial in product(METAL_FEATURIZERS, MACROCYCLE_FEATURIZERS, AXIAL_FEATURIZERS):
        print(f"** RUNNING metal={metal}, macro={macro}, axial={axial} **")
        features = pd.DataFrame(index=sids)
        # featurize metal
        if not metal in base_features["metal"]:
            print(f"featurizing metal with {metal}...")
            base_features["metal"][metal] = METAL_FEATURIZERS[metal].featurize(session, sids)
        features = pd.merge(features, base_features["metal"][metal], left_index=True, right_index=True)
        # featurize macrocycle
        if not macro in base_features["macrocycle"]:
            print(f"featurizing macro with {macro}...")
            base_features["macrocycle"][macro] = MACROCYCLE_FEATURIZERS[macro].featurize(session, sids)
        features = pd.merge(features, base_features["macrocycle"][macro], left_index=True, right_index=True)
        # featurize axial
        if not axial in base_features["axial"]:
            print(f"featurizing axial with {axial}...")
            base_features["axial"][axial] = AXIAL_FEATURIZERS[axial].featurize(session, sids)
        features = pd.merge(features, base_features["axial"][axial], left_index=True, right_index=True)
        args = []
        for target in targets.columns:
            # making output directory
            path = os.path.join(models_dir, f"metal={metal}_macro={macro}_axial={axial}_target={target}")
            if not os.path.isdir(path):
                os.mkdir(path)
            # building arguments for fit
            for i in range(nbootstraps):
                bootstrap_path = os.path.join(path, str(i))
                if not os.path.isdir(bootstrap_path):
                    os.mkdir(bootstrap_path)
                args.append([features, targets[target], 30, i, cv, bootstrap_path])
        print(f"TRAINING ON {len(targets.columns)} TARGETS WITH {nbootstraps} BOOTSTRAP EXPERIMENTS ON {nworkers} WORKERS")
        with Pool(processes=nworkers) as pool:
            for _ in tqdm(pool.imap_unordered(_run_bootstrap, args), total=len(args)):
                pass


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src import config
    utils.define_pallet()
    engine = create_engine("sqlite:///{}".format(os.environ["CRYSTAL_MAIN_DB"]))
    session = sessionmaker(bind=engine)()
    main(session, os.path.join(config.PROJECT_SRC_DIR, "models"), cv=5, nworkers=4, nbootstraps=10)
