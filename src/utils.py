import joblib
import json
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import networkx as nx
from scipy import stats
from networkx.algorithms import isomorphism
import os
from openbabel import openbabel as ob
from typing import List
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
import numpy as np
from read_to_sql import Structure
import config

def get_molecule(path: str) -> ob.OBMol:
    """Method to get a molecule by its name"""
    obmol = ob.OBMol()
    conv = ob.OBConversion()
    conv.ReadFile(obmol, path)
    return obmol

def get_counter_mols(mol_path: str):
    """Method to extract other molecules in the crystal structure. this is done mostly for best esitmation of complex charge"""
    dirpath, name = os.path.split(mol_path)
    dirpath = dirpath.replace("curated", "xyz")
    sid = name.split("_")[0]
    ajr = []
    for fname in os.listdir(dirpath):
        if fname.startswith(sid) and not fname.endswith("_0.xyz"):
            mol = get_molecule(os.path.join(dirpath, fname))
            ajr.append(mol)
    return ajr

def mol_to_smiles(obmol: ob.OBMol) -> str:
    """conert a molecule to SMILES string"""
    conv = ob.OBConversion()
    conv.SetOutFormat("smi")
    s = conv.WriteString(obmol)
    return s.split("\t")[0]

def mol_from_smiles(smiles: str) -> ob.OBMol:
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    mol = ob.OBMol()
    conv.ReadString(mol, smiles)
    return mol

def is_metal(atom: ob.OBAtom) -> bool:
    Z = atom.GetAtomicNum()
    return 10 < Z < 16 or 18 < Z < 35 or 36 < Z < 53 or 54 < Z

def ob_find_pyrrole_indices(mol: ob.OBMol) -> List[List[int]]:
    # matches the corrole ring by SMARTS matching
    # first, we match all the pyrrolic nitrogens (only aromatic nitrogens, rest are kekulized)
    smarts_pattern = ob.OBSmartsPattern()
    # smarts_pattern.Init("[n]")
    smarts_pattern.Init("c1cccn1")
    smarts_pattern.Match(mol)
    # building unique list of indices
    ajr = []
    for x in smarts_pattern.GetMapList():
        s = set(x)
        if s not in ajr:
            ajr.append(s)
    return [list(x) for x in ajr]

def find_pyrrole_indices(mol: ob.OBMol):
    return find_structure_indices(mol, "pyrrole")
    # fix indicis to fit with OB indices
    # return [map(lambda x: x + 1, iso) for iso in isos]
    
def find_pyrrolic_nitrogens(mol: ob.OBMol) -> List[ob.OBAtom]:
    ajr = []
    for atom_indices in find_pyrrole_indices(mol):
        for i in atom_indices:
            atom = mol.GetAtom(i)
            # if atom is nitrogen, add it to the results
            if atom.GetAtomicNum() == 7:
                ajr.append(atom)
    return ajr

def disect_complex(mol: ob.OBMol) -> List[List[ob.OBAtom]]:
    """Method to extract the indices of the beta carbons, meta carbons and central metal"""
    pyrrole_indices = find_pyrrole_indices(mol)
    nitrogens = find_pyrrolic_nitrogens(mol)
    # we analyze the first nearest neighbors - this is the metal and the atoms neighboring the meta and beta carbons
    carbon_neighbors = []
    metal_idxs = []
    for n in nitrogens:
        neighbors = [atom for atom in ob.OBAtomAtomIter(n)]
        metal_idxs += [atom.GetIdx() for atom in neighbors if is_metal(atom)]
        carbon_neighbors += [atom for atom in neighbors if atom.GetAtomicNum() == 6]
    # now we go to the second nearest neighbors of the carbon neighbors - these are the meta and beta carbons
    if len(metal_idxs) == 0:
        return None, [], []
    betas = set()
    metas = set()
    for c in carbon_neighbors:
        neighbors = [atom for atom in ob.OBAtomAtomIter(c)]
        for neighbor in neighbors:
            # beta carbon is the carbon neighbor that is in a pyrrolic ring
            if any([neighbor.GetIdx() in x for x in pyrrole_indices]) and neighbor.GetAtomicNum() == 6 and not neighbor.GetIdx() in [a.GetIdx() for a in carbon_neighbors]:
                betas.add(neighbor.GetIdx())
            # meta carbons are the neighbor outside of the ring
            elif all([neighbor.GetIdx() not in x for x in pyrrole_indices]):
                metas.add(neighbor.GetIdx())
    return metal_idxs[0], list(metas), list(betas)

def read_command_line_arguments(script_description: str="", return_args: bool=True):
    import argparse
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument("structure", type=str, help="structure you want to use (corrole, porphyrin etc.)")
    if return_args:
        return parser.parse_args()
    else:
        return parser

def get_directory(ftype: str, structure: str, create_dir: bool=False):
    """get the proper directory of a filetype (cif, xyz, curated...) and structure"""
    ftype_path = os.path.join(config.DATA_DIR, ftype)
    if os.path.isdir(ftype_path):
        struct_dir = os.path.join(ftype_path, structure) + "/"
        if os.path.isdir(struct_dir):
            return struct_dir
        elif create_dir:
            os.mkdir(struct_dir)
            return struct_dir
        else:
            raise ValueError("Structure not available for this file type. existing structures " + ", ".join(os.listdir(ftype_path)))
    else:
        raise ValueError("File type not available. existing file types " + ", ".join(os.listdir(config.DATA_DIR)))
    
def clean_directory(direactory: str, extension: str):
    """Make sure that a direactory contains only files of shape {cod_id}.{extension} and nothing else"""
    for fname in os.listdir(direactory):
        cid = fname.split(".")[0]
        if "{}.{}".format(cid, extension) != fname:
            os.remove(os.path.join(direactory, fname))

def node_matcher(node1_attr, node2_attr):
    return node1_attr["Z"] == node2_attr["Z"]

def mol_to_graph(obmol: ob.OBMol) -> nx.Graph:
    """Method to convert an openbabel molecule to a networkx graph with single-bonds only. meant as a subroutine for substructure search"""
    g = nx.Graph()
    # adding atoms to graph
    for i, atom in enumerate(ob.OBMolAtomIter(obmol)):
        g.add_node(i + 1, Z=atom.GetAtomicNum(), x=atom.GetX(), y=atom.GetY(), z=atom.GetZ())
    # adding bonds (edges to graph)
    for bond in ob.OBMolBondIter(obmol):
        x = bond.GetBondOrder()
        bo = x if x is not None or x != 0 else 1
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bo=bo)
    return g

def graph_to_mol(g: nx.Graph) -> ob.OBMol:
    obmol = ob.OBMol()
    for i, atom in enumerate(g):
        obatom = ob.OBAtom()
        obatom.SetAtomicNum(g.nodes[atom]["Z"])
        if not g.nodes[atom]["x"] is None:
            coord_vec = ob.vector3(g.nodes[atom]["x"], g.nodes[atom]["y"], g.nodes[atom]["z"])
            obatom.SetVector(coord_vec)
        obmol.InsertAtom(obatom)
        g.nodes[atom]["idx"] = i + 1
    for u, v in g.edges:
        begin_atom = g.nodes[u]["idx"]
        end_atom = g.nodes[v]["idx"]
        obmol.AddBond(begin_atom, end_atom, g.edges[(u, v)]["bo"])
    return obmol

def get_definition(structure: str):
    mol = get_molecule(os.path.join(config.DATA_DIR, "definitions", structure + ".mol"))
    return mol_to_graph(mol)

def find_structure_indices(obmol: ob.OBMol, structure: str):
    subgraph = get_definition(structure)
    g = mol_to_graph(obmol)
    iso = isomorphism.GraphMatcher(g, subgraph, node_match=node_matcher)
    # get non-interceting isomorph counts
    covered_atoms = set()
    isos = []
    for morph in iso.subgraph_isomorphisms_iter():
        atoms = list(morph.keys())
        if all([a not in covered_atoms for a in atoms]):
            covered_atoms = covered_atoms.union(atoms)
            isos.append(atoms)
    return isos


def validate_structure(obmol: ob.OBMol, structure: str, n_isomorphs: int=1) -> bool:
    isos = find_structure_indices(obmol, structure)
    # if there are too many isomorphs, return false
    if len(isos) != n_isomorphs:
        return False
    # else, make sure that the structure is valid (each atom in the macrocycle have exactly 3 neighbors)
    for iso in isos:
        for idx in iso:
            atom = obmol.GetAtom(idx)
            if len(list(ob.OBAtomAtomIter(atom))) != 3:
                return False
    return True

def get_displaced_structures(structure: str):
    """Get all displaced structures for a given structure"""
    path = os.path.join(config.DISPLACED_STRUCTS_DIR, structure)
    if not os.path.isdir(path):
        raise ValueError("{} is not a valid structure name".format(structure))
    ajr = {}
    for s in os.listdir(path):
        f = os.path.join(path, s)
        name = s.split(".")[0].lower()
        struct = get_molecule(f)
        ajr[name] = struct
    return ajr

def sids_by_type(session, stype: str="all"):
    if stype in ["corrole", "porphyrin"]:
        q = session.query(Structure.id).filter(Structure.type == stype)
    elif stype == "all":
        q = session.query(Structure.id)
    else:
        raise ValueError("Unknown structure type ({}). allowed values are 'corrole', 'porphyrin' or 'all'".format(stype))
    ajr = q.distinct().all()
    return [x[0] for x in ajr]



# ML related utils

def split_train_test(X, y, test_size: int, random_seed=1):
    org_state = np.random.get_state()
    # setting seed for uniform results
    np.random.seed(random_seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # returning to original random state (to keep consistancy with other subroutines)
    np.random.set_state(org_state)
    return x_train, x_test, y_train, y_test
    
def bootstrap_idxs(n, n_bootstraps, n_test):
    idxs = range(n)
    res = []
    for _ in range(n_bootstraps):
      train = resample(idxs, replace=True, n_samples=(len(idxs)-n_test))
      test = resample([x for x in idxs if not x in train], replace=True, n_samples=n_test)
      res.append((train, test))
    return res 
    
def bootstrap_data(X, y, n_bootstraps, test_size, random_seed=1):
    org_state = np.random.get_state()
    # setting seed for uniform results
    np.random.seed(random_seed)
    res = []
    bs = bootstrap_idxs(len(y), n_bootstraps, test_size)
    for train_idxs, test_idxs in bs:
        x_train = np.array([X[i] for i in train_idxs])
        x_test = np.array([X[i] for i in test_idxs])
        y_train = np.array([y[i] for i in train_idxs])
        y_test = np.array([y[i] for i in test_idxs])
        res.append((x_train, x_test, y_train, y_test))
    # returning to original random state (to keep consistancy with other subroutines)
    np.random.set_state(org_state)
    return res

def normalize(data: np.array, ref_data: np.ndarray):
    """Method to make a z-score normalization for data vectors.
    ARGS:
        - data (np.array): data on input batches to normalize. normalizes each batch.
        - params (list): list of [mean, std] to use for calculation"""
    m = np.mean(ref_data, axis=0)
    s = np.std(ref_data, axis=0)
    return (data - m) / s

def unnormalize(data: np.ndarray, ref_data: np.ndarray):
    """Method to undo a z-score normalization, given reference population data"""
    m = np.mean(ref_data, axis=0)
    s = np.std(ref_data, axis=0)
    return data * s + m


def _safe_calc_sum_of_binary_func(pred, true, func) -> float:
    """Method to calculate sum of binary function values on two vectors in a memory-safe way"""
    s = 0
    for p, t in zip(pred, true):
        val = func(p, t)
        if not val == [np.inf] and not val == np.inf:
            s = s + val
    return s


def calc_rmse(pred, true) -> float:
    f = lambda p, t: np.square(p - t)
    return np.sqrt(_safe_calc_sum_of_binary_func(pred, true, f) / len(pred))


def calc_mae(pred, true) -> float:
    f = lambda p, t: np.abs(p - t)
    return _safe_calc_sum_of_binary_func(pred, true, f) / len(pred)


def calc_mare(pred, true) -> float:
    f = lambda p, t: np.abs((p - t) / t) if not t == 0 else 0
    return _safe_calc_sum_of_binary_func(pred, true, f)/ len(pred)


def calc_r_squared(pred, true) -> float:
    avg_t = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: t) / len(true)
    avg_p = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: p) / len(pred)
    var_t = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: np.square(t - avg_t)) / len(true)
    var_p = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: np.square(p - avg_p)) / len(pred)
    cov = _safe_calc_sum_of_binary_func(pred, true, lambda p, t: (t - avg_t) * (p - avg_p)) / len(true)
    return cov**2 / (var_p * var_t)


def estimate_regression_fit(pred, true, prefix="") -> dict:
    return {
        prefix + "rmse": calc_rmse(pred, true)[0],
        prefix + "mae": calc_mae(pred, true)[0],
        prefix + "mare": calc_mare(pred, true)[0],
        prefix + "r_squared": calc_r_squared(pred, true)[0]
    }

def estimate_classification_fit(pred, true, prefix="") -> dict:
    d = classification_report(true, pred, output_dict=True)
    return {
        prefix + "accuracy": d["accuracy"],
        prefix + "true_f1": d["True"]["f1-score"],
        prefix + "false_f1": d["False"]["f1-score"],
    }
    

def train_model(task: str, model: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, n_bootstraps: int, test_size: float):
    # task input validation
    if task not in ["regression", "classification"]:
        raise ValueError("Unknown task {}. allowed values are 'regression' and 'classification'.".format(task))
    metric_data = []
    models = []
    # bootstrapping data and training model
    for xtrain, xtest, ytrain, ytest in bootstrap_data(X.to_numpy(), y.to_numpy(), n_bootstraps, test_size):
        # depending on task, fitting and calculating metrics
        # xtrain = normalize(xtrain, xtrain)
        # xtest = normalize(xtest, xtrain)
        if task == "regression":
            # model.fit(xtrain, normalize(ytrain, ytrain))
            model.fit(xtrain, ytrain)
            # calculating metrics
            # metrics = estimate_regression_fit(unnormalize(model.predict(xtrain), ytrain), ytrain, "train_")
            # metrics.update(estimate_regression_fit(unnormalize(model.predict(xtest), ytrain), ytest, "test_"))
            metrics = estimate_regression_fit(model.predict(xtrain), ytrain, "train_")
            metrics.update(estimate_regression_fit(model.predict(xtest), ytest, "test_"))
        else:
            model.fit(xtrain, ytrain)
            # calcualting metrics
            metrics = estimate_classification_fit(model.predict(xtrain), ytrain, "train_")
            metrics.update(estimate_classification_fit(model.predict(xtest), ytest, "test_"))
        # add metric data to list
        models.append(deepcopy(model))
        metric_data.append(metrics)
    return models, pd.DataFrame(metric_data)


def save_model(model: BaseEstimator, model_dir: str):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # saving model using joblib
    joblib.dump(model, os.path.join(model_dir, "model.sav"))
    # saving model parameters to json
    with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
        json.dump(model.get_params(), f)

def load_model(model_dir: str):
    sk_model = joblib.load(os.path.join(model_dir, "model.sav"))
    with open(os.path.join(model_dir, "model_parameters.json"), "r") as f:
        sk_model.set_params(**json.load(f))
    return sk_model

def analyze_bootstrap(df: pd.DataFrame, ci_alpha=0.025):
    """Method to calcualte average value of bootstrap experiments. optionally it adds CI information (in a separate column). if ci_alpha is none, returns a dataframe with average values"""
    avg = df.mean()
    std = df.std()
    ci = stats.t(len(df) - 1).isf(ci_alpha / 2) * std / np.sqrt(len(df))
    avg = avg.to_frame()
    avg.columns = ["avg"]
    avg["ci"] = ci
    return avg

def read_performance_resutls(display_metric: str, display_targets: List[str], display_features: List[str], models_dir: str, ci_alpha: float):
    data = {x: {} for x in display_features}
    for target in display_targets:
        for feat in display_features:
            csv_path = os.path.join(models_dir, "{}_{}".format(feat, target), "metrics.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, index_col=0)
            analyzed = analyze_bootstrap(df, ci_alpha)
            data[feat][target + "_avg"] = analyzed.loc[display_metric, "avg"]
            data[feat][target + "_ci"] = analyzed.loc[display_metric, "ci"]
    return pd.DataFrame(data)


def read_influence_results(models_dir: str, ci_alpha: float, target_features: str, feature_names: List[str]):
    influence_report = pd.DataFrame()
    for md in os.listdir(models_dir):
        if not md.startswith(target_features):
            continue
        target_property = md[(len(target_features) + 1):]
        # read models from directory
        data = []
        for directory in os.listdir(os.path.join(models_dir, md)):
            p = os.path.join(models_dir, md, directory)
            if os.path.isdir(p):
                model = load_model(p)
                data.append(model.feature_importances_)
        df = pd.DataFrame(data, columns=feature_names)
        analyzed = analyze_bootstrap(df, ci_alpha)
        analyzed.columns = ["{}_{}".format(target_property, c) for c in analyzed.columns]
        for c in analyzed.columns:
            influence_report[c] = analyzed[c]
        influence_report.index = analyzed.index
    return influence_report

# plotting utils

def define_pallet():
    # Define custom color palette
    blue_palette = ["#496989", "#58A399", "#A8CD9F", "#E2F4C5"]

    # Register custom colormap
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=blue_palette)

    # Define custom fonts
    plt.rcParams['font.size'] = 14