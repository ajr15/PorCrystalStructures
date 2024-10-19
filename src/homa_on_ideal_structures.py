# test homa scores on ideal displaced structures
import json
import os
from openbabel import openbabel as ob
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from parsers.homa_scores import AROMATIC_RING_INFO, calc_alpha, get_circuit_mol, submol_from_idxs, calc_homa_properties
import utils


def get_coords(mol: ob.OBMol, atom_num=None):
    if atom_num is None:
        return np.array([[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(mol)])
    else:
        return np.array([[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(mol) if a.GetAtomicNum() == atom_num])


def set_coords(mol: ob.OBMol, coords: np.ndarray):
    for i, atom in enumerate(ob.OBMolAtomIter(mol)):
        atom.SetVector(coords[i, 0], coords[i, 1], coords[i, 2])
    return mol

def get_mode_mol(mode: str, stype: str):
    return utils.get_molecule(os.environ("CRYSTAL_SRC_DIR") + f"/displaced_structures/{stype}/{mode}.mol2")


def make_planar_structure(stype: str):
    mol = get_mode_mol("Propellering", stype)
    coords = get_coords(mol)
    pca = PCA(n_components=2)
    planar = pca.fit_transform(coords)
    planar = np.hstack((planar, np.zeros(len(planar)).reshape(-1, 1)))
    mol = set_coords(mol, planar)
    conv = ob.OBConversion()
    conv.WriteFile(mol, os.environ("CRYSTAL_SRC_DIR") + f"/displaced_structures/{stype}/plannar.xyz")

def get_planar_mol(stype: str):
    return utils.get_molecule(os.environ("CRYSTAL_SRC_DIR") + f"/displaced_structures/{stype}/plannar.xyz")

def tota_out_of_plane(plane_coords, mol_coords: np.ndarray):
    # now fitting a plane by solving the equation Ax = b (x are the plane parameters)
    A = np.column_stack((plane_coords[:, 0], plane_coords[:, 1], np.ones(len(plane_coords))))
    x, _, _, _ = np.linalg.lstsq(A, plane_coords[:, 2], rcond=None)
    # the plane equation will be ax + by + c = z
    a, b, c = x
    normal = np.array([a, b, -1])
    # Calculate the signed distance of each point from the plane
    distances = (np.dot(mol_coords, normal) + c) / np.linalg.norm(normal)
    # Take the absolute value of distances
    abs_distances = np.abs(distances)
    # Calculate the mean absolute deviation
    mean_absolute_dev = np.mean(abs_distances)
    return mean_absolute_dev

# def displaced_mol(stype, mode, scaling_factor) -> ob.OBMol:
#     modmol = get_mode_mol(mode, stype)
#     planar = get_planar_mol(stype)
#     planar_coords = get_coords(planar)
#     mod_coords = get_coords(modmol)
#     transitioned = planar_coords + scaling_factor * (mod_coords - planar_coords)
#     return set_coords(modmol, transitioned)

def displacements_from_plane(plane_coords, mol_coords: np.ndarray):
    # now fitting a plane by solving the equation Ax = b (x are the plane parameters)
    A = np.column_stack((plane_coords[:, 0], plane_coords[:, 1], np.ones(len(plane_coords))))
    x, _, _, _ = np.linalg.lstsq(A, plane_coords[:, 2], rcond=None)
    # the plane equation will be ax + by + c = z
    a, b, c = x
    normal = np.array([a, b, -1])
    # Calculate the signed distance of each point from the plane
    distances = (np.dot(mol_coords, normal) + c) / np.linalg.norm(normal)
    # mutliplies distance by normal vector to get displacements vectors
    return distances.reshape(-1, 1) * normal

def get_bond_lengths(coords, mol):
    ajr = []
    for bond in ob.OBMolBondIter(mol):
        c1 = coords[bond.GetBeginAtomIdx() - 1]
        c2 = coords[bond.GetEndAtomIdx() - 1]
        ajr.append(np.linalg.norm(c1 - c2))
    return ajr


def displaced_mol(stype, mode, scaling_factor) -> ob.OBMol:
    modmol = get_mode_mol(mode, stype)
    mod_coords = get_coords(modmol)
    n_coords = get_coords(modmol, atom_num=7)
    displacements = displacements_from_plane(n_coords, mod_coords)
    planar = mod_coords - displacements
    plannar_bl = np.mean(get_bond_lengths(planar, modmol))
    transitioned = planar + scaling_factor * displacements
    transitioned = transitioned - np.mean(transitioned, axis=0)
    transitioned_bl = np.mean(get_bond_lengths(transitioned, modmol))
    transitioned = plannar_bl / transitioned_bl * transitioned
    # scale down coordinates to make sure max distance in ring is
    return set_coords(modmol, transitioned)

def calc_homa_dict(mol: ob.OBMol, stype: str):
    ajr = {}
    definition_mol = utils.get_definition(stype)
    for circuit_name, circuit_idxs in AROMATIC_RING_INFO[stype].items():
        alpha = calc_alpha(submol_from_idxs(definition_mol, circuit_idxs))
        circuit_mol = get_circuit_mol(mol, stype, circuit_idxs)
        homa_dict = calc_homa_properties(circuit_mol, alpha)
        for k, v in homa_dict.items():
            ajr["{} {}".format(circuit_name, k)] = v
    ajr["pyrrole homa"] = np.mean([ajr["pyrrole{} homa".format(i)] for i in range(1, 5)])
    return ajr

def porphystruct_out_of_plane(mol: ob.OBMol):
    path = "tmp.mol2"
    conv = ob.OBConversion()
    conv.WriteFile(mol, path)
    os.system("$CRYSTAL_SRC_DIR/porphystruct/PorphyStruct.CLI analyze -x test/tmp.mol2 > /dev/null")
    json_file = path[:-5] + "_analysis.json"
    if not os.path.isfile(json_file):
        return None
    toop = None
    with open(json_file, "r") as f:
        toop = json.load(f)["OutOfPlaneParameter"]["Value"]
    os.remove(json_file)
    os.remove(path[:-5] + "_analysis.md")
    os.remove(path)
    return toop

def homa_vs_mod(stype, modes, scaling_factors, key, mol_dir=None):
    for mode in modes:
        dists = []
        vals = []
        porphyvals = []
        for s in scaling_factors:
            print(mode, s)
            mol = displaced_mol(stype, mode, s)
            if mol_dir is not None:
                path = "{}/{}_{:.2f}.mol2".format(mol_dir, mode, s)
                conv = ob.OBConversion()
                conv.WriteFile(mol, path)
            vals.append(calc_homa_dict(mol, stype)[key])
            dists.append(tota_out_of_plane(get_coords(mol, atom_num=7), get_coords(mol)))
            x = porphystruct_out_of_plane(mol)
            if x is not None:
                porphyvals.append(x)
        plt.plot(porphyvals, vals, "-o", label=mode)
    plt.xlabel("total out of plane")
    plt.ylabel(key)
    plt.legend()

if __name__ == "__main__":
    utils.define_pallet()
    stype = "porphyrins"
    modes = ["Saddling", "Ruffling"]
    homa_vs_mod(stype, modes, np.linspace(0, 3, 15), "pyrrole homa", None)
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    plt.ylabel("HOMA")
    plt.title("Pyrrole HOMA")
    plt.figure()
    homa_vs_mod(stype, modes, np.linspace(0, 3, 15), "inner_circuit homa", None)
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    plt.ylabel("HOMA")
    plt.title("Inner Cirvuit HOMA")
    plt.show()