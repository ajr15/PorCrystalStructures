# script to calculate HOMA scores for aromaticity
import os
import numpy as np
from networkx.algorithms import isomorphism
from openbabel import openbabel as ob
import utils
from read_to_sql import StructureProperty

BOND_ORDER_DATA = {
    "CC": {"R1": 1.467, "R2": 1.349, "c": 0.1702, "ROPT": 1.388},
    "CN": {"R1": 1.465, "R2": 1.269, "c": 0.2828, "ROPT": 1.334}
}

AROMATIC_RING_INFO = {
    "corroles": {
        "pyrrole1": [18, 19, 21, 22, 20],
        "pyrrole2": [9, 11, 10, 8, 7],
        "pyrrole3": [17, 2, 4 ,3, 5],
        "pyrrole4": [14, 13, 15, 1, 16],
        "inner_circuit": [9, 7, 6, 2, 17, 5, 12, 13, 14, 16, 20, 18, 19, 23, 11],
        "outer_circuit": [10, 8, 7, 6, 2, 4, 3, 5, 12, 13, 15, 1, 16, 20, 22, 21, 19, 23, 11],
    },
    "porphyrins": {
        "pyrrole1": [17, 18, 19, 20, 23],
        "pyrrole2": [9, 11, 10, 8, 7],
        "pyrrole3": [16, 2, 4, 3, 5],
        "pyrrole4": [14, 13, 15, 1, 22],
        "inner_circuit": [9, 7, 6, 2, 16, 5, 12, 13, 14, 22, 24, 23, 17, 18, 21, 11],
        "outer_circuit": [10, 8, 7, 6, 2, 4, 3, 5, 12, 13, 15, 1, 22, 24, 23, 20, 19, 18, 21, 11],
    },
}


def standard_bond_length(bond: ob.OBBond):
    a1 = bond.GetBeginAtom().GetAtomicNum()
    a2 = bond.GetEndAtom().GetAtomicNum()
    if a1 == 6 and a2 == 6:
        params = BOND_ORDER_DATA["CC"]
    else:
        params = BOND_ORDER_DATA["CN"]
    key = "R{}".format(int(bond.GetBondOrder()))
    return params[key]

def calc_bond_order(bond: ob.OBBond, use_standard_length: bool=False):
    a1 = bond.GetBeginAtom().GetAtomicNum()
    a2 = bond.GetEndAtom().GetAtomicNum()
    if a1 == 6 and a2 == 6:
        params = BOND_ORDER_DATA["CC"]
    else:
        params = BOND_ORDER_DATA["CN"]
    bl = standard_bond_length(bond) if use_standard_length else bond.GetLength()
    return np.exp((params["R1"] - bl) / params["c"])

def calc_bond_length(bond: ob.OBBond, use_standard_length: bool=False):
    bo = calc_bond_order(bond, use_standard_length)
    # converting to a virtual CC bond length using the Pauling relation paremeters
    params = BOND_ORDER_DATA["CC"]
    return params["R1"] - params["c"] * np.log(bo)

def calc_alpha(mol: ob.OBMol):
    """calcualte the normalization constant (alpha) given a kekulized SMILES structure"""
    bls = list(map(lambda b: calc_bond_length(b, use_standard_length=True), ob.OBMolBondIter(mol)))
    # calcuate sum of square differences
    ss = np.sum((np.array(bls) - BOND_ORDER_DATA["CC"]["ROPT"])**2)
    # return alph - assuring HOMA value for kekulized value is 0
    return len(bls) / ss

def calc_homa_properties(mol: ob.OBMol, alpha: float):
    """return HOMA value of given molecule"""
    bls = np.array(list(map(calc_bond_length, ob.OBMolBondIter(mol))))
    # calculate EN
    ropt = BOND_ORDER_DATA["CC"]["ROPT"]
    mean_bl = np.mean(bls)
    en = alpha * np.sum((ropt - mean_bl)**2)
    # if the optimal length is greater then average, EN sign is flipped
    if ropt > np.mean(bls):
        en = - en
    # calculate GEO
    geo = alpha * np.var(bls)
    # return properties: EN, GEO and HOMA
    return {"en": en, "geo": geo, "homa": 1 - en - geo}

def submol_from_idxs(mol: ob.OBMol, idxs) -> ob.OBMol:
    """Get sub-molecule containing atoms at indices"""
    ajr = ob.OBMol()
    # adding atoms to molecule
    for idx in idxs:
        ajr.AddAtom(mol.GetAtom(idx))
    # adding all bonds
    for k1, i in enumerate(idxs):
        for k2, j in enumerate(idxs[(k1 + 1):]):
            oj_bond = mol.GetBond(i, j)
            if oj_bond:
                bond = ob.OBBond()
                bond.SetBegin(ajr.GetAtom(k1 + 1))
                bond.SetEnd(ajr.GetAtom(k1 + k2 + 2))
                bond.SetBondOrder(oj_bond.GetBondOrder())
                ajr.AddBond(bond)
    return ajr

def get_circuit_mol(mol: ob.OBMol, stype: str, idxs) -> ob.OBMol:
    # first, get a map of struct_idx -> definition_mol_idx
    subgraph = utils.get_definition(stype)
    g = utils.mol_to_graph(mol)
    iso = isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    idx_mapping = list(iso.subgraph_isomorphisms_iter())[0]
    # now invert to definition_mol_idx -> struct_idx
    idx_mapping = {v: k for k, v in idx_mapping.items()}
    # now collect all idxs of circuit
    circuit = [idx_mapping[i] for i in idxs]
    # return mol from circuit idxs
    return submol_from_idxs(mol, circuit)

def mol_entries(sid, path: str, stype: str):
    ajr = []
    mol = utils.get_molecule(path)
    definition_mol = utils.get_molecule(os.path.join("ccdc_data", "definitions", stype + ".mol"))
    for circuit_name, circuit_idxs in AROMATIC_RING_INFO[stype].items():
        alpha = calc_alpha(submol_from_idxs(definition_mol, circuit_idxs))
        circuit_mol = get_circuit_mol(mol, stype, circuit_idxs)
        homa_dict = calc_homa_properties(circuit_mol, alpha)
        for k, v in homa_dict.items():
            ajr.append(StructureProperty(structure=sid, source="homa", property="{} {}".format(circuit_name, k), value=v))
    return ajr

def entries_for_structure(stype: str):
    data_dir = utils.get_directory("curated", stype)
    ajr = []
    for fname in os.listdir(data_dir):
        sid = fname.split("_")[0]
        ajr += mol_entries(sid, os.path.join(data_dir, fname), stype)
    return ajr

def main(session, n):
    print("=" * 10, "CALCULATING HOMA SCORES", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    print("reading corrole details...")
    ajr = entries_for_structure("corroles")
    session.add_all(ajr)
    print("reading porphyrin details...")
    ajr = entries_for_structure("porphyrins")
    session.add_all(ajr)
    session.commit()
    print("ALL DONE")



if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt
    entries = mol_entries(0, "data/curated/porphyrins/4330777_0.xyz", "porphyrins")
    for e in entries:
        if "homa" in e.property:
            print(e.property, e.value)