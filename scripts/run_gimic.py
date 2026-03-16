# script to run GIMIC analysis on a molecule
import numpy as np
from itertools import product
import pandas as pd
from networkx.algorithms import isomorphism
from typing import Iterable
from openbabel import openbabel as ob
import os
from src import utils, config, gimic_utils

GIMIC_INPUT_TEMPLATE = """
# GENERATED FROM ORCA

calc=integral
title=""
basis=MOL
#dryrun=on
xdens=XDENS
debug=10
openshell=true
magnet=[$MAGNETIC_FIELD_VEC]

$GRID_BLOCK

Advanced {
    lip_order=5
    spherical=off
    diamag=on
    paramag=on
    GIAO=on
    screening=on
    screening_thrs=1.d-8
}

Essential {
    acid=on
}
"""

def run_bond(gimic_dir: str, mol: ob.OBMol, bond: ob.OBBond, width: float, height: float):
    magnetic_field = gimic_utils.find_magnetic_field(mol)
    try:
        # try to find a rectangle for integration
        grid = gimic_utils.gimic_bond_grid_definition(mol, bond, width, height)
    except ValueError:
        # if fails, return emtpy dict
        return {}
    # make sure that the magnetic field is aligned with the grid ("right handed system") - this is crucial for gimic consistency later on
    v = grid.xvec - grid.origin
    u = grid.yvec - grid.origin
    bond_vector = np.cross(v, u)
    alignment = np.dot(bond_vector, magnetic_field)
    left_handed = alignment > 0
    if left_handed:
        # if left handed, flip the bond atom indices, to ensure GIMIC consistency
        new_bond = ob.OBBond()
        new_bond.SetBegin(bond.GetEndAtom())
        new_bond.SetEnd(bond.GetBeginAtom())
        new_bond.SetBondOrder(bond.GetBondOrder())
        grid = gimic_utils.gimic_bond_grid_definition(mol, new_bond, width, height)
    # now write block normally
    input_text = GIMIC_INPUT_TEMPLATE.replace("$MAGNETIC_FIELD_VEC", ",".join([str(x) for x in magnetic_field]))
    input_text = input_text.replace("$GRID_BLOCK", grid.to_gimic_definition())
    with open("gimic.inp", "w") as f:
        f.write(input_text)
    os.system(f"cd {gimic_dir}; ~/Software/gimic/build/gimic > gimic.out")
    output = gimic_utils.parse_gimic_output("gimic.out")
    return output, left_handed

def find_structure_indices(obmol: ob.OBMol):
    """Find the indices of a substructure within a molecule."""
    subgraph = utils.get_definition("porphyrins")
    g = utils.mol_to_graph(obmol)
    iso = isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    morph = list(iso.subgraph_isomorphisms_iter())
    return morph


def get_metal_idx(mol: ob.OBMol, macrocycle_atoms) -> int:
    for atom_idx in macrocycle_atoms:
        atom = mol.GetAtom(atom_idx)
        if atom.GetAtomicNum() != 7:  # Skip non-nitrogen atoms
            continue
        for nbr in ob.OBAtomAtomIter(atom):
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in macrocycle_atoms and nbr.GetAtomicNum() != 7:
                return nbr_idx

def get_macrocycle_atoms(mol: ob.OBMol):
    """Get the atoms involved in the macrocycle, including the metal center."""
    # get mapper and add metal idx
    mol.ConnectTheDots()
    atom_mapper = find_structure_indices(mol)[0]
    metal_idx = get_metal_idx(mol, atom_mapper.keys())
    if metal_idx:
        atom_mapper[metal_idx] = -1
    return atom_mapper

def iterate_macrocycle_bonds(mol: ob.OBMol) -> Iterable[ob.OBBond]:
    atom_mapper = get_macrocycle_atoms(mol)
    for atom_idx in atom_mapper.keys():
        atom_obj = mol.GetAtom(atom_idx)
        for nbr in ob.OBAtomAtomIter(atom_obj):
            nbr_idx = nbr.GetIdx()
            if nbr_idx in atom_mapper.keys() and atom_idx < nbr_idx:
                bond = mol.GetBond(atom_obj.GetIdx(), nbr.GetIdx())
                begin= atom_mapper[bond.GetBeginAtomIdx()] 
                end = atom_mapper[bond.GetEndAtomIdx()]  
                yield (begin, end), bond

def prepare_gimic_run(sid: str):
    """Prepare GIMIC run for ORCA computation output directory"""
    orca_output_dir = os.path.join(config.DATA_DIR, "nmr", f"{sid}_0_out")
    gimic_dir = os.path.join(orca_output_dir, "gimic")
    xdens_path = os.path.join(gimic_dir, "XDENS")
    mol_path = os.path.join(gimic_dir, "MOL")
    if os.path.exists(xdens_path) and os.path.exists(mol_path):
        print("GIMIC files already exist for this computation! skipping...")
        return gimic_dir
    if not os.path.isdir(gimic_dir):
        os.mkdir(gimic_dir)
    print("=== MAKING GIMIC FILES ===")
    gimic_utils.convert_to_gimic("~/Software/ORCA5/orca_2json", os.path.join(orca_output_dir, f"{sid}_0.inp"), xdens_path, mol_path)
    return gimic_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GIMIC analysis on a molecule.")
    parser.add_argument("sid", type=str, help="structure id of the target molecule")
    args = parser.parse_args()
    sid = args.sid
    print(f"=== RUNNING GIMIC ANALYSIS FOR {sid} ===")
    mol = utils.get_molecule(os.path.join(config.DATA_DIR, "xyz", "dft", f"{sid}_0.xyz"))
    # mol = utils.get_molecule(os.path.join(config.DATA_DIR, "test", "HETDAL_0_out", f"clean.xyz"))
    # gimic_dir = os.path.join(config.DATA_DIR, "test", "HETDAL_0_out", "gimic")
    gimic_dir = prepare_gimic_run(sid)
    os.chdir(gimic_dir)
    data = []
    widths = [1.5 + i * 0.5 for i in range(7)]
    heights = [1.5 + i * 0.5 for i in range(7)]
    print("=== STARTING INTEGRAL COMPUTATIONS ===")
    for height, width in product(heights, widths):
        print(f"Running with height={height} and width={width}")
        for mapped_idxs, bond in iterate_macrocycle_bonds(mol):
            ajr, left_handed = run_bond(gimic_dir, mol, bond, width, height)
            ajr["width"] = width
            ajr["height"] = height
            # if left handed, we need to flip the bond atoms
            if left_handed:
                ajr["bond_start"] = bond.GetEndAtomIdx()
                ajr["bond_end"] = bond.GetBeginAtomIdx()
                ajr["mapped_bond_start"] = mapped_idxs[1]
                ajr["mapped_bond_end"] = mapped_idxs[0]
            else:
                ajr["bond_start"] = bond.GetBeginAtomIdx()
                ajr["bond_end"] = bond.GetEndAtomIdx()
                ajr["mapped_bond_start"] = mapped_idxs[0]
                ajr["mapped_bond_end"] = mapped_idxs[1]
            data.append(ajr)
            pd.DataFrame(data).to_csv("gimic.csv")
    print("ALL DONE!")