# script to read the structure details of each structure
import os
from typing import List
from read_to_sql import Structure
import utils

def read_structures(stype: str) -> List[Structure]:
    """read all structures from a given type"""
    struct_dir = utils.get_directory("curated", stype)
    cif_dir = utils.get_directory("cif", stype)
    ajr = []
    for fname in os.listdir(struct_dir):
        xyz = os.path.join(struct_dir, fname)
        sid = fname.split("_")[0]
        cif = os.path.join(cif_dir, "{}.cif".format(sid))
        mol = utils.get_molecule(xyz)
        smiles = utils.mol_to_smiles(mol)
        ajr.append(Structure(id=sid, type=stype[:-1], xyz=xyz, cif=cif, smiles=smiles))
    return ajr

def main(session, n: int):
    print("=" * 10, "READING STRUCTURE DETAILS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    print("reading corrole details...")
    ajr = read_structures("corroles")
    session.add_all(ajr)
    print("reading porphyrin details...")
    ajr = read_structures("porphyrins")
    session.add_all(ajr)
    session.commit()
    print("ALL DONE")
