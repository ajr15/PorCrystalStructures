# parser to read both NMR and NICS values from ORCA output
from openbabel import openbabel as ob
import pandas as pd
import os
import json
from typing import List
import numpy as np
from copy import copy
from src.orca_utils import Block, read_file_to_blocks, FinishedNormally
from src.sqlmodels import StructureProperty
from src.parsers.BaseParser import StructureParser, Session
from src import config, utils

class InputMolecule (Block):

    def block_start(self, line):
        return "INPUT FILE" in line
    
    def block_ends(self, line):
        return "END OF INPUT" in line
    
    def molecule(self) -> ob.OBMol:
        lines = self._content[3:-1]
        coords_idx = lines.index([l for l in lines if "* xyz" in l][0]) + 1
        lines = [l.split(">")[-1].split()[:4] for l in lines[coords_idx:]]
        mol = ob.OBMol()
        for line in lines:
            symbol, x, y, z = line
            atom_num = 0 if symbol.endswith(":") else ob.GetAtomicNum(symbol)
            atom = mol.NewAtom()
            atom.SetAtomicNum(atom_num)
            atom.SetVector(round(float(x), 4), round(float(y), 4), round(float(z), 4))
        return mol
    
class TotalShieldingTensors (Block):

    def block_start(self, line):
        return "CHEMICAL SHIFTS" in line
    
    def block_ends(self, line):
        return "CHEMICAL SHIELDING SUMMARY (ppm)" in line
    
    def tensors(self):
        lines = self._content[:-4]
        tensors = []
        tensor_block = False
        ajr = []
        for line in lines:
            if tensor_block and len(line) < 5:
                tensors.append(ajr)
                ajr = []
                tensor_block = False
            if tensor_block:
                ajr.append([float(x) for x in line.strip().split()])
            if "Total shielding tensor" in line:
                tensor_block = True
        return tensors
    
class SummaryShieldingValues (Block):

    def block_start(self, line):
        return "SHIELDING SUMMARY" in line
    
    def block_ends(self, line):
        return "Maximum" in line
    
    def to_dataframe(self):
        lines = self._content[6:-4]
        ajr = [line.split() for line in lines]
        df = pd.DataFrame(ajr, columns=["idx", "symbol", "isotropic", "anisotropic"])
        df["idx"] = df["idx"].astype(int)
        df["isotropic"] = df["isotropic"].astype(float)
        df["anisotropic"] = df["anisotropic"].astype(float)
        return df.set_index("idx")


def normal_and_center(atoms: List[ob.OBAtom]):
    """Calculate the best fitting plane and centerpoint for a list of atoms"""
    points = np.array([[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in atoms])
    centroid = points.mean(axis=0)

    # Perform Singular Value Decomposition (SVD) to find the normal vector
    _, _, vh = np.linalg.svd(points - centroid)

    # The normal vector is the last row of vh (corresponding to the smallest singular value)
    normal = vh[-1]

    return normal / np.linalg.norm(normal), centroid

def get_pyrrole_planes(mol: ob.OBMol):
    """Disecting the macrocyle using a graph match to a basic structure description. returns a dataframe with all the position indices and list of macrocycle atoms"""
    with open(os.path.join(config.DATA_DIR, "definitions", "pyrroles.json"), "r") as f:
        pyrrole_idxs = json.load(f)
    coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(mol)]
    mol.ConnectTheDots()
    subgraph = utils.get_definition("porphyrins")
    g = utils.mol_to_graph(mol)
    # reset atomic coordinates - apparently there is some bug around it
    for atom, c in zip(ob.OBMolAtomIter(mol), coords):
        atom.SetVector(*c)
    # find pyrroles using graph isomorphism
    iso = utils.isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    morph = list(iso.subgraph_isomorphisms_iter())[0]
    morph = {v: k for k, v in morph.items()}
    # read as normal vector and center point
    return {pyrrole: normal_and_center([mol.GetAtom(morph[i]) for i in idxs]) for pyrrole, idxs in pyrrole_idxs.items()}

def assign_atom_labels(mol: ob.OBMol, pyrrole_details: dict):
    """
    Assigns a pyrrole to each dummy atom in the molecule by determining whether it is on a line
    normal to the pyrrole plane passing through its origin.
    """
    labels = {}
    for atomidx, atom in enumerate(ob.OBMolAtomIter(mol)):
        if atom.GetAtomicNum() != 0:
            continue
        vec = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        for pyrrole, (normal, center) in pyrrole_details.items():
            # Check if the dummy atom lies on the normal line passing through the origin
            dist = np.linalg.norm(vec - center)
            cosangle = np.dot(vec - center, normal) / dist
            if (1 - cosangle ** 2 < 1e-3) or dist < 1e-3: 
                z = dist * cosangle / abs(cosangle)
                labels[atomidx] = f"pyrrole{pyrrole}/z{z:.3f}"
                break
    return labels

def calculate_nics_zz(tensors: np.ndarray, atom_labels: dict, pyrrole_details: dict):
    ajr = {}
    for atomidx, label in atom_labels.items():
        tensor = tensors[atomidx]
        pyrrole_idx = label.split("/")[0][7:]
        normal, center = pyrrole_details[pyrrole_idx]
        # Convert the tensor to the pyrrole's normal coordinates
        ajr[atomidx] = np.dot(normal, np.dot(tensor, normal))
    return ajr

class Parser (StructureParser):

    name = "orca_nmr"
    source_prefix = "orca_nmr/"

    # def fetch_structure_ids(self, session):
    #     return ["HETDAL"]


    def parse_structure(self, session, sid):
        outfile = os.path.join(config.DATA_DIR, "old_nmr", sid + "_0_out", sid + "_0.out")
        if not os.path.exists(outfile):
            return [], ["INFO: No NMR calculation for " + sid]
        finished_normally = read_file_to_blocks(outfile, [FinishedNormally("")])[0]
        if not finished_normally.value():
            return [], ["INFO: Bad NMR calculation for " + sid]
        # read molecule, tensors and shielding df
        mol, tensors, shielding_df = read_file_to_blocks(outfile, [InputMolecule("input"), TotalShieldingTensors("tensor"), SummaryShieldingValues("summary")])
        mol = mol.molecule()
        tensors = tensors.tensors()
        shielding_df = shielding_df.to_dataframe()
        # calculate properties
        pyrrole_details = get_pyrrole_planes(mol)
        atom_labels = assign_atom_labels(mol, pyrrole_details)
        nics_zz = calculate_nics_zz(tensors, atom_labels, pyrrole_details)
        # now start making entries for atoms
        entries = []
        for atomindex, atom in enumerate(ob.OBMolAtomIter(mol)):
            # if the atom is a dummy atom, it gets slightly different output 
            if atomindex in atom_labels:
                label = atom_labels[atomindex]
                entries.append(StructureProperty(structure=sid, property=label + "/zz", value=nics_zz[atomindex], units="ppm", source="nics"))
            else:
                symbol = ob.GetSymbol(atom.GetAtomicNum())
                label = f"{symbol}{atomindex}"
            entries.append(StructureProperty(structure=sid, property=label + "/isotropic", value=shielding_df.loc[atomindex, "isotropic"], units="ppm", source="shielding"))
            entries.append(StructureProperty(structure=sid, property=label + "/anisotropic", value=shielding_df.loc[atomindex, "anisotropic"], units="ppm", source="shielding"))
        return entries, [] if len(atom_labels) == 17 * 4 else ["WARNING: Didn't find all probes for " + sid]

if __name__ == "__main__":
    outfile = os.path.join(config.PROJECT_SRC_DIR, "test.out")
    blocks = [InputMolecule("inp")]
    blocks = read_file_to_blocks(outfile, blocks)
    mol = blocks[0].molecule()
    pyrrole_details = get_pyrrole_planes(mol)
    labels = assign_atom_labels(mol, pyrrole_details)
    xyz_file = os.path.join(config.PROJECT_SRC_DIR, "test.xyz")
    conv = ob.OBConversion()
    conv.SetOutFormat("xyz")
    conv.WriteFile(mol, xyz_file)