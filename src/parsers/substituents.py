# script to parse results from data/nonplanarity directory to a dataframe format
import os
from typing import List
import networkx as nx
from itertools import chain
from openbabel import openbabel as ob
from openbabel import pybel
from sqlalchemy import delete
import pandas as pd
from dataclasses import dataclass
from src import config, utils
from src.sqlmodels import Substituent, StructureSubstituents
from src.parsers.BaseParser import BaseParser, Session


class OpenbabelBuildError (Exception):
    pass

def guess_geometry(mol: ob.OBMol) -> ob.OBMol:
    builder = ob.OBBuilder()
    if not builder.Build(mol):
        conv = ob.OBConversion()
        raise OpenbabelBuildError("Failed building the smiles: ".format(conv.WriteString(mol)))
    return mol

class OpenbabelFfError (Exception):
    pass

def mm_geometry_optimization(mol: ob.OBMol, force_field: str="UFF", nsteps: int=1000) -> ob.OBMol:
    # optimization
    OBFF = ob.OBForceField.FindForceField(force_field)
    suc = OBFF.Setup(mol)
    if not suc == True:
        raise OpenbabelFfError("Could not set up force field for molecule")
    OBFF.ConjugateGradients(nsteps)
    OBFF.GetCoordinates(mol)
    return mol

def mol_to_xyz(mol: ob.OBMol, path: str, optimize: bool):
    mol.ConnectTheDots()
    mol.AddHydrogens()
    mol.PerceiveBondOrders()
    # obmol = guess_geometry(mol)
    if optimize:
        mol = mm_geometry_optimization(mol)
    conv = ob.OBConversion()
    conv.SetOutFormat("xyz")
    conv.WriteFile(mol, path)


@dataclass
class SubstituentMol:

    mol: ob.OBMol
    uid: int = None

    def smiles(self):
        return utils.mol_to_smiles(self.mol)

    def to_sql(self):
        # find the connected atom and dummy atom
        connected_atom = None
        dummy = None
        for atom in ob.OBMolAtomIter(self.mol):
            if atom.GetAtomicNum() == 0:
                dummy = atom
                bond = list(ob.OBAtomBondIter(atom))[0]
                connected_atom = bond.GetBeginAtomIdx() if bond.GetBeginAtom().GetAtomicNum() != 0 else bond.GetEndAtomIdx()
                # make sure index will be correct after removing dummy atom
                if connected_atom > dummy.GetIdx():
                    connected_atom -= 1
        if connected_atom is None and self.mol.NumAtoms() == 1:
            conv = ob.OBConversion()
            conv.SetOutFormat("smi")
            smiles = conv.WriteString(self.mol).strip()
            # if connected atom is not found it is often the metal atom
            return Substituent(
            id=self.uid,
            smiles=smiles,
            connected_atom=None,
            xyz_no_h=None,
            xyz_with_h=None
        )
        elif connected_atom is None and self.mol.NumAtoms() != 1:
            raise RuntimeError("Could not find dummy atom in molecule!")
        # replace dummy with hydrogen 
        mol = ob.OBMol(self.mol)
        h = mol.GetAtom(dummy.GetIdx())
        h.SetAtomicNum(1)
        h.SetType("H")
        # save the "with hydrogen" 
        xyz_path_with_h = os.path.join(config.DATA_DIR, "substituents", "xyz", f"{self.uid}_with_h.xyz")
        mol_to_xyz(mol, xyz_path_with_h, optimize=True)
        # now remove the dummy atom (hydrogen)
        mol = ob.OBMol(self.mol)
        mol.DeleteAtom(mol.GetAtom(dummy.GetIdx()))
        # save the "no hydrogen xyz"
        xyz_path_no_h = os.path.join(config.DATA_DIR, "substituents", "xyz", f"{self.uid}_no_h.xyz")
        mol_to_xyz(mol, xyz_path_no_h, optimize=False)
        # Generate SMILES string for the molecule (without dummy)
        conv = ob.OBConversion()
        conv.SetOutFormat("smi")
        smiles = conv.WriteString(mol).strip()
        # Create a Substituent object and add it to the database
        return Substituent(
            id=self.uid,
            smiles=smiles,
            connected_atom=connected_atom,
            xyz_no_h=xyz_path_no_h,
            xyz_with_h=xyz_path_with_h
        )


class MolSet:
    def __init__(self):
        self.mol_dict = {}
        self.counter = 0

    def get_uid(self) -> int:
        self.counter += 1
        return self.counter

    def __len__(self) -> int:
        return self.counter

    def _hash_molecule(self, mol: SubstituentMol) -> int:
        """Generate a hash for the molecule using its fingerprint."""
        # print(ob.OBFingerprint.FindFingerprint("FP2").GetFingerprint(mol.mol, ))
        # fp = ob.OBFingerprint.GetFingerprint(mol.mol, "FP2")
        fp = str(pybel.Molecule(mol.mol).calcfp())
        return fp

    def add(self, mol: SubstituentMol):
        """Add a SubstituentMol to the set if it is not already present."""
        mol_hash = self._hash_molecule(mol)
        if mol_hash not in self.mol_dict:
            self.mol_dict[mol_hash] = [mol]
            mol.uid = self.get_uid()
            return mol
        else:
            for existing_mol in self.mol_dict[mol_hash]:
                if same_molecules(existing_mol.mol, mol.mol):
                    return existing_mol
            self.mol_dict[mol_hash].append(mol)
            mol.uid = self.get_uid()
            return mol

    def has(self, mol: SubstituentMol) -> bool:
        """Check if a SubstituentMol is in the set."""
        mol_hash = self._hash_molecule(mol)
        if mol_hash in self.mol_dict:
            for existing_mol in self.mol_dict[mol_hash]:
                if same_molecules(existing_mol.mol, mol.mol):
                    return True
        return False
    
    def items(self):
        return chain(*self.mol_dict.values())


def same_molecules(mol1: ob.OBMol, mol2: ob.OBMol) -> bool:
    """Compare graph of 2 molecules to determine if they are the same"""
    g1 = utils.mol_to_graph(mol1)
    g2 = utils.mol_to_graph(mol2)
    iso = utils.isomorphism.GraphMatcher(g1, g2, node_match=utils.node_matcher)
    return iso.is_isomorphic()

def find_substituent(mol: ob.OBMol, substitution_idx: int, macrocycle_idxs: List[int]) -> List[str]:
    """Method to find a substituent's SMILES at a given substituted carbon index. gives the substitution bond as a dummy atom"""
    # convert the molecules to a graph for the analysis
    G = utils.mol_to_graph(mol)
    # find the neighbors of the substition point
    neighbors = [i for i in G.neighbors(substitution_idx) if not i in macrocycle_idxs]
    ajr = []
    for n in neighbors:
        cG = G.copy()
        # removing the subs bond
        cG.remove_edge(substitution_idx, n)
        # getting substituent by the connected components - its the one with the neighbor atom
        # note that there is only one susbtituent as we broke only one bond
        subs = [G.subgraph(x).copy() for x in nx.connected_components(cG) if n in x][0]
        # add dummy atom to graph
        subs.add_node(0, Z=0, x=subs.nodes[n]["x"] + 1, y=subs.nodes[n]["y"], z=subs.nodes[n]["z"])
        subs.add_edge(0, n, bo=1)
        # convert it back to a molecule
        m = utils.graph_to_mol(subs)
        # add the smiles and atomic idxs to the output
        ajr.append((SubstituentMol(m), list(subs.nodes.keys())))
    return ajr

def find_metal_idx(mol: ob.OBMol, nitrogens: List[int], macrocycle_idxs: List[int]) -> List[str]:
    """Method to find a substituent's SMILES at a given substituted carbon index. gives the substitution bond as a dummy atom"""
    # convert the molecules to a graph for the analysis
    G = utils.mol_to_graph(mol)
    # find the neighbors of the substition point
    for nitrogen in nitrogens:
        neighbors = [i for i in G.neighbors(nitrogen) if not i in macrocycle_idxs]
        if len(neighbors) > 0:
            return neighbors[0]
    

def disect_ring(mol: ob.OBMol, stype: str):
    """Disecting the macrocyle using a graph match to a basic structure description. returns a dataframe with all the position indices and list of macrocycle atoms"""
    subs_points = pd.read_csv(os.path.join(config.DATA_DIR, "definitions", stype + "_positions.csv"), index_col="atom_idx")
    subgraph = utils.get_definition(stype)
    g = utils.mol_to_graph(mol)
    iso = utils.isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    morph = list(iso.subgraph_isomorphisms_iter())[0]
    morph = {v: k for k, v in morph.items()}
    subs_points["target"] = [morph[i] for i in subs_points.index]
    metal = pd.DataFrame([{"position": "metal", "position_idx": None, "target": find_metal_idx(mol, subs_points[subs_points["position"] == "N"]["target"], morph.values())}], index=[-1])
    subs_points = pd.concat([subs_points, metal])
    return subs_points, morph.values()


def mol_to_entries(mol: ob.OBMol, stype: str, sid: int) -> List[StructureSubstituents]:
    df, macrocycle_atoms = disect_ring(mol, stype)
    # first we analyze the meso and beta positions
    # go over all rows in df, each row has a substitution
    entries = []
    for row in df.to_dict(orient="records"):
        if row["position"] == "N":
            continue
        if row["position"] == "metal":
            continue
        subs, idxs = find_substituent(mol, row["target"], macrocycle_atoms)[0]
        entries.append(StructureSubstituents(structure=sid, substituent=subs, position=row["position"], position_index=row["position_idx"], atom_indicis=",".join([str(x) for x in idxs])))
    # now, analyze for the metal idx
    metal_idx = df[df["position"] == "metal"]["target"].values[0]
    # add record for metal atom
    metal = ob.OBMol()
    metal.AddAtom(mol.GetAtom(int(metal_idx)))
    subs = SubstituentMol(metal, None)
    entries.append(StructureSubstituents(structure=sid, substituent=subs, position="metal", atom_indicis=str(int(metal_idx))))
    # now add the axial ligand (if exists)
    for i, (subs, atoms) in enumerate(find_substituent(mol, metal_idx, macrocycle_atoms)):
        entries.append(StructureSubstituents(structure=sid, substituent=subs, position="axial", position_index=i+1, atom_indicis=",".join([str(x) for x in atoms])))
    return entries


class Parser (BaseParser):

    name = "substituents"
    source_prefix = ""

    def parse(self, session: Session, n: int):
        """Parse the data to SQL entries"""
        substituents = MolSet()
        entries = []
        moldir = os.path.join(config.DATA_DIR, "xyz", "crystal")
        fnames = list(os.listdir(moldir))
        print("Reading structure files...")
        for i, fname in enumerate(fnames):
            sid = fname.split("_")[0]
            print("analyzing", sid, f"({i + 1} out of {len(fnames)})")
            mol = utils.get_molecule(os.path.join(moldir, fname))
            for entry in mol_to_entries(mol, "porphyrins", sid):
                # get unique substituent from set
                subs = substituents.add(entry.substituent)
                # update the substituent's ID
                entry.substituent = subs.uid
                # add entry
                entries.append(entry)
        print(f"Done reading structures! found {len(substituents)} unique substituents")
        print("Converting substituents to sql...")
        subs = []
        for s in substituents.items():
            print(f"converting {s.smiles()} (id={s.uid})")
            subs.append(s.to_sql())
        print("Done!")
        return subs + entries
    
    def delete(self, session: Session):
        stmt = delete(Substituent)
        session.execute(stmt)
        stmt = delete(StructureSubstituents)
        session.execute(stmt)

