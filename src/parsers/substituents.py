# script to parse results from data/nonplanarity directory to a dataframe format
import os
from typing import List
import networkx as nx
from openbabel import openbabel as ob
import pandas as pd
import utils
import config
from read_to_sql import Substituent

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
        # add a dummy atom instead of the macrocyle
        subs.add_node(0, Z=0, x=0, y=0, z=0)
        subs.add_edge(0, n, bo=1)
        # convert it back to a molecule, perceive proper bond orders and get its SMILES
        m = utils.graph_to_mol(subs)
        m.PerceiveBondOrders()
        smiles = utils.mol_to_smiles(m)
        # add the smiles and atomic idxs to the output
        ajr.append((smiles, list(subs.nodes.keys())))
        # adding edge back to G
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


def mol_to_entries(mol: ob.OBMol, stype: str, sid: int):
    df, macrocycle_atoms = disect_ring(mol, stype)
    # first we analyze the meso and beta positions
    # go over all rows in df, each row has a substitution
    entries = []
    for row in df.to_dict(orient="records"):
        if row["position"] == "N":
            continue
        if row["position"] == "metal":
            continue
        smiles, idxs = find_substituent(mol, row["target"], macrocycle_atoms)[0]
        entries.append(Substituent(structure=sid, substituent=smiles, position=row["position"], position_index=row["position_idx"], atom_indicis=",".join([str(x) for x in idxs])))
    # now, analyze for the metal idx
    metal = df[df["position"] == "metal"]["target"].values[0]
    # add record for metal atom
    metal_z = mol.GetAtom(int(metal)).GetAtomicNum()
    smiles = "[{}]".format(ob.GetSymbol(metal_z))
    entries.append(Substituent(structure=sid, substituent=smiles, position="metal", atom_indicis=str(int(metal))))
    # now add the axial ligand (if exists)
    for i, (smiles, atoms) in enumerate(find_substituent(mol, metal, macrocycle_atoms)):
        entries.append(Substituent(structure=sid, substituent=smiles, position="axial", position_index=i+1, atom_indicis=",".join([str(x) for x in atoms])))
    return entries

def entries_for_structure(stype: str):
    """Make all entries for a given structure type"""
    moldir = utils.get_directory("curated", stype)
    ajr = []
    for fname in os.listdir(moldir):
        sid = fname.split("_")[0]
        print("analyzing", sid)
        mol = utils.get_molecule(os.path.join(moldir, fname))
        ajr += mol_to_entries(mol, stype, sid)
    return ajr


def main(session, n):
    print("=" * 10, "PARSING SUBSTITUENTS INFORMATION", "=" * 10)
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
