from collections import defaultdict
import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
import numpy as np
import openbabel as ob
from read_to_sql import StructureProperty, Structure
import config


def get_position(atom: ob.OBAtom):
    """Extract XYZ coordinates from an OBAtom"""
    return np.array([atom.GetX(), atom.GetY(), atom.GetZ()])

def compute_bond_length(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_bond_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def compute_dihedral(p1, p2, p3, p4):
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def get_neighbors_dict(obmol: ob.OBMol):
    neighbors = defaultdict(set)
    for bond in ob.OBMolBondIter(obmol):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors

def calculate_bond_lengths(sid, obmol: ob.OBMol, source) -> List[StructureProperty]:
    entries = []
    atom_positions = {atom.GetIdx(): get_position(atom) for atom in ob.OBMolAtomIter(obmol)}
    for bond in ob.OBMolBondIter(obmol):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        p1 = atom_positions[i]
        p2 = atom_positions[j]
        entries.append(StructureProperty(structure=sid, property=f"bond({i}, {j})", value=compute_bond_length(p1, p2), units="A", source=source))
    return entries

def calculate_bond_angles(sid, obmol: ob.OBMol, source) -> List[StructureProperty]:
    atom_positions = {atom.GetIdx(): get_position(atom) for atom in ob.OBMolAtomIter(obmol)}
    neighbors = get_neighbors_dict(obmol)
    entries = []
    # Bond angles: i–j–k
    for j in neighbors:
        for i in neighbors[j]:
            for k in neighbors[j]:
                if i < k:
                    a = compute_bond_angle(atom_positions[i], atom_positions[j], atom_positions[k])
                    entries.append(StructureProperty(structure=sid, property=f"angle({i}, {j}, {k})", value=a, units="deg", source=source))
    return entries

def calculate_dihedral_angles(sid, obmol: ob.OBMol, source) -> List[StructureProperty]:
    atom_positions = {atom.GetIdx(): get_position(atom) for atom in ob.OBMolAtomIter(obmol)}
    neighbors = get_neighbors_dict(obmol)
    entries = []
    # Dihedral angles: i–j–k–l
    for j in neighbors:
        for k in neighbors[j]:
            for i in neighbors[j] - {k}:
                for l in neighbors[k] - {j}:
                    dih = compute_dihedral(
                        atom_positions[i],
                        atom_positions[j],
                        atom_positions[k],
                        atom_positions[l]
                    )
                    entries.append(StructureProperty(structure=sid, property=f"dihedral({i}, {j}, {k}, {l})", value=dih, units="deg", source=source))
    return entries

def read_xyz_file(path: str):
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.ReadFile(mol, path)
    return mol

def analyze_file(path: str, sid: str, source: str) -> List[StructureProperty]:
    obmol = read_xyz_file(path)
    entries = calculate_bond_lengths(sid, obmol, source)
    entries += calculate_bond_angles(sid, obmol, source)
    entries += calculate_dihedral_angles(sid, obmol, source)
    return entries


def main(session: Session, n):
    print("=" * 10, "READING STRUCTURE ORCA CALCULATION RESULTS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    # read only structures with orca_out property (successful calculation)
    structs = session.query(Structure).all()
    entries = []
    for struct in structs:
        print("analyzing", struct.id)
        # analyzing given crystal structure
        xyz = os.path.join(config.DATA_DIR, "xyz", struct.id + "_0.xyz")
        if not os.path.isfile(xyz):
            continue
        entries += analyze_file(xyz, struct.id, source="crystal")
        if struct.orca_xyz is not None:
            entries += analyze_file(struct.orca_xyz, struct.id, source="orca")
    print("Writing to DB...")
    session.add_all(entries)
    session.commit()
    print("ALL DONE")
