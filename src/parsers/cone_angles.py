# script to calculate cone angle descriptor for the different ligands
from copy import deepcopy
import numpy as np
from typing import List
from openbabel import openbabel as ob
from read_to_sql import Structure, Substituent, SubstituentProperty
from utils import get_molecule

def molecule_from_id(session, sid: str) -> ob.OBMol:
    """get an OBMol for a given structure ID"""
    xyz = session.query(Structure.xyz).filter(Structure.id == sid).all()[0][0]
    return get_molecule(xyz)

def submol_from_idxs(mol: ob.OBMol, idxs) -> ob.OBMol:
    """Get sub-molecule containing atoms at indices"""
    ajr = ob.OBMol()
    # adding atoms to molecule
    for idx in idxs:
        ajr.AddAtom(mol.GetAtom(idx))
    # adding all bonds
    for a1 in ob.OBMolAtomIter(ajr):
        for a2 in ob.OBMolAtomIter(ajr):
            bond = mol.GetBond(a1, a2)
            if bond:
                ajr.AddBond(bond)
    return ajr

def cone_angle(points: np.ndarray, radii: np.ndarray, origin: np.ndarray, origin_radius: float):
    """Calculate the cone angle of bunch of points in 3D having given radii. cone origin is given as a point """
    # Step 1: Calculate vectors from origin to each point
    vectors_to_points = points - origin
    
    # Step 2: Determine direction vector (average of all vectors)
    direction_vector = np.mean(vectors_to_points, axis=0)
    
    # Step 3: Normalize direction vector
    direction_vector /= np.linalg.norm(direction_vector)

    # calculate the norms of each vector to point, add the origin's radii 

    dists = np.linalg.norm(vectors_to_points, axis=1)
    
    # Step 4: Calculate cosine of angles between direction vector and vectors to points
    cosines = np.abs(np.dot(vectors_to_points, direction_vector) / dists)
    cosines = np.clip(cosines, 0, 1)

    # Step 5: Calculate the angle between the tangent to the sphere around the point, the origin and the point
    sines = radii / dists
    # in case the radii of some atoms is greater then the distance, replace it with adequate relation
    sines = np.where(sines >= 1, (radii - origin_radius) / dists, sines)
    # in case some sines values are negative (like for H atoms), set them to 0
    sines = np.clip(sines, 0, 1)
    # Step 6: calculate the total angle
    angles = np.arcsin(sines) + np.arccos(cosines)
    if any(np.isnan(angles)):
        print(radii)
        print(dists)
        print(sines)
        print(np.arccos(cosines))
        print(np.arcsin(sines))
        raise ValueError("YOU ARE AN IDIOT")
    
    # Convert angles from radians to degrees
    angles = np.degrees(angles)

    # return the max angle    
    return np.max(angles)

def get_ligands(session, sid, position) -> List[ob.OBMol]:
    """Get the ligand molecules as OBMol sorted by position index"""
    # fetch data from database
    ligand_idxs = session.query(Substituent.atom_indicis, Substituent.position_index, Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == position).order_by(Substituent.position_index).all()
    mol = molecule_from_id(session, sid)
    # get ligand molecules
    ajr = []
    for idxs, position_idx, smiles in ligand_idxs:
        idxs = [int(x) for x in idxs.split(",") if not x == "0"]
        # ligand = submol_from_idxs(mol, idxs)
        points = np.array([[mol.GetAtom(idx).GetX(), mol.GetAtom(idx).GetY(), mol.GetAtom(idx).GetZ()] for idx in idxs])
        radii = np.array([ob.GetCovalentRad(mol.GetAtom(idx).GetAtomicNum()) for idx in idxs])
        # get the macrocycle atom (nearest neighbor to ligand molecule)
        origin_point = None
        origin_rad = None
        for i in idxs:
            for a in ob.OBAtomAtomIter(mol.GetAtom(i)):
                if a.GetIdx() not in idxs:
                    atom = mol.GetAtom(a.GetIdx())
                    origin_point = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
                    origin_rad = ob.GetCovalentRad(atom.GetAtomicNum())
                    break
        ajr.append((points, radii, position_idx, origin_point, origin_rad, smiles))
    return ajr

def get_mol_entries(session, sid):
    """Get all entries for a given molecule"""
    positions = ["meso", "beta", "axial"]
    entries = []
    for pos in positions:
        for points, radii, pos_idx, origin, origin_radius, smiles in get_ligands(session, sid, pos):
            # print(sid, pos, pos_idx)
            if origin is None:
                print("NULL ORIGIN ATOM AT", sid)
                return []
            angle = cone_angle(points, radii, origin, origin_radius)
            entry = SubstituentProperty(smiles=smiles, 
                                        property="cone angle", 
                                        value=angle, 
                                        units="degree", 
                                        source="calculated", 
                                        structure=sid, 
                                        position=pos, position_index=pos_idx)
            entries.append(entry)
    
    return entries

def main(session, n):
    problematic_sids = []
    sids = session.query(Structure.id).all()
    for sid in sids:
        sid = sid[0]
        entries = get_mol_entries(session, sid)
        if len(entries) == 0:
            problematic_sids.append(sid)
        session.add_all(entries)
    session.commit()
    print("YOU HAVE", len(problematic_sids), "PROBLEMATIC STRUCTURES:")
    for sid in problematic_sids:
        print(sid)


if __name__ == "__main__":
    points = np.array([[-2.110375, 4.948447, 0.657674], [-3.299124, 4.383838, 1.116851]])
    radii = np.array([1.7, 1.7])
    origin = np.array([-0.914102, 4.048444, 0.59664])
    print(cone_angle(points, radii, origin))