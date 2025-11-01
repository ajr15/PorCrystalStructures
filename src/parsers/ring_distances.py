import pandas as pd
from sqlalchemy import text
from openbabel import openbabel as ob
import numpy as np
from src import utils
from src.sqlmodels import SubstituentProperty, Structure, Substituent
from src.parsers.BaseParser import StructureParser, Session

CONNECTOR_SMILES = {
    "beta": "*C=C*", 
    "meso": "*c1cccc2cccc(*)c12"
}

DISTANCES = {
    "corrole": [
        ("beta1", "beta2"), 
        ("beta2", "meta1"),
        ("meta1", "beta3"),
        ("beta3", "beta4"), 
        ("beta4", "meta2"),
        ("meta2", "beta5"),
        ("beta5", "beta6"), 
        ("beta6", "meta3"),
        ("meta3", "beta7"),
        ("beta7", "beta8")
    ],
    "porphyrin": [
        ("beta1", "beta2"), 
        ("beta2", "meso2"),
        ("meso2", "beta3"),
        ("beta3", "beta4"), 
        ("beta4", "meso3"),
        ("meso3", "beta5"),
        ("beta5", "beta6"), 
        ("beta6", "meso4"),
        ("meso4", "beta7"),
        ("beta7", "beta8"),
        ("beta8", "meso1"),
        ("meso1", "beta1")
    ],
}


def get_dummy_atom(mol: ob.OBMol):
    for atom in ob.OBMolAtomIter(mol):
        if atom.GetAtomicNum() == 0:
            return atom

def join_molecules(mol1: ob.OBMol, mol2: ob.OBMol) -> ob.OBMol:
    """Join two molecules together on a dummy atom. the function joins the two together, removes one dummy atom from each and creates a bond between the neighbor atoms"""
    # get dummy atoms
    dummy1 = get_dummy_atom(mol1)
    dummy2 = get_dummy_atom(mol2)
    # get neighbors
    atom1 = list(ob.OBAtomAtomIter(dummy1))[0]
    atom2 = list(ob.OBAtomAtomIter(dummy2))[0]
    # remove dummies
    mol1.DeleteAtom(dummy1)
    mol2.DeleteAtom(dummy2)
    nmol1 = mol1.NumAtoms()
    # adding atoms
    for atom in ob.OBMolAtomIter(mol2):
        mol1.AddAtom(atom)
    # adding bonds
    for bond in ob.OBMolBondIter(mol2):
        mol1.AddBond(bond.GetBeginAtomIdx() + nmol1, bond.GetEndAtomIdx() + nmol1, bond.GetBondOrder())
    # add bond between dummies
    mol1.AddBond(atom1.GetIdx(), atom2.GetIdx() + nmol1, 1)
    return mol1


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

def atomic_distance(atom1: ob.OBAtom, atom2: ob.OBAtom, radius="vwd") -> float:
    v1 = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    v2 = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])
    if radius == "vdw":
        r1 = ob.GetVdwRad(atom1.GetAtomicNum())
        r2 = ob.GetVdwRad(atom2.GetAtomicNum())
    elif radius == "covalent":
        r1 = ob.GetCovalentRad(atom1.GetAtomicNum())
        r2 = ob.GetCovalentRad(atom2.GetAtomicNum())
    else:
        r1 = 0
        r2 = 0
    return np.linalg.norm(v1 - v2) - r1 - r2


def measure_distance(smiles1, smiles2, position, force_field: str="UFF", n_steps: int=10000):
    # reading molecules
    connector = utils.mol_from_smiles(CONNECTOR_SMILES[position])
    connector.AddHydrogens()
    nconnector = connector.NumAtoms() - 2
    mol1 = utils.mol_from_smiles(smiles1)
    mol1.AddHydrogens()
    nmol1 = mol1.NumAtoms() - 1
    mol2 = utils.mol_from_smiles(smiles2)
    mol2.AddHydrogens()
    nmol2 = mol2.NumAtoms() - 1
    # joining
    connector = join_molecules(connector, mol1)
    connector = join_molecules(connector, mol2)
    # guessing geometry
    connector = guess_geometry(connector)
    # optimizing
    connector = mm_geometry_optimization(connector, force_field, n_steps)
    # measuring distance
    # going over all atoms of mol1 and mol2 in joined mol
    min_dist = {"vdw": np.inf, "covalent": np.inf, "None": np.inf}
    for i1 in range(nconnector + 1, nconnector + nmol1 + 1):
        atom1 = connector.GetAtom(i1)
        for i2 in range(nconnector + nmol1 + 1, nconnector + nmol1 + nmol2 + 1):
            atom2 = connector.GetAtom(i2)
            for k, v in min_dist.items():
                dist = atomic_distance(atom1, atom2, radius=k)
                if dist < v:
                    min_dist[k] = dist
    return min_dist

def xyz_to_smiles(xyz, connected_atom):
    mol = utils.get_molecule(xyz)
    dummy_atom = mol.NewAtom()
    dummy_atom.SetAtomicNum(0)  # Set atomic number to 0 for a dummy atom
    connected = mol.GetAtom(connected_atom)
    mol.AddBond(connected.GetIdx(), dummy_atom.GetIdx(), 1)  # Add a single bond
    return utils.mol_to_smiles(mol)

def sid_to_entries(session, stype, sid):
    q = "SELECT substituent, position || position_index AS pos FROM " +\
        "structure_substituents WHERE (position=\"beta\" OR position=\"meta\" OR position=\"meso\") AND structure=\"{}\"".format(sid)
    data = session.execute(text(q))
    df = pd.DataFrame(data, columns=["subid", "position"])
    df["xyz"] = [session.query(Substituent.xyz_no_h).filter(Substituent.id == subid).first()[0] for subid in df["subid"]]
    df["connected_atom"] = [session.query(Substituent.connected_atom).filter(Substituent.id == subid).first()[0] for subid in df["subid"]]
    df["smiles"] = [xyz_to_smiles(xyz, atom) for xyz, atom in df[["xyz", "connected_atom"]].values]
    df = df.set_index("position")
    entries = []
    for pos1, pos2 in DISTANCES[stype]:
        # getting substituents smiles
        smi1 = df.loc[pos1, "smiles"]
        subid = df.loc[pos1, "subid"]
        smi2 = df.loc[pos2, "smiles"]
        # choosing meso or beta
        connector = "meso" if any(["meso" in pos1, "meso" in pos2]) else "beta"
        dist_dict = measure_distance(smi1, smi2, connector)
        for k, v in dist_dict.items():
            entry = SubstituentProperty(
                substituent=subid,
                property="{} nn dist".format(k),
                value=v,
                units="A",
                source="",
                structure=sid,
                position=pos1[:-1],
                position_index=int(pos1[-1])
            )
            entries.append(entry)
    return entries

class Parser (StructureParser):

    name = "ring_distances"
    source_prefix = "ring_distances"

    def parse_structure(self, session: Session, sid: str) -> tuple:
        """Parse a single structure (given by structure id), return a tuple of list of sql entries and messages"""
        return sid_to_entries(session, "porphyrin", sid), []



# def entries_for_structure(session, stype: str):
#     sids = session.query(Structure.id).filter(Structure.type == stype).all()
#     res = []
#     for i, sid in enumerate(sids):
#         sid = sid[0]
#         print(i + 1, "out of", len(sids))
#         res += sid_to_entries(session, stype, sid)
#     return res

# def main(session, n):
#     print("=" * 10, "CALCULATING RING DISTANCES", "=" * 10)
#     if n > 1:
#         print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
#     print("reading porphyrin details...")
#     ajr = entries_for_structure(session, "porphyrin")
#     session.add_all(ajr)
#     print("reading corroles details...")
#     ajr = entries_for_structure(session, "corrole")
#     session.add_all(ajr)
#     session.commit()
#     print("ALL DONE")

def test():
    smiles1 = "*[H]"
    # smiles2 = "*c1ccc(CO)cc1"
    smiles2 = "C(=O)(c1ccccc1)Oc1ccc(cc1)*"
    position = "meso"
    force_field = "UFF"
    n_steps = 1000
    # reading molecules
    connector = utils.mol_from_smiles(CONNECTOR_SMILES[position])
    connector.AddHydrogens()
    nconnector = connector.NumAtoms() - 2
    mol1 = utils.mol_from_smiles(smiles1)
    mol1.AddHydrogens()
    nmol1 = mol1.NumAtoms() - 1
    mol2 = utils.mol_from_smiles(smiles2)
    mol2.AddHydrogens()
    nmol2 = mol2.NumAtoms() - 1
    # joining
    connector = join_molecules(connector, mol1)
    connector = join_molecules(connector, mol2)
    # guessing geometry
    connector = guess_geometry(connector)
    # optimizing
    connector = mm_geometry_optimization(connector, force_field, n_steps)
    conv = ob.OBConversion()
    conv.WriteFile(connector, "test.mol")
    # measuring distance
    # going over all atoms of mol1 and mol2 in joined mol
    min_dist = {"vdw": np.inf, "covalent": np.inf, "None": np.inf}
    min_idxs = {"vdw": None, "covalent": None, "None": None}
    for i1 in range(nconnector + 1, nconnector + nmol1 + 1):
        atom1 = connector.GetAtom(i1)
        for i2 in range(nconnector + nmol1 + 1, nconnector + nmol1 + nmol2 + 1):
            atom2 = connector.GetAtom(i2)
            for k, v in min_dist.items():
                dist = atomic_distance(atom1, atom2, radius=k)
                if dist < v:
                    min_dist[k] = dist
                    min_idxs[k] = [i1, i2]
    print(min_dist)
    print(min_idxs)



if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import os
    from src import config
    engine = create_engine("sqlite:///" + os.path.join(config.PROJECT_SRC_DIR, "main.db"))
    Session = sessionmaker(bind=engine)
    session = Session()
    p = Parser()
    p.parse_structure(session, "ADIQAI")
    