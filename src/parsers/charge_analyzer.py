# script to use the axial ligand data and structure type information to infer the metal charge in the complex
# first, analyze the charges of the axial ligands and then analyze metal charges
import numpy as np
from openbabel import openbabel as ob
from sqlalchemy.orm import Session
from src.sqlmodels import SubstituentProperty, Substituent, Structure, StructureSubstituents
from src.parsers.BaseParser import StructureParser
from src import utils

PERIODIC_TABLE_BLOCKS = {
    "S": [1, 3, 11, 19, 37, 55, 87],
    "P": [1000, 5, 13, 31, 49, 81, 113],
    "D": [1000, 1000, 1000, 21, 39, 71, 103],
    "F": [1000, 1000, 1000, 1000, 1000, 57, 89]
}

MAX_POPULATION = {
    "S": 2,
    "P": 6,
    "D": 10,
    "F": 14
}

S1_D_BLOCK = [24, 29, 41, 42, 44, 45, 47, 78, 79, 110, ]
S0_D_BLOCK = [46]

def number_of_available_bonds(atom: ob.OBAtom):
    nbonds = sum([b.GetBondOrder() for b in ob.OBAtomBondIter(atom)])
    # by the number of bonds, we estimate the formal charge of the binding atom
    max_bonds = ob.GetMaxBonds(atom.GetAtomicNum())
    # if atom is nitrogen make sure it has only 3 bonds
    if atom.GetAtomicNum() == 7:
        max_bonds = 3
    # make sure that silicon (Z=14) has only 4 bonds
    if atom.GetAtomicNum() == 14:
        max_bonds = 4
    # the formal charge is the number of bonds minus the max number of bonds 
    return max_bonds - nbonds

def get_neighbors(atom: ob.OBAtom):
    ajr = []
    for bond in ob.OBAtomBondIter(atom):
        if bond.GetBeginAtom() != atom:
            ajr.append(bond.GetBeginAtom())
        else:
            ajr.append(bond.GetEndAtom())
    return ajr

def axial_ligand_charge(mol: ob.OBMol, bounded_atom_idx: int):
    """Get the charge of the axial ligand, assuming the connecting site is a dummy atom"""
    mol.AddHydrogens()
    # LEGACY - finds bounded atom based on dummy
    # # find the dummy atom position
    # dummy = None
    # for atom in ob.OBMolAtomIter(mol):
    #     if atom.GetAtomicNum() == 0:
    #         dummy = atom
    # if dummy is None:
    #     print("AXIAL", utils.mol_to_smiles(mol), "HAS NO DUMMY")
    #     exit()
    # # find the neighboring atom of the atom
    # bounded_atom = get_neighbors(dummy)[0]
    # # removing dummy from molecule
    # mol.DeleteAtom(dummy)
    bounded_atom = mol.GetAtom(bounded_atom_idx)
    # fix for the NO, N2 and O2 molecules - should be neutral
    if mol.NumAtoms() == 2 and all([atom.GetAtomicNum() in [7, 8] for atom in ob.OBMolAtomIter(mol)]):
        return 0
    # getting available bonds of neighboring atoms - this is to ensure corrent prediction even if bond orders are wrong
    available = sum([number_of_available_bonds(n) for n in get_neighbors(bounded_atom)])
    # returning the charge - it is negative the available bonds of the bounded atom minus the available neighbor bonds
    # note that we take the max(0, ...) as not all available bonds might go to the bounded atom
    # fix the number of bonds for Sulfur (Z=16) and Phosphorous (Z=15) - only if they are bounded, make sure their available bonds are contained
    if bounded_atom.GetAtomicNum() == 16:
        nbonds = 2 - sum([b.GetBondOrder() for b in ob.OBAtomBondIter(bounded_atom)])
    elif bounded_atom.GetAtomicNum() == 15:
        nbonds = 3 - sum([b.GetBondOrder() for b in ob.OBAtomBondIter(bounded_atom)])
    elif bounded_atom.GetAtomicNum() == 14:
        nbonds = 4 - sum([b.GetBondOrder() for b in ob.OBAtomBondIter(bounded_atom)])
    # fix for carbene binding, should have 0 charge
    elif bounded_atom.GetAtomicNum() == 6 and (number_of_available_bonds(bounded_atom) - available) == 2:
        return 0        
    else:
        nbonds = number_of_available_bonds(bounded_atom)
    nelec = sum([atom.GetAtomicNum() for atom in ob.OBMolAtomIter(mol)])
    charge = - max(nbonds - available, 0)
    # if (nelec - charge) % 2 != 0:
    #     raise RuntimeError(f"Estimated charge {charge} to molecule {utils.mol_to_smiles(mol)}. but it will have odd number ({nelec - charge}) of electrons!")
    return charge - (nelec - charge) % 2


def get_all_axial_ligands(session):
    return session.query(Substituent).join(StructureSubstituents, Substituent.id == StructureSubstituents.substituent).filter(StructureSubstituents.position == 'axial').distinct().all()

def axial_ligand_analysis(session):
    ligands = get_all_axial_ligands(session)
    entries = []
    for ligand in ligands:
        mol = utils.mol_from_smiles(ligand.smiles)
        charge = axial_ligand_charge(mol, ligand.connected_atom)
        entry = SubstituentProperty(substituent=ligand.id, property="charge", value=charge, source="charge_analyzer") # IMPORTANT: give the full source here, otherwise it will not clear database properly
        entries.append(entry)
    session.add_all(entries)
    session.commit()

def get_macrocycle_charge(session, sid: int):
    stype = session.query(Structure.type).filter(Structure.id == sid).all()[0][0]
    if stype == "corrole":
        return -3
    elif stype == "porphyrin":
        return -2
    
def get_axial_charge(session, sid: int):
    axials = session.query(StructureSubstituents.substituent).filter(StructureSubstituents.structure == sid).filter(StructureSubstituents.position == "axial").all()
    tcharge = 0
    for a in axials:
        a = a[0]
        c = session.query(SubstituentProperty.value).filter(SubstituentProperty.substituent == a).filter(SubstituentProperty.property == "charge").all()[0][0]
        tcharge += c
    return tcharge


def get_metal(session, sid: int):
    subid = session.query(StructureSubstituents.substituent).filter(StructureSubstituents.structure == sid).filter(StructureSubstituents.position == "metal").all()[0][0]
    smiles = session.query(Substituent.smiles).filter(Substituent.id == subid).all()[0][0]
    return smiles, subid


def electron_configuration(z: int):
    """figure out the number of d electrons in an atom (given atomic number). returns the population as an n, (n-2)f, (n-1)d, (n)s, (n)p where n is the valence level"""
    # figure out the row in the periodic table
    row = len(PERIODIC_TABLE_BLOCKS["S"]) - [z >= x for x in reversed(PERIODIC_TABLE_BLOCKS["S"])].index(True) - 1
    # figure out the block in the table
    keys = list(PERIODIC_TABLE_BLOCKS.keys())
    values = [x[row] for x in PERIODIC_TABLE_BLOCKS.values()]
    block_limits = list(sorted(values, reverse=True))
    blocks = list(sorted(keys, key=lambda x: values[keys.index(x)], reverse=True))
    block_idx = [z >= x for x in block_limits].index(True)
    block = blocks[block_idx]
    # building basic cofiguration
    ajr = {"S": 0, "P": 0, "D": 0, "F": 0}
    for b in ajr.keys():
        if block != b and z > PERIODIC_TABLE_BLOCKS[b][row]:
            ajr[b] = MAX_POPULATION[b]
    ajr[block] = z - PERIODIC_TABLE_BLOCKS[block][row] + 1
    # fixing for unique D cases
    if block == "D" and z in S1_D_BLOCK:
        ajr["S"] = 1
        ajr["D"] += 1
    if block == "D" and z in S0_D_BLOCK:
        ajr["S"] = 0
        ajr["D"] += 2
    ajr["n"] = row + 1
    return [row + 1, ajr["F"], ajr["D"], ajr["S"], ajr["P"]]

def ionized_configuration(z, charge):
    base_configuration = electron_configuration(z)
    idx = -1
    counter = 0
    while counter < charge:
        if base_configuration[idx] > 0:
            base_configuration[idx] -= 1
            counter += 1
        else:
            idx = idx - 1
    return base_configuration


def metal_charge_analysis(session):
    sids = session.query(Structure.id).all()
    entries = []
    for sid in sids:
        sid = sid[0]
        base_c = -2
        axial_c = get_axial_charge(session, sid)
        metal_charge = - (base_c + axial_c)
        smiles, subid = get_metal(session, sid)
        z = ob.GetAtomicNum(smiles[1:-1])
        configuration = ionized_configuration(z, metal_charge)
        print(smiles, metal_charge, configuration)
        entries.append(SubstituentProperty(substituent=subid, property="charge", value=metal_charge, source="", structure=sid))
        entries.append(SubstituentProperty(substituent=subid, property="p_population", value=configuration[-1], source="", structure=sid))
        entries.append(SubstituentProperty(substituent=subid, property="s_population", value=configuration[-2], source="", structure=sid))
        entries.append(SubstituentProperty(substituent=subid, property="d_population", value=configuration[-3], source="", structure=sid))
        entries.append(SubstituentProperty(substituent=subid, property="f_population", value=configuration[-4], source="", structure=sid))
        entries.append(SubstituentProperty(substituent=subid, property="valence_level", value=configuration[-5], source="", structure=sid))

    session.add_all(entries)
    session.commit()

def main(session, n):
    print("cleaning database")
    stmt = "DELETE FROM structure_properties WHERE source='charge_analyzer'"
    session.execute(stmt)
    session.commit()
    print("======== ANALYZING AXIAL LIGAND CHARGES ========")
    axial_ligand_analysis(session)
    print("======== ANALYZING METAL CHARGES ========")
    metal_charge_analysis(session)

class Parser (StructureParser):

    name = "charge_analyzer"
    source_prefix = "charge_analyzer"

    def parse_structure(self, session: Session, sid: int):
        base_c = -2
        axial_c = get_axial_charge(session, sid)
        metal_charge = - (base_c + axial_c)
        smiles, subid = get_metal(session, sid)
        z = ob.GetAtomicNum(smiles[1:-1])
        configuration = ionized_configuration(z, metal_charge)
        return [
            SubstituentProperty(substituent=subid, property="charge", value=metal_charge, source="", structure=sid),
            SubstituentProperty(substituent=subid, property="p_population", value=configuration[-1], source="", structure=sid),
            SubstituentProperty(substituent=subid, property="s_population", value=configuration[-2], source="", structure=sid),
            SubstituentProperty(substituent=subid, property="d_population", value=configuration[-3], source="", structure=sid),
            SubstituentProperty(substituent=subid, property="f_population", value=configuration[-4], source="", structure=sid),
            SubstituentProperty(substituent=subid, property="valence_level", value=configuration[-5], source="", structure=sid)
        ], []

    def parse(self, session: Session, n: int):
        # add the axial ligand analysis before the structure-based analysis
        print("analyzing ligand charges...")
        axial_ligand_analysis(session)
        # normally run structure analysis
        print("analyzing metal charges...")
        return super().parse(session, n)



if __name__ == "__main__":
    print(electron_configuration(29))