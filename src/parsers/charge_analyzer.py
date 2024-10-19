# script to use the axial ligand data and structure type information to infer the metal charge in the complex
# first, analyze the charges of the axial ligands and then analyze metal charges
from openbabel import openbabel as ob
import utils
from read_to_sql import SubstituentProperty, Substituent, Structure

def number_of_available_bonds(atom: ob.OBAtom):
    nbonds = sum([b.GetBondOrder() for b in ob.OBAtomBondIter(atom)])
    # by the number of bonds, we estimate the formal charge of the binding atom
    max_bonds = ob.GetMaxBonds(atom.GetAtomicNum())
    # if atom is nitrogen make sure it has only 3 bonds
    if atom.GetAtomicNum() == 7:
        max_bonds = 3
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

def axial_ligand_charge(mol: ob.OBMol):
    """Get the charge of the axial ligand, assuming the connecting site is a dummy atom"""
    mol.AddHydrogens()
    # find the dummy atom position
    dummy = None
    for atom in ob.OBMolAtomIter(mol):
        if atom.GetAtomicNum() == 0:
            dummy = atom
    if dummy is None:
        print("AXIAL", utils.mol_to_smiles(mol), "HAS NO DUMMY")
        exit()
    # find the neighboring atom of the atom
    bounded_atom = get_neighbors(dummy)[0]
    # removing dummy from molecule
    mol.DeleteAtom(dummy)
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
    return - max(nbonds - available, 0)


def get_all_axial_ligands(session):
    return session.query(Substituent.substituent).filter(Substituent.position == "axial").distinct().all()

def axial_ligand_analysis(session):
    ligands = get_all_axial_ligands(session)
    entries = []
    for ligand in ligands:
        ligand = ligand[0]
        mol = utils.mol_from_smiles(ligand)
        charge = axial_ligand_charge(mol)
        entry = SubstituentProperty(smiles=ligand, property="charge", value=charge, source="openbabel")
        print(ligand, charge)
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
    axials = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "axial").all()
    tcharge = 0
    for a in axials:
        a = a[0]
        c = session.query(SubstituentProperty.value).filter(SubstituentProperty.smiles == a).filter(SubstituentProperty.property == "charge").all()[0][0]
        tcharge += c
    return tcharge

def get_metal(session, sid: int):
    return session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]

def metal_charge_analysis(session):
    sids = session.query(Structure.id).all()
    entries = []
    for sid in sids:
        sid = sid[0]
        base_c = get_macrocycle_charge(session, sid)
        axial_c = get_axial_charge(session, sid)
        metal_charge = - (base_c + axial_c)
        smiles = get_metal(session, sid)
        print(smiles, metal_charge)
        entries.append(SubstituentProperty(smiles=smiles, property="charge", value=metal_charge, source="openbabel", structure=sid))
    session.add_all(entries)
    session.commit()

def main(session, n):
    print("======== ANALYZING AXIAL LIGAND CHARGES ========")
    axial_ligand_analysis(session)
    print("======== ANALYZING METAL CHARGES ========")
    metal_charge_analysis(session)


if __name__ == "__main__":
    mol = utils.mol_from_smiles("C1=CN([CH]N1*)C")
    charge = axial_ligand_charge(mol)
    print(charge)
