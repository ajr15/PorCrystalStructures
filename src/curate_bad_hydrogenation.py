from shutil import copyfile
import os
from openbabel import openbabel as ob
import utils


def is_valid(mol: ob.OBMol) -> bool:
    """validate the structure of a molecule"""
    mol.AddHydrogens()
    z_in_bond = lambda b, z: b.GetBeginAtom().GetAtomicNum() == z or b.GetEndAtom().GetAtomicNum() == z
    for atom in ob.OBMolAtomIter(mol):
        z = atom.GetAtomicNum()
        bonds = ob.OBAtomBondIter(atom)
        # if atom is bound to a metal skip it - its binding is too hard to determine
        bound_to_metal = any([utils.is_metal(b.GetBeginAtom()) or utils.is_metal(b.GetEndAtom()) for b in bonds])
        if bound_to_metal:
            continue
        # starting to analyze carbons
        if z == 6:
            # if the carbon already has hydrogens attached to it, we assume it is well hydrogenated.
            # this is based on the fact that if a crystal structure has any hydrogens it should have the correct amount
            has_hydrogens = any([z_in_bond(b, 1) for b in ob.OBAtomBondIter(atom)])
            if has_hydrogens:
                continue
            # check if the neighboring atoms have hydrogens on them - if yes, chances are that there is a proper hydrogenation
            # also check if it has some neighbor hetero-atom connected to a metal - this can lead to bizzare bonding
            has_neighboring_hydrogens = False
            has_neighboring_hetero = False
            for a in ob.OBAtomAtomIter(atom):
                if any([z_in_bond(b, 1) for b in ob.OBAtomBondIter(a)]):
                    has_neighboring_hydrogens = True
                if a.GetAtomicNum() in [7, 8] and any([utils.is_metal(b.GetBeginAtom()) or utils.is_metal(b.GetEndAtom()) for b in ob.OBAtomBondIter(a)]):
                    has_neighboring_hetero = True
            if has_neighboring_hydrogens or has_neighboring_hetero:
                continue
            # if no hydrogens are present, count the number of bonds of the atom and compare it to its valence
            nbonds = sum([b.GetBondOrder() for b in ob.OBAtomBondIter(atom)])
            valence = ob.GetMaxBonds(z)
            if nbonds != valence:
                print("ATOM", atom.GetId(), "HAS BAD VALENCE")
                return False
    return True


if __name__ == "__main__":
    xyz_dir = "../data/xyz"
    curated_dir = "../data/curated_xyz"
    bad_files = []
    for fname in os.listdir(xyz_dir):
        mol = utils.get_molecule(os.path.join(xyz_dir, fname))
        x = is_valid(mol)
        if not x:
            bad_files.append(fname)
            print(fname, x)
        else:
            copyfile(os.path.join(xyz_dir, fname), os.path.join(curated_dir, fname))
    print("TOTAL", len(bad_files), "BAD MOLECULES")
    for f in bad_files:
        print(f)
    # structures DECKII, FIRYOW are curated but are OK - we do not want to deal with it now
