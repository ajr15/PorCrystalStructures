import json
from networkx.algorithms import isomorphism
from openbabel import openbabel as ob
from src import gimic_utils as gutils
from src import utils
import os


def get_definition(structure: str):
    """Get the graph definition of a structure from a .mol file."""
    mol = utils.get_molecule(os.path.join("data", "definitions", structure + ".mol"))
    return utils.mol_to_graph(mol)


def find_structure_indices(obmol: ob.OBMol, structure: str):
    """Find the indices of a substructure within a molecule."""
    subgraph = get_definition(structure)
    g = utils.mol_to_graph(obmol)
    iso = isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    morph = list(iso.subgraph_isomorphisms_iter())
    return morph


def get_macrocycle_atoms(mol: ob.OBMol):
    """Get the atoms involved in the macrocycle, including the metal center."""
    # Remove all hydrogen atoms
    while True:
        for atom in ob.OBMolAtomIter(mol):
            if atom.GetAtomicNum() == 1:
                mol.DeleteAtom(atom)
                break
        else:
            break

    mol.ConnectTheDots()
    atom_mapper = find_structure_indices(mol, "porphyrins")[0]
    metal_idx = None
    for atom_idx in atom_mapper.keys():
        atom = mol.GetAtom(atom_idx)
        if atom.GetAtomicNum() != 7:  # Skip non-nitrogen atoms
            continue
        for nbr in ob.OBAtomAtomIter(atom):
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in atom_mapper.keys() and nbr.GetAtomicNum() != 7:
                metal_idx = nbr_idx
    if metal_idx:
        atom_mapper[metal_idx] = -1
    return atom_mapper


def calculate_bond_integrals(mol: ob.OBMol, atom_mapper: dict, width, height, data):
    """Calculate bond integrals for bonds within the macrocycle."""
    bond_integrals = {}
    for atom_idx in atom_mapper.keys():
        atom_obj = mol.GetAtom(atom_idx)
        for nbr in ob.OBAtomAtomIter(atom_obj):
            nbr_idx = nbr.GetIdx()
            if nbr_idx in atom_mapper.keys() and atom_idx < nbr_idx:
                bond = mol.GetBond(atom_obj.GetIdx(), nbr.GetIdx())
                integral = gutils.calculate_flux_through_bond(
                    bond, data, width, height
                )
                bond_integrals[(atom_mapper[bond.GetBeginAtomIdx()], atom_mapper[bond.GetEndAtomIdx()])] = integral
    return bond_integrals


def parse_gimic_dir(gimic_dir, width, height):
    """Main function to get bond integral values for a molecule."""
    xyz_path = gimic_dir + "/mol.xyz"
    mol = utils.get_molecule(xyz_path)
    mol.DeleteHydrogens()
    porphyrin_atoms = get_macrocycle_atoms(mol)
    jvec_vti_file = gimic_dir + "/jvec.vti"
    vti_data = gutils.read_vti_file(jvec_vti_file)
    bond_integrals = calculate_bond_integrals(mol, porphyrin_atoms, width, height, vti_data)
    return bond_integrals

def process_parent_directory(parent_dir, width, height):
    """Process each directory in the parent directory to find and parse gimic subdirectories."""
    results = {}
    for root, dirs, files in os.walk(parent_dir):
        if "gimic" in dirs:
            gimic_dir = os.path.join(root, "gimic")
            if os.path.isfile(os.path.join(gimic_dir, "jvec.vti")):
                print(gimic_dir)
                bond_integrals = parse_gimic_dir(gimic_dir, width, height)
                results[gimic_dir] = [{"begin": i, "end": j, "val": val} for (i, j), val in bond_integrals.items()]
                with open("gimic_results.json", "w") as f:
                    json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    process_parent_directory("data/nmr", 2, 3)
