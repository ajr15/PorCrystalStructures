# parser to read both NMR and NICS values from ORCA output
from networkx.algorithms import isomorphism
from openbabel import openbabel as ob
import os
from src.sqlmodels import StructureProperty
from src.parsers.BaseParser import StructureParser
from src import gimic_utils as gutils
from src import config, utils



def find_structure_indices(obmol: ob.OBMol):
    """Find the indices of a substructure within a molecule."""
    subgraph = utils.get_definition("porphyrins")
    g = utils.mol_to_graph(obmol)
    iso = isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    morph = list(iso.subgraph_isomorphisms_iter())
    return morph


def get_metal_idx(mol: ob.OBMol, macrocycle_atoms) -> int:
    for atom_idx in macrocycle_atoms:
        atom = mol.GetAtom(atom_idx)
        if atom.GetAtomicNum() != 7:  # Skip non-nitrogen atoms
            continue
        for nbr in ob.OBAtomAtomIter(atom):
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in macrocycle_atoms and nbr.GetAtomicNum() != 7:
                return nbr_idx

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
    # get mapper and add metal idx
    mol.ConnectTheDots()
    atom_mapper = find_structure_indices(mol)[0]
    metal_idx = get_metal_idx(mol, atom_mapper.keys())
    if metal_idx:
        atom_mapper[metal_idx] = -1
    return atom_mapper


def get_macrocycle_bonds(mol: ob.OBMol):
    atom_idxs = get_macrocycle_atoms(mol).keys()
    mol.ConnectTheDots()
    bonds = []
    covered = set()
    for idx in atom_idxs:
        atom = mol.GetAtom(idx)
        covered.add(idx)
        for bond in ob.OBAtomBondIter(atom):
            bidx = bond.GetBeginAtomIdx()
            eidx = bond.GetEndAtomIdx()
            if eidx in covered and bidx in covered: 
                continue
            if bidx in atom_idxs and eidx in atom_idxs:
                bonds.append(bond)
    return bonds



def calculate_acid_records(sid, atom_mapper: dict, bonds, radius: float, distance: float, data):
    """Calculate bond integrals for bonds within the macrocycle."""
    res = []
    xs, ys, zs, acid_grid, spacing = gutils.get_scalar_values(data)
    acid_func = gutils.create_acid_interpolator(acid_grid, xs, ys, zs)
    for bond in bonds:
        acid = gutils.integrate_acid_around_bond(
            acid_func, bond, spacing, radius, distance
        )
        begin = atom_mapper[bond.GetBeginAtomIdx()] 
        end = atom_mapper[bond.GetEndAtomIdx()]  
        ajr = [
            StructureProperty(
                structure=sid, 
                property=f"mapped/{begin}->{end}",
                value=acid,
                source=f"radius={radius}&distance={distance}"
            ),
            StructureProperty(
                structure=sid, 
                property=f"original/{bond.GetBeginAtomIdx()}->{bond.GetEndAtomIdx()}",
                value=acid,
                source=f"radius={radius}&distance={distance}"
            )
        ]
        res.extend(ajr)
    return res


class Parser (StructureParser):

    name = "acid"
    source_prefix = "acid/"
    radii = [0.5 + 0.5 * i for i in range(8)]
    distances = [0.1 * i for i in range(8)]

    def parse_structure(self, session, sid):
        jvec_file = os.path.join(config.DATA_DIR, "nmr", sid + "_0_out", "gimic", "acid.vti")
        if not os.path.exists(jvec_file):
            return [], [f"INFO: No GIMIC calculation for {sid} ({jvec_file})"]
        mol_file = os.path.join(config.DATA_DIR, "nmr", sid + "_0_out", "gimic", "mol.xyz")
        mol = utils.get_molecule(mol_file)
        atom_mapper = get_macrocycle_atoms(mol)
        bonds = get_macrocycle_bonds(mol)
        vti_data = gutils.read_vti_file(jvec_file)
        entries = []
        for r in self.radii:
            for dist in self.distances:
                try:
                    entries.extend(calculate_acid_records(sid, atom_mapper, bonds, r, dist, vti_data))
                except ValueError:
                    continue
        return entries, [] 

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine('sqlite:///main.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    parser = Parser()
    entries, _ = parser.parse_structure(session, "ATEWUT")
    for entry in entries:
        print(entry.structure, entry.property, entry.value, entry.source)