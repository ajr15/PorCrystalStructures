# parser to read both NMR and NICS values from ORCA output
from itertools import chain
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


def calculate_bond_current_records(sid, mol: ob.OBMol, atom_mapper: dict, width, height, data):
    """Calculate bond integrals for bonds within the macrocycle."""
    res = []
    for atom_idx in atom_mapper.keys():
        atom_obj = mol.GetAtom(atom_idx)
        for nbr in ob.OBAtomAtomIter(atom_obj):
            nbr_idx = nbr.GetIdx()
            if nbr_idx in atom_mapper.keys() and atom_idx < nbr_idx:
                bond = mol.GetBond(atom_obj.GetIdx(), nbr.GetIdx())
                integral = gutils.calculate_flux_through_bond(
                    bond, data, width, height
                )
                begin= atom_mapper[bond.GetBeginAtomIdx()] 
                end = atom_mapper[bond.GetEndAtomIdx()]  
                ajr = [
                    StructureProperty(
                        structure=sid, 
                        property=f"current/mapped/{begin}->{end}",
                        value=integral,
                        source=f"h={height}&w={width}"
                    ),
                    StructureProperty(
                        structure=sid, 
                        property=f"current/original/{bond.GetBeginAtomIdx()}->{bond.GetEndAtomIdx()}",
                        value=integral,
                        source=f"h={height}&w={width}"
                    )
                ]
                res.extend(ajr)
    return res

def build_macrocycle_box(mol: ob.OBMol, buffer: float):
    atoms = get_macrocycle_atoms(mol)
    coords = [mol.GetAtom(idx).GetVector() for idx in atoms.keys()]
    x_coords = [coord.GetX() for coord in coords]
    y_coords = [coord.GetY() for coord in coords]
    z_coords = [coord.GetZ() for coord in coords]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    return min_x - buffer, max_x + buffer, min_y - buffer, max_y + buffer, min_z - buffer, max_z + buffer


class Parser (StructureParser):

    name = "gimic"
    source_prefix = "gimic/"
    # widths = [1.5 + 0.5 * i for i in range(8)]
    # heights = [1.5 + 0.5 * i for i in range(8)]
    widths = [3]
    heights = [2]

    def parse_structure(self, session, sid):
        jvec_file = os.path.join(config.DATA_DIR, "nmr", sid + "_0_out", "gimic", "jvec.vti")
        if not os.path.exists(jvec_file):
            return [], [f"INFO: No GIMIC calculation for {sid} ({jvec_file})"]
        mol_file = os.path.join(config.DATA_DIR, "nmr", sid + "_0_out", "gimic", "mol.xyz")
        mol = utils.get_molecule(mol_file)
        atom_mapper = get_macrocycle_atoms(mol)
        vti_data = gutils.read_vti_file(jvec_file)
        entries = []
        for width in self.widths:
            for height in self.heights:
                entries.extend(calculate_bond_current_records(sid, mol, atom_mapper, width, height, vti_data))
        return entries, [] 

    def parse(self, session, n):
        """Run the structure parser in parallel"""
        connection_string = str(session.get_bind().engine.url)
        sids = self.fetch_structure_ids(session)
        args = [(connection_string, sid) for sid in sids]
        # if n > 1:
        # with Pool(processes=n) as pool:
        #     ajr = list(tqdm(pool.imap(self._parse_structure, args), total=len(sids), desc="Processing structures"))
        # else:
        ajr = []
        for i, a in enumerate(args):
            _, sid = a
            print(f"Parsing {sid} ({i + 1} out of {len(args)})")
            ajr.append(self._parse_structure(a))
        results = [x[0] for x in ajr]
        messages = list(chain(*[x[-1] for x in ajr]))
        print("Done!")
        if len(messages) > 0:
            print("== Run Messages ==")
            for m in messages:
                print(m)
        return list(chain(*results))

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