# script to parse the CIF files achieved from COD into molecule files (XYZ or MOL) of the porphyrinoids
from time import time
import os
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import JmolNN as AnalyzerNN
import multiprocessing
import signal
from contextlib import contextmanager
from tqdm import tqdm
from src import config, utils

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class StructureNotOrderedError (Exception):
    pass

def find_molecules(struct: Structure):
    # running function with limit of 5 minutes - sometime the run gets stuck for some reason
    with time_limit(5 * 60):
        # analyzing nearest neighbohrs using pymatgen
        nn_analyzer = AnalyzerNN()
        structure_graph = nn_analyzer.get_bonded_structure(structure=struct)
        return structure_graph.get_subgraphs_as_molecules()

def order_structure(struct: Structure):
    if struct.is_ordered:
        return struct
    to_remove = []
    for i, site in enumerate(struct.sites):
        species = site.species.as_dict()
        if all([x < 0.5 for x in species.values()]):
            to_remove.append(i)
        if len(species) == 1 and list(species.values())[0] > 0.5:
            site.species = {list(species.keys())[0]: 1}
    struct.remove_sites(to_remove)
    # if still the structure is not ordered, return empty list
    if struct.is_ordered:
        return struct

# def real_coordinates(site: PeriodicSite, image: np.ndarray):
#     """return the real coordinates of the PeriodicSite given its image"""
#     shifted_frac_coords = site.frac_coords + np.array(image)
#     return site.lattice.get_cartesian_coords(shifted_frac_coords)

# def structure_to_molecule(struct: Structure):
#     """Method to convert a structure to a molecule, without any fancy graph stuff, just taking the appropriate NN sites from neighboring images"""
#     neighbors = struct.get_all_neighbors_py(3.5)
#     atoms = set()
#     sites = list(struct.sites)
#     for site, nn in zip(struct.sites, neighbors):
#         atom = tx.base.Atom(site.specie.symbol, site.coords)
#         atoms.add(atom)
#         for n in nn:
#             atom = tx.base.Atom(n.specie.symbol, real_coordinates(sites[n.index], n.image))
#             atoms.add(atom)
#     mol = tx.base.Molecule(list(atoms))
#     mol.save_to_file("all_atoms.xyz")
#     mol = tx.utils.openbabel.molecule_to_obmol(mol)
#     mol = utils.connect_the_dots(mol)
#     g = utils.mol_to_graph(mol)
#     components = list(nx.connected_components(g))
#     sizes = [len(c) for c in components]
#     max_comp = components[sizes.index(max(sizes))]
#     mol = utils.graph_to_mol(g.subgraph(max_comp).copy())
#     conv = ob.OBConversion()
#     conv.WriteFile(mol, "component.xyz")


def cif_to_xyz(args):
    cif_file, xyz_dir = args
    # print("converting", cif_file)
    filename = os.path.split(cif_file)[-1][:-4]
    base_xyz_file = os.path.join(xyz_dir, filename)
    output = {"file": cif_file, "read": None, "order": None, "extract_mol": None, "status": None}
    if not os.path.isfile(base_xyz_file + "_0.xyz"):
        t = time()
        # occupancy_tolerance relaxed: some CIFs place atoms on special positions without an
        # explicit occupancy column, causing symmetry-merged sites to sum to >1 occupancy
        try:
            with open(cif_file, "rt", errors="replace") as f:
                contents = f.read()
            parser = CifParser.from_str(contents, occupancy_tolerance=100)
            struct = parser.parse_structures(primitive=False, on_error="ignore")[0]
            output["read"] = time() - t
        except:
            output["status"] = "BAD_FILE"
            return output
        t = time()
        struct = order_structure(struct)
        output["order"] = time() - t
        if struct is None:
            output["status"] = "UNORDERED"
            return output
        t = time()
        try:
            molecules = find_molecules(struct)
        except TimeoutException:
            output["status"] = "TIMEOUT"
            return output
        except Exception:
            output["status"] = "ERROR"
            return output
        output["extract_mol"] = time() - t
        if len(molecules) == 0 or molecules is None:
            output["status"] = "NO_MOLS"
        # taking the largest molecule as the porphyrinoid molecule
        molecules = sorted(molecules, key=lambda m: len(m.sites), reverse=True)
        for i, mol in enumerate(molecules):
            # saving mol to xyz file
            mol.to("{}_{}.xyz".format(base_xyz_file, i), fmt="xyz")
        output["status"] = "OK"
        return output
    else:
        output["status"] == "OK"
    return output

def main(structure: str, nworkers: int=1):
    print("initializing...")
    # cif_dir = utils.get_directory("cif", structure)
    cif_dir = os.path.join(config.DATA_DIR, "archive", "ccdc_data", "cif", "porphyrins")
    # xyz_dir = utils.get_directory("xyz", structure, create_dir=True)
    xyz_dir = os.path.join(config.DATA_DIR, "archive", "ccdc_data", "curated_2")
    args = []
    for fname in os.listdir(cif_dir):
        args.append((os.path.join(cif_dir, fname), xyz_dir))
    # running parallel the conversion jobs
    print("starting conversion...")
    if nworkers > 1:
        with multiprocessing.Pool(nworkers) as pool:
            data = list(tqdm(
                pool.imap_unordered(cif_to_xyz, args),
                total=len(args),
                desc="Converting CIF files",
            ))
    else:
        data = list(tqdm(
            map(cif_to_xyz, args),
            total=len(args),
            desc="Converting CIF files",
        ))
    pd.DataFrame(data).to_csv('cif_to_xyz_report.csv')
    # cleaning garbage files 
    print("cleaning garbage...")
    utils.clean_directory(xyz_dir, "xyz")
    print("ALL DONE!")
    print("total converted XYZ files:", len(os.listdir(xyz_dir)))

def test():
    import numpy as np
    import pandas as pd
    print("initializing...")
    cif_dir = os.path.join(config.DATA_DIR, "archive", "ccdc_data", "cif", "porphyrins")
    sample = np.random.choice(list(os.listdir(cif_dir)), 1, replace=False)
    # sample = list(os.listdir(cif_dir))[:1]
    # sample = ["/home/shachar/repos/PorCrystalStructures/data/archive/ccdc_data/cif/porphyrins/KIZSUI.cif"]
    args = []
    for fname in sample:
        args.append((os.path.join(cif_dir, fname), ""))
    res = list(map(cif_to_xyz, args))
    df = pd.DataFrame(res, columns=["cif", "status"])
    # df["sid"] = [os.path.split(x).split(".")[0] for x in df["cif"]]
    print("status counts:")
    print(df["status"].value_counts())
    df.to_csv("cif_to_xyz_report.csv")
    

if __name__ == "__main__":
    parser = utils.read_command_line_arguments("convert all CIF files to XYZ files of single molecule", return_args=False)
    parser.add_argument("--nworkers", type=int, default=1, help="number of worker for parallel processing of files")
    args = parser.parse_args()
    main(args.structure, args.nworkers)