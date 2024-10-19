# script to parse the CIF files achieved from COD into molecule files (XYZ or MOL) of the porphyrinoids
import os
from pymatgen.core import Structure
from pymatgen.analysis.local_env import JmolNN as AnalyzerNN
import multiprocessing
import signal
from contextlib import contextmanager
import config
import utils

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
        # preprocessing structure to ensure no partial occupancies
        if not struct.is_ordered:
            to_remove = []
            for i, site in enumerate(struct.sites):
                species = site.species.as_dict()
                if all([x < 0.5 for x in species.values()]):
                    to_remove.append(i)
                if len(species) == 1 and list(species.values())[0] > 0.5:
                    site.species = {list(species.keys())[0]: 1}
            struct.remove_sites(to_remove)
            # if still the structure is not ordered, return empty list
            if not struct.is_ordered:
                raise StructureNotOrderedError
        # analyzing nearest neighbohrs using pymatgen
        nn_analyzer = AnalyzerNN()
        structure_graph = nn_analyzer.get_bonded_structure(structure=struct)
        return structure_graph.get_subgraphs_as_molecules()
    
def cif_to_xyz(args):
    cif_file, xyz_dir = args
    print("converting", cif_file)
    filename = os.path.split(cif_file)[-1][:-4]
    base_xyz_file = os.path.join(xyz_dir, filename)
    if not os.path.isfile(base_xyz_file + "_0.xyz"):
        try:
            struct = Structure.from_file(cif_file)
            molecules = find_molecules(struct)
        except TimeoutException:
            print("timeout occured at", cif_file)
            return
        except StructureNotOrderedError:
            print("structure cannot be ordered at", cif_file)
            return
        except Exception:
            print("errors occured at", cif_file)
            return
        if len(molecules) == 0:
            print("no molecules found in", cif_file)
            return
        # taking the largest molecule as the porphyrinoid molecule
        molecules = sorted(molecules, key=lambda m: len(m.sites), reverse=True)
        for i, mol in enumerate(molecules):
            # saving mol to xyz file
            mol.to("{}_{}.xyz".format(base_xyz_file, i), fmt="xyz")

def main(structure: str, nworkers: int=1):
    print("initializing...")
    cif_dir = utils.get_directory("cif", structure)
    xyz_dir = utils.get_directory("xyz", structure, create_dir=True)
    args = []
    for fname in os.listdir(cif_dir):
        args.append((os.path.join(cif_dir, fname), xyz_dir))
    # running parallel the conversion jobs
    print("starting conversion...")
    if nworkers > 1:
        with multiprocessing.Pool(nworkers) as pool:
            pool.map(cif_to_xyz, args)
    else:
        list(map(cif_to_xyz, args))
    # cleaning garbage files 
    print("cleaning garbage...")
    utils.clean_directory(xyz_dir, "xyz")
    print("ALL DONE!")
    print("total converted XYZ files:", len(os.listdir(xyz_dir)))

if __name__ == "__main__":
    parser = utils.read_command_line_arguments("convert all CIF files to XYZ files of single molecule", return_args=False)
    parser.add_argument("--nworkers", type=int, default=1, help="number of worker for parallel processing of files")
    args = parser.parse_args()
    main(args.structure, args.nworkers)