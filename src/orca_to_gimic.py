# script to convert ORCA NMR output file to GIMIC input
import json
import numpy as np
import os
from GimicBasisSet import SHELL, BasisSet
import utils
from openbabel import openbabel as ob
from torinax.io import OrcaIn
from torinax.utils.openbabel import molecule_to_obmol


GIMIC_INPUT_TEXT = """
# GENERATED FROM ORCA

calc=cdens
title=""
basis=$BASIS_FILE
xdens=$DENSITY_FILE
debug=10
openshell=$OPENSHELL 
magnet=$MAGNETIC_FIELD_VEC

Grid(base) {
    type=even
    origin=$GRID_ORIGIN
    ivec=$GRID_X_VEC
    jvec=$GRID_Y_VEC
    lengths=$GRID_LENGTH
    spacing=[$GRID_SPACING, $GRID_SPACING, $GRID_SPACING]
}

Advanced {
    lip_order=5
    spherical=off
    diamag=on
    paramag=on
    GIAO=on
    screening=on
    screening_thrs=1.d-8
}

Essential {
    acid=on
}
"""


ORBITAL_TO_L = {
    "s": 0,
    "p": 1,
    "d": -2, # use of negative values signifies spherical harmonics
    "f": -3,  # use of negative values signifies spherical harmonics
    "g": -4,  # use of negative values signifies spherical harmonics
    "h": -5,  # use of negative values signifies spherical harmonics
    "i": -6,  # use of negative values signifies spherical harmonics
}

def orca_to_json(orca_2json_path: str, fname: str):
    config_dict = {"Densities": ["all"]}
    json_config_path = os.path.join(os.path.dirname(fname), "orca.json.conf")
    with open(json_config_path, "w") as f:
        json.dump(config_dict, f)
    # ogpath = os.getcwd()
    # basepath = os.path.dirname(fname)
    # os.chdir(basepath)
    basename = os.path.splitext(fname)[0]
    os.system("{} {}".format(orca_2json_path, basename))
    # os.chdir(ogpath)


def read_orca_out(orca_2json_path: str, fname: str):
    """Read the unperturbed and perturbed density matrices"""
    # runs json reader
    orca_to_json(orca_2json_path, fname)
    # load json results
    basename = os.path.basename(fname).split(".")[0]
    json_path = os.path.join(os.path.dirname(fname), basename + ".json")
    with open(json_path, "r") as f:
        return json.load(f)

def is_openshell(results: dict) -> bool:
    """determine if a calculation is openshell"""
    return "scfr" in results["Molecule"]["Densities"]

def read_mol(results: dict):
    """Read the basis set info and atom coordinates"""
    coords = []
    shells = []
    for atom in results["Molecule"]["Atoms"]:
        for basisset in atom["BasisFunctions"]:
            shells.append(
                SHELL(
                    atom["Idx"], 
                    atom["ElementNumber"], 
                    ORBITAL_TO_L[basisset["Shell"]], 
                    np.array(basisset["Exponents"]), 
                    np.array(basisset["Coefficients"]),
                    coord=atom["Coords"]))
        coords.append(atom["Coords"])
    return coords, BasisSet(shells)

def read_densities(results: dict, open_shell: bool):
    ajr = results["Molecule"]["Densities"]
    full = np.array(ajr["scfp"])
    pfull = np.array([np.array(ajr["pbscf_0"]), np.array(ajr["pbscf_1"]), np.array(ajr["pbscf_2"])])
    if open_shell:
        spin = np.array(ajr["scfr"])
        pspin = np.array([np.array(ajr["rbscf_0"]), np.array(ajr["rbscf_1"]), np.array(ajr["rbscf_2"])])
        return 0.5 * (full + spin), 0.5 * (full - spin), 0.5 * (pfull + pspin), 0.5 * (pfull - pspin)
    else:
        return full, None, pfull, None

def format_density(unperturbed, perturbed):
    """Format the unperturbed (ground-state) and perturbed (magnetically) densities to desired array format.
    The perturbed density should have dimensions of (3, *perturbed) shape"""
    densities = np.zeros((1, 4) + unperturbed.shape)
    densities[0, 0] = unperturbed
    densities[0, 1:] = perturbed
    return densities

def format_density_open_shell(alpha, beta, alpha_p, beta_p):
    """in case of open shell, the density vector contains two "closed shell" vectors, one for each spin"""
    k = alpha.shape[-1]
    densities = np.zeros((2, 4, k, k))
    densities[0, 0] = alpha
    densities[1, 0] = beta
    densities[0, 1:] = alpha_p
    densities[1, 1:] = beta_p
    return densities

def format_basisset(basisset: BasisSet):
    """Read the atomic basis set info and cartesian transition matrix for density matrix set from an ORCA file"""
    CartOrdering = "cfour" # can be "turbomole" also
    tmat = basisset.Cart2Spher(square=False, real=True, order=CartOrdering, normalized=False)
    return basisset, tmat

def convert_to_cartesian(formatted_densities, tmat):
    """Convert the calculation's density matrix (in spherical orbitals) to cartesian orbitals (required for GIMIC)"""
    # apply the transformation to 'densities' array
    # the transformation reorganized the basis functions
    # with the shells organized by types
    # with the angular momenta organized like CartOrdering
    nspin = formatted_densities.shape[0]
    ndensities = np.zeros((nspin, 4, tmat.shape[0], tmat.shape[0]))
    for ispin in range(nspin):
        for idir in range(4):# 0, Bx, By, Bz
            ndensities[ispin, idir] = np.dot(np.dot(tmat, formatted_densities[ispin, idir]), tmat.transpose())
    return ndensities


def write_xdens(densities, xdens_path):
    nspin = densities.shape[0]
    # ATTENTION values in XDENS for Bx, By, Bz are 2 times bigger for closeshell and 4 times for openshell
    if nspin == 1:
        densities[:, 1:, :, :] *= 2
    else:
        densities[:, 1:, :, :] *= 4

    # open outputfile
    outfile=open(xdens_path, "w")

    # write densities matrices on file
    # write in a fortran way
    for ispin in range(nspin):
        for idir in range(4):# 0, Bx, By, Bz
            for j in range(densities.shape[3]):
                for i in range(densities.shape[2]):
                    outfile.write("%16.8e\n"%(densities[ispin, idir, i, j]))
            outfile.write("\n")
    outfile.close()


def parse_results(orca_2json_path, fname):
    """Parse ORCA results to GIMIC friendly information. 
    RETURNS: atomic coordinates, basisset, densities, openshell (bool)"""
    results = read_orca_out(orca_2json_path, fname)
    open_shell = is_openshell(results)
    alpha, beta, alpha_p, beta_p = read_densities(results, open_shell)
    coords, basisset = read_mol(results)
    basisset, mat = format_basisset(basisset)
    print("N ATOMS", len(coords))
    print("N SHELLS", len(basisset.basis))
    print("TMAT SHAPE", mat.shape)
    if open_shell:
        densities = format_density_open_shell(alpha, beta, alpha_p, beta_p)
    else:
        densities = format_density(alpha, alpha_p)
    print("FORMATTED DENSITY SHAPE (SPHERIC)", densities.shape)
    densities = convert_to_cartesian(densities, mat)
    print("FORMATTED DENSITY SHAPE (CARTESIAN)", densities.shape)
    return coords, basisset, densities, open_shell

def determine_grid(coords: np.ndarray) -> tuple:
    coords = np.array(coords)
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    origin = min_coords
    grid_x_vec = np.array([max_coords[0] - min_coords[0], 0, 0])
    grid_y_vec = np.array([0, max_coords[1] - min_coords[1], 0])
    grid_z_vec = np.array([0, 0, max_coords[2] - min_coords[2]])
    lengths = [np.linalg.norm(grid_x_vec), np.linalg.norm(grid_y_vec), np.linalg.norm(grid_z_vec)]
    return origin, grid_x_vec / np.linalg.norm(grid_x_vec), grid_y_vec / np.linalg.norm(grid_y_vec), grid_z_vec / np.linalg.norm(grid_z_vec), lengths

def determine_magnetic_field(fname: str):
    # count number of ghost atoms
    with open(fname, "r") as file:
        nghosts = sum(1 for line in file if line.startswith("H:"))
    # read geometry from input file
    infile = OrcaIn(fname)
    mol = molecule_to_obmol(infile.read_specie())
    # removes the ghost atoms from molecule
    for _ in range(nghosts):
        mol.DeleteAtom(mol.GetAtom(mol.NumAtoms()))
    core_idxs = utils.find_structure_indices(mol, "porphyrins")[0]
    core_coords = np.array([[mol.GetAtom(int(idx)).GetX(), mol.GetAtom(int(idx)).GetY(), mol.GetAtom(int(idx)).GetZ()] for idx in core_idxs])
    centroid = np.mean(core_coords, axis=0)
    centered_coords = core_coords - centroid
    _, _, vh = np.linalg.svd(centered_coords)
    normal = vh[-1]
    return normal


def write_gimic_files(densities, basisset, xdens_path, mol_path):
    # write xdens
    write_xdens(densities, xdens_path)
    # write basis set
    basisset.write_MOL(filename=mol_path, coords=None, turbomole=False)

def convert(gimic_comp_dir: str, orca_2json_path: str, fname: str, grid_spacing: int):
    # setup
    xdens_path = os.path.join(gimic_comp_dir, "XDENS")
    mol_path = os.path.join(gimic_comp_dir, "MOL")
    gimic_input_path = os.path.join(gimic_comp_dir, "gimic.inp")
    # read raw results from ORCA
    coords, basisset, densities, open_shell = parse_results(orca_2json_path, fname)
    print("COMPUTATION IS", "OPENS-SHELL" if open_shell else "CLOSE-SHELL")
    # write gimic files in gimic dir
    write_gimic_files(densities, basisset, xdens_path, mol_path)
    # making GIMIC input text
    # basic keywords
    s = GIMIC_INPUT_TEXT.replace("$BASIS_FILE", "MOL")
    s = s.replace("$DENSITY_FILE", "XDENS")
    s = s.replace("$OPENSHELL", "true" if open_shell else "false")
    s = s.replace("$GRID_SPACING", str(grid_spacing))
    # determine grid
    origin, xvec, yvec, zvec, lengths = determine_grid(coords)
    s = s.replace("$GRID_ORIGIN", "[{}, {}, {}]".format(*[str(x) for x in origin]))
    s = s.replace("$GRID_X_VEC", "[{}, {}, {}]".format(*[str(x) for x in xvec]))
    s = s.replace("$GRID_Y_VEC", "[{}, {}, {}]".format(*[str(x) for x in yvec]))
    s = s.replace("$GRID_LENGTH", "[{}, {}, {}]".format(*[str(x) for x in lengths]))
    # determive magnetic field
    field_normal = determine_magnetic_field(fname)
    s = s.replace("$MAGNETIC_FIELD_VEC", "[{}, {}, {}]".format(*[str(x) for x in field_normal]))
    # write input
    with open(gimic_input_path, "w") as f:
        f.write(s) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert ORCA NMR output file to GIMIC input.")
    parser.add_argument("input_file", type=str, help="Path to the ORCA input file.")
    parser.add_argument("--gimic_dir", type=str, default=None, help="Path of GIMIC directory (to create/exiting). defaults to make 'gimic' directory in input file's direcotry")
    parser.add_argument("--grid_spacing", type=float, default=0.1, help="Grid spacing for GIMIC input (default: 0.1).")
    parser.add_argument("--orca_2json_path", type=str, default="~/Software/ORCA5/orca_2json", help="Path to orca_2json executable (default: ~/Software/ORCA5/orca_2json).")

    # Parse arguments
    args = parser.parse_args()

    # parse & create gimic dir
    gimic_dir = args.gimic_dir
    if gimic_dir is None:
        gimic_dir = os.path.join(os.path.dirname(args.input_file), "gimic")
    if not os.path.isdir(gimic_dir):
        os.mkdir(gimic_dir)

    # Call the convert function with parsed arguments
    convert(gimic_dir, args.orca_2json_path, args.input_file, args.grid_spacing)
