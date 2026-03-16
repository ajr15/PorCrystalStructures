# script to convert ORCA NMR output file to GIMIC input
from dataclasses import dataclass
from typing import Iterable, Tuple
import json
from scipy.optimize import minimize
import numpy as np
import os
from openbabel import openbabel as ob
from src.GimicBasisSet import SHELL, BasisSet
from src import utils

ANGSTROM_TO_AU = 1.8897

# ================
#  GIMIC IO UTILS
# ================

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
    basename = os.path.splitext(fname)[0]
    json_config_path = os.path.join(os.path.dirname(fname), f"{basename}.json.conf")
    parent_dir = os.path.dirname(os.path.dirname(fname))
    with open(json_config_path, "w") as f:
        json.dump(config_dict, f)
    # ogpath = os.getcwd()
    # basepath = os.path.dirname(fname)
    # os.chdir(basepath)
    v = basename.split("/")
    basename = f"./{v[-2]}/{v[-1]}"
    os.system("cd {}; {} {}".format(parent_dir, orca_2json_path, basename))
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
    _, basisset = read_mol(results)
    basisset, mat = format_basisset(basisset)
    if open_shell:
        densities = format_density_open_shell(alpha, beta, alpha_p, beta_p)
    else:
        densities = format_density(alpha, alpha_p)
    densities = convert_to_cartesian(densities, mat)
    return basisset, densities

def convert_to_gimic(orca_2json_path: str, fname: str, xdens_file_path: str, mol_file_path: str):
    """Convert an ORCA magnietic computation output to GIMIC compatible format"""
    print("reading orca out...")
    basisset, densities = parse_results(orca_2json_path, fname)
    # write xdens
    print("writing density file...")
    write_xdens(densities, xdens_file_path)
    # write basis set
    print("writing mol file...")
    basisset.write_MOL(filename=mol_file_path, coords=None, turbomole=False)
    print("done writing GIMIC files!")
    print("deleting orca json output...")
    json_path = fname.split(".")[0] + ".json"
    os.remove(json_path)


# ========================
#  GIMIC INPUT GENERATION
# ========================

Point = Tuple[float, float]

def assign_rectangle_edges(
    points: Iterable[Point],
    point_radiuses: Iterable[float],
    center: Point,
    width: float,
    height: float,
    ntrails: int=100
):
    """
    Assign L, R, B, T such that:
    - R - L = width
    - T - B = height
    - rectangle contains `center`
    - rectangle contains no forbidden points

    Raises ValueError if impossible.
    """

    xc, yc = center
    W, H = width, height

    # Feasible ranges from center constraint
    L_min, L_max = xc - W, xc
    B_min, B_max = yc - H, yc

    # Forbidden rectangles in (L, B) space
    forbidden = []
    for x, y in points:
        forbidden.append((
            x - W, x,      # L interval
            y - H, y       # B interval
        ))

    # Objective: make the center point closest to the center of the rectangle
    def objective(x):
        L, B = x
        return (L + W/2 - xc)**2 + (B + H/2 - yc)**2

    # Constraints: L in [L_min, L_max], B in [B_min, B_max]
    bounds = [(L_min, L_max), (B_min, B_max)]

    # Forbidden regions: for each forbidden rectangle, (L, B) must be outside
    def make_forbidden_constraint(Lf0, Lf1, Bf0, Bf1, radius):
        # Returns a constraint function that is positive if (L, B) is outside the forbidden rectangle (expanded by radius)
        def constraint(x):
            L, B = x
            # Expand the forbidden rectangle by 'radius' in all directions
            Lf0_exp = Lf0 - radius
            Lf1_exp = Lf1 + radius
            Bf0_exp = Bf0 - radius
            Bf1_exp = Bf1 + radius
            # At least one of these must be true: L <= Lf0_exp or L >= Lf1_exp or B <= Bf0_exp or B >= Bf1_exp
            # So, we require max(L-Lf0_exp, Lf1_exp-L, B-Bf0_exp, Bf1_exp-B) >= 0 for being outside
            return max(L-Lf1_exp, Lf0_exp-L, B-Bf1_exp, Bf0_exp-B)
        return constraint

    constraints = []
    for (Lf0, Lf1, Bf0, Bf1), radius in zip(forbidden, point_radiuses):
        constraints.append({'type': 'ineq', 'fun': make_forbidden_constraint(Lf0, Lf1, Bf0, Bf1, radius)})

    for _ in range(ntrails):
        # random initial guess within the bounds
        x0 = [L_min + np.random.rand() * W, B_min + np.random.rand() * H]

        res = minimize(
            objective,
            x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1e6, "ftol": 1e-9, "disp": False}
        )

        if not res.success:
            continue

        best_L, best_B = res.x

        return best_L, best_L + W, best_B, best_B + H
    else:
        raise ValueError("Cannot find rectangle!")
    
    
@dataclass
class Rectangle:
    origin: np.ndarray
    xvec: np.ndarray
    yvec: np.ndarray
    width: float
    height: float
    gauss_order: int = 9
    grid_spacing: float = 0.4

    def to_gimic_definition(self):
        return f"""Grid(bond) {{
type=gauss
distance=0 # must have for this grid type, not used in practice
coord1=[{",".join([str(x) for x in self.xvec])}]
coord2=[{",".join([str(x) for x in self.yvec])}]
fixcoord=[{",".join([str(x) for x in self.origin])}]
height=[0, {self.height}]
width=[0, {self.width}]
gauss_order={self.gauss_order}
spacing=[{self.grid_spacing}, {self.grid_spacing}, 0]
}}"""


def find_bond_plane(mol: ob.OBMol, bond: ob.OBBond, width: float, height: float) -> Rectangle:
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    start = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    end = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    center = (start + end) / 2
    bond_vector = end - start
    normal = bond_vector / np.linalg.norm(bond_vector)

    macrocycle_norm, _, _ = utils.find_macrocyle_plane_vectors(mol, "porphyrins")
    macrocycle_norm /= np.linalg.norm(macrocycle_norm)

    v = np.cross(bond_vector, macrocycle_norm)
    v /= np.linalg.norm(v)
    u = np.cross(bond_vector, v)
    u /= np.linalg.norm(u)

    close_atoms = []
    close_radiuses = []

    for atom in ob.OBMolAtomIter(mol):
        if atom in [atom1, atom2]: 
            continue
        atom_pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        relative_pos = atom_pos - center

        distance = np.abs(np.dot(relative_pos, normal))
        r = ob.GetCovalentRad(atom.GetAtomicNum())
        if distance < r:
            s, t = np.dot(relative_pos, v), np.dot(relative_pos, u)
            r = np.sqrt(r ** 2 - distance ** 2)
            close_atoms.append((s, t))
            close_radiuses.append(r)

    proj_center = (0, 0)
    smin, smax, tmin, tmax = assign_rectangle_edges(close_atoms, close_radiuses, proj_center, width, height)

    # Adjust center to lower left corner
    origin = center + smin * v + tmin * u

    return Rectangle(
        origin = origin * ANGSTROM_TO_AU, 
        xvec = (origin + v) * ANGSTROM_TO_AU, 
        yvec = (origin + u) * ANGSTROM_TO_AU, 
        width = (smax - smin) * ANGSTROM_TO_AU, 
        height = (tmax - tmin) * ANGSTROM_TO_AU
    )

def find_magnetic_field(mol: ob.OBMol):
    macrocycle_norm, _, _ = utils.find_macrocyle_plane_vectors(mol, "porphyrins")
    macrocycle_norm /= np.linalg.norm(macrocycle_norm)
    return macrocycle_norm * ANGSTROM_TO_AU

def gimic_bond_grid_definition(mol: ob.OBMol, bond: ob.OBBond, width: float, height: float, grid_spacing: float=0.4, gauss_order: int=9) -> Rectangle:
    """
    Generates a grid definition string for GiMiC calculations on a specified bond in a molecule.
    This function defines a rectangular grid in the plane of a given bond, suitable for use with GiMiC
    (Gauge Including Magnetically Induced Currents) calculations. The grid is centered on the bond and 
    oriented according to the molecular geometry.
    Args:
        mol (ob.OBMol): The molecule containing the bond.
        bond (ob.OBBond): The bond for which the grid is defined.
        width (float): The width of the grid in angstroms.
        height (float): The height of the grid in angstroms.
        grid_spacing (float, optional): The spacing between grid points in angstroms. Default is 0.4.
        gauss_order (int, optional): The order of the Gaussian quadrature. Default is 9.
    Returns:
        str: A formatted string defining the grid for GIMIC input.
    """
    rect = find_bond_plane(mol, bond, width, height)
    rect.gauss_order = gauss_order
    rect.grid_spacing = grid_spacing
    return rect
    

def parse_gimic_output(path: str):
    """Parse output of a GIMIC calculation as a JSON format"""
    current_density_block = False
    ajr = {}
    with open(path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if "*** Integrating total density" in line:
            current_density_block = True
            continue
        if "Positive" in line and current_density_block:
            ajr["positive_current"] = float(line.split()[-2])
        if "Negative" in line and current_density_block:
            ajr["negative_current"] = float(line.split()[-2])
        if "Induced current (nA/T)  :" in line and current_density_block:
            ajr["total_current"] = float(line.split()[-1])
            current_density_block = False
        if "ACID (au) sqrt(delta J^2)" in line:
            ajr["acid_sqrt(J^2)"] = float(line.split()[-1])
        if "ACID (nA/T)" in line:
            ajr["acid_current"] = float(line.split()[-1])
    return ajr
    