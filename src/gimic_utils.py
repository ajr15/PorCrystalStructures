import threading
from typing import Tuple, Iterable
import vtk
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator
from openbabel import openbabel as ob
from scipy.optimize import minimize
from vtkmodules.util.numpy_support import vtk_to_numpy
from src import utils

def read_vti_file(file_path, timeout=None) -> vtk.vtkImageData:
    """
    Reads a .vti file and returns the vtkImageData object.

    Args:
        file_path (str): Path to the .vti file.
        timeout (float, optional): Timeout in seconds for reading the file.

    Returns:
        vtk.vtkImageData: The image data from the .vti file.

    Raises:
        TimeoutError: If reading the file takes longer than the specified timeout.
    """
    result = {}
    exception = {}

    def worker():
        try:
            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(file_path)
            reader.Update()
            result['output'] = reader.GetOutput()
        except Exception as e:
            exception['error'] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Reading VTI file exceeded timeout of {timeout} seconds.")
    if 'error' in exception:
        raise exception['error']
    return result['output']


def get_scalar_values(vti_data: vtk.vtkImageData, scalar_name: str="scalars") -> np.ndarray:
    """Load a VTI grid and scalar field"""
    dims = vti_data.GetDimensions()
    origin = np.array(vti_data.GetOrigin())
    spacing = np.array(vti_data.GetSpacing())

    # convert coords
    xs = origin[0] + spacing[0] * np.arange(dims[0])
    ys = origin[1] + spacing[1] * np.arange(dims[1])
    zs = origin[2] + spacing[2] * np.arange(dims[2])
    # read ACID field
    point_data = vti_data.GetPointData()
    if scalar_name:
        arr = point_data.GetArray(scalar_name)
    else:  
        arr = point_data.GetArray(0)  # assume first array
    # read scalar data and reshpe to grid
    flat = vtk_to_numpy(arr)
    grid = flat.reshape((dims[2], dims[1], dims[0]))
    grid = np.transpose(grid, (2,1,0))
    # grid = flat.reshape(dims + (-1,), order='F')
    return xs, ys, zs, grid, spacing
    

def get_closest_vector_value(vti_data: vtk.vtkImageData, x: float, y: float, z: float, vector_name: str="vectors"):
    """
    Finds the closest point in the VTI data grid to the given coordinates and returns its vector value.

    Args:
        vti_data (vtk.vtkImageData): The VTI data object.
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        z (float): Z-coordinate of the point.

    Returns:
        tuple: The vector value (vx, vy, vz) at the closest grid point.
    """
    origin = np.array(vti_data.GetOrigin())
    spacing = np.array(vti_data.GetSpacing())
    dims = np.array(vti_data.GetDimensions())

    # Calculate the indices of the closest grid point
    indices = np.round((np.array([x, y, z]) - origin) / spacing).astype(int)

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, dims - 1)

    # Get the vector data
    vector_array = vti_data.GetPointData().GetArray(vector_name)
    if not vector_array:
        raise ValueError("No vector data found in the VTI file.")

    # Convert to numpy array and reshape to grid dimensions
    vector_values = vti_data.GetPointData().GetArray(vector_name)
    if not vector_values:
        raise ValueError(f"Vector field '{vector_name}' not found in the .vti file.")
    
    return vector_values.GetTuple(indices[2] * dims[1] * dims[0] + indices[1] * dims[0] + indices[0])

def get_vector_function(vti_data: vtk.vtkImageData, vector_name: str="vectors"):
    """
    Converts a vector field in the .vti file to a vector-valued function in 3D space
    using linear interpolation.
    
    Args:
        vti_data (vtk.vtkImageData): The image data from the .vti file.
        vector_name (str): Name of the vector field.
    
    Returns:
        function: A vector-valued function f(x, y, z) -> (vx, vy, vz).
    """
    
    def vector_function(x, y, z):
        return get_closest_vector_value(vti_data, x, y, z, vector_name)
    
    return vector_function

# ======= ANALYZE GIMIC OUTPUT =======

def _gl_on_subrect(vector_function, center, u, v, s0, s1, t0, t1, normal, N):
    """
    Compute GL tensor-product on subrectangle with s in [s0,s1], t in [t0,t1].
    s,t are coordinates along u and v (signed distances from center).
    """
    # nodes & weights on [-1,1]
    x, w = leggauss(N)

    # map x->s,t
    half_s = 0.5 * (s1 - s0)
    mid_s = 0.5 * (s1 + s0)
    half_t = 0.5 * (t1 - t0)
    mid_t = 0.5 * (t1 + t0)

    ws = half_s * w
    wt = half_t * w
    s_nodes = mid_s + half_s * x
    t_nodes = mid_t + half_t * x

    flux = 0.0
    # loop small N^2 (N up to ~12 is fine)
    for i in range(N):
        si = s_nodes[i]
        wsi = ws[i]
        for j in range(N):
            tj = t_nodes[j]
            wtj = wt[j]

            # map to 3D point
            r = center + si * u + tj * v
            Jx, Jy, Jz = vector_function(r[0], r[1], r[2])
            J = np.array([Jx, Jy, Jz], float)
            flux += wsi * wtj * np.dot(J, normal)

    # area Jacobian: for u,v not unit or not orthogonal
    jac = np.linalg.norm(np.cross(u, v))
    return flux * jac

def calculate_flux_through_plane(
    vector_function,
    center,
    normal,
    s_min,
    s_max,
    t_min,
    t_max,
    u,
    v,
    N=8,
    tol_rel=1e-3,
    tol_abs=0.0,
    max_depth=6,
):
    """
    Adaptive GL quadrature over rectangular plane.

    Args:
        vector_function (function): A vector-valued function f(x, y, z) -> (Jx, Jy, Jz).
        center (array-like): 3D coordinates of the rectangle center.
        normal (array-like): Normal vector of the plane.
        s_min (float): Minimum value of the s-coordinate along the u direction.
        s_max (float): Maximum value of the s-coordinate along the u direction.
        t_min (float): Minimum value of the t-coordinate along the v direction.
        t_max (float): Maximum value of the t-coordinate along the v direction.
        u (array-like): Vector defining the u direction in the plane.
        v (array-like): Vector defining the v direction in the plane.
        N (int, optional): Base Gauss-Legendre order used per subrectangle. Default is 8.
        tol_rel (float, optional): Relative tolerance for local error. Default is 1e-3.
        tol_abs (float, optional): Absolute tolerance for local error. Default is 0.0.
        max_depth (int, optional): Maximum subdivision depth. Default is 6.

    Returns:
        float: The computed flux through the plane.
    """

    center = np.array(center, float)
    normal = np.array(normal, float)
    normal /= np.linalg.norm(normal)

    # recursive adaptive routine
    def recurse(s0, s1, t0, t1, depth):
        # coarse estimate (N) and fine estimate (2N)
        f_coarse = _gl_on_subrect(vector_function, center, u, v, s0, s1, t0, t1, normal, N)
        f_fine = _gl_on_subrect(vector_function, center, u, v, s0, s1, t0, t1, normal, 2*N)

        # error estimate
        err = abs(f_fine - f_coarse)
        tol_local = max(tol_abs, tol_rel * max(abs(f_fine), 1.0))

        if (err <= tol_local) or (depth >= max_depth):
            # accept fine value
            return f_fine
        else:
            # subdivide into 4 subrectangles (bisect s and t)
            sm = 0.5 * (s0 + s1)
            tm = 0.5 * (t0 + t1)
            return (
                recurse(s0, sm, t0, tm, depth+1)
                + recurse(sm, s1, t0, tm, depth+1)
                + recurse(s0, sm, tm, t1, depth+1)
                + recurse(sm, s1, tm, t1, depth+1)
            )

    total_flux = recurse(s_min, s_max, t_min, t_max, depth=0)
    return total_flux

def find_plane_limits(mol: ob.OBMol, bond: ob.OBBond, width: float, height: float, u: np.ndarray, v: np.ndarray) -> tuple:
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    start = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    end = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    # Calculate the bond's center and direction
    center = (start + end) / 2
    bond_vector = end - start
    normal = bond_vector / np.linalg.norm(bond_vector)
    u = np.array(u)
    v = np.array(v)

    close_atoms = []
    close_radiuses = []

    # Iterate over all atoms in the molecule
    for atom in ob.OBMolAtomIter(mol):
        if atom in [atom1, atom2]: 
            continue
        atom_pos = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        relative_pos = atom_pos - center

        # Project the atom position onto the u and v directions
        distance = np.abs(np.dot(relative_pos, normal))
        r = ob.GetCovalentRad(atom.GetAtomicNum())
        if distance < r:
            s, t = np.dot(relative_pos, v), np.dot(relative_pos, u)
            r = np.sqrt(r ** 2 - distance ** 2) # fix radius of cut sphere (pythagorian theorem)
            close_atoms.append((s, t))
            close_radiuses.append(r)

    proj_center = (0, 0) # the center is always at the origin of the plane

    return assign_rectangle_edges(close_atoms, close_radiuses, proj_center, width, height)


def calculate_flux_through_bond(mol: ob.OBMol, bond: ob.OBBond, vti_data: vtk.vtkImageData, surface_width: float, surface_height: float, n_gl: int=10):
    """
    Calculates the flux of a vector field through a bond.

    Args:
        mol (ob.OBMol): The molecule containing the bond.
        bond (ob.OBBond): An OpenBabel bond object.
        vti_data (vtk.vtkImageData): The VTI data containing the vector field.
        surface_width (float): Width of the integration surface.
        surface_height (float): Height of the integration surface.
        n_gl (int): Gauss-Legendre order for integration.

    Returns:
        float: The flux of the vector field through the bond.
    """


    # Get the bond's start and end points
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    start = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    end = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    # Calculate the bond's center and direction
    center = (start + end) / 2
    bond_vector = end - start
    bond_vector = bond_vector / np.linalg.norm(bond_vector)
    
    macrocycle_norm, _, _ = utils.find_macrocyle_plane_vectors(mol, "porphyrins")
    macrocycle_norm /= np.linalg.norm(macrocycle_norm)

    # Find vector roughly in the macrocycle plane orthogonal to both bond_vector and macrocycle normal
    v = np.cross(bond_vector, macrocycle_norm)
    v /= np.linalg.norm(v)
    # find vector orthogonal to the macrocycle plane and the bond vector
    u = np.cross(bond_vector, v)
    u /= np.linalg.norm(u)


    # find required integration bounds (s and t) to avoid overlap with neighboring atoms
    smin, smax, tmin, tmax = find_plane_limits(mol, bond, surface_width, surface_height, u, v)

    # Extract the vector function
    vector_function = get_vector_function(vti_data)

    # Calculate the flux through the bond
    return calculate_flux_through_plane(
        vector_function,
        center,
        bond_vector, # Normal to the plane
        smin,
        smax,
        tmin,
        tmax,
        u,
        v,
        N=n_gl
    )

# ======= ANALYZE ACID OUTPUT =======

def proj_dist_points_to_segment(points, a, b):
    """
    Vectorized distance of many points to one segment.
    points: (N,3)
    a,b: (3,)
    returns distances (N,)
    """
    ab = b - a
    ap = points - a
    bp = points - b
    ab_norm = np.dot(ab, ab)
    t = np.sum(ap * ab, axis=1) / ab_norm

    # Calculate orthogonal projection distances
    proj = a + np.outer(t, ab)
    orthogonal_distances = np.linalg.norm(points - proj, axis=1)

    # Check if projection is within the segment
    within_segment = (t >= 0.0) & (t <= 1.0)

    # Calculate distances to endpoints
    distances_to_a = np.linalg.norm(ap, axis=1)
    distances_to_b = np.linalg.norm(bp, axis=1)

    # Combine distances based on projection position
    result_distances = np.where(within_segment, orthogonal_distances, np.minimum(distances_to_a, distances_to_b))
    return result_distances


def compute_bond_voronoi(xs, ys, zs, bonds):
    """
    xs,ys,zs: coordinate vectors from VTK
    bonds: list of (atom_i_coords, atom_j_coords)

    returns bond_idx_grid, shape = (Nx,Ny,Nz)
    """
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    # points = np.column_stack([xs, ys, zs])


    nbonds = len(bonds)
    distances = np.zeros((len(points), nbonds))

    for j,(a,b) in enumerate(bonds):
        distances[:,j] = proj_dist_points_to_segment(points, np.array(a), np.array(b))

    nearest = np.argmin(distances, axis=1)  
    return nearest.reshape(len(xs), len(ys), len(zs))


def integrate_acid_per_bond(acid_grid, bond_map, spacing, nbonds):
    dV = spacing[0] * spacing[1] * spacing[2]
    bond_integrals = np.zeros(nbonds)

    for b in range(nbonds):
        mask = (bond_map == b)
        bond_integrals[b] = np.sum(acid_grid[mask]) * dV

    return bond_integrals

def create_acid_interpolator(acid_grid, xs, ys, zs):
    """
    Creates an interpolator function for the ACID grid.

    Args:
        acid_grid (np.ndarray): The scalar field grid (ACID values).
        xs, ys, zs (np.ndarray): The grid coordinates along x, y, z axes.

    Returns:
        function: A function that takes x, y, z and returns the interpolated ACID value.
    """
    interpolator = RegularGridInterpolator((xs, ys, zs), acid_grid, bounds_error=False, fill_value=None)

    def acid_function(x, y, z):
        return interpolator((x, y, z))

    return acid_function

def integrate_acid_around_bond(acid_function, bond: ob.OBBond, spacing: float, R: float, nuclie_distance: float):
    """
    Integrates the ACID values around a bond within a cylindrical region of radius R.

    Args:
        acid_function (function): A function that takes (x, y, z) and returns the interpolated ACID value.
        bond (ob.OBBond): An OpenBabel bond object.
        spacing (tuple): The grid spacing along x, y, z axes.
        R (float): Radius of the cylinder around the bond.

    Returns:
        float: Integrated ACID value for the bond.
    """
    # Get the bond's start and end points
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    a = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    c = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    ab = c - a
    ab_norm = np.linalg.norm(ab)
    ab_unit = ab / ab_norm

    if ab_norm < nuclie_distance * 2:
        raise ValueError("Cannot take distance larger than bond length!")

    # take distance from each atom nucleaus
    a = a + ab_unit * nuclie_distance
    c = c - ab_unit * nuclie_distance
    
    # recalculate
    ab = c - a
    ab_norm = np.linalg.norm(ab)
    ab_unit = ab / ab_norm

    # Define a grid of points along the bond axis and within the cylinder radius
    num_points_along_bond = int(ab_norm / spacing[0]) + 1
    num_points_radial = int(R / spacing[0]) + 1

    integral = 0.0

    for i in range(num_points_along_bond):
        t = i / (num_points_along_bond - 1)
        point_on_bond = a + t * ab

        for j in range(num_points_radial):
            for k in range(num_points_radial):
                # Generate points in the radial plane
                theta = 2 * np.pi * j / num_points_radial
                r = R * k / num_points_radial
                offset = r * np.array([np.cos(theta), np.sin(theta), 0])

                # Rotate offset to align with the bond direction
                rotation_matrix = np.eye(3)
                rotation_matrix[:2, :2] = [[ab_unit[0], -ab_unit[1]], [ab_unit[1], ab_unit[0]]]
                rotated_offset = rotation_matrix @ offset

                # Calculate the final point
                final_point = point_on_bond + rotated_offset
                integral += acid_function(*final_point) * spacing[0] * spacing[1] * spacing[2]

    return integral


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



if __name__ == "__main__":
    import os
    from src import config
    vti_file = os.path.join(config.DATA_DIR, "nmr", "ATUSOX_0_out", "gimic", "acid.vti")
    res = None
    print("starting to read...")
    try:
        res = read_vti_file(vti_file, 60 * 20)
    except TimeoutError:
        print("HEY! i had a timout error")
    print("HEY! i finished normally")
    print(res)
    import sys; sys.exit()
    
    
    from matplotlib import pyplot as plt
    mol_file = os.path.join(config.DATA_DIR, "xyz", "dft", "ATUSOX" + "_0.xyz")
    mol = utils.get_molecule(mol_file)
    bond = mol.GetBond(18, 19)

    # Get the bond's start and end points
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    start = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    end = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    # Calculate the bond's center and direction
    center = (start + end) / 2
    bond_vector = end - start
    bond_vector = bond_vector / np.linalg.norm(bond_vector)
    
    macrocycle_norm, _, _ = utils.find_macrocyle_plane_vectors(mol, "porphyrins")
    macrocycle_norm /= np.linalg.norm(macrocycle_norm)

    # Find vector roughly in the macrocycle plane orthogonal to both bond_vector and macrocycle normal
    v = np.cross(bond_vector, macrocycle_norm)
    v /= np.linalg.norm(v)
    # find vector orthogonal to the macrocycle plane and the bond vector
    u = np.cross(bond_vector, v)
    u /= np.linalg.norm(u)

    l, r, b, t = find_plane_limits(mol, bond, 5, 10, u, v)
    
    # adds hydrogen atoms to the molecule on the rectangle edges
    borders = [(l, b), (l, t), (r, b), (r, t)]
    for x, y in borders:
        coords = center + v * x + u * y
        hydrogen = ob.OBAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetVector(coords[0], coords[1], coords[2])
        mol.AddAtom(hydrogen)
    out_file = "test.xyz"
    obConversion = ob.OBConversion()
    obConversion.SetOutFormat("xyz")
    obConversion.WriteFile(mol, out_file)


