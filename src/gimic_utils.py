import vtk
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import List
from scipy.interpolate import RegularGridInterpolator
from openbabel import openbabel as ob
from functools import lru_cache
from vtkmodules.util.numpy_support import vtk_to_numpy

def read_vti_file(file_path) -> vtk.vtkImageData:
    """
    Reads a .vti file and returns the vtkImageData object.
    
    Args:
        file_path (str): Path to the .vti file.
    
    Returns:
        vtk.vtkImageData: The image data from the .vti file.
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def get_vti_bounds(vti_data: vtk.vtkImageData):
    """
    Extracts the bounds of the .vti data.

    Args:
        vti_data (vtk.vtkImageData): The image data from the .vti file.

    Returns:
        tuple: A tuple of ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    """
    origin = vti_data.GetOrigin()
    spacing = vti_data.GetSpacing()
    dims = vti_data.GetDimensions()

    xmin = origin[0]
    xmax = origin[0] + (dims[0] - 1) * spacing[0]
    ymin = origin[1]
    ymax = origin[1] + (dims[1] - 1) * spacing[1]
    zmin = origin[2]
    zmax = origin[2] + (dims[2] - 1) * spacing[2]

    return ((xmin, xmax), (ymin, ymax), (zmin, zmax))


def get_vti_grid(vti_data: vtk.vtkImageData):
    dims = vti_data.GetDimensions()
    origin = vti_data.GetOrigin()
    spacing = vti_data.GetSpacing()
    
    x = np.linspace(origin[0], origin[0] + (dims[0] - 1) * spacing[0], dims[0])
    y = np.linspace(origin[1], origin[1] + (dims[1] - 1) * spacing[1], dims[1])
    z = np.linspace(origin[2], origin[2] + (dims[2] - 1) * spacing[2], dims[2])
    return x, y, z


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
    vector_field = vti_data.GetPointData().GetArray(vector_name)
    if not vector_field:
        raise ValueError(f"Vector field '{vector_name}' not found in the .vti file.")
    
    dims = vti_data.GetDimensions()
    origin = vti_data.GetOrigin()
    spacing = vti_data.GetSpacing()
    
    x = np.linspace(origin[0], origin[0] + (dims[0] - 1) * spacing[0], dims[0])
    y = np.linspace(origin[1], origin[1] + (dims[1] - 1) * spacing[1], dims[1])
    z = np.linspace(origin[2], origin[2] + (dims[2] - 1) * spacing[2], dims[2])
    
    vector_values = vtk_to_numpy(vector_field).reshape(dims + (-1,), order='F')
    interpolators = [RegularGridInterpolator((x, y, z), vector_values[..., i], bounds_error=True)
                     for i in range(vector_values.shape[-1])]
    
    def vector_function(x, y, z):
        return tuple(interpolator((x, y, z)) for interpolator in interpolators)
    
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

def calculate_flux_through_plane_adaptive(
    vector_function,
    center,
    width,
    height,
    normal,
    N=8,
    tol_rel=1e-3,
    tol_abs=0.0,
    max_depth=6,
):
    """
    Adaptive GL quadrature over rectangular plane.
    Args:
        vector_function: f(x,y,z)->(Jx,Jy,Jz)
        center: 3-array, rectangle center in 3D
        width: total width along u (float)
        height: total height along v (float)
        normal: plane normal (3-array)
        N: base GL order (int) used per subrectangle
        tol_rel: relative tolerance for local error
        tol_abs: absolute tolerance for local error
        max_depth: maximum subdivision depth
    Returns:
        flux (float)
    """

    center = np.array(center, float)
    normal = np.array(normal, float)
    normal /= np.linalg.norm(normal)

    # build robust u,v basis in plane
    tmp = np.array([1.0, 0.0, 0.0])
    if np.allclose(normal, tmp):
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # s,t ranges relative to center
    s_min, s_max = -0.5 * width, 0.5 * width
    t_min, t_max = -0.5 * height, 0.5 * height

    # simple cache wrapper around vector_function to avoid re-evaluating same points
    # key by rounded coordinates (you may adapt precision)
    # @lru_cache(maxsize=lru_maxcash)
    # def vf_cached(x, y, z):
    #     return tuple(vector_function(float(x), float(y), float(z)))

    # def vf_wrapper(x, y, z):
    #     return vf_cached(round(x,9), round(y,9), round(z,9))

    # recursive adaptive routine
    def recurse(s0, s1, t0, t1, depth):
        # coarse estimate (N) and fine estimate (2N)
        # f_coarse = _gl_on_subrect(vf_wrapper, center, u, v, s0, s1, t0, t1, normal, N)
        # f_fine = _gl_on_subrect(vf_wrapper, center, u, v, s0, s1, t0, t1, normal, 2*N)
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

def cut_vti_data_to_box(vti_data: vtk.vtkImageData, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float) -> vtk.vtkImageData:
    # Extract the dimensions, origin, and spacing of the original VTI data
    dims = vti_data.GetDimensions()
    origin = vti_data.GetOrigin()
    spacing = vti_data.GetSpacing()

    # Calculate the indices corresponding to the box bounds
    i_min = max(0, int((xmin - origin[0]) / spacing[0]))
    i_max = min(dims[0] - 1, int((xmax - origin[0]) / spacing[0]))
    j_min = max(0, int((ymin - origin[1]) / spacing[1]))
    j_max = min(dims[1] - 1, int((ymax - origin[1]) / spacing[1]))
    k_min = max(0, int((zmin - origin[2]) / spacing[2]))
    k_max = min(dims[2] - 1, int((zmax - origin[2]) / spacing[2]))

    # Extract the sub-image
    extract = vtk.vtkExtractVOI()
    extract.SetInputData(vti_data)
    extract.SetVOI(i_min, i_max, j_min, j_max, k_min, k_max)
    extract.Update()
    return extract.GetOutput()



def calculate_flux_through_bond(bond: ob.OBBond, vti_data: vtk.vtkImageData, surface_width: float, surface_height: float, n_gl: int=10):
    """
    Calculates the flux of a vector field through a bond.

    Args:
        bond (ob.OBBond): An OpenBabel bond object.
        vti_file (str): Path to the .vti file containing the vector field.
        bond_radius (flat): Radius of bond to calcualte flux

    Returns:
        float: The flux of the vector field through the bond.
    """

    # extract the vector function
    vector_function = get_vector_function(vti_data)

    # Get the bond's start and end points
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    start = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    end = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    # Calculate the bond's center and direction
    center = (start + end) / 2
    direction = end - start
    normal = direction / np.linalg.norm(direction)

    # Calculate the flux through the bond
    return calculate_flux_through_plane_adaptive(vector_function, center, surface_width, surface_height, normal, N=n_gl)

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

def integrate_acid_around_bond(acid_function, bond: ob.OBBond, spacing, R):
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


# def calculate_bond_integrals(vti_data: vtk.vtkImageData, bonds: List[ob.OBBond]):
#     # extract data from vti data
#     xs, ys, zs, acid_grid, spacing = get_scalar_values(vti_data)
#     # extract atom coordinates from all bonds
#     ajr = []
#     for bond in bonds:
#         a1 = bond.GetBeginAtom()
#         a2 = bond.GetEndAtom()
#         ajr.append((
#             (a1.GetX(), a1.GetY(), a1.GetZ()),
#             (a2.GetX(), a2.GetY(), a2.GetZ())
#         ))
#     # assign each grid point to bond (using voronoi nearest neighbors algorithm)
#     bond_map = compute_bond_voronoi(xs, ys, zs, ajr)
#     # return bond integral values
#     return integrate_acid_per_bond_vtk(acid_grid, bond_map, spacing, len(bonds))

if __name__ == "__main__":
    import utils
    from openbabel import openbabel as ob
    import matplotlib.pyplot as plt

    acid_vti_file = "data/test/benzene/hr_acid.vti"
    vti_data = read_vti_file(acid_vti_file)
    xs, ys, zs, acid_grid, spacing = get_scalar_values(vti_data)
    mol = utils.get_molecule("data/test/benzene/mol.xyz")
    for _ in range(2):
        mol.DeleteAtom(mol.GetAtom(mol.NumAtoms()))
    # 2. Define bonds as ((x1,y1,z1),(x2,y2,z2)) list
    bonds = []
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        # if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
        bonds.append((
            (a1.GetX(), a1.GetY(), a1.GetZ()),
            (a2.GetX(), a2.GetY(), a2.GetZ())
        ))


    

    # Extract points where z=0
    z_index = np.argmin(np.abs(zs))  # Find the index where z is closest to 0
    xy_points = np.column_stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])
    acid_values = acid_grid[:, :, z_index].ravel()

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xy_points[:, 0], xy_points[:, 1], c=acid_values, cmap='Oranges', s=10)
    plt.colorbar(scatter, label="Bond Assignment")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Plot of Points Colored by Bond Assignment (z=0)")
    # Plot the bonds as lines
    for i, (a, b) in enumerate(bonds):
        a = np.array(a)
        b = np.array(b)
        plt.plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=2)
        # Calculate the midpoint of the bond
        midpoint = (a + b) / 2
        # Annotate the bond index at the midpoint
        # plt.text(midpoint[0], midpoint[1], str(i), color='red', fontsize=12, ha='center', va='center')

    # bond_acid = calculate_bond_integrals(vti_data, list(ob.OBMolBondIter(mol)))
    # for i, acid in enumerate(bond_acid):
    #     print(f"{i:2d} | {acid:.4f}")

    acid_func = create_acid_interpolator(acid_grid, xs, ys, zs)
    acids = []
    for bond in ob.OBMolBondIter(mol):
        acids.append(integrate_acid_around_bond(acid_func, bond, spacing, R=1))
    print(acids)

    # # Create a figure to visualize bonds colored by their ACID value
    # plt.figure(figsize=(10, 8))
    # norm = plt.Normalize(vmin=np.min(bond_acid), vmax=np.max(bond_acid))
    # cmap = plt.cm.Greens

    # for i, (a, b) in enumerate(bonds):
    #     a = np.array(a)
    #     b = np.array(b)
    #     color = cmap(norm(bond_acid[i]))
    #     plt.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=4)

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label="ACID Value", ax=plt.gca())
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title("Bonds Colored by ACID Value")
    # print(f"Bond | Points | ACID")
    plt.show()
    # 0, 3, 5, 7, 9, 1