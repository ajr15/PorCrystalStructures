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
    """
    Converts a scalar field in the .vti file to a scalar-valued function in 3D space
    using linear interpolation.
    
    Args:
        vti_data (vtk.vtkImageData): The image data from the .vti file.
        scalar_name (str): Name of the scalar field.
    
    Returns:
        np.array: array with scalar values on the vti data grid
    """
    scalar_field = vti_data.GetPointData().GetArray(scalar_name)
    if not scalar_field:
        raise ValueError(f"Scalar field '{scalar_name}' not found in the .vti file.")
    
    dims = vti_data.GetDimensions()
    
    return vtk_to_numpy(scalar_field).reshape(dims, order='F')
    

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
    @lru_cache(maxsize=100000)
    def vf_cached(x, y, z):
        return tuple(vector_function(float(x), float(y), float(z)))

    def vf_wrapper(x, y, z):
        return vf_cached(round(x,9), round(y,9), round(z,9))

    # recursive adaptive routine
    def recurse(s0, s1, t0, t1, depth):
        # coarse estimate (N) and fine estimate (2N)
        f_coarse = _gl_on_subrect(vf_wrapper, center, u, v, s0, s1, t0, t1, normal, N)
        f_fine = _gl_on_subrect(vf_wrapper, center, u, v, s0, s1, t0, t1, normal, 2*N)

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

def calculate_flux_through_circle(vector_function, center: np.ndarray, radius: np.ndarray, normal: np.ndarray):
    """
    Calculates the flux of a vector function through a circle with a given center, radius, and normal.

    Args:
        vector_function (function): A vector-valued function f(x, y, z) -> (vx, vy, vz).
        center (tuple): The (x, y, z) coordinates of the circle's center.
        radius (float): The radius of the circle.
        normal (tuple): The (nx, ny, nz) normal vector of the circle.
        num_points (int): Number of points to sample on the circle.

    Returns:
        float: The flux of the vector function through the circle.
    """

    # Normalize the normal vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    # Generate two orthogonal vectors in the plane of the circle
    if np.allclose(normal, [1, 0, 0]):
        u = np.array([0, 1, 0])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Approximate the circle as a polygon with N sides
    N = 1000  # Number of sides for the polygon approximation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = [center + radius * (np.cos(angle) * u + np.sin(angle) * v) for angle in angles]

    # Calculate the flux using the polygon approximation
    flux = 0.0
    for i in range(N):
        p1 = points[i]
        p2 = points[(i + 1) % N]  # Next point, wrapping around
        midpoint = (p1 + p2) / 2
        vector_value = vector_function(midpoint[0], midpoint[1], midpoint[2])
        edge = p2 - p1
        edge_normal = np.cross(normal, edge)
        edge_normal = edge_normal / np.linalg.norm(edge_normal) * np.linalg.norm(edge)
        flux += np.dot(vector_value, edge_normal)

    flux /= 2  # Divide by 2 to account for the polygon approximation

    return flux

def create_atomic_radius_function(atoms: List[ob.OBAtom]):
    """
    Creates a function that checks if a point is within the radius of any atom.

    Args:
        atoms (list of ob.OBAtom): A list of openbabel atoms

    Returns:
        function: A scalar-valued function f(x, y, z) that returns 1 if the point is within
                  the radius of any atom, and 0 otherwise.
    """
    centers = np.array([[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in atoms])
    radii_values = np.array([ob.GetCovalentRad(atom.GetAtomicNum()) for atom in atoms])

    def atomic_radius_function(x, y, z):
        point = np.array([x, y, z])
        distances_squared = np.sum((centers - point)**2, axis=1)
        within_radius = distances_squared <= radii_values**2
        return 1 if np.any(within_radius) else 0

    return atomic_radius_function

def calculate_masked_integral(atoms: List[ob.OBAtom], vti_file: str):
    """
    Calculates the integral of a scalar function masked by the atomic radius function of the atoms.

    Args:
        obmol (openbabel.OBMol): An OpenBabel molecule object.
        vti_file (str): Path to the .vti file containing the scalar field.
        scalar_name (str): Name of the scalar field in the .vti file.
        radii (dict): A dictionary mapping atomic symbols to their respective radii.

    Returns:
        float: The masked integral of the scalar function.
    """
    # Read the .vti file and extract the scalar function
    vti_data = read_vti_file(vti_file)
    function_values = get_scalar_values(vti_data)
    
    # Create the atomic radius function
    atomic_radius_function = create_atomic_radius_function(atoms)

    # Create a grid of points
    x, y, z = get_vti_grid(vti_data)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Evaluate the atomic radius function on the grid
    atomic_mask = np.vectorize(atomic_radius_function)(X, Y, Z)

    # Perform element-wise multiplication of the scalar values and the mask
    masked_values = function_values * atomic_mask

    # Calculate the integral as the sum of the masked values multiplied by the voxel volume
    integral = np.sum(masked_values) / np.sum(atomic_mask)

    return integral


def calculate_flux_through_bond(bond: ob.OBBond, vti_data: vtk.vtkImageData, bond_radius: float):
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
    return calculate_flux_through_circle(vector_function, center, bond_radius, normal)
