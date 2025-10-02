import vtk
import numpy as np
from typing import List
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import tplquad
from scipy.integrate import dblquad
from openbabel import openbabel as ob
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


def calculate_flux_through_bond(bond: ob.OBBond, vti_file: str, bond_radius: float):
    """
    Calculates the flux of a vector field through a bond.

    Args:
        bond (ob.OBBond): An OpenBabel bond object.
        vti_file (str): Path to the .vti file containing the vector field.
        bond_radius (flat): Radius of bond to calcualte flux

    Returns:
        float: The flux of the vector field through the bond.
    """
    # Read the .vti file and extract the vector function
    vti_data = read_vti_file(vti_file)
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
