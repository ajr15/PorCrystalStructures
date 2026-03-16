# script to build starting XYZ geometries for substituents 
import os
import openbabel as ob
from sqlalchemy import create_engine
from src import utils, config

def safe_smiles_str(smiles: str) -> str:
    """Method to encode a smiles string in a safe way for using in ORCA I/O files"""
    return smiles.replace("[", "!").replace("]", "!").replace("(", ".").replace(")", ".").replace("/", "|").replace("@", "0")

def replace_dummy_with_hydrogen(obmol: ob.OBMol):
    atoms_to_replace = [atom for atom in ob.OBMolAtomIter(obmol) if atom.GetAtomicNum() == 0]
    for atom in atoms_to_replace:
        # Replace dummy with hydrogen
        atom.SetAtomicNum(1)
        atom.SetType("H")
        atom.SetImplicitValence(1)
        atom.SetFormalCharge(0)
    obmol.ConnectTheDots()
    obmol.AddHydrogens()
    obmol.PerceiveBondOrders()
    return obmol

class OpenbabelBuildError (Exception):
    pass

def guess_geometry(mol: ob.OBMol) -> ob.OBMol:
    builder = ob.OBBuilder()
    if not builder.Build(mol):
        conv = ob.OBConversion()
        raise OpenbabelBuildError("Failed building the smiles: ".format(conv.WriteString(mol)))
    return mol

class OpenbabelFfError (Exception):
    pass

def mm_geometry_optimization(mol: ob.OBMol, force_field: str="UFF", nsteps: int=1000) -> ob.OBMol:
    # optimization
    OBFF = ob.OBForceField.FindForceField(force_field)
    suc = OBFF.Setup(mol)
    if not suc == True:
        raise OpenbabelFfError("Could not set up force field for molecule")
    OBFF.ConjugateGradients(nsteps)
    OBFF.GetCoordinates(mol)
    return mol

if __name__ == "__main__":
    print("initializing...")
    engine = create_engine("sqlite:///{}".format("../main.db"))
    smiles = engine.execute("SELECT DISTINCT substituent FROM substituents").all()
    target_dir = os.path.join(config.DATA_DIR, "substituents", "xyz")
    conv = ob.OBConversion()
    conv.SetOutFormat("xyz")
    for i, s in enumerate(smiles):
        target_file = os.path.join(target_dir, safe_smiles_str(s[0]) + ".xyz")
        if os.path.isfile(target_file):
            continue
        print("building", s[0], "({} out of {})".format(i + 1, len(smiles)))
        # writing dummy file for future skipping of bad files
        with open(target_file, "w") as f:
            f.write("")
        try:
            obmol = utils.mol_from_smiles(s[0])
            obmol = replace_dummy_with_hydrogen(obmol)
            obmol = guess_geometry(obmol)
            obmol = mm_geometry_optimization(obmol)
            conv.WriteFile(obmol, target_file)
        except:
            print("errors building", s[0])
    print("ALL DONE!")
