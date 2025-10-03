# script to parse results from data/nonplanarity directory to a dataframe format
import os
from shutil import copyfile
import json
from src import config
from src.sqlmodels import StructureProperty, Structure
from src.parsers.BaseParser import StructureParser, Session

def json_to_dicts(parameters: dict):
    """Convert Porphystruct JSON results file to list of dict entries"""
    entries = []
    entries.append({"property": "total out of plane (exp)", "value": parameters["OutOfPlaneParameter"]["Value"], "units": "A"})
    entries.append({"property": "total out of plane (fit)", "value": parameters["Simulation"]["OutOfPlaneParameter"]["Value"], "units": "A"})
    entries.append({"property": "metal cavity size", "value": parameters["Cavity"]["Value"], "units": "A^2"})
    for d in parameters["Simulation"]["SimulationResult"]:
        entries.append({"property": "{} non planarity".format(d["Key"].lower()), "value": d["Value"], "units": "A"})
    for d in parameters["Simulation"]["SimulationResultPercentage"]:
        entries.append({"property": "{} non planarity".format(d["Key"].lower()), "value": d["Value"], "units": "%"})
    for d in parameters["Distances"]:
        # formatting bond length info to a standard form
        pname = d["Key"].replace(" - ", "-")
        if pname != "N-N":
            pname = "M-N"
        entries.append({"property": "{} distance".format(pname), "value": d["Value"], "units": "A"})
    for d in parameters["PlaneDistances"]:
        # formatting bond length info to a standard form
        pname = d["Key"].split(" - ")[-1].lower()
        entries.append({"property": "metal - {} distance".format(pname), "value": d["Value"], "units": "A"})
    return entries

def _entries_for_structure(json_dir):
    ajr = []
    source = os.path.split(json_dir)[-1]
    for fname in os.listdir(json_dir):
        sid = fname.split("_")[0]
        with open(os.path.join(json_dir, fname), "r") as f:
            parameters = json.load(f)
            ajr += [StructureProperty(structure=sid, source="porphystruct-" + source, **kwargs) for kwargs in json_to_dicts(parameters)]
    return ajr


def entries_for_structure(sid: str, source: str):
    xyz = os.path.join(config.DATA_DIR, "xyz", source, sid + "_0.xyz")
    if not os.path.exists(xyz):
        return [], [f"INFO: no xyz file for {sid} (should be {xyz})"]
    porphystruct_output = os.path.join(config.DATA_DIR, "nonplanarity", source, sid + "_0_analysis.json")
    if not os.path.exists(porphystruct_output):
        os.system(f"bash $CRYSTAL_SRC_DIR/scripts/porphystruct_analysis.bash {sid} {source}")
    if not os.path.exists(porphystruct_output):
        return [], [f"ERROR: Porphystruct calculation failed for {sid}"]
    with open(porphystruct_output, "r") as f:
        parameters = json.load(f)
        return [StructureProperty(structure=sid, source=source, **kwargs) for kwargs in json_to_dicts(parameters)], []


def main(session, n):
    print("=" * 10, "READING STRUCTURE PORPHYSTRUCT CALCULATION RESULTS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    # removing all previous readings of HOMA
    session.execute("DELETE FROM structure_properties WHERE source LIKE 'porphystruct-%'")
    session.commit()
    # fetch all the xyz files from finished calculations
    xyz_files = session.query(Structure.orca_xyz).filter(Structure.orca_xyz != None).all()
    for path in xyz_files:
        path = path[0]
        fname = os.path.split(path)[-1]
        copyfile(path, os.path.join(config.DATA_DIR, "xyz", "dft", fname))
    # run porphystruct on the dft xyz files
    os.system("bash src/porphystruct_analysis.bash")
    # now read to database
    json_dir = os.path.join(config.DATA_DIR, "nonplanarity", "dft")
    ajr = entries_for_structure(json_dir)
    session.add_all(ajr)
    session.commit()
    json_dir = os.path.join(config.DATA_DIR, "nonplanarity", "crystal")
    ajr = entries_for_structure(json_dir)
    session.add_all(ajr)
    session.commit()
    print("ALL DONE")

class Parser (StructureParser):

    name = "porphystruct"
    source_prefix = "porphystruct/"

    def parse_structure(self, session: Session, sid: str):
        """Parse the data to SQL entries"""
        ajr = entries_for_structure(sid, "crystal")
        ajr += entries_for_structure(sid, "dft")
        return ajr

