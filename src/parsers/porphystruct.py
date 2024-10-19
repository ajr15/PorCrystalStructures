# script to parse results from data/nonplanarity directory to a dataframe format
import os
import json
import utils
from read_to_sql import StructureProperty

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

def entries_for_structure(stype: str):
    json_dir = utils.get_directory("nonplanarity", stype)
    ajr = []
    for fname in os.listdir(json_dir):
        sid = fname.split("_")[0]
        with open(os.path.join(json_dir, fname), "r") as f:
            parameters = json.load(f)
            ajr += [StructureProperty(structure=sid, source="porphystruct", **kwargs) for kwargs in json_to_dicts(parameters)]
    return ajr


def main(session, n):
    print("=" * 10, "READING STRUCTURE PORPHYSTRUCT CALCULATION RESULTS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    print("reading corrole details...")
    ajr = entries_for_structure("corroles")
    session.add_all(ajr)
    print("reading porphyrin details...")
    ajr = entries_for_structure("porphyrins")
    session.add_all(ajr)
    session.commit()
    print("ALL DONE")
