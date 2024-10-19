# script to scape all the CIF files and metadata from the COD. we want to download only corrole complexes (and possibly porphyrins)
import os
import json
import re
from rdkit.Chem import rdchem
import openbabel as ob
import config
import utils

def find_all_structures(keyword: str, output_file: str):
    """Method to fetch all metadata for files with matching keyword (corrole, porphyrin, etc.). outputs the results to JSON file"""
    # getting the raw response from search API (for some reason the requests library does not work well here)
    if len(keyword.split()) > 1:
        raise ValueError("Cannot search with more than one word")
    cmd = "curl --data \"text={}&format=json\" https://www.crystallography.net/cod/result --output {}".format(keyword, "tmp.json")
    os.system(cmd)
    # now reading the text and formatting it 
    response = ""
    with open("tmp.json", "r") as f:
        response = f.read()
    with open(output_file, "w") as f:
        f.write("{\"structures\":" + response + "}")
    os.remove("tmp.json")

def has_metal(formula_str: str):
    """Method to check if a given formula string has metal atom"""
    symbols = ["".join(re.findall("[a-zA-Z]", x)) for x in formula_str.split()]
    for symbol in symbols:
        if len(symbol) == 0:
            continue
        try:
            Z = rdchem.GetPeriodicTable().GetAtomicNumber(symbol)
        except Exception as e:
            print("exception occured in formula", formula_str, "and symbol", symbol)
            continue
        # makes custom criteria to get all possible central metals. including from S and P blocks
        if 10 < Z < 16 or 18 < Z < 35 or 36 < Z < 53 or 54 < Z:
            return True
        # atom = ob.OBAtom()
        # atom.SetAtomicNum(Z)
        # if atom.IsMetal():
        #     return True
    else:
        return False


def find_structures_with_metals(output_file: str):
    with open(output_file, "r") as f:
        d = json.load(f)
    print("TOTAL NUMBER OF STRUCTURES =", len(d["structures"]))
    metal_structs = []
    for strcut in d["structures"]:
        formula_str = strcut["formula"]
        if has_metal(formula_str):
            metal_structs.append(strcut["file"])
    print("TOTAL NUMBER OF STRUCTURES WITH METAL =", len(metal_structs))
    return metal_structs


def download_cif_files(codids, target_dir):
    """download all the cifs in list"""
    os.chdir(target_dir)
    for i, codid in enumerate(codids):
        print("downloading {} out of {}".format(i + 1, len(codids)))
        if not os.path.exists("{}.cif".format(codid)):
            os.system("wget https://www.crystallography.net/cod/{}.cif".format(codid))


def scrape_structures(search_term, structure):
    json_path = os.path.join(config.DATA_DIR, structure + "_structures.json")
    cif_dir = utils.get_directory("cif", structure, create_dir=True)
    print("getting full strcuture list...")
    find_all_structures(search_term, json_path)
    print("getting only structures with metal...")
    codids = find_structures_with_metals(json_path)
    print("downloading files...")
    download_cif_files(codids, cif_dir)
    # clean directory from garbage files (like .cif.1)
    print("cleaning garbage...")
    utils.clean_directory(cif_dir, "cif")
    print("ALL DONE!")

if __name__ == "__main__":
    args = utils.read_command_line_arguments("scrape all files containing metal from the COD database. download as cif files")
    struct = args.structure
    scrape_structures(struct, struct)
