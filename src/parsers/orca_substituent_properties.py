import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
from src.sqlmodels import Substituent, SubstituentProperty
from src.orca_utils import FinishedNormally, MoEnergies, read_file_to_blocks
from src.parsers.BaseParser import BaseParser
from src import config 

def safe_smiles_str(smiles: str) -> str:
    """Method to encode a smiles string in a safe way for using in ORCA I/O files"""
    return smiles.replace("[", "!").replace("]", "!").replace("(", ".").replace(")", ".").replace("/", "|").replace("@", "0")

def substituent_properties_from_file(output_file: str) -> dict:
    """Reads the ORCA output file and returns a dictionary with the substituent properties."""
    # Read the file and split it into blocks
    blocks = [FinishedNormally(""), MoEnergies("mo_energy", 1)]
    blocks = read_file_to_blocks(output_file, blocks=blocks)
    finished_normally, mo_energy = blocks
    finished_normally = finished_normally.to_sql("", "")[0].value

    if finished_normally == 0:
        return {
        "symbol": os.path.split(output_file)[-1][:-4],
        "finished_normally": 0,
        "HOMO": None,
        "LUMO": None,
    }
    else:
        df = mo_energy.to_dataframe()
        homo = df[df["occ"] == 1].iloc[-1, 1]
        lumo = df[df["occ"] == 0].iloc[0, 1]
        return {
        "symbol": os.path.split(output_file)[-1][:-4],
        "finished_normally": 1,
        "HOMO": homo,
        "LUMO": lumo,
    }


def read_substituent_data(output_dir: str) -> pd.DataFrame:
    data = []
    for outdir in os.listdir(output_dir):
        print(outdir)
        if not outdir.endswith("_out"):
            continue
        name = os.path.split(outdir)[-1][:-4]
        outfile = os.path.join(output_dir, outdir, name + ".out")
        if not os.path.exists(outfile):
            continue
        data.append(substituent_properties_from_file(outfile))
    df = pd.DataFrame(data)
    df = df.set_index("symbol")
    return df


# def main(session: Session, n):
#     print("=" * 10, "READING SUBSTITUENT ORCA CALCULATION RESULTS", "=" * 10)
#     if n > 1:
#         print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
#     substituents = session.query(Substituent).all()
#     df = read_substituent_data("data/substituents/dft")
#     for sub in substituents:
#         symbol = safe_smiles_str(sub.substituent)
#         if symbol not in df.index:
#             continue
#         properties = df.loc[symbol, :].to_dict()
#         if properties["finished_normally"] == 0:
#             continue
#         for k, v in properties.items():
#             if k == "finished_normally":
#                 continue
#             prop = SubstituentProperty(
#                 smiles=sub.substituent, 
#                 structure=sub.structure, 
#                 property=k, 
#                 value=v, 
#                 position=sub.position, 
#                 position_index=sub.position_index, 
#                 source="orca"
#             )
#             session.add(prop)
#     session.commit()
#     print("ALL DONE")

class Parser (BaseParser):

    name = "orca_substituent_properties"
    source_prefix = "orca_substituent_property"

    def parse(self, session: Session, n: int):
        """Parse the data to SQL entries"""
        entries = []
        substituents = session.query(Substituent).all()
        df = read_substituent_data(os.path.join(config.DATA_DIR, "substituents", "dft"))
        for sub in substituents:
            symbol = safe_smiles_str(sub.substituent)
            if symbol not in df.index:
                continue
            properties = df.loc[symbol, :].to_dict()
            if properties["finished_normally"] == 0:
                continue
            for k, v in properties.items():
                if k == "finished_normally":
                    continue
                entries.append(SubstituentProperty(
                    smiles=sub.substituent, 
                    structure=sub.structure, 
                    property=k, 
                    value=v, 
                    position=sub.position, 
                    position_index=sub.position_index, 
                    source=""
                ))
        return entries


