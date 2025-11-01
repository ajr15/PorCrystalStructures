import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
from src.sqlmodels import Substituent, SubstituentProperty
from src.orca_utils import FinishedNormally, MoEnergies, read_file_to_blocks, Block
from src.parsers.BaseParser import BaseParser
from src import config 

class PartialCharges (Block):

    """Block to return the partial charges (Mulliken or Loewdin) of atom in orca output"""

    def __init__(self, name: str, kind: str="mulliken"):
        super().__init__(name)
        self.kind = kind

    def block_start(self, line: str) -> bool: 
        """Checks if a line is the start of a block"""
        return self.kind.upper() + " ATOMIC CHARGES AND SPIN POPULATIONS" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of a block"""
        return "Sum of" in line or len(line) < 5
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for line in self._content[2:-1]:
            numbers = [float(x) for x in line.split(":")[-1].split()]
            symbol = line.split(":")[0].split()[-1]
            data.append([symbol] + numbers)
        return pd.DataFrame(data, columns=["symbol", "charge", "spin"])

    def to_sql(self, sid: str, source: str) -> List[SubstituentProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        pass


def substituent_properties_from_file(output_file: str) -> dict:
    """Reads the ORCA output file and returns a dictionary with the substituent properties."""
    # Read the file and split it into blocks
    blocks = [PartialCharges("mulliken_charges", "mulliken"), PartialCharges("loewdin_charges", "loewdin"), FinishedNormally(""), MoEnergies("mo_energy", 1)]
    blocks = read_file_to_blocks(output_file, blocks=blocks)
    mulliken, loewdin, finished_normally, mo_energy = blocks
    finished_normally = finished_normally.to_sql("", "")[0].value
    basename = os.path.split(os.path.dirname(output_file))[-1][:-4]
    subid = basename.split("_")[0]
    with_h = "with_h" in basename
    if finished_normally == 0:
        return basename, {
        "subid": subid,
        "with_h": with_h,
        "finished_normally": 0,
    }
    else:
        df = mo_energy.to_dataframe()
        homo = df[df["occ"] == 1].iloc[-1, 1]
        lumo = df[df["occ"] == 0].iloc[0, 1]
        return basename, {
        "subid": subid,
        "with_h": with_h,
        "finished_normally": 1,
        "HOMO": homo,
        "LUMO": lumo,
        "mulliken": mulliken.to_dataframe(),
        "loewdin": loewdin.to_dataframe()
    }


def read_substituent_data(output_dir: str) -> pd.DataFrame:
    data = {}
    for outdir in os.listdir(output_dir):
        if not outdir.endswith("_out"):
            continue
        name = os.path.split(outdir)[-1][:-4]
        outfile = [x for x in os.listdir(os.path.join(output_dir, outdir)) if x.endswith(".out")]
        if len(outfile) == 0:
            print("No output file at", outdir)
            continue
        outfile = os.path.join(os.path.join(output_dir, outdir, outfile[0]))
        if not os.path.exists(outfile):
            print("No output file at", outdir)
            continue
        print("reading", outfile)
        basename, ajr = substituent_properties_from_file(outfile)
        data[basename] = ajr
    return data

def substituent_to_entries(substituent: Substituent, orca_data: dict, comp_with_h: bool):
    basename = "_with_h" if comp_with_h else "_no_h"
    basename = str(substituent.id) + basename
    if basename not in orca_data:
        return []
    entries = []
    source = "with_h" if comp_with_h else "no_h"
    for key, value in orca_data[basename].items():
        if key in ["subid", "with_h"]: 
            continue
        if key not in ["mulliken", "loewdin"]:
            entries.append(SubstituentProperty(
                substituent=substituent.id,
                property=key,
                value=value,
                source=source
            ))
            continue
        # if the key is partial charge, add entries for both connected atom and hydrogen (if comp_with_h is true)
        # add for connected atom
        entries.append(SubstituentProperty(
            substituent=substituent.id,
            property=f"connected_atom_{key}",
            value=value.iloc[substituent.connected_atom - 1, 1],
            source=source
        ))
        # if we read comp with h, add register for hydrogen partial charge
        if comp_with_h:
          entries.append(SubstituentProperty(
                substituent=substituent.id,
                property=f"hydrogen_{key}",
                value=value.iloc[-1, 1], # hydrogen is the last atom
                source=source
            ))
    return entries


class Parser (BaseParser):

    name = "orca_substituent_properties"
    source_prefix = "orca_substituent_property/"

    def parse(self, session: Session, n: int):
        """Parse the data to SQL entries"""
        entries = []
        substituents = session.query(Substituent).filter(Substituent.connected_atom.isnot(None)).all()
        ajr = read_substituent_data(os.path.join(config.DATA_DIR, "substituents", "dft"))
        for sub in substituents:
            entries += substituent_to_entries(sub, ajr, comp_with_h=True)
            entries += substituent_to_entries(sub, ajr, comp_with_h=False)
        return entries

if __name__ == "__main__":
    from src import config
    block = PartialCharges("", kind="mulliken")
    block = read_file_to_blocks(config.DATA_DIR + "/test/test.out", [block])[0]
    print(block.to_dataframe())
