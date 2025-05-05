import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
from read_to_sql import Substituent, SubstituentProperty
from parsers.orca_details import Block, file_to_sql, FinishedNormally, read_file_to_blocks


def safe_smiles_str(smiles: str) -> str:
    """Method to encode a smiles string in a safe way for using in ORCA I/O files"""
    return smiles.replace("[", "!").replace("]", "!").replace("(", ".").replace(")", ".").replace("/", "|").replace("@", "0")


class MoEnergies (Block):

    """Block to return the MO energies of the structure. the N parameter says how many levels beyond the HOMO-LUMO to return.
    1= HOMO-1-LUMO+1, 2= HOMO-2-LUMO+2..."""

    def __init__(self, name: str, source: str, n_levels: int=0):
        super().__init__(name)
        self.n_levels = n_levels
        self.source = source

    def block_start(self, line: str) -> bool: 
        """Checks if a line is the start of a block"""
        return "NO   OCC          E(Eh)            E(eV)" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of a block"""
        return len(line) < 5
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for line in self._content[1:-1]:
            data.append([float(line.split()[1]), float(line.split()[-1])])
        return pd.DataFrame(data, columns=["occ", "E"])
    
    def get_energy(self, level: str) -> float:
        df = self.to_dataframe()
        if "HOMO" in level:
            homo_idx = df[df["occ"] == 1].last_valid_index()
            delta = int(level.split("-")[-1])
            if homo_idx - delta < 0:
                return None
            return df.loc[homo_idx - delta, "E"]
        elif "LUMO" in level:
            lumo_idx = df[df["occ"] == 1].last_valid_index() + 1
            delta = int(level.split("+")[-1])
            return df.loc[lumo_idx + delta, "E"]
        else:
            raise ValueError("Unknown level " + level)

    def to_sql(self, sid: str):
        """Parse block content to a list of StructureProperty SQL entries"""
        pass    

class FinalEnergy(Block):
    """Block to extract the final single-point energy from the ORCA output file."""

    def __init__(self, name, source: str):
        super().__init__(name)
        self.source = source

    def block_start(self, line: str) -> bool:
        """Checks if a line is the start of the block."""
        return "FINAL SINGLE POINT ENERGY" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of the block."""
        return "FINAL SINGLE POINT ENERGY" in line
    
    def parse_value(self) -> float:
        return float(self._content[0].split()[4])

    def to_sql(self, sid: str):
        pass


def substituent_properties_from_file(output_file: str) -> dict:
    """Reads the ORCA output file and returns a dictionary with the substituent properties."""
    # Read the file and split it into blocks
    blocks = [FinishedNormally(""), MoEnergies("mo_energy", "base_calc", 2), FinalEnergy("final_energy", "base_calc")]
    blocks = read_file_to_blocks(output_file, blocks=blocks)
    finished_normally, mo_energy, final_energy = blocks
    finished_normally = finished_normally.to_sql("")[0].value
    if finished_normally == 0:
        return {
        "symbol": os.path.split(output_file)[-1][:-4],
        "finished_normally": 0,
        "HOMO": None,
        "LUMO": None,
    }
    else:
        return {
        "symbol": os.path.split(output_file)[-1][:-4],
        "finished_normally": 1,
        "HOMO": mo_energy.get_energy("HOMO-0"),
        "LUMO": mo_energy.get_energy("LUMO+0"),
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


def main(session: Session, n):
    print("=" * 10, "READING SUBSTITUENT ORCA CALCULATION RESULTS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    substituents = session.query(Substituent).all()
    df = read_substituent_data("data/substituents/dft")
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
            prop = SubstituentProperty(
                smiles=sub.substituent, 
                structure=sub.structure, 
                property=k, 
                value=v, 
                position=sub.position, 
                position_index=sub.position_index, 
                source="orca"
            )
            session.add(prop)
    session.commit()
    print("ALL DONE")
