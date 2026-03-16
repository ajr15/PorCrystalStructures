from src.sqlmodels import StructureProperty
from typing import List
import pandas as pd
import os

class Block:

    """Block object to handle different output blocks from the computation output. a block always contains its start and end lines."""

    def __init__(self, name: str):
        self.name = name
        self._content = []

    def add_line(self, line: str):
        """Add a line to block's content"""
        self._content.append(line)

    def reset(self):
        """Reset current block's content"""
        self._content = []

    def block_start(self, line: str) -> bool: 
        """Checks if a line is the start of a block"""
        pass

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of a block"""
        pass

    def to_sql(self, sid: str, source: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        pass


class FinishedNormally (Block):

    def block_start(self, line: str) -> bool: 
        """Checks if a line is the start of a block"""
        return "****ORCA TERMINATED NORMALLY****" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of a block"""
        return "****ORCA TERMINATED NORMALLY****" in line

    def value(self) -> bool:
        return len(self._content) > 0

    def to_sql(self, sid: str, source: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        value = len(self._content)
        return [StructureProperty(structure=sid, property="finished_normally", value=value, source=source)]


class MoEnergies (Block):

    """Block to return the MO energies of the structure. the N parameter says how many levels beyond the HOMO-LUMO to return.
    1= HOMO-1-LUMO+1, 2= HOMO-2-LUMO+2..."""

    def __init__(self, name: str, n_levels: int=0):
        super().__init__(name)
        self.n_levels = n_levels

    def block_start(self, line: str) -> bool: 
        """Checks if a line is the start of a block"""
        return "NO   OCC          E(Eh)            E(eV)" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of a block"""
        return len(line) < 5
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for line in self._content[1:-1]:
            # print("*", line)
            data.append([float(line.split()[1]), float(line.split()[-1])])
        return pd.DataFrame(data, columns=["occ", "E"])

    def single_entry(self, sid: str, level: str, energy: float, source: str) -> StructureProperty:
        return StructureProperty(structure=sid, property="{}_energy".format(level), value=energy, source=source)

    def to_sql(self, sid: str, source: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        df = self.to_dataframe()
        ajr = []
        for level in range(self.n_levels + 1):
            homo = df[df["occ"] == 1].iloc[-(level + 1), 1]
            ajr.append(self.single_entry(sid, "HOMO-{}".format(level), homo, source))
            lumo = df[df["occ"] == 0].iloc[level, 1]
            ajr.append(self.single_entry(sid, "LUMO+{}".format(level), lumo, source))
            if level == 0:
                ajr.append(StructureProperty(structure=sid, property="HOMO-LUMO_gap".format(level), value=lumo-homo, source=source))
        return ajr
    

class FinalEnergy(Block):
    """Block to extract the final single-point energy from the ORCA output file."""

    def block_start(self, line: str) -> bool:
        """Checks if a line is the start of the block."""
        return "FINAL SINGLE POINT ENERGY" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of the block."""
        return "FINAL SINGLE POINT ENERGY" in line

    def to_sql(self, sid: str, source: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries."""
        energy_line = self._content[0]
        energy = float(energy_line.split()[4])
        return [
            StructureProperty(
                structure=sid,
                property="final_energy",
                value=energy,
                source=source
            )
        ]


def read_file_to_blocks(path: str, blocks: List[Block]):
    block_status = {b.name: False for b in blocks}
    with open(path, "r") as f:
        for line in f.readlines():
            for block in blocks:
                if block.block_start(line):
                    block_status[block.name] = True
                    block.reset()
                if block_status[block.name]:
                    block.add_line(line)
                if block_status[block.name] and block.block_ends(line):
                    block_status[block.name] = False
    return blocks


def file_to_sql(path: str, blocks: List[Block], source: str) -> List[StructureProperty]:
    # read all file contents to blocks
    blocks = read_file_to_blocks(path, blocks)
    # get the structure ID from path
    sid = os.path.split(path)[-1].split("_")[0]
    ajr = []
    for b in blocks:
        ajr += b.to_sql(sid, source)
    return ajr
