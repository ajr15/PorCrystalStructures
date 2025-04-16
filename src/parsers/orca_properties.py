import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
from read_to_sql import StructureProperty, Structure
from parsers.orca_details import Block, file_to_sql

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

    def single_entry(self, sid: str, level: str, energy: float) -> StructureProperty:
        return StructureProperty(structure=sid, property="{}_energy".format(level), value=energy, source="orca_geometry_optimization")

    def to_sql(self, sid: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        df = self.to_dataframe()
        ajr = []
        for level in range(self.n_levels + 1):
            homo = df[df["occ"] == 1].iloc[-(level + 1), 1]
            ajr.append(self.single_entry(sid, "HOMO-{}".format(level), homo))
            lumo = df[df["occ"] == 0].iloc[level, 1]
            ajr.append(self.single_entry(sid, "LUMO+{}".format(level), lumo))
            if level == 0:
                ajr.append(StructureProperty(structure=sid, property="HOMO-LUMO_gap".format(level), value=lumo-homo, source="orca_geometry_optimization"))
        return ajr


def main(session: Session, n):
    print("=" * 10, "READING STRUCTURE ORCA CALCULATION RESULTS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    # read only structures with orca_out property (successful calculation)
    sids_outfile = session.query(Structure.id, Structure.orca_out).filter(Structure.orca_out != None).all()
    # define blocks
    blocks = [
        MoEnergies("mo_energy", 2)

    ]
    # read properties of each structure
    entries = []
    for sid, outfile in sids_outfile:
        sid = sid[0]
        entries += file_to_sql(outfile, blocks)
    session.add_all(entries)
    session.commit()
    print("ALL DONE")
