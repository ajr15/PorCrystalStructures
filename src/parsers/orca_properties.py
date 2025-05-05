import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
from read_to_sql import StructureProperty, Structure
from parsers.orca_details import Block, file_to_sql, FinishedNormally

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
            # print("*", line)
            data.append([float(line.split()[1]), float(line.split()[-1])])
        return pd.DataFrame(data, columns=["occ", "E"])

    def single_entry(self, sid: str, level: str, energy: float, source: str) -> StructureProperty:
        return StructureProperty(structure=sid, property="{}_energy".format(level), value=energy, source="orca_property/{}".format(source))

    def to_sql(self, sid: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        df = self.to_dataframe()
        ajr = []
        for level in range(self.n_levels + 1):
            homo = df[df["occ"] == 1].iloc[-(level + 1), 1]
            ajr.append(self.single_entry(sid, "HOMO-{}".format(level), homo, self.source))
            lumo = df[df["occ"] == 0].iloc[level, 1]
            ajr.append(self.single_entry(sid, "LUMO+{}".format(level), lumo, self.source))
            if level == 0:
                ajr.append(StructureProperty(structure=sid, property="HOMO-LUMO_gap".format(level), value=lumo-homo, source="orca_property/{}".format(self.source)))
        return ajr
    

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

    def to_sql(self, sid: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries."""
        energy_line = self._content[0]
        energy = float(energy_line.split()[4])
        return [
            StructureProperty(
                structure=sid,
                property="final_energy",
                value=energy,
                source="orca_property/{}".format(self.source),
            )
        ]

def read_electronic_structure(base_dir: str, sid: str) -> List[StructureProperty]:
    entries = []
    for multiplicity in [1, 3, 5]:
        dir_pattern = os.path.join(base_dir, f"{sid}_0_S{multiplicity}_out")
        if os.path.exists(dir_pattern):
            output_file = os.path.join(dir_pattern, f"{sid}_0_S{multiplicity}.out")
            if os.path.isfile(output_file):
                finished_normally = file_to_sql(output_file, [FinishedNormally("")])[0]
                if finished_normally.value == 0:
                    continue
                blocks = [
                    MoEnergies("mo_energy", f"S{multiplicity}", 2),
                    FinalEnergy("final_energy", f"S{multiplicity}")
                ]
                entries += file_to_sql(output_file, blocks)
    return entries


def main(session: Session, n):
    print("=" * 10, "READING STRUCTURE ORCA CALCULATION RESULTS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    # read only structures with orca_out property (successful calculation)
    sids_outfile = session.query(Structure.id, Structure.orca_out).filter(Structure.orca_out != None).all()
    # define blocks
    blocks = [
        MoEnergies("mo_energy", "base_calc", 2),
        FinalEnergy("final_energy", "base_calc")
    ]
    # read properties of each structure
    entries = []
    for sid, outfile in sids_outfile:
        print("reading", sid)
        print("reading base calculation")
        entries += file_to_sql(outfile, blocks)
        print("reading multiplicity calculations")
        entries += read_electronic_structure("data/electron_structure", sid)
    session.add_all(entries)
    session.commit()
    print("ALL DONE")
