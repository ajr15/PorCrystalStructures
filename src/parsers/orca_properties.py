import pandas as pd
import os
from typing import List
from sqlalchemy.orm import Session
from src.sqlmodels import StructureProperty, Structure
from src.orca_utils import Block, file_to_sql, FinishedNormally, MoEnergies, FinalEnergy
from src.parsers.BaseParser import StructureParser

OUTPUT_BLOCKS = [
    MoEnergies("mo_energy", 2),
    FinalEnergy("final_energy")
]

def read_electronic_structure(base_dir: str, sid: str) -> List[StructureProperty]:
    entries = []
    for multiplicity in [1, 3, 5]:
        dir_pattern = os.path.join(base_dir, f"{sid}_0_S{multiplicity}_out")
        if os.path.exists(dir_pattern):
            output_file = os.path.join(dir_pattern, f"{sid}_0_S{multiplicity}.out")
            if os.path.isfile(output_file):
                finished_normally = file_to_sql(output_file, [FinishedNormally("")], "")[0]
                if finished_normally.value == 0:
                    continue
                entries += file_to_sql(output_file, OUTPUT_BLOCKS, "S" + str(multiplicity))
    return entries


# def main(session: Session, n):
#     print("=" * 10, "READING STRUCTURE ORCA CALCULATION RESULTS", "=" * 10)
#     if n > 1:
#         print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
#     # read only structures with orca_out property (successful calculation)
#     sids_outfile = session.query(Structure.id, Structure.orca_out).filter(Structure.orca_out != None).all()
#     # define blocks
#     blocks = [
#         MoEnergies("mo_energy", "base_calc", 2),
#         FinalEnergy("final_energy", "base_calc")
#     ]
#     # read properties of each structure
#     entries = []
#     for i, (sid, outfile) in enumerate(sids_outfile):
#         print("reading", sid, "({} out of {})".format(i + 1, len(sids_outfile)))
#         print("reading base calculation")
#         entries += file_to_sql(outfile, blocks)
#         print("reading multiplicity calculations")
#         entries += read_electronic_structure("data/electron_structure", sid)
#     session.add_all(entries)
#     session.commit()
#     print("ALL DONE")

class Parser (StructureParser):

    name = "orca_properties"
    source_prefix = "parser/"


    def parse_structure(self, session: Session, sid: str) -> tuple:
        """Parse a single structure (given by structure id), return a tuple of list of sql entries and messages"""
        outfile = session.query(Structure.orca_out).filter(Structure.id == sid).all()[0][0]
        if outfile is None:
            return [], ["INFO: no output file for " + sid]
        entries = file_to_sql(outfile, OUTPUT_BLOCKS, source="base_calculation")
        entries += read_electronic_structure("data/electron_structure", sid)
        return entries, []
        


