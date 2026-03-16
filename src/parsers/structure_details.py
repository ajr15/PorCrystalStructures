# script to read the structure details of each structure
import os
from typing import List
from sqlalchemy import delete
from src.sqlmodels import Structure
from src import utils, config
from src.parsers.BaseParser import BaseParser, Session

def read_structures(path: str) -> List[Structure]:
    """read all structures from a given type"""
    ajr = []
    for fname in os.listdir(path):
        xyz = os.path.join(path, fname)
        sid = fname.split("_")[0]
        mol = utils.get_molecule(xyz)
        smiles = utils.mol_to_smiles(mol)
        ajr.append(Structure(id=sid, xyz=xyz, smiles=smiles))
    return ajr

# def main(session, n: int):
#     UPDATE_DB = True # update manually if needed, this updates the database other tables to only have indexed structures
#     print("=" * 10, "READING STRUCTURE DETAILS", "=" * 10)
#     if n > 1:
#         print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
#     path = os.path.join(config.DATA_DIR, "xyz", "crystal")
#     print("reading structures from", os.path.abspath(path))
#     if UPDATE_DB:
#         stmt = delete(Structure)
#         session.execute(stmt)
#     ajr = read_structures(path)
#     session.add_all(ajr)
#     session.commit()
#     if UPDATE_DB: 
#         print("dropping bad structure IDs from dataset")
#         # removing from StructureProperty table
#         stmt = "DELETE FROM structure_properties WHERE structure NOT IN (SELECT id FROM structures)"
#         session.execute(stmt)
#         # removing from Substituent table
#         stmt = "DELETE FROM substituents WHERE structure NOT IN (SELECT id FROM structures)"
#         session.execute(stmt, execution_options={"synchronize_session": False})
#         # removing from SubstituentProperty table
#         stmt = "DELETE FROM substituents_properties WHERE structure NOT IN (SELECT id FROM structures)"
#         session.execute(stmt, execution_options={"synchronize_session": False})
#         session.commit()
#     print("ALL DONE")

class Parser (BaseParser):

    name = "structure_details"
    source_prefix = ""

    def parse(self, session: Session, n: int):
        """Parse the data to SQL entries"""
        path = os.path.join(config.DATA_DIR, "xyz", "crystal")
        return read_structures(path)

    def delete(self, session: Session):
        stmt = delete(Structure)
        session.execute(stmt)