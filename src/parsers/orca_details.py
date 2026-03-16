import pandas as pd
import os
from shutil import copyfile
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from typing import List
from src.sqlmodels import StructureProperty, Structure
from src import config
from src.parsers.BaseParser import BaseParser
from src.orca_utils import FinishedNormally, file_to_sql

def update_structure_schema(session: Session):
    """Update structure schema if necessary"""
    inspector = inspect(session.bind)
    existing_columns = set(c['name'] for c in inspector.get_columns(Structure.__tablename__))
    # Get model columns
    model_columns = {col.name: col for col in Structure.__table__.columns}
    missing_columns = set(model_columns.keys()) - existing_columns
    # Add missing columns using session connection
    conn = session.connection()
    for col_name in missing_columns:
        col = model_columns[col_name]
        col_type = col.type.compile(dialect=session.bind.dialect)
        nullable = "NULL" if col.nullable else "NOT NULL"
        default = f"DEFAULT {col.default.arg}" if col.default is not None else ""
        sql = f"ALTER TABLE {Structure.__tablename__} ADD COLUMN {col.name} {col_type} {nullable} {default}"
        conn.execute(text(sql))
    session.commit()

# def main(session: Session, n):
#     print("=" * 10, "READING STRUCTURE ORCA CALCULATION DETAILS", "=" * 10)
#     if n > 1:
#         print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
#     update_structure_schema(session)
#     orca_out_dir = os.path.join(config.DATA_DIR, "dft")
#     orca_xyz_dir = os.path.join(config.DATA_DIR, "xyz", "dft")
#     sids = session.query(Structure.id).all()
#     for i, sid in enumerate(sids):
#         sid = sid[0]
#         print(sid, "({} out of {})".format(i + 1, len(sids)))
#         outdir = os.path.join(orca_out_dir, sid + "_0_out")
#         if not os.path.isdir(outdir):
#             continue
#         outfile = os.path.join(outdir, sid + "_0.out")
#         xyzfile = os.path.join(outdir, sid + "_0.xyz")
#         # we add to sql only successful output files
#         finished_normally = file_to_sql(outfile, [FinishedNormally("")], "")[0]
#         if finished_normally.value == 1:
#             # update details
#             session.query(Structure).filter(Structure.id == sid).update({"orca_out": outfile, "orca_xyz": xyzfile})
#             # copy file to ORCA output file dir
#             copyfile(xyzfile, orca_xyz_dir + "/" + sid + "_0.xyz")
#     session.commit()
#     print("ALL DONE")

class Parser (BaseParser):

    name = "orca_details"
    source_prefix = "orca_details/"

    def parse(self, session: Session, n: int):
        """Parse the data to SQL entries"""
        update_structure_schema(session)
        orca_out_dir = os.path.join(config.DATA_DIR, "dft")
        orca_xyz_dir = os.path.join(config.DATA_DIR, "xyz", "dft")
        sids = session.query(Structure.id).all()
        for i, sid in enumerate(sids):
            sid = sid[0]
            print(sid, "({} out of {})".format(i + 1, len(sids)))
            outdir = os.path.join(orca_out_dir, sid + "_0_out")
            if not os.path.isdir(outdir):
                continue
            outfile = os.path.join(outdir, sid + "_0.out")
            xyzfile = os.path.join(outdir, sid + "_0.xyz")
            # we add to sql only successful output files
            finished_normally = file_to_sql(outfile, [FinishedNormally("")], "")[0]
            if finished_normally.value == 1:
                # update details
                session.query(Structure).filter(Structure.id == sid).update({"orca_out": outfile, "orca_xyz": xyzfile})
                # copy file to ORCA output file dir
                copyfile(xyzfile, orca_xyz_dir + "/" + sid + "_0.xyz")
        session.commit()
        # it does not return any new entries to the database, only updating existing ones
        return []
    
    def delete(self, session: Session):
        """Remove the updadtes to the database made by the orca details parser"""
        session.query(Structure).update({"orca_out": None, "orca_xyz": None})
        session.commit()
        # also clear out the xyz directory
        orca_xyz_dir = os.path.join(config.DATA_DIR, "xyz", "dft")
        if os.path.isdir(orca_xyz_dir):
            os.system("rm -r " + orca_xyz_dir)
            os.mkdir(orca_xyz_dir)


if __name__ == "__main__":
    orca_out_dir = os.path.join(config.DATA_DIR, "dft")
    for dirname in os.listdir(orca_out_dir):
        dirpath = os.path.join(orca_out_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith("_0.out"):
                continue
            path = os.path.join(dirpath, fname)
            # read only files that ended normally
            finished_normally = file_to_sql(path, [FinishedNormally("")], "")[0]
            if finished_normally.value == 0:
                continue
            # entries = file_to_sql(path, [MoEnergies("", 2)])
            # print("***", fname, "***")
            # for e in entries:
            #     print(e.property, e.value)

