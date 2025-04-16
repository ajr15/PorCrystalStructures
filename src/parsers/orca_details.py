import pandas as pd
import os
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from typing import List
from read_to_sql import StructureProperty, Structure
import config

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

    def to_sql(self, sid: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        pass

class FinishedNormally (Block):

    def block_start(self, line: str) -> bool: 
        """Checks if a line is the start of a block"""
        return "****ORCA TERMINATED NORMALLY****" in line

    def block_ends(self, line: str) -> bool:
        """Checks if a line is the end of a block"""
        return "****ORCA TERMINATED NORMALLY****" in line

    def to_sql(self, sid: str) -> List[StructureProperty]:
        """Parse block content to a list of StructureProperty SQL entries"""
        value = len(self._content)
        return [StructureProperty(structure=sid, property="finished_normally", value=value, source="orca_geometry_optimization")]

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


def file_to_sql(path: str, blocks: List[Block]) -> List[StructureProperty]:
    # read all file contents to blocks
    blocks = read_file_to_blocks(path, blocks)
    # get the structure ID from path
    sid = os.path.split(path)[-1].split("_")[0]
    ajr = []
    for b in blocks:
        ajr += b.to_sql(sid)
    return ajr

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

def main(session: Session, n):
    print("=" * 10, "READING STRUCTURE ORCA CALCULATION DETAILS", "=" * 10)
    if n > 1:
        print("WARNING: you requested more than 1 process for this parser, it cannot be parallelized, so we use 1.")
    update_structure_schema(session)
    orca_out_dir = os.path.join(config.DATA_DIR, "dft")
    sids = session.query(Structure.id).all()
    for sid in sids:
        sid = sid[0]
        outdir = os.path.join(orca_out_dir, sid + "_0_out")
        if not os.path.isdir(outdir):
            continue
        outfile = os.path.join(outdir, sid + "_0.out")
        xyzfile = os.path.join(outdir, sid + "_0.xyz")
        # we add to sql only successful output files
        finished_normally = file_to_sql(outfile, [FinishedNormally("")])[0]
        if finished_normally.value == 1:
            # update details
            session.query(Structure).filter(Structure.id == sid).update({"orca_out": outfile, "orca_xyz": xyzfile})
    session.commit()
    print("ALL DONE")

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
            finished_normally = file_to_sql(path, [FinishedNormally("")])[0]
            if finished_normally.value == 0:
                continue
            entries = file_to_sql(path, [MoEnergies("", 2)])
            print("***", fname, "***")
            for e in entries:
                print(e.property, e.value)

