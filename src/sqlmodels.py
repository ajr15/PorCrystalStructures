# script to parse data from all the sources (XYZ, non-planarity...) to a single SQLite database
# this is to ensure a consistant and convenient access to processed data, to be used in statistical models
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

SqlBase = declarative_base()

class Structure (SqlBase):

    """Simple details on each structure: its XYZ file, CIF, type (corrole, porphyrin)..."""

    __tablename__ = "structures"
    id = Column(String, primary_key=True)
    type = Column(String) # porphyrin or corrole
    xyz = Column(String)
    cif = Column(String)
    smiles = Column(String)
    orca_out = Column(String)
    orca_xyz = Column(String)


class Substituent (SqlBase):

    """Details on the substituents of a given macrocycle. a relationship table specifying relation between the substituents table and the details table"""

    __tablename__ = "substituents"
    id = Column(Integer, primary_key=True)
    structure = Column(String, ForeignKey("structures.id"))
    substituent = Column(String)
    position = Column(String) # metal, meso, beta or axial
    position_index = Column(Integer) # to specify index of each substituent (e.g. meta1, beta4...)
    atom_indicis = Column(String) # to specify the atomic indices of the substituent's atoms in the parent molecule (for easy future reference)


class SubstituentProperty (SqlBase):

    """Various calculated/measured properties of the structure's substituents"""

    __tablename__ = "substituents_properties"
    id = Column(Integer, primary_key=True)
    smiles = Column(String)
    property = Column(String)
    value = Column(Float)
    units = Column(String)
    source = Column(String) # calculated, experimental...
    structure = Column(String, ForeignKey("structures.id"), nullable=True) # optionally specify value of property for a given structure, good for metal charges for example
    position = Column(String) # optionally specify the position of the substituent in the macrocycle
    position_index = Column(Integer) # optionally specify the position of the substituent in the macrocycle


class StructureProperty (SqlBase):

    """Various calcualted / measured numerical properties on the structures. the table has the structure of a structure_id, property, value, source"""

    __tablename__ = "structure_properties"
    id = Column(Integer, primary_key=True)
    structure = Column(String, ForeignKey("structures.id"))
    property = Column(String)
    value = Column(Float)
    units = Column(String)
    source = Column(String) # calculated, experimental...

