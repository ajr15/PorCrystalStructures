# script to parse data from all the sources (XYZ, non-planarity...) to a single SQLite database
# this is to ensure a consistant and convenient access to processed data, to be used in statistical models
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

SqlBase = declarative_base()

class Structure (SqlBase):

    """Simple details on each structure: its XYZ file, CIF, type (corrole, porphyrin)..."""

    __tablename__ = "structures"
    id = Column(String, primary_key=True)
    type = Column(String) # porphyrin or corrole
    xyz = Column(String)
    cif = Column(String)
    smiles = Column(String)


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

def run_parser(parser_name: str, db_path: str, nworkers: int):
    # trying to import the main function from the parsing file
    try:
        parser_module = importlib.import_module("parsers.{}".format(parser_name))
        parser_func = getattr(parser_module, "main")
    except ImportError:
        raise ValueError("The required parser ({}) does not exist or does not contain a 'main' function".format(parser_name))
    # connect to the databse
    engine = create_engine("sqlite:///{}".format(db_path))
    SqlBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    # run the parsing function
    parser_func(session, nworkers)
    

if __name__ == "__main__":
    # when running the script, one can choose what to read to the SQL database
    import argparse
    import importlib
    import os
    parser = argparse.ArgumentParser("Main script to read data into the main SQL database")
    parser.add_argument("parser", type=str, help="name of the parser to use")
    parser.add_argument("-db", "--database", type=str, default="main.db", help="path to the database file")
    parser.add_argument("-n", "--n_workers", type=int, default=1, help="number of processes to be used in parsing")
    parser.add_argument("--delete", type=bool, default=False, help="delete existing database file")
    args = parser.parse_args()
    if args.parser == "all":
        parsers = [fname.split(".")[0] for fname in os.listdir("parsers") if fname.endswith(".py") and not fname.startswith("_")]
        parsers.remove("structure_details")
        parsers.remove("substituents")
        parsers = ["structure_details", "substituents"] + parsers
    else:
        parsers = [args.parser]
    # if requested delete, remove the database
    if args.delete and os.path.isfile(args.database):
        os.remove(args.database)
    for parser in parsers:
        print("running with", parser)
        run_parser(parser, args.database, args.n_workers)
