# script to parse data from all the sources (XYZ, non-planarity...) to a single SQLite database
# this is to ensure a consistant and convenient access to processed data, to be used in statistical models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.sqlmodels import SqlBase
from src import config

def fetch_parser(parser_name: str, raise_error: bool):
    # trying to import the main function from the parsing file
    try:
        parser_module = importlib.import_module("src.parsers.{}".format(parser_name))
        if hasattr(parser_module, "Parser"):
            return getattr(parser_module, "Parser")()
        else:
            raise ImportError()
    except ImportError or AttributeError:
        if raise_error:
            raise ValueError("The required parser ({}) does not exist or does not contain a 'Parser' object".format(parser_name))
        else:
            return

def all_parsers():
    # method to find all available parsers
    module_dir = os.path.join(config.PROJECT_SRC_DIR, "src", "parsers")
    ajr = []
    for fname in os.listdir(module_dir):
        if fname.endswith(".py"):
            parser = fetch_parser(fname[:-3], raise_error=False)
            if parser is not None:
                ajr.append(fname[:-3])
    return ajr


def run_parser(parser_name, db_path, nworkers, clear_db):
    # connect to the databse
    engine = create_engine("sqlite:///{}".format(db_path))
    SqlBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    # run the parsing function
    parser = fetch_parser(parser_name, raise_error=True)
    parser.execute(session, nworkers, clear_db)
    

if __name__ == "__main__":
    # when running the script, one can choose what to read to the SQL database
    import argparse
    import importlib
    import os
    parser = argparse.ArgumentParser("Main script to read data into the main SQL database")
    parser.add_argument("parser", type=str, nargs="+", help="name(s) of the parser(s) to use (use 'all' to run all parsers or '_parser' to run all but a parser)")
    parser.add_argument("-db", "--database", type=str, default="main.db", help="path to the database file")
    parser.add_argument("-n", "--n_workers", type=int, default=1, help="number of processes to be used in parsing")
    parser.add_argument("--clear_db", default=True, action="store_false", help="clear db from previous parser read (default: True)")
    parser.add_argument("--delete_db", default=False, action="store_true", help="delete existing database file (default: False)")
    args = parser.parse_args()
    if "all" in args.parser:
        parsers = [parser for parser in all_parsers()]
        for excluded_parser in [p[1:] for p in args.parser if p.startswith("_")]:
            parsers = [p for p in parsers if p != excluded_parser]
    else:
        parsers = args.parser
    # Ensure specific order of parsers if they exist
    ordered_parsers = []
    for required_parser in ["structure_details", "substituents", "orca_details"]:
        if required_parser in parsers:
            ordered_parsers.append(required_parser)
            parsers.remove(required_parser)
    parsers = ordered_parsers + parsers
    # if requested delete, remove the database
    if args.delete_db and os.path.isfile(args.database):
        os.remove(args.database)
    for parser in parsers:
        print("=== running with", parser, "===")
        run_parser(parser, args.database, args.n_workers, args.clear_db)
