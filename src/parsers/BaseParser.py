# a base class for all parsers
from abc import ABC, abstractmethod
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.sqlmodels import Structure
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class BaseParser (ABC):

    name = "parser"
    source_prefix = "parser/"

    @abstractmethod
    def parse(self, session: Session, n: int):
        """Parse the data to SQL entries"""
        pass

    def delete(self, session: Session):
        """Remove entries originating from this parser"""
        stmt = f"DELETE FROM structure_properties WHERE source LIKE '{self.source_prefix}%'"
        session.execute(text(stmt))
        session.commit()
        stmt = f"DELETE FROM substituents_properties WHERE source LIKE '{self.source_prefix}%'"
        session.execute(text(stmt))
        session.commit()

    def execute(self, session: Session, n: int, clear_db: bool):
        """Internal method to execute parser. clear_db is bool to make sure DB is clean of previous reads before parsing"""
        if clear_db: 
            self.delete(session)
        entries = self.parse(session, n)
        # add prefix to entries
        for entry in entries:
            if hasattr(entry, "source"):
                entry.source = self.source_prefix + entry.source
        session.add_all(entries)
        session.commit()


class StructureParser (BaseParser):

    """Base class for parsers that run on per-structure basis (sid)"""

    @abstractmethod
    def parse_structure(self, session: Session, sid: str) -> tuple:
        """Parse a single structure (given by structure id), return a tuple of list of sql entries and messages"""
        pass

    def fetch_structure_ids(self, session: Session):
        """Method to fetch all structure IDs in a database"""
        return [x[0] for x in session.query(Structure.id).all()]
    
    def _parse_structure(self, args):
        connection_str, sid = args
        try:
            engine = create_engine(connection_str)
            Session = sessionmaker(bind=engine)
            session = Session()
            return self.parse_structure(session, sid)
        except Exception as e:
            print("Errors with", sid)
            raise e

    def parse(self, session: Session, n: int):
        """Run the structure parser in parallel"""
        connection_string = str(session.get_bind().engine.url)
        sids = self.fetch_structure_ids(session)
        with Pool(processes=n) as pool:
            args = [(connection_string, sid) for sid in sids]
            ajr = list(tqdm(pool.imap(self._parse_structure, args), total=len(sids), desc="Processing structures"))
            results = [x[0] for x in ajr]
            messages = list(chain(*[x[-1] for x in ajr]))
        print("Done!")
        if len(messages) > 0:
            print("== Run Messages ==")
            for m in messages:
                print(m)
        return list(chain(*results))