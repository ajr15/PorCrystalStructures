# parser to read both NMR and NICS values from ORCA output
import os
import pandas as pd
from src.sqlmodels import StructureProperty
from src.parsers.BaseParser import BaseParser
from src import config


# ,Unnamed: 0,positive_current,negative_current,total_current,acid_sqrt(J^2),acid_current,bond_start,bond_end,mapped_bond_start,mapped_bond_end,width,height,sid

def row_to_entries(row: dict):
    """Calculate bond integrals for bonds within the macrocycle."""
    return [
        StructureProperty(
            structure=row["sid"], 
            property=f"current/mapped/{row['mapped_bond_start']}->{row['mapped_bond_end']}",
            value=row["positive_current"],
            source=f"h={row['width']}&w={row['height']}"
        ),
        StructureProperty(
            structure=row["sid"], 
            property=f"current/mapped/{row['mapped_bond_end']}->{row['mapped_bond_start']}",
            value=-row["negative_current"],
            source=f"h={row['width']}&w={row['height']}"
        ),
        StructureProperty(
            structure=row["sid"], 
            property=f"current/original/{row['bond_start']}->{row['bond_end']}",
            value=row["positive_current"],
            source=f"h={row['width']}&w={row['height']}"
        ),
        StructureProperty(
            structure=row["sid"], 
            property=f"current/original/{row['bond_end']}->{row['bond_start']}",
            value=-row["negative_current"],
            source=f"h={row['width']}&w={row['height']}"
        ),
        StructureProperty(
            structure=row['sid'], 
            property=f"acid/mapped/{row['mapped_bond_start']}->{row['mapped_bond_end']}",
            value=row['acid_sqrt(J^2)'],
            source=f"h={row['width']}&w={row['height']}"
        ),
        StructureProperty(
            structure=row['sid'], 
            property=f"acid/original/{row['bond_start']}->{row['bond_end']}",
            value=row['acid_sqrt(J^2)'],
            source=f"h={row['width']}&w={row['height']}"
        ),
    ]


class Parser (BaseParser):

    name = "gimic"
    source_prefix = "gimic/"


    def parse(self, session, n):
        gimic_df = pd.read_csv(os.path.join(config.PROJECT_SRC_DIR, "all_gimic.csv"))
        ajr = []
        for row in gimic_df.to_dict(orient="records"):
            ajr += row_to_entries(row)
        return ajr

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import pandas as pd
    engine = create_engine('sqlite:///main.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    parser = Parser()
    entries, _ = parser.parse_structure(session, "ATEWUT")
    print(_)
    data = []
    for entry in entries:
        print(entry.structure, entry.property, entry.value, entry.source)
        data.append([entry.structure, entry.property, entry.value, entry.source])
    df = pd.DataFrame(data, columns=["structure", "property", "value", "source"])
    df.to_csv("test.csv")