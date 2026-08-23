import pandas as pd
import os
from src import config, utils
from src.sqlmodels import Structure, StructureProperty
from src.featurizers import StructurePropertyFeaturizer
import requests
import json

def find_abstract_and_title(doi):
    """
    Given a DOI, returns the title and abstract of a paper.
    
    Args:
        doi: Digital Object Identifier string
        
    Returns:
        tuple: (title, abstract) or (None, None) if not found
    """
    try:
        # Use Crossref API to fetch metadata
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        message = data.get("message", {})
        
        title = message.get("title", [None])[0] if message.get("title") else None
        abstract = message.get("abstract", None)
        
        return title, abstract
    except Exception as e:
        print(f"Error fetching data for DOI {doi}: {e}")
        return None, None
    

def extract_journal_name(d):
    try:
        if pd.isna(d) or len(d) == 0:
            return None
        return json.loads(d)["journal_name_full"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    
def orca_property_names(session):
    names = session.query(StructureProperty.property).filter(StructureProperty.source == "parser/base_calculation").distinct().all()
    return [x[0] for x in names]

def orca_properties_df(session, sids, source: str):
    # reading the data from SQL
    props = orca_property_names(session)
    units = [None for _ in range(len(props))]
    feat = StructurePropertyFeaturizer(props, units, navalue=None, property_source=source)
    df = pd.DataFrame(data=feat.featurize(session, sids), columns=feat.feature_names, index=sids)
    df = df.dropna()
    return df

def get_electron_structure_data(sids):
    singlet = orca_properties_df(session, sids, "parser/S1")
    triplet = orca_properties_df(session, sids, "parser/S3")
    quintet = orca_properties_df(session, sids, "parser/S5")
    # df = complex_details_df(session, sids)
    df = pd.DataFrame(index=sids)
    df["singlet_energy"] = singlet["final_energy"]
    df["triplet_energy"] = triplet["final_energy"]
    df["quintet_energy"] = quintet["final_energy"]
    return df

def check_sid(sid):
    xyzdir = utils.get_directory("curated_xyz", "porphyrins")
    return os.path.exists(os.path.join(xyzdir, sid + "_0.xyz"))

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    dbpath = os.path.join(config.PROJECT_SRC_DIR, "main.db")
    engine = create_engine("sqlite:///{}".format(dbpath))
    session = sessionmaker(bind=engine)()

    df = pd.read_sql_query("SELECT * FROM structures", engine.connect())
    elect_details = get_electron_structure_data(df["id"])
    df = pd.merge(df, elect_details, left_on="id", right_index=True)
    
    df["journal"] = [extract_journal_name(d) for d in df["structure_metadata"]]

    valid_sids = [x for x in df["id"] if check_sid(x)]

    df = df[df['id'].isin(valid_sids)]

    df.to_csv("structure_details.csv")

    # for row in df.to_dict(orient="records"):
    #     print(row["id"], row["doi"])
    #     title, abstract = find_abstract_and_title(row["doi"])
    #     print(title)
    #     print(abstract)