# script to read the structure details of each structure
import os
from typing import List
from sqlalchemy import delete
from src.sqlmodels import Structure
from src import utils, config
from src.parsers.BaseParser import BaseParser, Session

def find_by_key(cif_text: str, key: str, islist: bool=False):
    lines = cif_text.strip().split('\n')
    for i, line in enumerate(lines):
        if line.startswith(key):
            if not islist:
                # Return the row without the key name
                return line[len(key):].strip().replace("'", "").replace("\"", "")
            else:
                # Return all rows below the key row until a row starts with _
                result = []
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('_'):
                        break
                    if lines[j].strip():
                        result.append(lines[j].replace("'", "").replace("\"", ""))
                return result
    
    return None

CIF_KEYS = {
    "_audit_creation_date": False,
    "_audit_creation_method": False,
    "_database_code_CSD": False,
    "_database_code_depnum_ccdc_archive": False,
    "_chemical_formula_sum": False,
    "_chemical_formula_moiety": False,
    "_journal_volume": False,
    "_journal_year": False,
    "_journal_page_first": False,
    "_journal_name_full": False,
    "_publ_author_name": True,
}

def search_doi(journal_name, journal_volume, journal_year, journal_page_first, authors):
    """
    Search for a DOI using bibliographic data via CrossRef API.
    
    Args:
        journal_name: Full journal name
        journal_volume: Journal volume number
        journal_year: Publication year
        journal_page_first: First page number
        authors: List of author names
    
    Returns:
        DOI string if found, None otherwise
    """
    import requests
    
    if not journal_name or not journal_year:
        return None
    
    # Prepare CrossRef API query
    first_author = authors[0].split(',')[0].strip() if authors and len(authors) > 0 else ""
    
    query_parts = []
    if first_author:
        query_parts.append(first_author)
    if journal_name:
        query_parts.append(journal_name)
    if journal_volume:
        query_parts.append(str(journal_volume))
    if journal_page_first:
        query_parts.append(str(journal_page_first))
    
    query_params = {
        'query': ' '.join(query_parts),
        'rows': 5
    }
    
    if journal_year:
        query_params['filter'] = f"from-pub-date:{journal_year},until-pub-date:{journal_year}"
    
    try:
        response = requests.get(
            'https://api.crossref.org/works',
            params=query_params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('message', {}).get('items'):
            for item in data['message']['items']:
                # Check if volume and page match
                if (journal_volume and str(journal_volume) in str(item.get('volume', ''))) or \
                   (journal_page_first and str(journal_page_first) in str(item.get('page', ''))):
                    return item.get('DOI')
            
            # Return first result if no exact match
            return data['message']['items'][0].get('DOI')
    
    except Exception as e:
        print(f"Error searching for DOI: {e}")
    
    return None


def read_structures(path: str) -> List[Structure]:
    """read all structures from a given type"""
    ajr = []
    for fname in os.listdir(path):
        xyz = os.path.join(path, fname)
        sid = fname.split("_")[0]
        print(sid)
        mol = utils.get_molecule(xyz)
        smiles = utils.mol_to_smiles(mol)
        cif = os.path.join(config.DATA_DIR, "ccdc_data", "cif", "porphyrins", sid + '.cif')
        if os.path.isfile(cif):
            with open(cif, "r") as f:
                cif_text = f.read()
            metadata = {k[1:]: find_by_key(cif_text, k, CIF_KEYS[k]) for k in CIF_KEYS}
            doi = search_doi(metadata["journal_name_full"], metadata["journal_volume"], metadata["journal_year"], metadata["journal_page_first"], metadata["publ_author_name"])
        else:
            print("No cif file for", sid)
            metadata = None
            doi = None
            cif = None
        ajr.append(Structure(id=sid, xyz=xyz, smiles=smiles, cif=cif, structure_metadata=metadata, doi=doi))
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