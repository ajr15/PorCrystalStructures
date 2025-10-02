COD_SEARCH_API_ENDPOINT = "https://www.crystallography.net/cod/result"
COD_FILE_API_ENDPOINT = "https://www.crystallography.net/cod/$CODID.cif"

import os

PROJECT_SRC_DIR = os.environ["CRYSTAL_SRC_DIR"] if "CRYSTAL_SRC_DIR" in os.environ else ""
DATA_DIR = os.environ["CRYSTAL_DATA_DIR"] if "CRYSTAL_DATA_DIR" in os.environ else ""
DISPLACED_STRUCTS_DIR = os.environ["CRYSTAL_SRC_DIR"] + "/displaced_structures" if "CRYSTAL_SRC_DIR" in os.environ else ""