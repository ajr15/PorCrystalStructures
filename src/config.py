COD_SEARCH_API_ENDPOINT = "https://www.crystallography.net/cod/result"
COD_FILE_API_ENDPOINT = "https://www.crystallography.net/cod/$CODID.cif"

import os
DATA_DIR = os.environ("CRYSTAL_DATA_DIR")
DISPLACED_STRUCTS_DIR = os.environ("CRYSTAL_SRC_DIR") + "/displaced_structures"