#!/bin/bash

# Step 1: Get the directory where the setup script is located and use it as the project's source directory
PROJECT_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 2: Define a unique prefix for environment variables (for example, using PROJECT_NAME or your own)
PREFIX="CRYSTAL"

# Step 3: Set up the environment variable for the project's source directory
export ${PREFIX}_SRC_DIR="$PROJECT_SRC_DIR"

# Step 4: Add the "src" directory of the project to the PYTHONPATH
export PYTHONPATH="$PROJECT_SRC_DIR/src:$PYTHONPATH"

# Step 5: Set up the "data_dir" environment variable based on the project source directory
export ${PREFIX}_DATA_DIR="$PROJECT_SRC_DIR/data"

# Step 6: Set up the "main.db" file acesss
export ${PREFIX}_MAIN_DB="$PROJECT_SRC_DIR/main.db"

# Step 7: Create dedicated directories for analysis results
mkdir $PROJECT_SRC_DIR/models
mkdir $PROJECT_SRC_DIR/results

# Optional: Display the setup information for confirmation
echo "SETUP DONE SUCCESSFULLY !"
echo "Project source directory set to: ${PROJECT_SRC_DIR}"
echo "Data directory set to: ${PROJECT_SRC_DIR}/data"
