#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <structure_id> <type (dft|crystal)>"
    exit 1
fi

STRUCTURE_ID=$1
TYPE=$2

# Validate the type argument
if [[ "$TYPE" != "dft" && "$TYPE" != "crystal" ]]; then
    echo "Error: Type must be either 'dft' or 'crystal'."
    exit 1
fi

# Define the input and output directories
INPUT_DIR="$CRYSTAL_DATA_DIR/xyz/$TYPE"
OUTPUT_DIR="$CRYSTAL_DATA_DIR/nonplanarity/$TYPE"

# Check if the input file exists
INPUT_FILE="$INPUT_DIR/${STRUCTURE_ID}_0.xyz"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File $INPUT_FILE does not exist."
    exit 1
fi

# Run PorphyStruct analysis
echo "Analyzing $INPUT_FILE..."
$CRYSTAL_SRC_DIR/porphystruct/PorphyStruct.CLI analyze -x "$INPUT_FILE"

# Move the generated JSON file to the output directory
mv "$INPUT_DIR/${STRUCTURE_ID}_0_analysis.json" "$OUTPUT_DIR/"

# Remove the generated Markdown file
rm "$INPUT_DIR/${STRUCTURE_ID}_0_analysis.md"

echo "Analysis complete. Results moved to $OUTPUT_DIR."