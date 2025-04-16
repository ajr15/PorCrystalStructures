# script to use PorphyStruct to analyze corrole and porphyrin non-planarity
for file in $CRYSTAL_DATA_DIR/curated_xyz/*; do
    echo $file
    ./porphystruct/PorphyStruct.CLI analyze -x $file
    mv $CRYSTAL_DATA_DIR/curated_xyz/*.json $CRYSTAL_DATA_DIR/nonplanarity/
    rm $CRYSTAL_DATA_DIR/curated_xyz/*.md
done