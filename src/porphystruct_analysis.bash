# script to use PorphyStruct to analyze corrole and porphyrin non-planarity
for file in $CRYSTAL_DATA_DIR/curated/$1/*; do
    echo $file
    ./porphystruct/PorphyStruct.CLI analyze -x $file
    mv $CRYSTAL_DATA_DIR/curated/$1/*.json $CRYSTAL_DATA_DIR/nonplanarity/$1/
    rm $CRYSTAL_DATA_DIR/curated/$1/*.md
done