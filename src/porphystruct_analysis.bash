# script to use PorphyStruct to analyze corrole and porphyrin non-planarity
for file in $CRYSTAL_DATA_DIR/xyz/dft/*.xyz; do
    echo $file
    ./porphystruct/PorphyStruct.CLI analyze -x $file
    mv $CRYSTAL_DATA_DIR/xyz/dft/*.json $CRYSTAL_DATA_DIR/nonplanarity/dft/
    rm $CRYSTAL_DATA_DIR/xyz/dft/*.md
done