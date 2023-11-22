datadir="datasets"

mkdir datasets/processed/ViNewsQA
mkdir datasets/processed/ViQuAD

unzip "$datadir/ViNewsQA.zip" -d "$datadir"
chmod a+rx "$datadir"/ViNewsQA
unzip "$datadir/ViQuAD1.0.zip" -d "$datadir"
chmod a+rx "$datadir"/ViQuAD1.0
