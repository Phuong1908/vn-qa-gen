datadir="Datasets"

mkdir Datasets/processed
mkdir Datasets/processed/ViNewsQA
mkdir Datasets/processed/ViQuAD

unzip "$datadir/ViNewsQA.zip" -d "$datadir"
chmod a+rx "$datadir"/ViNewsQA
unzip "$datadir/ViQuAD1.0.zip" -d "$datadir"
chmod a+rx "$datadir"/ViQuAD1.0
