#!/bin/bash

WD=$(pwd)
PUNKT_DIR=$NLTK_DATA/tokenizers/punkt

mkdir -p $PUNKT_DIR

cd $PUNKT_DIR

wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
unzip punkt.zip
rm punkt.zip

cd $WD