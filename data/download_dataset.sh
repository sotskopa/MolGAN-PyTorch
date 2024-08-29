#!/bin/bash
mkdir -p data

wget -P data http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz 
tar xvzf data/gdb9.tar.gz -C data
rm data/gdb9.tar.gz

wget -P data https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
wget -P data https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz