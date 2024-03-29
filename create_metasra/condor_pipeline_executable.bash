#!/bin/bash

source /ua/mnbernstein/software/python2_env/bin/activate 

results_file=$1
samples_list=$2
assay=$3
species=$4

tar -zxf create_metasra_condor_bundle.$assay.$species.tar.gz
export PYTHONPATH=$(pwd)/create_metasra_condor_bundle.$assay.$species:$PYTHONPATH
export PYTHONPATH=$(pwd)/create_metasra_condor_bundle.$assay.$species/map_sra_to_ontology:$PYTHONPATH
export PYTHONPATH=/ua/mnbernstein/projects/tbcp/metadata/ontology/src/bktree:$PYTHONPATH

python $(pwd)/create_metasra_condor_bundle.$assay.$species/condor_run_pipeline.py -s $samples_list -m $(pwd)/create_metasra_condor_bundle.$assay.$species/sample_to_raw_metadata.$assay.$species.json -o $results_file


