#!/bin/bash

results_file=$1
samples_list=$2

tar -zxf create_metasra_condor_bundle.tar.gz
export PYTHONPATH=$(pwd)/create_metasra_condor_bundle:$PYTHONPATH
export PYTHONPATH=$(pwd)/create_metasra_condor_bundle/map_sra_to_ontology:$PYTHONPATH
export PYTHONPATH=/ua/mnbernstein/projects/tbcp/metadata/ontology/src/bktree:$PYTHONPATH

~/software/python_environment/python_build/bin/python $(pwd)/create_metasra_condor_bundle/condor_run_pipeline.py -s $samples_list -m $(pwd)/create_metasra_condor_bundle/sample_to_raw_metadata.json -o $results_file


