#!/bin/bash

# Set the pythonpath as per README instructions

MAP_SRA_TO_ONTOLOGY_PATH=$(realpath map_sra_to_ontology)
BKTREE_PATH=$(realpath bktree)
PIP_PACKAGES_PATH=$(realpath pip-packages)

export PYTHONPATH="$MAP_SRA_TO_ONTOLOGY_PATH:$BKTREE_PATH:$PIP_PACKAGES_PATH:$PYTHONPATH"

export NLTK_DATA=$(realpath nltk_data)