#!/bin/bash

# TODO add the map_sra_to_ontology module to the PYTHONPATH
# Example:
#   export PYTHONPATH=<path to directory containing map_sra_to_ontology>:$PYTHONPATH

# TODO add the bktree module to the PYTHONPATH
# Example:
#   export PYTHONPATH=<path to directory containing bktree.py script>:$PYTHONPATH

# Download ontologies
echo "Downloading ontologies..."
python download_ontologies.py

# Download SPECIALIST Lexicon
echo "Downloading SPECIALIST Lexicon..."
python download_specialist_lexicon.py 

# Build BK-tree for fuzzy string matching
echo "Building the BK-tree from the ontologies..."
mkdir ../map_sra_to_ontology/fuzzy_matching_index
python build_bk_tree.py
mv fuzzy_match_bk_tree.pickle ../map_sra_to_ontology/fuzzy_matching_index 
mv fuzzy_match_string_data.json ../map_sra_to_ontology/fuzzy_matching_index

# Link the terms between ontologies 
echo "Linking ontologies..."
python link_ontologies.py
python superterm_linked_terms.py 
cp term_to_superterm_linked_terms.json ../map_sra_to_ontology/metadata

# Generate cell-line to disease implications
echo "Generating cell-line to disease implications..."
python generate_implications.py
cp cellline_to_disease_implied_terms.json ../map_sra_to_ontology/metadata


