import os
import shutil
import sys

# Add the parent directory to the python path to allow imports of other modules
# This replaces the need to manually set PYTHONPATH
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# It is assumed that the following imported scripts can be run by calling a main() function.
# If they run on import, the .main() calls can be removed.
import download_ontologies
import reformat_cellosaurus
import download_specialist_lexicon
import build_bk_tree
import link_ontologies
import superterm_linked_terms
import generate_implications

def main():
    """Runs the setup process for the map_sra_to_ontology pipeline."""

    # Download ontologies
    print("Downloading ontologies...")
    download_ontologies.main()

    # Reformat Cellosaurus
    print("Reformatting Cellosaurus...")
    reformat_cellosaurus.main()

    # Download SPECIALIST Lexicon
    print("Downloading SPECIALIST Lexicon...")
    download_specialist_lexicon.main()

    # Build BK-tree for fuzzy string matching
    print("Building the BK-tree from the ontologies...")
    fuzzy_index_path = '../map_sra_to_ontology/fuzzy_matching_index'
    os.makedirs(fuzzy_index_path, exist_ok=True)
    
    build_bk_tree.main()
    
    os.replace('fuzzy_match_bk_tree.pickle', os.path.join(fuzzy_index_path, 'fuzzy_match_bk_tree.pickle'))
    os.replace('fuzzy_match_bk_tree_candidate_mentions.pickle', os.path.join(fuzzy_index_path, 'fuzzy_match_bk_tree_candidate_mentions.pickle'))
    os.replace('fuzzy_match_string_data.json', os.path.join(fuzzy_index_path, 'fuzzy_match_string_data.json'))
 
    # Link the terms between ontologies
    print("Linking ontologies...")
    link_ontologies.main()
    superterm_linked_terms.main()
    
    metadata_path = '../map_sra_to_ontology/metadata'
    shutil.copy('term_to_superterm_linked_terms.json', metadata_path)

    # Generate cell-line to disease implications
    print("Generating cell-line to disease implications...")
    generate_implications.main()
    shutil.copy('cellline_to_disease_implied_terms.json', metadata_path)

    print("Setup complete.")

if __name__ == "__main__":
    main()