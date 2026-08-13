import os
import shutil
import sys
from pathlib import Path

# Add the parent directory to the python path to allow imports of other modules
# This replaces the need to manually set PYTHONPATH
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# It is assumed that the following imported scripts can be run by calling a main() function.
# If they run on import, the .main() calls can be removed.
from . import (download_ontologies, 
               reformat_cellosaurus,
               download_specialist_lexicon,
               build_bk_tree,
               link_ontologies,
               superterm_linked_terms,
               generate_implications,
               )

from map_sra_to_ontology.config import Config

def main():
    """Runs the setup process for the map_sra_to_ontology pipeline."""
    # this script takes one optional argument, the path to where all reference files should be stored.
    # use argparse to handle this argument which does not require a flag
    import argparse
    parser = argparse.ArgumentParser(description="Setup the map_sra_to_ontology pipeline.")
    parser.add_argument("ref_path", nargs="?", type=Path,
                        default="./metasra_ref", 
                        help="Path to where all reference files should be stored.")
    args = parser.parse_args()

    os.makedirs(args.ref_path, exist_ok=True)

    # Create a config object
    config = Config(args.ref_path)

    # Download ontologies
    print("Downloading ontologies...")
    download_ontologies.main(config)

    # Reformat Cellosaurus
    print("Reformatting Cellosaurus...")
    reformat_cellosaurus.main(config)

    # Download SPECIALIST Lexicon
    print("Downloading SPECIALIST Lexicon...")
    download_specialist_lexicon.main(config)

    # Build BK-tree for fuzzy string matching
    print("Building the BK-tree from the ontologies...")
    build_bk_tree.main(config)

    # Link the terms between ontologies
    print("Linking ontologies...")
    link_ontologies.main(config)
    superterm_linked_terms.main(config)
    
    # Generate cell-line to disease implications
    print("Generating cell-line to disease implications...")
    generate_implications.main(config)

    print("Setup complete.")

if __name__ == "__main__":
    main()