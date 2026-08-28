import os

from . import (download_ontologies, 
               reformat_cellosaurus,
               download_specialist_lexicon,
               build_bk_tree,
               link_ontologies,
               superterm_linked_terms,
               generate_implications,
               )

from metasra.map_sra_to_ontology.config import Config

def generate_metasra_reference(ref_path):
    """Runs the setup process for the map_sra_to_ontology pipeline."""
    os.makedirs(ref_path, exist_ok=True)

    # Create a config object
    config = Config(ref_path)

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
