import argparse
from pathlib import Path

from setup_map_sra_to_ontology.setup import generate_metasra_reference

def main():
    """Runs the setup process for the map_sra_to_ontology pipeline."""
    # this script takes one optional argument, the path to where all reference files should be stored.

    parser = argparse.ArgumentParser(description="Setup the map_sra_to_ontology pipeline.")
    parser.add_argument("ref_path", nargs="?", type=Path,
                        default="./metasra_ref", 
                        help="Path to where all reference files should be stored.")
    args = parser.parse_args()

    generate_metasra_reference(args.ref_path)

if __name__ == "__main__":
    main()
