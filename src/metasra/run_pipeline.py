########################################################################
#
# Run the ontology mapping pipeline on a set of key-value pairs 
# that describe a biological sample
#
########################################################################

import argparse

from metasra.map_sra_to_ontology.run_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run the ontology mapping pipeline on a sample metadata JSON file."
    )
    parser.add_argument("input_f", help="JSON file with list of sample metadata, each sample represented as an object with key-value pairs")
    parser.add_argument("-r", "--ref_path", default="metasra_ref",
                        help="Path to where reference files are stored (default: %(default)s).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file to which to write results (default: stdout)")
    parser.add_argument("-f", "--fine_grained", action="store_true",
                        help="Output mappings stratified by explicit and consequent")
    parser.add_argument("-c", "--candidate_mentions", action="store_true",
                        help="Output candidate mentions only")
    args = parser.parse_args()

    run_pipeline(
        args.input_f,
        ref_path=args.ref_path,
        output=args.output,
        fine_grained=args.fine_grained,
        candidate_mentions=args.candidate_mentions
    )

if __name__ == "__main__":
    main()
