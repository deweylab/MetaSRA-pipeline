import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="metasra",
        description="Set up or run the MetaSRA ontology mapping pipeline.",
    )
    modes = parser.add_subparsers(dest="mode", required=True)

    setup_parser = modes.add_parser("setup", help="Download and configure reference resources.")
    setup_parser.add_argument(
        "ref_path",
        nargs="?",
        type=Path,
        default=Path("./metasra_ref"),
        help="Path to where reference files are stored (default: %(default)s).",
    )

    run_parser = modes.add_parser("run", help="Map sample metadata to ontology terms.")
    run_parser.add_argument(
        "input_f",
        help="JSON file with a list of sample metadata objects represented as key-value pairs",
    )
    run_parser.add_argument(
        "-r",
        "--ref_path",
        default="metasra_ref",
        help="Path to where reference files are stored (default: %(default)s).",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file to which to write results (default: stdout)",
    )
    run_parser.add_argument(
        "-f",
        "--fine_grained",
        action="store_true",
        help="Output mappings stratified by explicit and consequent",
    )
    run_parser.add_argument(
        "-c",
        "--candidate_mentions",
        action="store_true",
        help="Output candidate mentions only",
    )

    args = parser.parse_args()

    if args.mode == "setup":
        from metasra.setup_map_sra_to_ontology.setup import generate_metasra_reference

        generate_metasra_reference(args.ref_path)
    else:
        from metasra.map_sra_to_ontology.run_pipeline import run_pipeline

        run_pipeline(
            args.input_f,
            ref_path=args.ref_path,
            output=args.output,
            fine_grained=args.fine_grained,
            candidate_mentions=args.candidate_mentions,
        )
