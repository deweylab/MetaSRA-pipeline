###############################################################
#   Kicked off by the Condor executable, this script runs the 
#   pipeline for a given list of sample accessions.
###############################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
import json
import sys
import  os
from os.path import realpath
from optparse import OptionParser

from map_sra_to_ontology import jsonio

def get_script_path():
    return os.path.dirname(realpath(sys.argv[0]))
#sys.path.append( join(get_script_path(), "..") )

sys.path.append(get_script_path())

import pipeline

def main():
    parser = OptionParser()
    parser.add_option(
        "-s", 
        "--sample_accessions_file", 
        help="Path to JSON containing sample accessions"
    )
    parser.add_option(
        "-m",
        "--metadata_input",
        help="JSON file mapping each sample to its raw key-value pairs"
    )
    parser.add_option(
        "-o", 
        "--pipeline_results_file", 
        help="File to which to write the matches."
    )
    (options, args) = parser.parse_args()

    sample_accs = None
    with open(options.sample_accessions_file, 'r') as f:
        j = json.load(f)
        sample_accs = j["sample_accessions"]

    with open(options.metadata_input, 'r') as f:
        sample_to_metadata = json.load(f)

    # If specified in options, run the pipeline
    pipeline_func = pipeline.build_pipeline()
    sample_acc_to_matches = run_pipeline(
        pipeline_func, 
        sample_accs,
        sample_to_metadata
    )
    with open(options.pipeline_results_file, 'w') as f:
        f.write(jsonio.dumps(sample_acc_to_matches))


def run_pipeline(pipeline_func, sample_accs, sample_to_metadata):
    sample_acc_to_matches = {}
    c = 1
    for sample_acc in sample_accs:
        try:
            print("Sample #%d: %s" % (c, sample_acc))
            c += 1
            tag_to_val = sample_to_metadata[sample_acc]

            # Make sure the key-value pairs are unicode strings
            decoded_tag_to_val = {}
            for tag, val in tag_to_val.items():
                decoded_tag_to_val[tag] = val
            tag_to_val = decoded_tag_to_val
            mapped_terms, real_props = pipeline_func.run(tag_to_val)
            sample_acc_to_matches[sample_acc] = {
                "mapped_terms":[
                    x.to_dict() 
                    for x in mapped_terms
                ], 
                "real_value_properties": [
                    x.to_dict() 
                    for x in real_props
                ]
            }
        except Exception as e:
            print("ERROR! An error occurred processing sample %s. %s" % (sample_acc, e))
            print(traceback.print_exc())
    return sample_acc_to_matches

if __name__ == "__main__":
    main()
