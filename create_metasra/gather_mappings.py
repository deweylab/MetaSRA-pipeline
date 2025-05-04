##############################################################################################
#   Gather all of the term-mappings and real-value-property-mappings from the output of each
#   Condor job and aggregate them into a single JSON file.
##############################################################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import os
from os.path import isdir, join
import json

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()

    condor_root = args[0]
    job_out_fname = args[1]
    out_file = args[2]
    log_file = args[3]

    n_found = 0
    all_metasra_mappings = {}
    log_data = {'Failed job output files': []}
    for d in os.listdir(condor_root):
        path_d = join(condor_root, d)
        if not isdir(path_d):
            continue
        metasra_mappings_f = join(path_d, job_out_fname)
        try:
            with open(metasra_mappings_f, 'r') as f:
                #print "loading %s..." % metasra_mappings_f
                mappings = json.load(f)
                for k,v in mappings.items():
                    all_metasra_mappings[k] = v
                    n_found += 1
        except ValueError as e:
            print("Could not decode %s. Error: %s" % (metasra_mappings_f, str(e)))
            log_data['Failed job output files'].append(
                metasra_mappings_f
            )
            
    print("Extracted %d mappings..." % n_found)
    log_data['Total samples with mappings'] = n_found

    print(len(all_metasra_mappings))    
    with open(out_file, 'w') as f:
        json.dump(
            all_metasra_mappings,
            f, 
            sort_keys=True, 
            indent=True 
        )
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=True) 


if __name__ == "__main__":
    main()
