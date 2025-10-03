###########################################################################################
# Filter the 'raw matches files' to MetaSRA output.
###########################################################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
import os
from os.path import join
from optparse import OptionParser
import json
from collections import defaultdict
import sqlite3

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology
from map_sra_to_ontology import jsonio

ONT_NAME_TO_ONT_ID = {"UBERON":"12", "CL":"1", "DOID":"2", "EFO":"16", "CVCL":"4"}
ONT_ID_TO_OG = {x:load_ontology.load(x)[0] for x in ONT_NAME_TO_ONT_ID.values()}

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    mappings_f = args[0]
    out_json = args[1]

    # Write mappable terms to JSON file
    #mappable_terms = gather_mappable_terms()
    #with open(join(OUTPUT_LOC, "mappable_terms.json"), "w") as f:
    #    f.write(json.dumps(mappable_terms, indent=4, separators=(',', ': ')))

    build_metasra_json(mappings_f, out_json)

def gather_mappable_terms():
    mappable_terms = set()
    for og in ONT_ID_TO_OG.values():
        mappable_terms.update(og.get_mappable_term_ids())
    return sorted(list(mappable_terms))

def gather_mapped_terms(mappings_f):
    sample_to_mapped_terms = defaultdict(lambda: set())
    sample_to_real_val_props = defaultdict(lambda: [])
    #for fname in os.listdir(matches_file_dir):
    with open(mappings_f, 'r') as f:
        j = json.load(f)
        for sample_acc, mapping_data in j.items():
            sample_to_mapped_terms[sample_acc] = set()
            sample_to_real_val_props[sample_acc] = []
            if len(mapping_data) == 0:
                #print "Sample %s has mapped to no terms." % sample_acc
                pass
            for term_id in mapping_data:
                for ont in ONT_ID_TO_OG.values():
                    if term_id in ont.get_mappable_term_ids():
                        sample_to_mapped_terms[sample_acc].add(term_id)
                        break
            
    # Why was this assert here?
    #assert 'SRS440532' in sample_to_mapped_terms
    if 'SRS440532' not in sample_to_mapped_terms:
        print('SRS440532 is not in our mapped terms!')
    return sample_to_mapped_terms, sample_to_real_val_props

def build_metasra_json(mappings_f, out_f):
    sample_to_mapped_terms, sample_to_real_val_props = gather_mapped_terms(mappings_f)
    print("Gathered %d samples" % len(sample_to_mapped_terms))  
 
    sample_to_annotated_data = {
        x: {
            "mapped ontology terms": list(sample_to_mapped_terms[x]), 
            "real-value properties": sample_to_real_val_props[x], 
        } 
        for x in sample_to_mapped_terms
    }

    with open(out_f, 'w') as f:
        f.write(jsonio.dumps(sample_to_annotated_data))

if __name__ == "__main__":
    main()
