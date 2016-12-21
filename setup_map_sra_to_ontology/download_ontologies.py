from optparse import OptionParser
import datetime
import subprocess
import json
import os
from os.path import join

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    obo_rel_loc = "../map_sra_to_ontology/obo"
    name_to_prefix = {
        "uberon":"UBERON",
        "cl": "CL",
        "doid":"DOID",
        "cellosaurus":"CVCL",
        "uo":"UO",
        "chebi":"CHEBI",
        "efo":"EFO"}

    prefix_to_filename = {}
    date_str = datetime.datetime.now().strftime("%y-%m-%d")
    with open("ontology_name_to_url.json", "r") as f:
        for ont_name, url in json.load(f).iteritems():
            obo_f_name = join(obo_rel_loc, "%s.%s.obo" % (ont_name, date_str))
            output_f = open(obo_f_name, "w")
            subprocess.call(["curl", url], stdout=output_f)   
            prefix_to_filename[name_to_prefix[ont_name]] = "%s.%s.obo" % (ont_name, date_str)

    with open("../map_sra_to_ontology/ont_prefix_to_filename.json", "w") as f:
        f.write(json.dumps(prefix_to_filename, indent=4, separators=(',', ': '))) 
    
    

if __name__ == "__main__":
    main()
