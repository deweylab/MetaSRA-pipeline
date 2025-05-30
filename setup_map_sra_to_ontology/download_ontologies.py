from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import datetime
import subprocess
import json
import os
from os.path import join
from map_sra_to_ontology import jsonio

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()

    obo_rel_loc = "../map_sra_to_ontology/obo"
    
    prefix_to_filename = {}
    date_str = datetime.datetime.now().strftime("%y-%m-%d")
    with open("ontology_name_to_url.json", "r") as f:
        for ont_prefix, url in json.load(f).items():
            obo_f_name = join(obo_rel_loc, "%s.%s.obo" % (ont_prefix, date_str))
            output_f = open(obo_f_name, "w")
            subprocess.call(["curl", url], stdout=output_f)   
            prefix_to_filename[ont_prefix] = "%s.%s.obo" % (ont_prefix, date_str)

    with open("../map_sra_to_ontology/ont_prefix_to_filename.json", "w") as f:
        f.write(jsonio.dumps(prefix_to_filename))
    
    

if __name__ == "__main__":
    main()
