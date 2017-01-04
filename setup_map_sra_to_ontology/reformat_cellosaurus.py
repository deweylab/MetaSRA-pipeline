from optparse import OptionParser
import json
import os
from os.path import join

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()
    
    with open("../map_sra_to_ontology/ont_prefix_to_filename.json", "r") as f:
        ont_prefix_to_filename = json.load(f)
    cvcl_f = join("../map_sra_to_ontology/obo", ont_prefix_to_filename["CVCL"])

    with open(cvcl_f, "r") as f:
        cvcl_content = f.read()
        cvcl_content = cvcl_content.replace("CVCL_", "CVCL:")

    with open(cvcl_f, "w") as f:
        f.write(cvcl_content)

if __name__ == "__main__":
    main()
