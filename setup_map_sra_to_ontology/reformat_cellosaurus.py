from optparse import OptionParser
import json
import os
from os.path import join

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()
    
    # Reformat Cellosaurus
    with open("../map_sra_to_ontology/ont_prefix_to_filename.json", "r") as f:
        ont_prefix_to_filename = json.load(f)
    cvcl_f = join("../map_sra_to_ontology/obo", ont_prefix_to_filename["CVCL"])

    with open(cvcl_f, "r") as f:
        cvcl_content = f.read()
        cvcl_content = cvcl_content.replace("CVCL_", "CVCL:")

    with open(cvcl_f, "w") as f:
        f.write(cvcl_content)

    # Reformat EFO
    with open("../map_sra_to_ontology/ont_prefix_to_filename.json", "r") as f:
        ont_prefix_to_filename = json.load(f)
    efo_f = join("../map_sra_to_ontology/obo", ont_prefix_to_filename["EFO"])

    with open(efo_f, "r") as f:
        efo_content = f.read()
        efo_content = efo_content.replace("UBERON:", "EFO_UBERON:")
        efo_content = efo_content.replace("CL:", "EFO_CL:")
        efo_content = efo_content.replace("DOID:", "EFO_DOID:")
        efo_content = efo_content.replace("PATO:", "EFO_PATO:")

    with open(efo_f, "w") as f:
        f.write(efo_content)

    

if __name__ == "__main__":
    main()
