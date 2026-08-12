from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import json
import os
from os.path import join

def main(config):

    with open(config.ref_path / "ont_prefix_to_filename.json", "r") as f:
        ont_prefix_to_filename = json.load(f)

    # Reformat Cellosaurus

    cvcl_f = config.OBO_DIR / ont_prefix_to_filename["CVCL"]

    with open(cvcl_f, "r") as f:
        cvcl_content = f.read()
        cvcl_content = cvcl_content.replace("CVCL_", "CVCL:")

    with open(cvcl_f, "w") as f:
        f.write(cvcl_content)

    # Reformat EFO
    efo_f = config.OBO_DIR / ont_prefix_to_filename["EFO"]

    with open(efo_f, "r") as f:
        efo_content = f.read()
        efo_content = efo_content.replace("UBERON:", "EFO_UBERON:")
        efo_content = efo_content.replace("CL:", "EFO_CL:")
        efo_content = efo_content.replace("DOID:", "EFO_DOID:")
        efo_content = efo_content.replace("PATO:", "EFO_PATO:")

    with open(efo_f, "w") as f:
        f.write(efo_content)
