from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import datetime
import subprocess
import json
import os
from os.path import join
from importlib import resources
from metasra.map_sra_to_ontology import jsonio

def main(config):
    obo_rel_loc = config.OBO_DIR
    os.makedirs(obo_rel_loc, exist_ok=True)
    
    prefix_to_filename = {}
    date_str = datetime.datetime.now().strftime("%y-%m-%d")
    ontology_name_to_url_f = resources.files(__package__) / "ontology_name_to_url.json"
    with ontology_name_to_url_f.open() as f:
        for ont_prefix, url in json.load(f).items():
            obo_f_name = join(obo_rel_loc, "%s.%s.obo" % (ont_prefix, date_str))
            output_f = open(obo_f_name, "w")
            subprocess.call(["curl", "-L", url], stdout=output_f)   
            prefix_to_filename[ont_prefix] = "%s.%s.obo" % (ont_prefix, date_str)

    with open(config.ref_path / "ont_prefix_to_filename.json", "w") as f:
        f.write(jsonio.dumps(prefix_to_filename))
    
    


