from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import datetime
import subprocess
import json
import os
from os.path import join

def main(config):
    lex_rel_loc = config.LEX_DIR
    os.makedirs(lex_rel_loc, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%y-%m-%d")
    with open("lex_file_to_url.json", "r") as f:
        for lex_f, url in json.load(f).items():
            lex_f_name = lex_rel_loc / lex_f
            output_f = open(lex_f_name, "w")
            subprocess.call(["curl", url], stdout=output_f)   
