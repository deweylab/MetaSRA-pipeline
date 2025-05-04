from __future__ import print_function
from io import open # Python 2/3 compatibility
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

    lex_rel_loc = "../map_sra_to_ontology/LEX"
    subprocess.call("mkdir %s" % lex_rel_loc, shell=True, env=None)

    date_str = datetime.datetime.now().strftime("%y-%m-%d")
    with open("lex_file_to_url.json", "r") as f:
        for lex_f, url in json.load(f).items():
            lex_f_name = join(lex_rel_loc, "%s" % lex_f)
            output_f = open(lex_f_name, "w")
            subprocess.call(["curl", url], stdout=output_f)   

if __name__ == "__main__":
    main()
