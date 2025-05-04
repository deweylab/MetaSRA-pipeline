#########################################################
#   Given a set of JSON files from MetaSRA outputs
#   for various species and assays, this script joins
#   them into a single JSON file with new fields for
#   species and assay.
#########################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import os
from os.path import isdir, join
import json

def main():
    parser = OptionParser()
    parser.add_option("-o", "--out_file", help="Output JSON file")
    (options, args) = parser.parse_args()

    dbfs = args[0].split(',')
    assays = args[1].split(',')
    species = args[2].split(',')
    out_f = options.out_file

    dbs = []
    result_db = {}
    for dbf, assay, spec in zip(dbfs, assays, species):
        with open(dbf, 'r') as f:
            db = json.load(f)
        for k in db.keys():
            db[k]['assay'] = assay
            db[k]['species'] = spec
        result_db.update(db)

    with open(out_f, 'w') as f:
        json.dump(result_db, f, indent=4)

if __name__ == '__main__':
    main()


