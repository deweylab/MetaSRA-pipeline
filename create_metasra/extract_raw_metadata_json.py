#############################################################################
#   Create JSON file that maps each sample to their raw key-value paired
#   metadata.
#############################################################################

from __future__ import print_function
from optparse import OptionParser
import sqlite3
import json

def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()

    db_loc = args[0]
    out_file = args[1]
    sql_cmd = "SELECT sample_accession, tag, value FROM sample_attribute"

    sample_to_tag_to_value = {}    
    with sqlite3.connect(db_loc) as db_conn:
        db_cursor = db_conn.cursor()
        returned = db_cursor.execute(sql_cmd)
        for r in returned:
            sample_acc = r[0].encode('utf-8')
            tag = r[1].encode('utf-8')
            value = r[2].encode('utf-8')

            if sample_acc not in sample_to_tag_to_value:   
                sample_to_tag_to_value[sample_acc] = {}    
   
            sample_to_tag_to_value[sample_acc][tag] = value

    with open(out_file, 'w') as f:
        f.write(json.dumps(
            sample_to_tag_to_value, 
            sort_keys=True, 
            indent=4, 
            separators=(',', ': ')
        ))



if __name__ == "__main__":
    main()
