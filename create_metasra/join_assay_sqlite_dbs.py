#########################################################
#   Given a set of SQLite files from MetaSRA outputs
#   for various species and assays, this script joins
#   them into a single SQLite file with new a table for
#   species and assay.
#########################################################

from __future__ import print_function
from optparse import OptionParser
import os
from os.path import isdir, join
import sqlite3

CREATE_MAPPED_ONTOLOGY_TABLE_SQL = """CREATE TABLE mapped_ontology_terms 
        (sample_accession text, term_id text, 
        PRIMARY KEY (sample_accession, term_id))"""

CREATE_REAL_VAL_PROP_TABLE_SQL = """CREATE TABLE real_value_properties
    (sample_accession TEXT, property_term_id TEXT, value NUMERIC, 
    unit_id TEXT, PRIMARY KEY (sample_accession, property_term_id, value, unit_id))"""

CREATE_SAMPLE_TYPE_TABLE_SQL = """CREATE TABLE sample_type 
    (sample_accession TEXT, sample_type TEXT, confidence NUMERIC, 
    PRIMARY KEY (sample_accession))"""

CREATE_SAMPLE_INFO_TABLE_SQL = """ 
    CREATE TABLE sample_info 
    (sample_accession TEXT, species TEXT, assay TEXT, 
    PRIMARY KEY (sample_accession, assay))
"""

INSERT_ONTOLOGY_TERM_SQL = """INSERT OR REPLACE INTO 
    mapped_ontology_terms VALUES(?, ?)"""

INSERT_REAL_VAL_PROP_SQL = """INSERT OR REPLACE INTO
    real_value_properties VALUES(?, ?, ?, ?)"""

INSERT_SAMPLE_TYPE_SQL = """INSERT OR REPLACE INTO 
    sample_type VALUES (?, ?, ?)"""

INSERT_SAMPLE_INFO_SQL = """INSERT OR REPLACE INTO
    sample_info VALUES (?, ?, ?)"""

GET_ONTOLOGY_TERM_SQL = """SELECT sample_accession, term_id FROM mapped_ontology_terms"""

GET_REAL_VAL_PROP_SQL = """SELECT sample_accession, property_term_id, value, unit_id FROM real_value_properties"""

GET_SAMPLE_TYPE_SQL = """SELECT sample_accession, sample_type, confidence FROM sample_type"""

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
    with sqlite3.connect(out_f) as out_db_conn:
        c_write = out_db_conn.cursor()
        c_write.execute(CREATE_MAPPED_ONTOLOGY_TABLE_SQL)
        c_write.execute(CREATE_REAL_VAL_PROP_TABLE_SQL)
        c_write.execute(CREATE_SAMPLE_TYPE_TABLE_SQL)
        c_write.execute(CREATE_SAMPLE_INFO_TABLE_SQL)
        for dbf, assay, spec in zip(dbfs, assays, species):
            print('Concatenating data from %s' % dbf)
            with sqlite3.connect(dbf) as in_db_conn:
                c_read = in_db_conn.cursor()
                # Add ontology term data
                res = c_read.execute(GET_ONTOLOGY_TERM_SQL)
                for r in res:
                    c_write.execute(INSERT_ONTOLOGY_TERM_SQL, r)
                # Add real-value property data
                res = c_read.execute(GET_REAL_VAL_PROP_SQL)
                for r in res:
                    c_write.execute(INSERT_REAL_VAL_PROP_SQL, r)
                # Add sample-type data and species & assay data
                res = c_read.execute(GET_SAMPLE_TYPE_SQL)
                for r in res:
                    c_write.execute(INSERT_SAMPLE_TYPE_SQL, r)
                    sample = r[0]
                    insert_tuple = (sample, spec, assay)
                    c_write.execute(INSERT_SAMPLE_INFO_SQL, insert_tuple)
            print('done.')                 



if __name__ == '__main__':
    main()


