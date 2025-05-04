###########################################################################################
# Build the MetaSRA JSON file from the 'raw matches files' that come from the Condor
# jobs.
###########################################################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
import os
from os.path import join
from optparse import OptionParser
import json
from collections import defaultdict
import sqlite3

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology
from map_sra_to_ontology import jsonio

ONT_NAME_TO_ONT_ID = {"UBERON":"12", "CL":"1", "DOID":"2", "EFO":"16", "CVCL":"4"}
ONT_ID_TO_OG = {x:load_ontology.load(x)[0] for x in ONT_NAME_TO_ONT_ID.values()}

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    mappings_f = args[0]
    sample_type_predictions_f = args[1]
    out_json = args[2]
    out_sql = args[3]    

    # Write mappable terms to JSON file
    #mappable_terms = gather_mappable_terms()
    #with open(join(OUTPUT_LOC, "mappable_terms.json"), "w") as f:
    #    f.write(json.dumps(mappable_terms, indent=4, separators=(',', ': ')))

    build_metasra_json(mappings_f, sample_type_predictions_f, out_json)
    build_metasra_sqlite(mappings_f, sample_type_predictions_f, out_sql)


def gather_mappable_terms():
    mappable_terms = set()
    for og in ONT_ID_TO_OG.values():
        mappable_terms.update(og.get_mappable_term_ids())
    return sorted(list(mappable_terms))

def gather_mapped_terms(mappings_f):
    sample_to_mapped_terms = defaultdict(lambda: set())
    sample_to_real_val_props = defaultdict(lambda: [])
    #for fname in os.listdir(matches_file_dir):
    with open(mappings_f, 'r') as f:
        j = json.load(f)
        for sample_acc, mapping_data in j.items():
            sample_to_mapped_terms[sample_acc] = set()
            sample_to_real_val_props[sample_acc] = []
            if len(mapping_data["mapped_terms"]) == 0:
                #print "Sample %s has mapped to no terms." % sample_acc
                pass
            for mapped_term_data in mapping_data["mapped_terms"]:
                term_id = mapped_term_data["term_id"]
                for ont in ONT_ID_TO_OG.values():
                    if term_id in ont.get_mappable_term_ids():
                        sample_to_mapped_terms[sample_acc].add(term_id)
                        break
            for real_val_data in mapping_data["real_value_properties"]:
                real_val_prop = {
                    "unit_id": real_val_data["unit_id"], 
                    "value": real_val_data["value"], 
                    "property_id": real_val_data["property_id"]
                }
                sample_to_real_val_props[sample_acc].append(real_val_prop)
    assert 'SRS440532' in sample_to_mapped_terms
    return sample_to_mapped_terms, sample_to_real_val_props

def build_metasra_json(mappings_f, sample_type_predictions_f, out_f, date_str=None):
    sample_to_mapped_terms, sample_to_real_val_props = gather_mapped_terms(mappings_f)
    print("Gathered %d samples" % len(sample_to_mapped_terms))  
 
    raw_pred_to_sample_type = {
        "cell_line":"cell line",
        "stem_cells":"stem cells",
        "in_vitro_differentiated_cells":"in vitro differentiated cells",
        "primary_cells":"primary cells",
        "induced_pluripotent_stem_cells": "induced pluripotent stem cell line",
        "tissue":"tissue"}
    with open(sample_type_predictions_f, "r") as f:
        sample_to_predictions = json.load(f)
    mod_sample_to_predictions = defaultdict(lambda: [None, None])
    for sample, prediction in sample_to_predictions.items():
        mod_sample_to_predictions[sample] = (
            raw_pred_to_sample_type[prediction[0]], 
            prediction[1]
        )
    sample_to_predictions = mod_sample_to_predictions 

    print("Hmm... now there are %d samples" % len(sample_to_predictions))

    sample_to_annotated_data = {
        x: {
            "mapped ontology terms": list(sample_to_mapped_terms[x]), 
            "real-value properties": sample_to_real_val_props[x], 
            "sample type": sample_to_predictions[x][0], 
            "sample-type confidence": sample_to_predictions[x][1]
        } 
        for x in sample_to_mapped_terms
    }

    with open(out_f, 'w') as f:
        f.write(jsonio.dumps(sample_to_annotated_data))

def build_metasra_sqlite(mappings_f, sample_type_predictions_f, out_f):
    sample_to_mapped_terms, sample_to_real_val_props = gather_mapped_terms(mappings_f)

    raw_pred_to_sample_type = {
        "cell_line":"cell line",
        "stem_cells":"stem cells",
        "in_vitro_differentiated_cells":"in vitro differentiated cells",
        "primary_cells":"primary cells",
        "induced_pluripotent_stem_cells": "induced pluripotent stem cell line",
        "tissue":"tissue"}
    with open(sample_type_predictions_f, "r") as f:
        sample_to_predictions = json.load(f)
    mod_sample_to_predictions = defaultdict(lambda: [None, None])
    for sample, prediction in sample_to_predictions.items():
        mod_sample_to_predictions[sample] = (raw_pred_to_sample_type[prediction[0]], prediction[1])
    sample_to_predictions = mod_sample_to_predictions


    sample_to_annotated_data = {
        x: {
            "mapped ontology terms": list(sample_to_mapped_terms[x]), 
            "real-value properties": sample_to_real_val_props[x]
        } 
        for x in sample_to_mapped_terms
    }

    create_mapped_ontology_table_sql = """CREATE TABLE mapped_ontology_terms 
        (sample_accession text, term_id text, 
        PRIMARY KEY (sample_accession, term_id))"""

    create_real_val_prop_table_sql = """CREATE TABLE real_value_properties
        (sample_accession TEXT, property_term_id TEXT, value NUMERIC, 
        unit_id TEXT, PRIMARY KEY (sample_accession, property_term_id, value, unit_id))"""

    create_sample_type_table_sql = """CREATE TABLE sample_type 
        (sample_accession TEXT, sample_type TEXT, confidence NUMERIC, 
        PRIMARY KEY (sample_accession)) """

    insert_ontology_term_sql = """INSERT OR REPLACE INTO 
        mapped_ontology_terms VALUES(?, ?)"""

    insert_real_val_prop_sql = """INSERT OR REPLACE INTO
        real_value_properties VALUES(?, ?, ?, ?)"""

    insert_sample_type_sql = """INSERT OR REPLACE INTO 
        sample_type VALUES (?, ?, ?)"""

    with sqlite3.connect(out_f) as db_conn:
        c = db_conn.cursor()
        c.execute(create_mapped_ontology_table_sql)
        c.execute(create_real_val_prop_table_sql)
        c.execute(create_sample_type_table_sql)
        for sample_acc, term_ids in sample_to_mapped_terms.items():
            for term_id in term_ids:
                insert_tuple = (sample_acc, term_id)
                c.execute(insert_ontology_term_sql, insert_tuple)
        
        for sample_acc, real_val_props in sample_to_real_val_props.items():
            for real_val_prop in real_val_props: 
                insert_tuple = (
                    sample_acc, 
                    real_val_prop["property_id"],
                    real_val_prop["value"], 
                    real_val_prop["unit_id"]
                )
                c.execute( insert_real_val_prop_sql, insert_tuple)
         
        for sample_acc, prediction in sample_to_predictions.items():
            insert_tuple = (sample_acc, prediction[0], prediction[1])
            c.execute(insert_sample_type_sql, insert_tuple)

if __name__ == "__main__":
    main()
