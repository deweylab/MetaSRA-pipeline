from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
import json
import sys
import os
from os.path import join
import pickle
from collections import Counter, defaultdict

import numpy as np
import pkg_resources as pr
resource_package = __name__
sys.path.append(pr.resource_filename(resource_package, ".."))
import learn_classifier as lc
import load_ontology

ONT_IDS = ["12", "1", "2", "16", "4"]
OGS = [load_ontology.load(ont_id)[0] for ont_id in ONT_IDS]

SAMPLE_TO_TAG_TO_VALUES_F = "/ua/mnbernstein/projects/tbcp/metadata/ontology/src/map_sra_to_ontology/metadata/sample_to_tag_to_values.json"

def get_all_samples_to_mappings(mappings_f):
    print("loading sample to predicted ontology term mappings...")
    sample_to_predicted_terms = {}
    sample_to_real_val_props = {}
    #for fname in os.listdir(matches_file_dir):
    with open(mappings_f, 'r') as f:
        j = json.load(f)
        for sample_acc, map_data in j.items():
            sample_to_predicted_terms[sample_acc] = set()
            mapped_term_ids = [
                x["term_id"] 
                for x in map_data["mapped_terms"]
            ]
            term_in_onts = False
            for term in mapped_term_ids:
                for og in OGS:
                    if term in og.mappable_term_ids:
                        sample_to_predicted_terms[sample_acc].add(term)
                        break
            real_val_props = [
                {
                    "property_id": x["property_id"], 
                    "unit_id": x["unit_id"], 
                    "value": x["value"]
                } 
                for x in map_data["real_value_properties"]
            ]
            sample_to_real_val_props[sample_acc] = real_val_props

        for sample_acc, predicted_terms in sample_to_predicted_terms.items():
            sup_terms = set()
            for og in OGS:
                for term in predicted_terms:
                    sup_terms.update(
                        og.recursive_relationship(
                            term, 
                            ['is_a', 'part_of']
                        )
                    )
            sample_to_predicted_terms[sample_acc].update(sup_terms)
    return sample_to_predicted_terms, sample_to_real_val_props

     
def get_dataset(val_set_file):
    data_set = []
    with open(val_set_file, "r") as f:
        val_data = json.load(f)
        for v in val_data["annotated_samples"]:
            if v["sample_type"] != "TODO":
                data_set.append((
                    v["attributes"], 
                    v["sample_type"], 
                    v["sample_accession"]))
    return data_set

def main():
    parser = OptionParser()
    parser.add_option(
        "-m", 
        "--mapping_output_dir", 
        help="Location of mappings file output by pipeline"
    )
    (options, args) = parser.parse_args()
   
    sample_to_metadata_f = args[0]
    mapping_f = args[1]
    out_f = args[2]
    log_f = args[3]
 
    # Build sample to predicted terms and real-value properties
    sample_to_predicted_terms_all, sample_to_real_val_props_all = get_all_samples_to_mappings(
        mapping_f
    )

    vectorizer_f = pr.resource_filename(
        __name__, 
        "sample_type_vectorizer.pickle"
    )
    classifier_f = pr.resource_filename(
        __name__, 
        "sample_type_classifier.pickle"
    ) 
    with open(vectorizer_f, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(classifier_f, 'rb') as f:
        if sys.version_info[0] == 2:
            model = pickle.load(f)
        else:
            model = pickle.load(f, encoding='latin1')

    # Build sample to tag to values
    with open(sample_to_metadata_f, 'r') as f:
        sample_to_tag_to_values = json.load(f) 

    # Make predictions
    sample_to_prediction = {}
    not_found = 0
    pred_none = 0
    for sample_acc, tag_to_values in sample_to_tag_to_values.items():
        if sample_acc not in sample_to_predicted_terms_all:
            # The mapping process may have failed for this sample
            not_found += 1
            continue

        #print "\nPredicting %s" % sample_acc
        try:
            n_grams = lc.get_ngrams_from_tag_to_val(
                sample_to_tag_to_values[sample_acc]
            )
        except:
            print("Error retrieving n-grams!")
            n_grams = []
        feat_v = vectorizer.convert_to_features(
            n_grams, 
            sample_to_predicted_terms_all[sample_acc]
        )
        predicted, confidence = model.predict(
            feat_v, 
            sample_to_predicted_terms_all[sample_acc], 
            sample_to_real_val_props_all[sample_acc]
        )
        if predicted == None or confidence == None:
            #print "HUH? Sample %s was predicted as None..." % sample_acc
            pred_none += 1
        sample_to_prediction[sample_acc] = (predicted, confidence)

    print("%d samples were not found in mappings file" % not_found)
    print("%d samples were predicted as none" % pred_none)
    log_data = {
        'Number samples in metadata, but not mapping file': not_found,
        'Number of samples with prediction errors': pred_none 
    }

    #print sample_to_prediction
    with open(out_f, 'w') as f:
        json.dump(
            sample_to_prediction,
            f, 
            indent=True, 
            sort_keys=True 
        )

    with open(log_f, 'w') as f:
        json.dump(
            log_data,
            f,
            indent=True
        )

        


if __name__ == "__main__":
    main()
