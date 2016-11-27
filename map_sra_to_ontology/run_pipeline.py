########################################################################
#
# Run the ontology mapping pipeline on a set of key-value pairs 
# that describe a biological sample
#
########################################################################

from optparse import OptionParser
import json
from sets import Set
import sys
from collections import defaultdict, deque
import json
import dill

import pkg_resources as pr
resource_package = __name__

import ontology_graph
import load_ontology as lo
import predict_sample_type
from predict_sample_type.run_on_entire_dataset import * 
import pipeline_components as pc

def main():
    parser = OptionParser()
    parser.add_option("-f", "--key_value_file", help="JSON file storing key-value pairs describing sample")
    (options, args) = parser.parse_args()
    
    # Map key-value pairs to ontologies
    with open(options.key_value_file, "r") as f:
        tag_to_val = json.load(f)
    mapping_data = run_pipeline(tag_to_val)
    
    # Filter for biologically significant terms
    ont_name_to_ont_id = {
        "UBERON":"12", 
        "CL":"1", 
        "DOID":"2", 
        "EFO":"16", 
        "CVCL":"4"}
    ont_id_to_og = {x:load_ontology.load(x)[0] for x in ont_name_to_ont_id.values()}
    mapped_terms = []
    real_val_props = []
    for mapped_term_data in mapping_data["mapped_terms"]:
        term_id = mapped_term_data["term_id"]
        for ont in ont_id_to_og.values():
            if term_id in ont.get_mappable_term_ids():
                mapped_terms.append(term_id)
                break
    for real_val_data in mapping_data["real_value_properties"]:
        real_val_prop = {
            "unit_id":real_val_data["unit_id"], 
            "value":real_val_data["value"], 
            "property_id":real_val_data["property_id"]}
        real_val_props.append(real_val_prop)
   
    predicted, confidence = run_sample_type_prediction(tag_to_val, mapped_terms, real_val_props)

    mapping_data = {
        "mapped ontology terms": mapped_terms, 
        "real-value properties": real_val_props, 
        "sample type": predicted, 
        "sample-type confidence": confidence}
    print json.dumps(mapping_data, indent=4, separators=(',', ': '))

def run_sample_type_prediction(tag_to_val, mapped_terms, real_props):
    # Load the dilled vectorizer and model
    vectorizer_f = pr.resource_filename(resource_package, join("predict_sample_type", "sample_type_vectorizor.dill"))
    classifier_f = pr.resource_filename(resource_package, join("predict_sample_type", "sample_type_classifier.dill"))
    with open(vectorizer_f, "rb") as f:
        vectorizer = dill.load(f)
    with open(classifier_f, "rb") as f:
        model = dill.load(f)

    # Make sample-type prediction
    feat_v = vectorizer.convert_to_features(
        get_ngrams_from_tag_to_val(tag_to_val),
        mapped_terms)
    predicted, confidence = model.predict(
        feat_v,
        mapped_terms,
        real_props)

    return predicted, confidence



def run_pipeline(tag_to_val):
    pipeline = p_41()
    sample_acc_to_matches = {}
    mapped_terms, real_props = pipeline.run(tag_to_val)
    mappings = {
        "mapped_terms":[x.to_dict() for x in mapped_terms], 
        "real_value_properties": [x.to_dict() for x in real_props]}
    return mappings
    
def p_41():
    spec_lex = pc.SpecialistLexicon()
    inflec_var = pc.SPECIALISTLexInflectionalVariants(spec_lex)
    spell_var = pc.SPECIALISTSpellingVariants(spec_lex)
    key_val_filt = pc.KeyValueFilter_Stage()
    init_tokens_stage = pc.InitKeyValueTokens_Stage()
    ngram = pc.NGram_Stage()
    lower_stage = pc.Lowercase_Stage()
    man_at_syn = pc.ManuallyAnnotatedSynonyms_Stage()
    infer_cell_line = pc.InferCellLineTerms_Stage()
    prop_spec_syn = pc.PropertySpecificSynonym_Stage()
    infer_dev_stage = pc.ImpliedDevelopmentalStageFromAge_Stage()
    linked_super = pc.LinkedTermsOfSuperterms_Stage()
    cell_culture = pc.ConsequentCulturedCell_Stage()
    filt_match_priority = pc.FilterOntologyMatchesByPriority_Stage()
    real_val = pc.ExtractRealValue_Stage()
    match_cust_targs = pc.ExactMatchCustomTargets_Stage()
    cust_conseq = pc.CustomConsequentTerms_Stage()
    delimit_plus = pc.Delimit_Stage('+')
    delimit_underscore = pc.Delimit_Stage('_')
    delimit_dash = pc.Delimit_Stage('-')
    delimit_slash = pc.Delimit_Stage('/')
    block_cell_line_key = pc.BlockCellLineNonCellLineKey_Stage()
    subphrase_linked = pc.RemoveSubIntervalOfMatchedBlockAncestralLink_Stage()
    cellline_to_implied_disease = pc.CellLineToImpliedDisease_Stage()
    acr_to_expan = pc.AcronymToExpansion_Stage()
    exact_match = pc.ExactStringMatching_Stage(["1", "2", "4", "5", "7", "8", "9"], query_len_thresh=3)
    fuzzy_match = pc.FuzzyStringMatching_Stage(0.1, query_len_thresh=3)

    stages = [
        key_val_filt,
        init_tokens_stage,
        ngram,
        lower_stage,
        delimit_plus,
        delimit_underscore,
        delimit_dash,
        delimit_slash,
        inflec_var,
        spell_var,
        man_at_syn,
        acr_to_expan,
        exact_match,
        prop_spec_syn,
        fuzzy_match,
        match_cust_targs,
        block_cell_line_key,
        linked_super,
        cellline_to_implied_disease,
        subphrase_linked,
        cust_conseq,
        real_val,
        filt_match_priority,
        infer_cell_line,
        infer_dev_stage,
        cell_culture]
    return pc.Pipeline(stages, defaultdict(lambda: 1.0))



if __name__ == "__main__":
    main()



