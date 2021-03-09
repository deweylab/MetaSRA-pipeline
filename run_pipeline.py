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
import os
from os.path import join

import map_sra_to_ontology
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
from map_sra_to_ontology import config
import predict_sample_type
from predict_sample_type import run_sample_type_predictor
from predict_sample_type.learn_classifier import *
from map_sra_to_ontology import pipeline_components as pc

def main():
    parser = OptionParser()
    #parser.add_option("-f", "--key_value_file", help="JSON file storing key-value pairs describing sample")
    (options, args) = parser.parse_args()
   
    input_f = args[0]
     
    # Map key-value pairs to ontologies
    with open(input_f, "r") as f:
        tag_to_vals = json.load(f)

    # Load ontologies
    ont_name_to_ont_id = {
        "UBERON":"12",
        "CL":"1",
        "DOID":"2",
        "EFO":"16",
        "CVCL":"4"}
    ont_id_to_og = {x:load_ontology.load(x)[0] for x in ont_name_to_ont_id.values()}
    pipeline = p_53()

    all_mappings = []
    for tag_to_val in tag_to_vals:
        sample_acc_to_matches = {}
        mapped_terms, real_props = pipeline.run(tag_to_val)
        mappings = {
            "mapped_terms":[x.to_dict() for x in mapped_terms],
            "real_value_properties": [x.to_dict() for x in real_props]
        }
        all_mappings.append(mappings)

    outputs = []
    for tag_to_val, mappings in zip(tag_to_vals, all_mappings):
        outputs.append(
            run_pipeline_on_key_vals(tag_to_val, ont_id_to_og, mappings)
        )
    print json.dumps(outputs, indent=4, separators=(',', ': '))

def run_pipeline_on_key_vals(tag_to_val, ont_id_to_og, mapping_data): 
    
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
            "property_id":real_val_data["property_id"]
        }
        real_val_props.append(real_val_prop)

    # Add super-terms of mapped terms to the list of ontology term features   
    sup_terms = Set()
    for og in ont_id_to_og.values():
        for term_id in mapped_terms:
            sup_terms.update(og.recursive_relationship(term_id, ['is_a', 'part_of']))
    mapped_terms = list(sup_terms)

    predicted, confidence = run_sample_type_predictor.run_sample_type_prediction(
        tag_to_val, 
        mapped_terms, 
        real_val_props
    )

    mapping_data = {
        "mapped ontology terms": mapped_terms, 
        "real-value properties": real_val_props, 
        "sample type": predicted, 
        "sample-type confidence": confidence}

    return mapping_data
    #print json.dumps(mapping_data, indent=4, separators=(',', ': '))


#def run_pipeline(tag_to_val, pipeline):
#    pipeline = p_48()
#    sample_acc_to_matches = {}
#    mapped_terms, real_props = pipeline.run(tag_to_val)
#    mappings = {
#        "mapped_terms":[x.to_dict() for x in mapped_terms], 
#        "real_value_properties": [x.to_dict() for x in real_props]
#    }
#    return mappings
    

def p_53():
    spec_lex = SpecialistLexicon(config.specialist_lex_location())
    inflec_var = SPECIALISTLexInflectionalVariants(spec_lex)
    spell_var = SPECIALISTSpellingVariants(spec_lex)
    key_val_filt = KeyValueFilter_Stage()
    init_tokens_stage = InitKeyValueTokens_Stage()
    ngram = NGram_Stage()
    lower_stage = Lowercase_Stage()
    man_at_syn = ManuallyAnnotatedSynonyms_Stage()
    infer_cell_line = InferCellLineTerms_Stage()
    prop_spec_syn = PropertySpecificSynonym_Stage()
    infer_dev_stage = ImpliedDevelopmentalStageFromAge_Stage()
    linked_super = LinkedTermsOfSuperterms_Stage()
    cell_culture = ConsequentCulturedCell_Stage()
    filt_match_priority = FilterOntologyMatchesByPriority_Stage()
    real_val = ExtractRealValue_Stage()
    match_cust_targs = ExactMatchCustomTargets_Stage()
    cust_conseq = CustomConsequentTerms_Stage()
    delimit_plus = Delimit_Stage('+')
    delimit_underscore = Delimit_Stage('_')
    delimit_dash = Delimit_Stage('-')
    delimit_slash = Delimit_Stage('/')
    block_cell_line_key = BlockCellLineNonCellLineKey_Stage()
    subphrase_linked = RemoveSubIntervalOfMatchedBlockAncestralLink_Stage()
    cellline_to_implied_disease = CellLineToImpliedDisease_Stage()
    acr_to_expan = AcronymToExpansion_Stage()
    exact_match = ExactStringMatching_Stage(
        [
            "1",
            "2",
            "5",
            "7",
            "8",
            "9",
            "18" # Cellosaurus restricted to human cell lines
        ],
        query_len_thresh=3
    )
    fuzzy_match = FuzzyStringMatching_Stage(0.1, query_len_thresh=3)
    two_char_match = TwoCharMappings_Stage()
    time_unit = ParseTimeWithUnit_Stage()
    prioritize_exact = PrioritizeExactMatchOverFuzzyMatch()
    artifact_term_combo = TermArtifactCombinations_Stage()

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
        time_unit,
        two_char_match,
        prop_spec_syn,
        fuzzy_match,
        match_cust_targs,
        block_cell_line_key,
        linked_super,
        cellline_to_implied_disease,
        subphrase_linked,
        cust_conseq,
        artifact_term_combo,
        real_val,
        filt_match_priority,
        infer_cell_line,
        infer_dev_stage,
        cell_culture,
        prioritize_exact
    ]
    return Pipeline(stages, defaultdict(lambda: 1.0))

if __name__ == "__main__":
    main()



