from optparse import OptionParser
import json
from sets import Set
import sys
from collections import defaultdict, deque
import time
import traceback
import os
from os.path import join, realpath

import map_sra_to_ontology
from map_sra_to_ontology import config
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import query_metadata
from map_sra_to_ontology import load_ontology as lo
from map_sra_to_ontology.pipeline_components import *

INCLUDED_ONTOLOGIES = ["CL", "DOID", "UBERON"]

def build_pipeline():
    """
    Improvement: Added stage TermArtifactCombinations_Stage

    Pipeline Version 53
    """
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
