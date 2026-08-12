from __future__ import print_function
from io import open # Python 2/3 compatibility
import os
from pybktree import BKTree
from map_sra_to_ontology import load_ontology
from map_sra_to_ontology import string_metrics
from map_sra_to_ontology import jsonio

import json
import pickle
from collections import defaultdict

def main(config):
    fuzzy_index_path = config.ref_path / 'fuzzy_matching_index'
    os.makedirs(fuzzy_index_path, exist_ok=True)

    fuzzy_match_bk_tree_filename = fuzzy_index_path / 'fuzzy_match_bk_tree.pickle'
    fuzzy_match_bk_tree_candidate_mentions_filename = fuzzy_index_path / 'fuzzy_match_bk_tree_candidate_mentions.pickle'
    fuzzy_match_string_data_filename = fuzzy_index_path / 'fuzzy_match_string_data.json'

    og_ids = [
        "1", 
        "2", 
        "18", # Cellosaurus with relavent terms for human biology 
        "5", 
        "7", 
        "9"
    ]
    ogs = [load_ontology.load(x, config)[0] for x in og_ids]
    str_to_terms = defaultdict(lambda: [])

    print("Gathering all term string identifiers in ontologies...")
    string_identifiers = set()
    for og in ogs:
        for id, term in og.id_to_term.items():
            str_to_terms[term.name].append([term.id, "TERM_NAME"])
            string_identifiers.add(term.name)
            for syn in term.synonyms:
                str_to_terms[syn.syn_str].append([term.id, "SYNONYM_%s" % syn.syn_type])
                string_identifiers.add(syn.syn_str)

    print("Building the BK-Tree...")
    bk_tree = BKTree(string_metrics.bag_dist_multiset, string_identifiers)

    print("Building the BK-Tree for candidate mentions pipeline...")
    bk_tree_candidate_mentions = BKTree(string_metrics.CasePermissiveAlnumWeightedBagDistance(0.2, 0.2), string_identifiers)

    with open(fuzzy_match_bk_tree_filename, "wb") as f:
        pickle.dump(bk_tree, f)

    with open(fuzzy_match_bk_tree_candidate_mentions_filename, "wb") as f:
        pickle.dump(bk_tree_candidate_mentions, f)

    with open(fuzzy_match_string_data_filename, "w") as f:
        f.write(jsonio.dumps(str_to_terms))
