from bktree import BKTree
import load_ontology
import string_metrics

from sets import Set
import json
import pickle
from collections import defaultdict

def main():
    
    og_ids = ["1", "2", "4", "5", "7", "9"]
    ogs = [load_ontology.load(x)[0] for x in og_ids]
    str_to_terms = defaultdict(lambda: [])

    print "Gathering all term string identifiers in ontologies..."
    string_identifiers = Set()
    for og in ogs:
        for id, term in og.id_to_term.iteritems():
            str_to_terms[term.name].append([term.id, "TERM_NAME"])
            string_identifiers.add(term.name)
            for syn in term.synonyms:
                str_to_terms[syn.syn_str].append([term.id, "SYNONYM_%s" % syn.syn_type])
                string_identifiers.add(syn.syn_str)

    print "Building the BK-Tree..."
    bk_tree = BKTree(string_metrics.bag_dist_multiset, string_identifiers)

    with open("fuzzy_match_bk_tree.pickle", "w") as f:
        pickle.dump(bk_tree, f)

    with open("fuzzy_match_string_data.json", "w") as f:
        f.write(json.dumps(str_to_terms, indent=4, separators=(',', ': ')))


if __name__ == "__main__":
    main() 

