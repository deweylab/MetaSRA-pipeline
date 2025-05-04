###############################################################
# Generate candidate terms in the Experimental Factor Ontology
# that have a synonym that should be removed. The final output
# of this script was refined manually.
################################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
from optparse import OptionParser
from collections import defaultdict
import json
from collections import deque

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology
from map_sra_to_ontology import jsonio

def main():
    og, x, y = load_ontology.load("13")
    problematic_terms = deque()
    for t_id, term in og.id_to_term.items():
        if "carcinoma" in term.name or "adenocarcinoma" in term.name:
            problematic_terms.append(term)

    term_to_syns = {}
    for term in problematic_terms:
        term_to_syns[term.id] = {"name": term.name, "synonyms":[x.syn_str for x in term.synonyms]}

    with open("candidate_term_to_remove_synonyms.json", "w") as f:
        f.write(jsonio.dumps(term_to_syns))


if __name__ == "__main__":
    main()
