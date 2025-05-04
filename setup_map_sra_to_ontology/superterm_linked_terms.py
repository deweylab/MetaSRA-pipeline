###################################################################
# Finds terms between the ontologies, term_a and term_b for which 
# term_b is linked to an ancestor term of term_a.
###################################################################

from __future__ import print_function
from io import open # Python 2/3 compatibility
from collections import defaultdict
import json
from collections import deque
import marisa_trie as mt

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology
from map_sra_to_ontology import jsonio

def main():
    efo_cell_og, x,y = load_ontology.load("11")
    #efo_disease_og, x,y = load_ontology.load("3")
    #efo_anatomy_og, x,y, = load_ontology.load("14")
    #efo_og, x,y = load_ontology.load("13")
    cl_uberon_doid_og, x,y = load_ontology.load("0")
    efo_cellline_og, x,y = load_ontology.load("10")
    cvcl_og,x,y = load_ontology.load("4")

    term_to_linkedsup_terms = term_to_linked_superterms(cl_uberon_doid_og)   
    #term_to_linkedsup_terms = term_to_linked_superterms(efo_og) 
    term_to_linkedsup_terms.update(term_to_linked_superterms(efo_cell_og))
    term_to_linkedsup_terms.update(term_to_linked_superterms(efo_cellline_og))
    term_to_linkedsup_terms.update(term_to_linked_superterms(cvcl_og))
    #term_to_linkedsup_terms.update(term_to_linked_superterms(efo_disease_og))
    #term_to_linkedsup_terms.update(term_to_linked_superterms(efo_anatomy_og))

    # Remove known incorrect mappings
    if "CVCL:1240" in term_to_linkedsup_terms["EFO:0003045"]:
        term_to_linkedsup_terms["EFO:0003045"].remove("CVCL:1240")
        

    with open("term_to_superterm_linked_terms.json", "w") as f:
        f.write(jsonio.dumps(term_to_linkedsup_terms))

def term_to_linked_superterms(og):
    term_to_linked_terms = None
    with open("term_to_linked_terms.json", "r") as f:
        term_to_linked_terms = json.load(f)

    term_to_linkedsup_terms = defaultdict(lambda: [])

    for t_id in og.id_to_term:
        for sup_term in og.recursive_relationship(t_id, ["is_a", "part_of"]):
            if sup_term in term_to_linked_terms:
                for linked_term in term_to_linked_terms[sup_term]:
                    print("Linking term %s --> %s" % (t_id, linked_term)) 
                    term_to_linkedsup_terms[t_id].append(linked_term)

    return term_to_linkedsup_terms


if __name__ == "__main__":
    main()
