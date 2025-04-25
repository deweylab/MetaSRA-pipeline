#################################################################
#   Add extra synonyms to the Experimental Factor Ontology
#################################################################

from __future__ import print_function
from optparse import OptionParser
from collections import defaultdict
import json

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology

def main():
    uncaps_id_to_syns = uncaps_EFO_syns()
    cvcl_id_to_syns = efo_cvcl_syns()

    term_id_to_syns = uncaps_id_to_syns
    for t_id, syns in cvcl_id_to_syns.iteritems():
        term_id_to_syns[t_id] += syns

    with open("term_to_extra_synonyms.json", "w") as f:
        f.write(json.dumps(term_id_to_syns, indent=4, sort_keys=True, separators=(',', ': ')))

def efo_cvcl_syns():
    """
    Use the Cellosaurus to generate extra cell-line synonyms
    for the EFO.
    """
    og, x, y = load_ontology.load("10")
    syn_sets = None
    with open("../map_sra_to_ontology/synonym_sets/cvcl_syn_sets.json", "r") as f:
        syn_sets = [set(x) for x in json.load(f)]

    # Add synonyms
    total_terms = len(og.get_mappable_terms())
    c = 1
    term_id_to_syns = defaultdict(lambda: [])
    for term in og.get_mappable_terms():

        print("Adding synonyms to term %d/%d with id %s" % (c, total_terms, term.id))
        c += 1
        for syn_set in syn_sets:
            current_term_strs = [x.syn_str for x in term.synonyms]
            current_term_strs.append(term.name)
            current_term_strs = set(current_term_strs)

            for c_str in current_term_strs:
                if c_str in syn_set:
                    for syn in syn_set:
                        if syn not in current_term_strs and syn not in set(term_id_to_syns[term.id]):
                            print("Added synonym %s to term with name %s" % (syn, term.name))
                            term_id_to_syns[term.id].append(syn)
    return term_id_to_syns

def uncaps_EFO_syns():
    """
    For all synonyms in the EFO, check if the first character is
    upper case (and only the first character). If so, convert to
    lower case.
    """    

    def uncap_str(r_str):
        tokens = r_str.split()
        new_tokens = []
        for tok in tokens:
            if len(tok) == 1:
                new_tokens.append(tok)
            else:
                first_upper = tok[0].isupper()
                rest_lower = True
                for t in tok[1:]:
                    if t.isupper():
                        rest_lower = False

                if first_upper and rest_lower:
                    new_tokens.append(tok[0].lower() + tok[1:])
                else:
                    new_tokens.append(tok)
        return " ".join(new_tokens)

    og, x, y = load_ontology.load("9")
    term_id_to_syns = defaultdict(lambda: [])
    for term in og.id_to_term.values():
        
        print("Looking at term %s" % term.id)

        # Names of term
        ref_strs = [term.name]
        ref_strs += [x.syn_str for x in term.synonyms]
        ref_strs = set(ref_strs)

        for r_str in ref_strs:
            new_str = uncap_str(r_str)
            if new_str not in ref_strs:
                print("Derived '%s' from '%s'" % (new_str, r_str))
                term_id_to_syns[term.id].append(new_str) 
        
    return term_id_to_syns 

if __name__ == "__main__":
    main()
