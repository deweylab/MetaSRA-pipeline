#######################################################################
# Links terms between the ontologies. Terms are linked if they share
# the same name, exact synonym, or related synonym.
#######################################################################

from optparse import OptionParser
from collections import defaultdict
import json
from sets import Set
from collections import deque
import marisa_trie as mt

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology

def main():
    efo_celltype_og, x,y = load_ontology.load("11")
    cl_celltype_og, x,y = load_ontology.load("1")

    doid_disease_og, x,y = load_ontology.load("2")
    efo_disease_og, x,y = load_ontology.load("3")

    efo_anatomy_og, x,y = load_ontology.load("14")
    uberon_anatomy_og, x,y = load_ontology.load("5")

    efo_cellline_og, x,y = load_ontology.load("10")
    cvcl_og, x,y = load_ontology.load("4")

    term_to_linked_terms = {}
    term_to_linked_terms.update(linked_terms(efo_celltype_og, cl_celltype_og))
    term_to_linked_terms.update(linked_terms(efo_disease_og, doid_disease_og))
    term_to_linked_terms.update(linked_terms(efo_anatomy_og, uberon_anatomy_og))
    term_to_linked_terms.update(linked_terms(cl_celltype_og, efo_celltype_og))
    term_to_linked_terms.update(linked_terms(doid_disease_og, efo_disease_og))
    term_to_linked_terms.update(linked_terms(uberon_anatomy_og, efo_anatomy_og)) 
    term_to_linked_terms.update(linked_terms(efo_cellline_og, cvcl_og, link_syn_types=["EXACT", "RELATED"]))
    term_to_linked_terms.update(linked_terms(cvcl_og, efo_cellline_og, link_syn_types=["EXACT", "RELATED"]))

    with open("term_to_linked_terms.json", "w") as f:
        f.write(json.dumps(term_to_linked_terms, indent=4, sort_keys=True, separators=(',', ': ')))
    

class Mapper:
    def __init__(self, og, link_syn_types=None):
        self.link_syn_types = Set(link_syn_types)
        self.map_trie, self.terms_array = self._trie_from_ontology(og)

    def map_string(self, query):
        mapped = []
        try:
            results = self.map_trie[query]
            for r in results:
                term = self.terms_array[r[0]]
                mapped.append(term)

        except KeyError:
            #print "Query '%s' not in trie" % query
            pass
        return mapped

    def _trie_from_ontology(self, og):
        terms_array = deque()
        tups = deque()
        curr_i = 0
        for term in og.get_mappable_terms():
            terms_array.append(term)
            tups.append((term.name.decode('utf-8'), [curr_i]))
            #for syn in [x for x in term.synonyms if x.syn_type == "EXACT"]:
            for syn in [x for x in term.synonyms if x.syn_type in self.link_syn_types]:
                try:
                    tups.append((syn.syn_str.decode('utf-8'), [curr_i]))
                except UnicodeEncodeError:
                    print "Warning! Unable to decode unicode of a synonym for term %s" % term.id
            curr_i += 1
        return mt.RecordTrie("<i", tups), terms_array

def linked_terms(og_a, og_b, link_syn_types=None):
    if not link_syn_types:
        link_syn_types = Set(["EXACT"])
    else:
        link_syn_types = Set(link_syn_types)

    a_mapper = Mapper(og_a, link_syn_types=link_syn_types)
    b_to_a = defaultdict(lambda: Set())
    for b_term in og_b.id_to_term.values():
        b_ref_strs = [b_term.name]
        #b_ref_strs += [syn.syn_str for syn in b_term.synonyms if syn.syn_type == "EXACT"]
        b_ref_strs += [syn.syn_str for syn in b_term.synonyms if syn.syn_type in link_syn_types]
        for b_str in b_ref_strs:
            for a_term in a_mapper.map_string(b_str):
                print "LINKING terms: %s=%s: '%s' '%s'='%s'" % (b_term.id, a_term.id, b_str, b_term.name, a_term.name)
                b_to_a[b_term.id].add(a_term.id)
    return {k:list(v) for k,v in b_to_a.iteritems()}


if __name__ == "__main__":
    main()
