from __future__ import print_function
from optparse import OptionParser
from collections import defaultdict
import json
from sets import Set
from collections import deque
import marisa_trie as mt

import map_sra_to_ontology
from map_sra_to_ontology import load_ontology

def generate_implications(og_a, og_b):
    """
    Given two ontologies og_a and og_b, find all terms in
    og_a that imply the term og_b. For example, the term
    'prostate cancer' implies 'cancer'. 
    """

    term_to_implications = defaultdict(lambda: [])
    temp = subterm_consequent_terms(og_a, og_b)
    for term, implied_terms in temp.iteritems():
        term_to_implications[term] += implied_terms

    # Filter terms 
    term_to_superterm_linked_terms = None
    with open("term_to_superterm_linked_terms.json", "r") as f:
        term_to_superterm_linked_terms = json.load(f)
    new_term_to_implications = defaultdict(lambda: [])
    for term, conseq_terms in term_to_implications.iteritems():
        if term in term_to_superterm_linked_terms:
            linked_terms = Set(term_to_superterm_linked_terms[term])
            conseq_terms = Set(conseq_terms).difference(linked_terms)
            if len(conseq_terms) > 0:
                new_term_to_implications[term] = list(conseq_terms)
        else:
            new_term_to_implications[term] = conseq_terms
    return new_term_to_implications
    
    

def main():
    doid_disease_og, x,y = load_ontology.load("2")
    efo_disease_og, x,y = load_ontology.load("3")
    efo_cellline_og, x,y = load_ontology.load("10")

    print("Generating cell line to disease implications...")
    term_to_implications = generate_implications(efo_disease_og, efo_cellline_og)
    temp = generate_implications(doid_disease_og, efo_cellline_og)
    for term, implied_terms in temp.iteritems():
        term_to_implications[term] += implied_terms
    with open("cellline_to_disease_implied_terms.json", "w") as f:
        f.write(json.dumps(term_to_implications, indent=4, separators=(',', ': '), sort_keys=True))

class Mapper:
    def __init__(self, og):
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
            for syn in [x for x in term.synonyms if x.syn_type == "EXACT"]:
                try:
                    tups.append((syn.syn_str.decode('utf-8'), [curr_i]))
                except UnicodeEncodeError:
                    print("Warning! Unable to decode unicode of a synonym for term %s" % term.id)
            curr_i += 1
        return mt.RecordTrie("<i", tups), terms_array

def subterm_consequent_terms(og_a, og_b):
    """
    b implies a
    """

    term_to_implications = defaultdict(lambda: [])

    a_mapper = Mapper(og_a)

    b_terms_not_in_a = deque()
    for b_term in og_b.id_to_term.values():
        b_strs = [b_term.name]
        b_strs += [x.syn_str for x in b_term.synonyms]
        b_in_a = False
        for b_str in b_strs:
            if len(a_mapper.map_string(b_str)) > 0:
                b_in_a = True
                break
        if not b_in_a:
            b_terms_not_in_a.append(b_term)

    total = len(og_a.id_to_term)
    c = 1

    for a_term in og_a.id_to_term.values():

        if c % 100 == 0:
            print("Examined %d/%d terms" % (c, total))
        c += 1


        for b_term in b_terms_not_in_a:

            a_strs = [a_term.name]
            a_strs += [x.syn_str for x in a_term.synonyms if x.syn_type == "EXACT"]

            b_strs = [b_term.name]
            b_strs += [x.syn_str for x in b_term.synonyms if x.syn_type == "EXACT"]

            for b_str in b_strs:
                for a_str in a_strs:
                    try:
                        # Make sure substring is of full tokens
                        a_str_in_b_str = True
                        b_toks = Set(b_str.split(" "))
                        a_toks = a_str.split(" ")
                        for a_tok in a_toks:
                            if a_tok not in b_toks:
                                a_str_in_b_str = False
                                break
 
                        if a_str_in_b_str and a_str in b_str and a_str != b_str and len(a_str) > 2 and len(b_str) > 2:
                            term_to_implications[b_term.id].append(a_term.id)
                            print("Found match %s --> %s: '%s' --> '%s'" % (b_term.id, a_term.id, b_str, a_str))
                    except UnicodeDecodeError:
                        pass
                        #print "Error decoding strings trying match %s with %s" % (b_term.id, a_term.id)

    return term_to_implications





if __name__ == "__main__":
    main()
