###############################################################################
#   Functionality for loading the NLM's SPECIALIST Lexicon 
#   (https://www.nlm.nih.gov/pubs/factsheets/umlslex.html) into an in-memory
#   a trie data structure. Allows for fast query of the lexicon.
###############################################################################

from optparse import OptionParser
import os
from os.path import join
from collections import deque, defaultdict
import json
import marisa_trie
import pkg_resources as pr

resource_package = __name__

def main():
    # Test that module is functioning
    s = SpecialistLexicon("/scratch/mnbernstein/LEX")
    print s.search('tumor')


class SpecialistLexicon:
    def __init__(self, lex_loc):
        print "loading SPECIALIST Lexicon..."
        self.lexicon = load_lexicon(lex_loc)
        self.eui_array = []

        print "building SPECIALIST Lexicon trie..."
        tups = []
        curr_i = 0
        for eui, lex_info in self.lexicon.iteritems():
            self.eui_array.append(eui)

            tups.append((lex_info["base"].decode('utf-8'), [curr_i]))

            if "spelling variants" in lex_info:
                for spell_var in lex_info["spelling variants"]:
                    tups.append((spell_var.decode('utf-8'), [curr_i]))

            if "nominalization" in lex_info:
                for nom in lex_info["nominalization"]:
                    tups.append((nom.decode('utf-8'), [curr_i]))

            if "inflection variants" in lex_info:
                for infl_var in lex_info["inflection variants"]:
                    tups.append((infl_var.decode('utf-8'), [curr_i]))

            curr_i += 1

        self.trie = marisa_trie.RecordTrie("<i", tups)


    def search(self, query):
        mapped = []
        try:
            results = self.trie[query]
            for r in results:
                eui = self.eui_array[r[0]]
                mapped.append(eui)
        except KeyError:
            #print "Query '%s' not in trie" % query
            pass
        return mapped

    def spelling_variants(self, query):
        spell_vars = []
        mapped = self.search(query)
        for m in mapped:
            spell_vars.append(self.lexicon[m]["base"])
            if "spelling variants" in self.lexicon[m]:
                spell_vars += [x for x in self.lexicon[m]["spelling variants"]]
        return spell_vars

    def inflection_variants(self, query):
        infl_vars = []
        mapped = self.search(query)
        for m in mapped:
            infl_vars.append(self.lexicon[m]["base"])
            if "inflection variants" in self.lexicon[m]:
                infl_vars += [x for x in self.lexicon[m]["inflection variants"]]
        return infl_vars

    def nominalizations(self, query):
        noms = []
        mapped = self.search(query)
        for m in mapped:
            if "nominalization" in self.lexicon[m]:
                noms = [x for x in self.lexicon[m]["nominalization"]]
        return noms

def load_lexicon(lex_loc):
    lexicon = parse_LEXICON(lex_loc)
    lexicon = add_spelling_variants(lexicon)
    lexicon = add_inflection_variants(lexicon)
    lexicon = add_nominalization(lexicon)
    lexicon = add_trademarks(lexicon)

    return lexicon

def add_trademarks(lexicon):

    f_name = pr.resource_filename(resource_package, join("LEX", "LRTRM"))
    with open(f_name, "r") as f:
        for l in f:
            vals = l.strip().split('|')
            eui = vals[0]
            chem = vals[2]

            if eui not in lexicon:
                print "WARNING! Attempt trademarks, but %s is not in the lexicon!" % eui
                continue

            if "trademark" not in lexicon[eui]:
                lexicon[eui]["trademark"] = []
            lexicon[eui]["trademark"].append(chem)
    return lexicon


def add_nominalization(lexicon):

    f_name = pr.resource_filename(resource_package, join("LEX", "LRNOM"))
    with open(f_name, "r") as f:
        for l in f:
            vals = l.strip().split('|')
            eui = vals[0]
            nom = vals[1]

            if eui not in lexicon:
                print "WARNING! Attempt nominalization, but %s is not in the lexicon!" % eui
                continue

            if "nominalization" not in lexicon[eui]:
                lexicon[eui]["nominalization"] = []
            lexicon[eui]["nominalization"].append(nom)
    return lexicon


def add_spelling_variants(lexicon):

    f_name = pr.resource_filename(resource_package, join("LEX", "LRSPL"))
    with open(f_name, "r") as f:
        for l in f:
            vals = l.strip().split('|')
            eui = vals[0]
            spell_var = vals[1]

            if eui not in lexicon:
                print "WARNING! Attempt spelling variant, but %s is not in the lexicon!" % eui
                continue

            if "spelling variants" not in lexicon[eui]:
                lexicon[eui]["spelling variants"] = []
            lexicon[eui]["spelling variants"].append(spell_var)
    return lexicon

def add_inflection_variants(lexicon):

    f_name = pr.resource_filename(resource_package, join("LEX", "LRAGR"))
    with open(f_name, "r") as f:
        for l in f:
            vals = l.strip().split('|')
            eui = vals[0]
            infl_var = vals[1]

            if eui not in lexicon:
                print "WARNING! Attempting inflection variant, %s is not in the lexicon!" % eui
                continue
        
            if infl_var == lexicon[eui]["base"]:
                continue

            if "inflection variants" not in lexicon[eui]:
                lexicon[eui]["inflection variants"] = []
            lexicon[eui]["inflection variants"].append(infl_var)       
    return lexicon

    
def parse_LEXICON(lex_loc):
    def process_curr_lines(c_lines):
        if not c_lines:
            return     

        entries = {}

        c_lines[0] = c_lines[0][1:]
        c_lines = [x.strip() for x in c_lines]
        
        eui = None
        base = None
        for l in c_lines:
            if l == "}":
                continue
            if len(l.split('=')) < 2:
                continue

            key = l.split('=')[0]
            val = l.split('=')[1]

            if key == "entry":
                eui = val
            elif key == "base":
                base = val

        return eui, base

    f_name = join(lex_loc, "LEXICON")
    with open(f_name, "r") as f:
        lexicon = defaultdict(lambda: {})

        curr_lines = []
        for line in f:
            if line[0] == "{":
                result = process_curr_lines(curr_lines)
                if result:
                    lexicon[result[0]]["base"] = result[1]
                curr_lines = []
            curr_lines.append(line)
        result = process_curr_lines(curr_lines)
        if result:
            lexicon[result[0]]["base"] = result[1]
    return lexicon
   
if __name__ == "__main__":
    main()
