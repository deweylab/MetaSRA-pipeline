###########################################################################
#   Components for building metadata mapping pipelines. 
###########################################################################

import json
from collections import defaultdict, deque, Counter
import re
import pickle

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.metrics.distance import edit_distance
from scipy.spatial import KDTree
import sklearn
from sklearn.neighbors import BallTree

import pkg_resources as pr
import json
from sets import Set
import os
from os.path import join

import load_ontology
from text_reasoning_graph import *
import ball_tree_distance
from load_specialist_lex import SpecialistLexicon

import bktree
from bktree import BKTree
import marisa_trie as mt

# Relative paths to resources
resource_package = __name__
FILTER_KEYS_JSON = pr.resource_filename(resource_package, join("metadata", "filter_key_val_rules.json"))
CELL_LINE_FILTER_KEYS_JSON = pr.resource_filename(resource_package, join("metadata", "cell_line_filter_key_val_rules.json"))
PROPERTY_SPECIFIC_SYNONYMS_JSON = pr.resource_filename(resource_package, join("metadata", "has_val_syn_term_ids.json"))
TERM_TO_LINKED_ANCESTOR_JSON =  pr.resource_filename(resource_package, join("metadata", "term_to_superterm_linked_terms.json"))
NOUN_PHRASES_JSON = pr.resource_filename(resource_package, join("metadata", "noun_phrases.json"))
CELL_LINE_TO_IMPLIED_DISEASE_JSON = pr.resource_filename(resource_package, join("metadata", "cellline_to_disease_implied_terms.json"))
ACRONYM_TO_EXPANSION_JSON = pr.resource_filename(resource_package, join("metadata", "acronym_to_expansions.json"))
REAL_VALUE_PROPERTIES = pr.resource_filename(resource_package, join("metadata", "real_valued_properties.json"))
CUST_TERM_TO_CONSEQ_TERMS_JSON = pr.resource_filename(resource_package, join("metadata", "custom_term_to_consequent_terms.json"))
CELL_LINE_TERMS_JSON = pr.resource_filename(resource_package, join("metadata", "cvcl_mappings.json"))
TWO_CHAR_MAPPINGS_JSON = pr.resource_filename(resource_package, join("metadata", "two_char_mappings.json"))


TOKEN_SCORING_STRATEGY = defaultdict(lambda: 1) # TODO We want an explicit score dictionary

class MappedTerm:
    def __init__(self, term_id, consequent, orig_key, orig_val, mapping_path):
        self.term_id = term_id
        self.orig_key = orig_key
        self.orig_val = orig_val
        #self.match_score = match_score
        self.mapping_path = mapping_path   
        self.consequent = consequent

    def to_dict(self):
        path = str(self.mapping_path)
        return {"term_id": self.term_id, 
            "original_key":self.orig_key,
            "original_value": self.orig_val,
            "consequent": self.consequent,
            "path_to_mapping": path}
      
    def __str__(self):
        return str(self.to_dict())  
 
class RealValueProperty:
    def __init__(self, property_id, consequent, value, unit_id, orig_key, orig_val, mapping_path):
        self.property_id = property_id
        self.consequent = consequent
        self.value = value
        self.unit_id = unit_id
        self.orig_key = orig_key
        self.orig_val = orig_val
        self.mapping_path = mapping_path

    def to_dict(self):
        path = str(self.mapping_path)
        return {"property_id": self.property_id,
            "value": self.value,
            "unit_id": self.unit_id,
            "original_key":self.orig_key,
            "original_value": self.orig_val,
            "path_to_mapping": path,
            "consequent": self.consequent}

    def __str__(self):
        return str(self.to_dict())


class Pipeline:
    def __init__(self, stages, scoring_strategy):
        self.stages = stages
        self.scoring_strategy = scoring_strategy

    def run(self, tag_to_val):
        tm_graph = TextReasoningGraph(prohibit_cycles=False)
                
        # Create initial text-mining-graph
        tokens = Set()
        for tag, val in tag_to_val.iteritems():
            kv_node = KeyValueNode(tag.encode('utf-8'), val.encode('utf-8'))    
            tm_graph.add_node(kv_node)

        # Process stages of pipeline
        for stage in self.stages:
            tm_graph = stage.run(tm_graph)

        print
        print "---------GRAPH----------"
        print tm_graph
        print "------------------------"
        print 
        #print "---------Graphviz-------"
        #print tm_graph.graphviz()
        #print "------------------------"

        return self.extract_mapped_terms(tm_graph)

    def extract_mapped_terms(self, text_mining_graph):

        # TODO EXTRACT MULTIPLE MAPPINGS FOR real-value property!!
        def extract_mapping(mapped_node):

            result = []
            dist, prev = text_mining_graph.shortest_path(mapped_node, use_reverse_edges=True)

            kv_node_w_dists = []
            for kv_node in text_mining_graph.key_val_nodes:
                if kv_node in dist and dist[kv_node] < float('inf'):
                    kv_node_w_dists.append((kv_node, dist[kv_node]))

            if not kv_node_w_dists: # No path from ontology term node to a key-value
                return None

            # Find the minimum weight path to a key-value
            m = min(kv_node_w_dists, key=lambda x: x[1])
            orig_kv_node = m[0]
            path_weight = m[1]

            # Exctract path from key-value to node TODO This might not work for all matches
            path = []
            c_node = orig_kv_node
            while c_node != mapped_node:
                path.append((c_node, prev[c_node][1], prev[c_node][0]))
                c_node = prev[c_node][0]
         
            try:
                print "Path from Ontology node '%s' to closest key-value %s is %s" % (str(mapped_node).encode('utf-8'), str(orig_kv_node).encode('utf-8'), str(path).encode('utf-8'))
            except:
                pass

            return path[0][0].key, path[0][0].value, path

        def is_consequent(mapped_node):
            consequent_edges = Set([Inference("Custom consequent term"), 
                Inference("Linked term of superterm"), Inference("Cell culture from cell line"),
                Inference("Infer developmental stage"), Inference("Inferred from cell line data")])
            for edge in text_mining_graph.reverse_edges[mapped_node]:
                if edge not in consequent_edges:
                     return False
            return True

        mapped_terms = []
        exclude_ids = Set([x.property_term_id for x in text_mining_graph.real_value_nodes])
        exclude_nodes = Set([x for x in text_mining_graph.ontology_term_nodes if x.term_id in exclude_ids])
        for o_node in text_mining_graph.ontology_term_nodes:
            r = extract_mapping(o_node)
            if r:
                consequent = is_consequent(o_node)
                mapped_terms.append(MappedTerm(o_node.term_id, consequent, r[0], r[1], r[2]))
        
        real_value_properties = []
        for rv_node in text_mining_graph.real_value_nodes:
            r = extract_mapping(rv_node)
            if r:
                consequent = is_consequent(rv_node)
                real_value_properties.append(RealValueProperty(rv_node.property_term_id, consequent, rv_node.value, rv_node.unit_term_id, r[0], r[1], r[2]))

        return mapped_terms, real_value_properties

 

#########################################################################################
#   Graph transformation stages
#########################################################################################

class InitKeyValueTokens_Stage_OLD:
    """
    From each key-value node, create a token node for the key and a token node for the 
    value with a DerivesInto edge between the key-value node and each new token node.
    """
    def run(self, text_mining_graph):
        for kv_node in text_mining_graph.key_val_nodes:
            key_node = TokenNode(kv_node.key)
            val_node = TokenNode(kv_node.value)
            text_mining_graph.add_edge(kv_node, key_node, DerivesInto("key"))
            text_mining_graph.add_edge(kv_node, val_node, DerivesInto("val"))
        return text_mining_graph



class InitKeyValueTokens_Stage:
    def run(self, text_mining_graph):
        curr_index = 0
        for kv_node in text_mining_graph.key_val_nodes:

            ind_start = curr_index
            ind_end = curr_index + len(kv_node.key)
            key_node = TokenNode(kv_node.key, ind_start, ind_end)
            print "KEY NODE %s HAS STARTING INTERVALS %d - %d" % (key_node, ind_start, ind_end)
            text_mining_graph.add_edge(kv_node, key_node, DerivesInto("key"))

            curr_index = ind_end
            ind_start = curr_index
            ind_end = curr_index + len(kv_node.value)
            val_node = TokenNode(kv_node.value, ind_start, ind_end)
            print "VAL NODE %s HAS STARTING INTERVALS %d - %d" % (val_node, ind_start, ind_end)
            text_mining_graph.add_edge(kv_node, val_node, DerivesInto("val"))

            curr_index = ind_end
        return text_mining_graph


class KeyValueFilter_Stage:
    def __init__(self, perform_filter_keys=True, perform_filter_values=True):
        with open(FILTER_KEYS_JSON, "r") as f:
            j = json.load(f)
            self.filter_keys = Set(j["filter_keys"])
            self.filter_values = Set(j["filter_values"])
            self.perform_filter_keys = perform_filter_keys
            self.perform_filter_values = perform_filter_values

    def run(self, text_mining_graph):
        if self.perform_filter_keys:
            remove_kv_nodes = [x for x in text_mining_graph.key_val_nodes if x.key in self.filter_keys]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        if self.perform_filter_values:
            remove_kv_nodes = [x for x in text_mining_graph.key_val_nodes if x.value in self.filter_values]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        return text_mining_graph


class TwoCharMappings_Stage():
    def __init__(self):
        with open(TWO_CHAR_MAPPINGS_JSON, "r") as f:
            self.str_to_mappings = json.load(f)
    
    def run(self, text_mining_graph):
        for t_node in text_mining_graph.token_nodes:
            if t_node.token_str in self.str_to_mappings:
                for t_id in self.str_to_mappings[t_node.token_str]:
                    match_node = OntologyTermNode(t_id) 
                    edge = FuzzyStringMatch(t_node.token_str, t_node.token_str, "CUSTOM_TWO_CHAR_MATCH")
                    text_mining_graph.add_edge(t_node, match_node, edge)
        return text_mining_graph
        

class Synonyms_Stage(object):
    def __init__(self, syn_set_name, syn_f):
        self.syn_set_name = syn_set_name
        syn_sets_f = pr.resource_filename(resource_package, join("synonym_sets", syn_f))
        with open(syn_sets_f, "r") as f:
            self.syn_sets = [Set(x) for x in json.load(f)]

    def run(self, text_mining_graph):
        edge = DerivesInto("synonym via %s synonym set" % self.syn_set_name)
        tnode_to_edges = defaultdict(lambda: []) # Buffer to store new edges
        for t_node in text_mining_graph.token_nodes:
            for syn_set in self.syn_sets:
                if t_node.token_str in syn_set:
                    for syn in syn_set:
                        tnode_to_edges[t_node].append(TokenNode(syn, t_node.origin_gram_start, t_node.origin_gram_end))

        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)

        return text_mining_graph



class CellosaurusSynonyms_Stage(Synonyms_Stage):
    def __init__(self):
        super(CellosaurusSynonyms_Stage, self).__init__("Cellosaurus", "cvcl_syn_sets.json")



class ManuallyAnnotatedSynonyms_Stage(Synonyms_Stage):
    def __init__(self):
        super(ManuallyAnnotatedSynonyms_Stage, self).__init__("Manually Annotated", "custom_syn_sets.json")



class NGram_Stage:
    def __init__(self):
        self.n_thresh = 8

    def run(self, text_mining_graph):
        edge = DerivesInto("N-Gram")                # All edges will be of this type
        tnode_to_edges = defaultdict(lambda: [])    # Buffer to store new edges

        # Compute the length of each gram
        for t_node in text_mining_graph.token_nodes:
            grams, intervals = get_ngrams(t_node.token_str, 1)
            max_n = min(self.n_thresh, len(grams))
            for n in range(1, max_n):
                n_gram_strs, intervals = get_ngrams(t_node.token_str, n)
                for i in range(len(n_gram_strs)):
                    g_str = n_gram_strs[i]
                    interval = intervals[i]
                    new_t_node = TokenNode(g_str, t_node.origin_gram_start + interval[0], t_node.origin_gram_start + interval[1])
                    tnode_to_edges[t_node].append(new_t_node)
        
        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)

        return text_mining_graph


class HierarchicalNGram_Stage:
    def __init__(self):
        self.n_thresh = 8

    def run(self, text_mining_graph):
        edge = DerivesInto("N-Gram")                # All edges will be of this type
        tnode_to_edges = defaultdict(lambda: [])    # Buffer to store new edges

        curr_t_nodes = text_mining_graph.token_nodes
        while len(curr_t_nodes) > 0:
            new_nodes = []
            for t_node in curr_t_nodes:

                # Compute the length of each gram
                grams, intervals = get_ngrams(t_node.token_str, 1)
                n = min(self.n_thresh, len(grams) - 1)
                if n < 1:
                    continue

                n_gram_strs, intervals = get_ngrams(t_node.token_str, n)
                for i in range(len(n_gram_strs)):
                    g_str = n_gram_strs[i]
                    interval = intervals[i]
                    new_t_node = TokenNode(g_str, t_node.origin_gram_start + interval[0], t_node.origin_gram_start + interval[1])
                    new_nodes.append(new_t_node)
                    tnode_to_edges[t_node].append(new_t_node)

            curr_t_nodes = new_nodes

        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)

        return text_mining_graph


      
class Lowercase_Stage:
    def run(self, text_mining_graph):
        edge = DerivesInto("Lowercase")
        tnode_to_edges = defaultdict(lambda: [])
        for t_node in text_mining_graph.token_nodes:
            tnode_to_edges[t_node] = TokenNode(t_node.token_str.lower(), t_node.origin_gram_start, t_node.origin_gram_end)
 
        for source_node, target_node in tnode_to_edges.iteritems():
            text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph



class PropertySpecificSynonym_Stage:
    def __init__(self):
        with open(PROPERTY_SPECIFIC_SYNONYMS_JSON, "r") as f:
            self.property_id_to_syn_sets = json.load(f)

    def run(self, text_mining_graph):

        for kv_node in text_mining_graph.key_val_nodes:
            # Find all downstream nodes of the 'key' token-nodes 
            key_term_nodes = Set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        key_term_nodes.update( text_mining_graph.downstream_nodes(t_node))

            key_term_nodes = [x for x in key_term_nodes if isinstance(x, OntologyTermNode) and x.term_id in self.property_id_to_syn_sets]

            if len(key_term_nodes) == 0:
                continue

            # Gather all nodes that are children of the this key-value's value
            for key_term_node in key_term_nodes:
                nodes_check_syn = Set()
                edge = DerivesInto("Property-specific synonym")
                for edge in text_mining_graph.forward_edges[kv_node]:
                    if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                        for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                            for down_node in text_mining_graph.downstream_nodes(t_node):
                                if isinstance(down_node, TokenNode):
                                    for syn_set in self.property_id_to_syn_sets[key_term_node.term_id]:
                                        if down_node.token_str in syn_set:
                                            for syn in syn_set:
                                                if syn != down_node.token_str:
                                                    new_node = TokenNode(syn, down_node.origin_gram_start, down_node.origin_gram_end)
                                                    text_mining_graph.add_edge(down_node, new_node, edge)

        return text_mining_graph


class BlockCellLineNonCellLineKey_Stage:
    def __init__(self):
        self.cell_line_keys = Set(["EFO:0000322", "EFO:0000324"])
        self.cell_line_phrases = Set(["source_name"])

        cvcl_og, x,y = load_ontology.load("4") 

        # Cell line terms are all CVCL terms and those terms in the EFO they link to
        self.cell_line_terms = Set(cvcl_og.id_to_term.keys())
        with open(TERM_TO_LINKED_ANCESTOR_JSON, "r") as f:
            term_to_suplinked = json.load(f)
            for t_id in cvcl_og.id_to_term:
                if t_id in term_to_suplinked:
                    self.cell_line_terms.update(term_to_suplinked[t_id])

    def run(self, text_mining_graph):

        kv_nodes_cellline_val = deque()
        for kv_node in text_mining_graph.key_val_nodes:
            # Find children of the key that indicate they encode a cell-line value 
            key_term_nodes = Set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        key_term_nodes.update( text_mining_graph.downstream_nodes(t_node))

            print "KEY TERM NODES ARE: %s" % key_term_nodes

            key_term_nodes = [x for x in key_term_nodes if (isinstance(x, OntologyTermNode) and x.term_id in self.cell_line_keys) or (isinstance(x, CustomMappingTargetNode) and x.rep_str in self.cell_line_phrases)]

            print "FOUND CELL-LINE KEYS: %s" % key_term_nodes
            if len(key_term_nodes) > 0:
                kv_nodes_cellline_val.append(kv_node)

    
        remove_nodes = deque()
        for kv_node in text_mining_graph.key_val_nodes:
            if kv_node in kv_nodes_cellline_val:
                continue

            # Gather all nodes that are children of the key-nodes that do not contain a cell-line value
            # Remove them if they represent a cell line
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        for down_node in text_mining_graph.downstream_nodes(t_node):
                            if isinstance(down_node, OntologyTermNode):
                                if down_node.term_id in self.cell_line_terms:

                                    # Check whether this node has a path to a cell line term node
                                    dist, prev = text_mining_graph.shortest_path(down_node, use_reverse_edges=True)
                                    path_from_cell_line_key = False
                                    for cl_kv_node in kv_nodes_cellline_val:
                                        if dist[cl_kv_node] < float('inf'):
                                            path_from_cell_line_key = True
                                            break

                                    if not path_from_cell_line_key: 
                                        remove_nodes.append(down_node)

        for remove_node in remove_nodes:
            text_mining_graph.delete_node(remove_node)

        return text_mining_graph



class RemoveStem_Stage:

    def run(self, text_mining_graph):
        stemmer = PorterStemmer()
        edge = DerivesInto("Remove stem")
        tnode_to_edges = defaultdict(lambda: [])
        for t_node in text_mining_graph.token_nodes:
            unigrams = nltk_n_grams(t_node.token_str, 1)
            if len(unigrams) == 0: # TODO make sure this can't happen 
                continue
            gram_replace = unigrams[-1]
            len_last_gram = len(gram_replace)
            try:
                destemmed = t_node.token_str[:-len_last_gram] + stemmer.stem(gram_replace).encode('utf-8') 
                tnode_to_edges[t_node].append(TokenNode(destemmed, t_node.origin_gram_start, t_node.origin_gram_end))
            except KeyError:
                print "%s not found in stemming dictionary" % gram_replace
            except UnicodeDecodeError:
                print "Error decoding gram"        

        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph


class SPECIALISTLexInflectionalVariants:
    def __init__(self, specialist_lex):
        self.specialist_lex = specialist_lex

    def run(self, text_mining_graph):
        
        edge = DerivesInto("Inflectional variant")
        tnode_to_edges = defaultdict(lambda: [])
        for t_node in text_mining_graph.token_nodes:

            unigrams = nltk_n_grams(t_node.token_str, 1)
            if len(unigrams) == 0: # TODO make sure this can't happen 
                continue
            gram_replace = unigrams[-1]
            len_last_gram = len(gram_replace)
            infl_vars = self.specialist_lex.inflection_variants(gram_replace)
            for infl_var in infl_vars:
                new_str = t_node.token_str[:-len_last_gram] + infl_var
                tnode_to_edges[t_node].append(TokenNode(new_str, t_node.origin_gram_start, t_node.origin_gram_end))
            
        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph
 
class SPECIALISTSpellingVariants:
    def __init__(self, specialist_lex):
        self.specialist_lex = specialist_lex

    def run(self, text_mining_graph):

        edge = DerivesInto("Spelling variant")
        tnode_to_edges = defaultdict(lambda: [])
        for t_node in text_mining_graph.token_nodes:

            unigrams = nltk_n_grams(t_node.token_str, 1)
            if len(unigrams) == 0: # TODO make sure this can't happen 
                continue
            gram_replace = unigrams[-1]
            len_last_gram = len(gram_replace)
            infl_vars = self.specialist_lex.spelling_variants(gram_replace)
            for infl_var in infl_vars:
                new_str = t_node.token_str[:-len_last_gram] + infl_var
                tnode_to_edges[t_node].append(TokenNode(new_str, t_node.origin_gram_start, t_node.origin_gram_end))

        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph



class AnnotatedSynonyms_Stage:
    """
    Maps a token to a synonym from a specified 
    synonym mapping.
    """
    def __init__(self, synonyms):
        """
        Args:
            synonyms: dictionary mapping a string to its
            set of synonyms
        """
        self.name = "annotated_synonym"
        self.syns = synonyms

    def run(self, tokens):
        new_tokens = Set()
        for t in tokens:
            if t.t_str in self.syns:
                for syn in self.syns[t.t_str]:
                    history = list(t.history)
                    history.append((self.name, syn))
                    new_tokens.add(Token(syn, history))
        tokens.update(new_tokens)
        return tokens    

class Delimit_Stage:
    """
    Delimits each token by a given regex and sequence
    of delimited substrings are used to generate new
    set of tokens.
    """
    def __init__(self, delimiter):
        self.delimiter = delimiter

    def run(self, text_mining_graph):
        node_to_next_nodes = defaultdict(lambda: [])
        for t_node in text_mining_graph.token_nodes:
            split_t_strs = t_node.token_str.split(self.delimiter)
            if len(split_t_strs) == 1:
                continue

            curr_interval_begin = t_node.origin_gram_start
            for split_t_str in split_t_strs:
                new_t_node = TokenNode(split_t_str, curr_interval_begin, curr_interval_begin + len(split_t_str))
                node_to_next_nodes[t_node].append(new_t_node)
                curr_interval_begin += len(split_t_str) + len(self.delimiter)

        edge = DerivesInto("Delimiter")
        for s_node, next_nodes in node_to_next_nodes.iteritems():
            for t_node in next_nodes:
                text_mining_graph.add_edge(s_node, t_node, edge)

        return text_mining_graph



class FilterOntologyMatchesByPriority_Stage:
    
    def run(self, text_mining_graph):
        def is_edge_direct_match(edge):
            return edge.match_target == "TERM_NAME" \
                or edge.match_target == "EXACT_SYNONYM" \
                or edge.match_target == "ENRICHED_SYNONYM"

        def is_edge_to_node_a_match(edge, targ_node, id_space):
            return isinstance(edge, FuzzyStringMatch) \
                and isinstance(targ_node, OntologyTermNode) \
                and targ_node.term_id.split(":")[0] == id_space

        id_spaces = Set([x.term_id.split(":")[0] for x in text_mining_graph.ontology_term_nodes])
        for id_space in id_spaces:
            for t_node in text_mining_graph.token_nodes:
                # Detect whether this token matched to a term-name or exact-synonym
                discard = False
                for edge, target_nodes in text_mining_graph.forward_edges[t_node].iteritems():
                    for targ_node in target_nodes:
                        if is_edge_to_node_a_match(edge, targ_node, id_space) and is_edge_direct_match(edge):
                            discard = True
                            break
                    if discard:
                        break

                # If match to term-name or exact synonym, then delete all non-term-name and 
                # non-exact-synonym matches
                if discard:
                    del_edges = []
                    for edge, target_nodes in text_mining_graph.forward_edges[t_node].iteritems():
                        for targ_node in target_nodes:
                            if is_edge_to_node_a_match(edge, targ_node, id_space) and not is_edge_direct_match(edge):
                                del_edges.append((t_node, targ_node, edge))
        
                    for e in del_edges:
                        text_mining_graph.delete_edge(e[0], e[1], e[2])

                    # Remove orphan ontology-term-nodes
                    #del_nodes = []
                    #for o_node in text_mining_graph.ontology_term_nodes:
                    #    if o_node not in text_mining_graph.reverse_edges:
                    #        print "DELETING ORPHAN ONTOLOGY TERM NODE! %s" % o_node
                    #        del_nodes.append(o_node)
                    #for n in del_nodes:
                    #    text_mining_graph.delete_node(n)

        return text_mining_graph




class ExactStringMatching_Stage:
    def __init__(self, target_og_ids, query_len_thresh=None, match_numeric=False):
          
        #self.mappable_ogs = load_mappable_ontologies(target_og_ids)
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric

        print "Building trie..."
        tups = deque()
        self.terms_array = deque()
        curr_i = 0

        ontology_graphs = [load_ontology.load(x)[0] for x in target_og_ids]
        for og in ontology_graphs:
            for term in og.get_mappable_terms():
                self.terms_array.append(term)
                tups.append((term.name.decode('utf-8'), [curr_i]))
                for syn in term.synonyms:
                    try:
                        tups.append((syn.syn_str.decode('utf-8'), [curr_i]))
                    except UnicodeEncodeError:
                        print "Warning! Unable to decode unicode of a synonym for term %s" % term.id
                curr_i += 1
        self.map_trie = mt.RecordTrie("<i", tups)

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

    def run(self, text_mining_graph):
        c = 0

        for t_node in text_mining_graph.token_nodes:

            if c % 1000 == 0:
                print "Processed %d/%d token nodes..." % (c, len(text_mining_graph.token_nodes))
            c += 1

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) < self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            # TODO Check for matches in the Trie            
            terms = self.map_string(t_node.token_str)
            for term in terms:
                match_node = OntologyTermNode(term.id)
                if t_node.token_str == term.name:
                    text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "TERM_NAME"))
                else:
                    for syn in term.synonyms:
                        if t_node.token_str == syn.syn_str:
                            text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, syn.syn_str, "%s_SYNONYM" % syn.syn_type))

        return text_mining_graph




class FuzzyStringMatching_Stage:
    # USE PRECONSTRUCTED BK TREE TO PERFORM FUZZY MATCHING
    
    def __init__(self, thresh, query_len_thresh=None, match_numeric=False):
       
        fname = pr.resource_filename(resource_package, join("fuzzy_matching_index", "fuzzy_match_string_data.json"))
        with open(fname, "r") as f:
            self.str_to_terms = json.load(f)

        fname = pr.resource_filename(resource_package, join("fuzzy_matching_index", "fuzzy_match_bk_tree.pickle"))
        with open(fname, "r") as f:
            self.bk_tree = pickle.load(f)
        
        self.query_len_thresh = query_len_thresh
        self.thresh = thresh
        self.match_numeric = match_numeric


    def _edit_below_thresh(self, query):

        matched = []

        print "Searching for '%s' in the BK-Tree..." % query
        try:
            within_edit_thresh = self.bk_tree.query(query, 2)
        except UnicodeDecodeError:
            print "Encoding error querying BK-tree for query: '%s'" % query        
            return matched

        str1 = query
        for result in within_edit_thresh:
            
            str2 = result[1]
            #dist = result[0]
            dist = edit_distance(str1, str2)
            if dist > 2:
                continue
        
            print "Retrieved '%s' from BK-tree. It has edit distance of %f" % (str2.encode('utf-8'), dist)
            len1 = len(str1)
            len2 = len(str2)
            max_len = max([len1, len2])

            # If the length difference between the two strings 
            # is greater than the threshold, we can return false.
            len_diff = abs(len1-len2)
            if len_diff / max_len > self.thresh:
                continue

            norm_dist = float(dist)/float(max_len)
            if norm_dist <= self.thresh:
                for match_data in self.str_to_terms[str2]:
                    # First element of 'match_data' is term_id, second is match type
                    matched.append((str2, dist, match_data[0], match_data[1]))
        return matched

    def run(self, text_mining_graph):
        c = 0
        print "Mapping %d token nodes" % len(text_mining_graph.token_nodes)
        for t_node in text_mining_graph.token_nodes:
            c += 1
            print "Searched %d nodes in the BK-tree." % c

            print "Attempting to map token node '%s'" % t_node.token_str

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) <= self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            matched = self._edit_below_thresh(t_node.token_str)
            if len(matched) == 0:
                continue
            min_edit = min([m[1] for m in matched])
            for m in matched:
                matched_str = m[0]
                edit_dist = m[1]
                term_id = m[2]
                match_type = m[3]

                # Only map to the best matches
                if edit_dist > min_edit:
                    continue

                match_node = OntologyTermNode(term_id)
                print "Mapping %s to %s" % (matched_str, term_id)
                text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, matched_str, match_type))

        return text_mining_graph

    





class FuzzyStringMatching_Stage_THRESH_EDIT:
    def __init__(self, target_og, thresh, query_len_thresh=None, match_numeric=False):
        """
        Args:
            target_og: target ontology graph
            thresh: the edit-distance threshold for which   
                if edit_dist < thresh, then call it a match
            query_len_thresh: the threshold for which if a token-node's
                token-string is less than or equal to this threshold, it won't be 
                matched
            match_numberic: if False, then don't match a token-string that is
                numeric. Otherwise, perform matching as usual
        """
        self.target_og = target_og
        self.thresh = thresh
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric

    def run(self, text_mining_graph):
        c = 0
        for t_node in text_mining_graph.token_nodes:

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) <= self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            for term in self.target_og.get_mappable_terms():
                matched = edit_below_thresh(t_node.token_str, term.name, self.thresh)
                if matched:
                    match_node = OntologyTermNode(term.id)
                    text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "TERM_NAME"))
                for syn in term.synonyms:
                    matched = edit_below_thresh(t_node.token_str, syn.syn_str, self.thresh)
                    if matched:
                        match_node = OntologyTermNode(term.id)
                        text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "%s_SYNONYM" % syn.syn_type))
        return text_mining_graph


class FuzzyStringMatching_Stage_OLD:
    def __init__(self, ontology_ids, thresh, query_len_thresh=None, match_numeric=False):
        """
        Args:
            ontology_ids: list of ontology ids this stage will map to
            thresh: the edit-distance threshold for which   
                if edit_dist < thresh, then call it a match
            query_len_thresh: the threshold for which if a token-node's
                token-string is less than or equal to this threshold, it won't be 
                matched
            match_numberic: if False, then don't match a token-string that is
                numeric. Otherwise, perform matching as usual
        """
        self.ontology_graphs = [load_ontology.load(x)[0] for x in ontology_ids]
        self.thresh = thresh
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric

    def run(self, text_mining_graph):
        c = 0
        print "Mapping %d token nodes" % len(text_mining_graph.token_nodes)
        for t_node in text_mining_graph.token_nodes:
            c += 1
            print "Mapped %d nodes" % c

            print "Attempting to map token node '%s'" % t_node.token_str

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) <= self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            for og in self.ontology_graphs:
                mappable_terms = og.get_mappable_terms()
                #print "Comparing edit distance to %d ontology terms..." % len(mappable_terms)
                #c = 1 
                for term in mappable_terms:
                    #if c == 500:
                    #    print "Compared to %d/%d terms" % (c, len(mappable_terms))
                    #c += 1
                    # REMOVE #######
                    matched = edit_below_thresh(t_node.token_str, term.name, self.thresh)
                    if matched:
                        match_node = OntologyTermNode(term.id)
                        text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "TERM_NAME"))
                    for syn in term.synonyms:
                        matched = edit_below_thresh(t_node.token_str, syn.syn_str, self.thresh)
                        if matched:
                            match_node = OntologyTermNode(term.id)
                            text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "%s_SYNONYM" % syn.syn_type))
        return text_mining_graph


class FuzzyStringMatching_Stage_NEW:
    # PRECOMPUTE ONTOLOGY TERM CHARACTER FREQUENCIES
    def __init__(self, ontology_ids, thresh, query_len_thresh=None, match_numeric=False):
        """
        Args:
            ontology_ids: list of ontology ids this stage will map to
            thresh: the edit-distance threshold for which   
                if edit_dist < thresh, then call it a match
            query_len_thresh: the threshold for which if a token-node's
                token-string is less than or equal to this threshold, it won't be 
                matched
            match_numberic: if False, then don't match a token-string that is
                numeric. Otherwise, perform matching as usual
        """
        self.ontology_graphs = [load_ontology.load(x)[0] for x in ontology_ids]
        
        print "Precomputing character frequencies for terms in ontologies..."
        self.str_to_char_freq = {}
        for og in self.ontology_graphs:
            for id, term in og.id_to_term.iteritems():
                self.str_to_char_freq[term.name] = Counter(term.name)
                for syn in term.synonyms:
                    self.str_to_char_freq[syn.syn_str] = Counter(syn.syn_str) 

        self.thresh = thresh
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric

    def run(self, text_mining_graph):
        c = 0
        print "Mapping %d token nodes" % len(text_mining_graph.token_nodes)
        for t_node in text_mining_graph.token_nodes:
            c += 1
            print "Mapped %d nodes" % c

            print "Attempting to map token node '%s'" % t_node.token_str

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) <= self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            for og in self.ontology_graphs:
                mappable_terms = og.get_mappable_terms()
                #print "Comparing edit distance to %d ontology terms..." % len(mappable_terms)
                #c = 1 
                for term in mappable_terms:
                    term_char_freqs = Counter(t_node.token_str)
                    matched = edit_below_thresh_precomputed(t_node.token_str, term.name, self.thresh, term_char_freqs, self.str_to_char_freq)
                    #matched = edit_below_thresh_precomputed(t_node.token_str, term.name, self.thresh, self.str_to_char_freq)
                    if matched:
                        match_node = OntologyTermNode(term.id)
                        text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "TERM_NAME"))
                    for syn in term.synonyms:
                        matched = edit_below_thresh_precomputed(t_node.token_str, syn.syn_str, self.thresh, term_char_freqs, self.str_to_char_freq)
                        #matched = edit_below_thresh_precomputed(t_node.token_str, syn.syn_str, self.thresh, self.str_to_char_freq)
                        if matched:
                            match_node = OntologyTermNode(term.id)
                            text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "%s_SYNONYM" % syn.syn_type))
        return text_mining_graph


class FuzzyStringMatching_Stage_KDTREE:
    # USE KD-TREE TO FIND CANDIDATE CLOSEST POINTS. LOAD KD_TREE FROM PICKLE FILE
    def __init__(self, thresh, query_len_thresh=None, match_numeric=False):
        """
        Args:
            thresh: the edit-distance threshold for which   
                if edit_dist < thresh, then call it a match
            query_len_thresh: the threshold for which if a token-node's
                token-string is less than or equal to this threshold, it won't be 
                matched
            match_numberic: if False, then don't match a token-string that is
                numeric. Otherwise, perform matching as usual
        """

        fname = pr.resource_filename(resource_package, join("fuzzy_matching_index", "fuzzy_match_string_data.json"))
        with open(fname, "r") as f:
            self.string_data = json.load(f)

        fname = pr.resource_filename(resource_package, join("fuzzy_matching_index", "fuzzy_match_ball_tree.pickle"))
        with open(fname, "r") as f:
            self.ball_tree = pickle.load(f)
        
        self.vec_scaffold = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890,./:;'()"
        self.thresh = thresh
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric

    def _vectorize_string(self, stri):
        freqs = Counter(stri)
        vec = np.zeros(len(self.vec_scaffold))
        for i,c in enumerate(self.vec_scaffold):
            if c in freqs:
                vec[i] = freqs[c]
        return vec

    def _edit_below_thresh(self, stri, thresh):

        print
        print
        print "Matching '%s' to KD-Tree..." % stri

        # Get all strings within edit distance of 2 from query string
        stri_vec = self._vectorize_string(stri)
        print "The vector is %s" % stri_vec

        rad = 2.0
        within_rad = self.ball_tree.query_radius([stri_vec], r=2)

        str1 = stri
        matched = []
        for str2_i in within_rad[0]:

            str2 = self.string_data[str2_i][0]
            print "Retrieved '%s' from kd-tree" % str2.encode('utf-8')

            len1 = len(str1)
            len2 = len(str2)
            max_len = max([len1, len2])

            # If the length difference between the two strings 
            # is greater than the threshold, we can return false.
            len_diff = abs(len1-len2)
            if len_diff / max_len > thresh:
                continue

            print "Attempting to compute edit distance..."
            dist = edit_distance(str1, str2)
            print "Edit distance is %s" % str(dist)
            norm_dist = float(dist)/float(max_len)
            if norm_dist <= thresh:
                matched.append((str2, dist, str2_i))

        return matched

    def run(self, text_mining_graph):
        c = 0
        print "Mapping %d token nodes" % len(text_mining_graph.token_nodes)
        for t_node in text_mining_graph.token_nodes:
            c += 1
            print "Mapped %d nodes" % c
            print "Attempting to map token node '%s'" % t_node.token_str

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) < self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            matched = self._edit_below_thresh(t_node.token_str, 0.1)
            print "Matced are %s" % matched
            if len(matched) == 0:
                continue
            min_edit = min([m[1] for m in matched])
            for m in matched:
                matched_str = m[0]
                edit_dist = m[1]
                str_i = m[2]
               
                # Only map to the best matches
                if edit_dist > min_edit:
                    continue 

                og_id = self.string_data[str_i][1]
                term_id = self.string_data[str_i][2]
                match_type = self.string_data[str_i][3]

                match_node = OntologyTermNode(term_id)
                print "Mapping %s to %s" % (matched_str, term_id)
                text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, matched_str, match_type))

        return text_mining_graph



class FuzzyStringMatching_Stage_KD_TREE_OLD:
    # USE KD-TREE TO FIND CANDIDATE CLOSEST POINTS
    def __init__(self, ontology_ids, thresh, query_len_thresh=None, match_numeric=False):
        """
        Args:
            ontology_ids: list of ontology ids this stage will map to
            thresh: the edit-distance threshold for which   
                if edit_dist < thresh, then call it a match
            query_len_thresh: the threshold for which if a token-node's
                token-string is less than or equal to this threshold, it won't be 
                matched
            match_numberic: if False, then don't match a token-string that is
                numeric. Otherwise, perform matching as usual
        """
        self.vec_scaffold = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890,./:;'()"

        self.ontology_graphs = [load_ontology.load(x)[0] for x in ontology_ids]

        self.str_to_char_freq = {}
        
        print "Constructing string-vectors from ontologies..."
        self.ont_strings = deque()
        vecs = deque()
        self.terms = deque()
        for og in self.ontology_graphs:
            for id, term in og.id_to_term.iteritems():
                vec = self._vectorize_string(term.name)
                self.ont_strings.append(term.name)
                vecs.append(vec)
                self.terms.append(term)
                for syn in term.synonyms:   
                    vec = self._vectorize_string(syn.syn_str)
                    self.ont_strings.append(syn.syn_str)
                    vecs.append(vec)
                    self.terms.append(term)

        print "Building KD-Tree..."
        #self.kd_tree = KDTree(vecs)
        self.ball_tree = BallTree(vecs, metric='manhattan')

        self.thresh = thresh
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric

    def _vectorize_string(self, stri):
        freqs = Counter(stri)
        vec = np.zeros(len(self.vec_scaffold))
        for i,c in enumerate(self.vec_scaffold):
            if c in freqs:
                vec[i] = freqs[c]
        return vec 


    def _edit_below_thresh(self, stri, thresh):

        print 
        print
        print "Matching '%s' to KD-Tree..." % stri

        # Get all strings within edit distance of 3 from
        # target string

        stri_vec = self._vectorize_string(stri)

        rad = 2.0
        #within_rad = self.kd_tree.query_ball_point([stri_vec], rad, p=1)
        within_rad = self.ball_tree.query_radius([stri_vec], r=2)
        #print "WITHIN_RAD: %s" % str(within_rad)  
    
 
        str1 = stri

        matched = []
        for str2_i in within_rad[0]:

            str2 = self.ont_strings[str2_i]
            print "Retrieved '%s' from kd-tree" % str2

            len1 = len(str1)
            len2 = len(str2)
            max_len = max([len1, len2])

            # If the length difference between the two strings 
            # is greater than the threshold, we can return false.
            len_diff = abs(len1-len2)
            if len_diff / max_len > thresh:
                continue
 

            #c1 = str1_char_freqs
            #c2 = str_to_char_freq[str2]
            #lowerbound_dist = 0
            #for ch, fr in c1.iteritems():
            #    if ch not in c2:
            #        lowerbound_dist += (0.5 * fr)
            #    else:
            #        lowerbound_dist += (0.5 * abs(c1[ch] - c2[ch]))
            #
            #for ch, fr in c2.iteritems():
            #    if ch not in c1:
            #      lowerbound_dist += (0.5 * fr)
            #
            #if lowerbound_dist / max_len > thresh:
            #    continue

            print "Attempting to compute edit distance..."
            dist = edit_distance(str1, str2)
            print "Edit distance is %s" % str(dist)
            norm_dist = float(dist)/float(max_len)
            if norm_dist <= thresh:
                matched.append((str2, dist, str2_i))

        return matched


    def run(self, text_mining_graph):
        c = 0
        print "Mapping %d token nodes" % len(text_mining_graph.token_nodes)
        for t_node in text_mining_graph.token_nodes:
            c += 1
            print "Mapped %d nodes" % c

            print "Attempting to map token node '%s'" % t_node.token_str

            # Skip matching tokens according to fuzzy-matching parameters
            if self.query_len_thresh and len(t_node.token_str) <= self.query_len_thresh:
                continue
            if not self.match_numeric and is_number(t_node.token_str):
                continue

            matched = self._edit_below_thresh(t_node.token_str, 0.1)
            print "Matced are %s" % matched
            for m in matched:
                matched_str = m[0]
                edit_dist = m[1]
                term_i = m[2]


                term = self.terms[term_i]
                match_node = OntologyTermNode(term.id)
                if matched_str == term.name:
                    print "Mapping %s to %s" % (matched_str, term.id)
                    text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "TERM_NAME"))
                else:
                    for syn in term.synonyms:
                        if matched_str == syn.syn_str:
                            print "Mapping %s to %s" % (matched_str, term.id)
                            text_mining_graph.add_edge(t_node, match_node, FuzzyStringMatch(t_node.token_str, term.name, "%s_SYNONYM" % syn.syn_type))
                            break
                            
        return text_mining_graph




class RemoveSubIntervalOfMatchedBlockAncestralLink_Stage:
    def run(self, text_mining_graph):

        def is_superphrase(super_node, sub_node):
            matched_interval = (super_node.origin_gram_start, super_node.origin_gram_end)
            if sub_node.origin_gram_start == matched_interval[0] and sub_node.origin_gram_end < matched_interval[1]:
                #print "1. sub_node %s in interval (%s, %s)" % (sub_node, matched_interval[0], matched_interval[1])
                return True
            elif sub_node.origin_gram_start > matched_interval[0] and sub_node.origin_gram_end == matched_interval[1]:
                #print "2. sub_node %s in interval (%s, %s)" % (sub_node, matched_interval[0], matched_interval[1])
                return True
            elif sub_node.origin_gram_start > matched_interval[0] and sub_node.origin_gram_end < matched_interval[1]:
                #print "3. sub_node %s in interval (%s, %s)" % (sub_node, matched_interval[0], matched_interval[1])
                return True
            else:
                return False

        mapped_t_nodes = deque()
        for mt_node in text_mining_graph.mapping_target_nodes:
            if not mt_node in text_mining_graph.reverse_edges:
                continue
            for edge in text_mining_graph.reverse_edges[mt_node]:
                for source_node in text_mining_graph.reverse_edges[mt_node][edge]:
                    if not isinstance(source_node, TokenNode):
                        continue
                    mapped_t_nodes.append(source_node)
                    #matched_intervals.append((source_node.origin_gram_start, source_node.origin_gram_end))

        for t_node in mapped_t_nodes:
            superphrase_nodes = Set()
            for mapped_t_node in mapped_t_nodes:
                if is_superphrase(mapped_t_node, t_node):
                    superphrase_nodes.add(mapped_t_node)
            if len(superphrase_nodes) == 0:
                continue


            print
            print
            print "Node %s has superphrase nodes: %s" % (t_node, superphrase_nodes)

            exclude_edges = Set([DerivesInto("N-Gram"),  DerivesInto("Delimiter")])
            superphrase_node_to_reachable = {x:text_mining_graph.downstream_nodes(x, exclude_edges=exclude_edges) for x in superphrase_nodes}
           
            mapped_from_t = [x for x in text_mining_graph.get_children(t_node) if isinstance(x, MappingTargetNode)]
            
            keep_as_mappable = Set()
            for mft in mapped_from_t:
                # Check if this mapped node from this token node is also reachable from all superphrase nodes.
                # If so, we want to maintain its reachability from the current token node.
                reachable_from_all_supernodes = True
                for superphrase_node, reachable_from_superphrase in superphrase_node_to_reachable.iteritems():
                    if mft not in reachable_from_superphrase:
                        reachable_from_all_supernodes = False
                        break
                if reachable_from_all_supernodes:
                    print "The node %s that is reachable from the token node is reachable from all of the superphrase nodes." % mft
                    keep_as_mappable.add(mft)

            del_edges = deque()
            for edge in text_mining_graph.forward_edges[t_node]:
                for targ_node in text_mining_graph.forward_edges[t_node][edge]:
                    reachable_from_targ_node = Set(text_mining_graph.downstream_nodes(targ_node))
                    if len(reachable_from_targ_node.intersection(keep_as_mappable)) == 0:
                        del_edges.append((t_node, targ_node, edge))
                    else:
                        print "Target node from the current node, %s, can reach a node that we want to keep as mappable: %s" % (targ_node, reachable_from_targ_node.intersection(keep_as_mappable))

            for d in del_edges:
                #print "This edge did not make the cut! %s --%s--> %s" % (t_node, edge, targ_node)
                text_mining_graph.delete_edge(d[0], d[1], d[2])
            

        return text_mining_graph


# TODO Deprecated. Remove this
class RemoveSubIntervalOfMatchedBlockAncestralLink_Stage_OLD:
    def run(self, text_mining_graph):

        def is_superphrase(super_node, sub_node):
            matched_interval = (super_node.origin_gram_start, super_node.origin_gram_end)
            if sub_node.origin_gram_start == matched_interval[0] and sub_node.origin_gram_end < matched_interval[1]:
                print "1. sub_node %s in interval (%s, %s)" % (sub_node, matched_interval[0], matched_interval[1])
                return True
            elif sub_node.origin_gram_start > matched_interval[0] and sub_node.origin_gram_end == matched_interval[1]:
                print "2. sub_node %s in interval (%s, %s)" % (sub_node, matched_interval[0], matched_interval[1])
                return True
            elif sub_node.origin_gram_start > matched_interval[0] and sub_node.origin_gram_end < matched_interval[1]:
                print "3. sub_node %s in interval (%s, %s)" % (sub_node, matched_interval[0], matched_interval[1])
                return True
            else:
                return False


        # Get matches that are achieved through linked-ancestral terms
        #dont_remove = Set()
        #for o_node in text_mining_graph.ontology_term_nodes:
        #    for edge in text_mining_graph.reverse_edges[o_node]:
        #        if isinstance(edge, Inference) and edge.inference_type == "Linked term of superterm":
        #            print "Don't remove %s. It has a %s edge to %s" % (o_node, edge, text_mining_graph.reverse_edges[o_node][edge])
        #            dont_remove.add(o_node)
        #matched_intervals = deque()

        mapped_t_nodes = deque()
        for mt_node in text_mining_graph.mapping_target_nodes:
            if not mt_node in text_mining_graph.reverse_edges:
                continue
            for edge in text_mining_graph.reverse_edges[mt_node]:
                for source_node in text_mining_graph.reverse_edges[mt_node][edge]:
                    if not isinstance(source_node, TokenNode):
                        continue
                    mapped_t_nodes.append(source_node)
                    #matched_intervals.append((source_node.origin_gram_start, source_node.origin_gram_end))

        to_remove = deque()
        for t_node in text_mining_graph.token_nodes:
            remove_maps = False
            for mapped_t_node in mapped_t_nodes:
                if is_superphrase(mapped_t_node, t_node):
                    remove_maps = True
                    break
                    

            #for matched_interval in matched_intervals:
            #    if t_node.origin_gram_start == matched_interval[0] and t_node.origin_gram_end < matched_interval[1]:
            #        print "1. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
            #        remove_maps = True
            #        break
            #    elif t_node.origin_gram_start > matched_interval[0] and t_node.origin_gram_end == matched_interval[1]:
            #        print "2. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
            #        remove_maps = True
            #        break
            #    elif t_node.origin_gram_start > matched_interval[0] and t_node.origin_gram_end < matched_interval[1]:
            #        print "3. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
            #        remove_maps = True
            #        break

            if remove_maps:
                
                # Find all nodes the current token node maps to
                mapped_targets_w_edges = []
                for edge in text_mining_graph.forward_edges[t_node]:
                    for targ_node in text_mining_graph.forward_edges[t_node][edge]:
                        if isinstance(targ_node, MappingTargetNode):
                            mapped_targets_w_edges.append((targ_node, edge))
 
                dont_remove = False # If this is true, we leave the mappings from the current token node alone
                                    # Otherwise, we delete mapping edges from this token node

                # Look at all the nodes that this target maps to. For each of target node, check which other mapping
                # targets in the graph consequently map to this target node. For each of the nodes that consequently map to a
                # target node, check which token nodes derive into them. If that token node is a superphrase
                # of the current token node, then we leave the current token node alone and don't delete its 
                # mappings.
                for mapped_node, edge in mapped_targets_w_edges:
                    for edge in text_mining_graph.reverse_edges[mapped_node]:
                        if isinstance(edge, Inference):  
                            print "1. Mapped node %s is mapped to via the edge %s" % (mapped_node, edge)
                            for linked_super_node in text_mining_graph.reverse_edges[mapped_node][edge]:
                                print "2. From node %s" % linked_super_node
                                for linked_super_edge in text_mining_graph.reverse_edges[linked_super_node]:
                                    if isinstance(linked_super_edge, FuzzyStringMatch):
                                        print "3. That node is then mapped to via the fuzzy string match edge %s" % linked_super_edge
                                        for in_linked_super_node in text_mining_graph.reverse_edges[linked_super_node][linked_super_edge]:
                                            print "4. Which maps from %s" % in_linked_super_node
                                            if isinstance(in_linked_super_node, TokenNode) and is_superphrase(in_linked_super_node, t_node):
                                                print "5. Well it turns out to be a token-node and also a superphrase! We don't want to remove this!!"
                                                dont_remove = True
                                                break
                                        else:
                                            continue  # executed if the loop ended normally (no break)
                                        break  
                                else:
                                    continue
                                break  
                            else:
                                continue
                            break
                    else:
                        continue
                    break            

                if not dont_remove:
                    for targ_node, edge in mapped_targets_w_edges:
                        print "Node %s has a larger mapping token node. Deleting edge %s --%s--> %s" % (t_node, t_node, edge, targ_node)
                        text_mining_graph.delete_edge(t_node, targ_node, edge)

        #print "BLOCK REMOVAL FROM: %s" % dont_remove
        #print "Will be removing the nodes: %s" % to_remove
        #for node in to_remove:
        #    dont_remove = False
        #    for edge in text_mining_graph.reverse_edges[node]:
        #        if isinstance(edge, Inference) and edge.inference_type == "Linked term of superterm":
        #            for o_node in text_mining_graph.reverse_edges[node][edge]:
        #                for o_node_edge in text_mining_graph.reverse_edges[o_node]:
        #                    for t_node in [x for x in text_mining_graph.reverse_edges[o_node][o_node_edge] if isinstance(x, TokenNode)]:
        #                 
        #
        #    if node not in dont_remove:
        #        text_mining_graph.delete_node(node)

        return text_mining_graph


# TODO Deprecated. Remove this
class RemoveSubIntervalOfMatchedBlockAncestralLink_Stage_OLD:
    def run(self, text_mining_graph):

        # Get matches that are achieved through linked-ancestral terms
        dont_remove = Set()
        for o_node in text_mining_graph.ontology_term_nodes:
            for edge in text_mining_graph.reverse_edges[o_node]:
                if isinstance(edge, Inference) and edge.inference_type == "Linked term of superterm":
                    print "Don't remove %s. It has a %s edge to %s" % (o_node, edge, text_mining_graph.reverse_edges[o_node][edge])
                    dont_remove.add(o_node)
        

        matched_intervals = deque()

        for mt_node in text_mining_graph.mapping_target_nodes:
            if not mt_node in text_mining_graph.reverse_edges:
                continue
            for edge in text_mining_graph.reverse_edges[mt_node]:
                for source_node in text_mining_graph.reverse_edges[mt_node][edge]:
                    if not isinstance(source_node, TokenNode):
                        continue
                    matched_intervals.append((source_node.origin_gram_start, source_node.origin_gram_end))

        to_remove = deque()
        for t_node in text_mining_graph.token_nodes:
            remove_maps = False
            for matched_interval in matched_intervals:
                if t_node.origin_gram_start == matched_interval[0] and t_node.origin_gram_end < matched_interval[1]:
                    print "1. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
                    remove_maps = True
                    #to_remove.append(t_node)
                    break
                elif t_node.origin_gram_start > matched_interval[0] and t_node.origin_gram_end == matched_interval[1]:
                    print "2. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
                    remove_maps = True
                    #to_remove.append(t_node)
                    break
                elif t_node.origin_gram_start > matched_interval[0] and t_node.origin_gram_end < matched_interval[1]:
                    print "3. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
                    remove_maps = True
                    #to_remove.append(t_node)
                    break
            if remove_maps:
                for edge in text_mining_graph.forward_edges[t_node]:
                    for targ_node in text_mining_graph.forward_edges[t_node][edge]:
                        if isinstance(targ_node, MappingTargetNode):
                            to_remove.append(targ_node)

        print "BLOCK REMOVAL FROM: %s" % dont_remove
        print "Will be removing the nodes: %s" % to_remove
        for node in to_remove:
            if node not in dont_remove:
                text_mining_graph.delete_node(node)

        return text_mining_graph


# TODO Deprecated. Remove this
class RemoveSubIntervalOfMatched_Stage:
    def run(self, text_mining_graph):

        matched_intervals = deque()

        for mt_node in text_mining_graph.mapping_target_nodes:
            if not mt_node in text_mining_graph.reverse_edges:
                continue
            for edge in text_mining_graph.reverse_edges[mt_node]:
                for source_node in text_mining_graph.reverse_edges[mt_node][edge]:
                    if not isinstance(source_node, TokenNode):
                        continue
                    matched_intervals.append((source_node.origin_gram_start, source_node.origin_gram_end))

        to_remove = deque()
        for t_node in text_mining_graph.token_nodes:
            for matched_interval in matched_intervals:
                if t_node.origin_gram_start == matched_interval[0] and t_node.origin_gram_end < matched_interval[1]:
                    print "1. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
                    to_remove.append(t_node)
                    break
                elif t_node.origin_gram_start > matched_interval[0] and t_node.origin_gram_end == matched_interval[1]:
                    print "2. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
                    to_remove.append(t_node)
                    break
                elif t_node.origin_gram_start > matched_interval[0] and t_node.origin_gram_end < matched_interval[1]:
                    print "3. t_node %s in interval (%s, %s)" % (t_node, matched_interval[0], matched_interval[1])
                    to_remove.append(t_node)
                    break
      
        print "Will be removing the nodes: %s" % to_remove  
        for node in to_remove:
            text_mining_graph.delete_node(node)

        return text_mining_graph


class ExactMatchCustomTargets_Stage:
    def __init__(self):
        with open(NOUN_PHRASES_JSON, "r") as f:
            self.noun_phrases = Set(json.load(f)) 
     
    def run(self, text_mining_graph):
        for t_node in text_mining_graph.token_nodes:
            if t_node.token_str in self.noun_phrases:
                c_node = CustomMappingTargetNode(t_node.token_str)
                edge = FuzzyStringMatch(t_node.token_str, t_node.token_str, "CUSTOM_NOUN_PHRASE")
                text_mining_graph.add_edge(t_node, c_node, edge)
        return text_mining_graph 
        

class RemoveDownstreamEdgesOfMappedTokenNodes_Stage:
    def run(self, text_mining_graph):

        def get_source_key_val_nodes(node):
            """
            Get the key-value nodes for which there exists a path from the node
            to that key-value node in the text mining graph.
            """
            source_kv_nodes = Set()
            dist, prev = text_mining_graph.shortest_path(node, use_reverse_edges=True)
            for kv_node in text_mining_graph.key_val_nodes:
                if kv_node in dist and dist[kv_node] < float('inf'):
                    source_kv_nodes.add(kv_node)
            return source_kv_nodes

        to_remove = Set()
        deletion_occured = True
        while deletion_occured:
            deletion_occured = False

            for kv_node in text_mining_graph.key_val_nodes:

                breadth_first_nodes = text_mining_graph.downstream_nodes(kv_node, depth_first=False)
                breadth_first_nodes = [x for x in breadth_first_nodes if isinstance(x, TokenNode)]

                for t_node in breadth_first_nodes:

                    #matched_to_ontology = False
                    matched_mt_nodes = Set()
                    for edge in text_mining_graph.forward_edges[t_node]:
                        for targ_node in text_mining_graph.forward_edges[t_node][edge]:
                            if isinstance(targ_node, MappingTargetNode):
                                matched_mt_nodes.add(targ_node)
                    if len(matched_mt_nodes) == 0:
                        continue

                    print "Node %s matched to ontology terms %s! Removing downstream edges of this node..." % (t_node, matched_mt_nodes)

                    # Get the key-value nodes that derive into this node
                    source_kv_nodes = get_source_key_val_nodes(t_node)

                    # Remove all downstream nodes without a path to any other key-value node            
                    # besides the key value node that derives into this node
                    downstream_nodes = text_mining_graph.downstream_nodes(t_node)
                    print "Downstream nodes are: %s" % downstream_nodes

                    for d_node in downstream_nodes:
                        if d_node in matched_mt_nodes or d_node == t_node:
                            continue

                        d_source_kv_nodes = get_source_key_val_nodes(d_node)
                        d_source_kv_nodes = d_source_kv_nodes - source_kv_nodes
                        if len(d_source_kv_nodes) == 0:
                        #    print "Deleted node %s..." % d_node
                            text_mining_graph.delete_node(d_node)
                            deletion_occured = True
                            break

                    if deletion_occured:
                        break 
                if deletion_occured:
                    break
                #deletion_occured = False
           

        return text_mining_graph


class RemoveDownstreamEdgesOfMappedTokenNodes_Stage_OLD:
    def run(self, text_mining_graph):

        def get_source_key_val_nodes(node):
            """
            Get the key-value nodes for which there exists a path from the node
            to that key-value node in the text mining graph.
            """
            source_kv_nodes = Set()
            dist, prev = text_mining_graph.shortest_path(node, use_reverse_edges=True)
            for kv_node in text_mining_graph.key_val_nodes:
                if kv_node in dist and dist[kv_node] < float('inf'):
                    source_kv_nodes.add(kv_node)
            return source_kv_nodes

        to_remove = Set()
        for t_node in text_mining_graph.token_nodes:
            #matched_to_ontology = False
            matched_o_nodes = Set()
            for edge in text_mining_graph.forward_edges[t_node]:
                for targ_node in text_mining_graph.forward_edges[t_node][edge]:
                    if isinstance(targ_node, MappingTargetNode):
                        matched_o_nodes.add(targ_node)
            if len(matched_o_nodes) == 0:
                continue

            print "Node %s matched to ontology terms %s! Removing downstream edges of this node..." % (t_node, matched_o_nodes) 

            # Get the key-value nodes that derive into this node
            source_kv_nodes = get_source_key_val_nodes(t_node)

            # Remove all downstream nodes without a path to any other key-value node            
            # besides the key value node that derives into this node
            downstream_nodes = text_mining_graph.downstream_nodes(t_node)

            #print "The downstream nodes are: %s" % downstream_nodes

            for d_node in downstream_nodes:
                if d_node in matched_o_nodes or d_node == t_node:
                    continue

                d_source_kv_nodes = get_source_key_val_nodes(d_node)
                d_source_kv_nodes = d_source_kv_nodes - source_kv_nodes
                if len(d_source_kv_nodes) == 0:
                    print "We want to remove %s..." % d_node
                    to_remove.add(d_node)

        print "The nodes we really want to remove are %s" % to_remove
        
        for node in to_remove:
            print "Deleted node %s..." % node
            text_mining_graph.delete_node(node)

        return text_mining_graph


class CellLineToImpliedDisease_Stage:
    def __init__(self):
        with open(CELL_LINE_TO_IMPLIED_DISEASE_JSON, "r") as f:
            self.term_to_implied_terms = json.load(f)

    def run(self, text_mining_graph):

        node_to_new_edges = defaultdict(lambda: [])
        edge = Inference("Cell line to implied disease")
        for ont_node in text_mining_graph.ontology_term_nodes:
            if ont_node.term_id in self.term_to_implied_terms:
                for implied_term_id in self.term_to_implied_terms[ont_node.term_id]:
                    new_ont_node = OntologyTermNode(implied_term_id)
                    node_to_new_edges[ont_node].append((new_ont_node, edge))

        for node, new_edges in node_to_new_edges.iteritems():
            for e in new_edges:
                text_mining_graph.add_edge(node, e[0], e[1])

        return text_mining_graph

class AcronymToExpansion_Stage:
    def __init__(self):
        with open(ACRONYM_TO_EXPANSION_JSON, "r") as f:
            self.acr_to_expansions = json.load(f)

    def run(self, text_mining_graph):
        node_to_new_edges = defaultdict(lambda: [])
        edge = Inference("Acronym to expansion")

        for t_node in text_mining_graph.token_nodes:
            if t_node.token_str in self.acr_to_expansions:
                for expansion in self.acr_to_expansions[t_node.token_str]:
                    new_t_node = TokenNode(expansion, t_node.origin_gram_start, t_node.origin_gram_end)
                    node_to_new_edges[t_node].append((new_t_node, edge))

        for node, new_edges in node_to_new_edges.iteritems():
            for e in new_edges:
                text_mining_graph.add_edge(node, e[0], e[1])

        return text_mining_graph




####################################
#   Only to be used with ATCC data
####################################

class ATCCKeyValueFilter_Stage:
    def __init__(self, perform_filter_keys=True, perform_filter_values=True):
        with open(CELL_LINE_FILTER_KEYS_JSON, "r") as f:
            j = json.load(f)
            self.filter_keys = Set(j["filter_keys"])
            self.filter_values = Set(j["filter_values"])
            self.perform_filter_keys = perform_filter_keys
            self.perform_filter_values = perform_filter_values

    def run(self, text_mining_graph):
        if self.perform_filter_keys:
            remove_kv_nodes = [x for x in text_mining_graph.key_val_nodes if x.key in self.filter_keys]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        if self.perform_filter_values:
            remove_kv_nodes = [x for x in text_mining_graph.key_val_nodes if x.value in self.filter_values]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        return text_mining_graph



#####################################
#   'Real value' extractions
#####################################

class ExtractRealValue_Stage:

    def __init__(self):
        with open(REAL_VALUE_PROPERTIES, "r") as f:
            j = json.load(f)
            self.real_val_tids = j["property_term_ids"]
            self.default_units = j["default_units"]

    def run(self, text_mining_graph):
        print
        print "Extracting real-value properties..."
    
        for kv_node in text_mining_graph.key_val_nodes:
            print "Checking whether key-value pair node %s encodes a real-value property."

            # Find ontology-terms that refer to real-valued properties 
            real_val_term_nodes = Set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        real_val_term_nodes.update( text_mining_graph.downstream_nodes(t_node))

            real_val_term_nodes = [x for x in real_val_term_nodes if isinstance(x, OntologyTermNode) and x.term_id in self.real_val_tids]
            print "Set of real-value properties we are searching for: %s" % self.real_val_tids
            print "Found property nodes: %s" % real_val_term_nodes
            if len(real_val_term_nodes) == 0:
                continue
        
            # Gather all nodes that are children of the this key-value's value
            real_val_candidates = Set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        real_val_candidates.update(text_mining_graph.downstream_nodes(t_node))    

            print "The real-value candidates are: %s" % real_val_candidates
            numeric_nodes = [x for x in real_val_candidates if isinstance(x, TokenNode) and is_number(x.token_str)]
            print "Found numeric nodes: %s" % numeric_nodes
            unit_nodes = [x for x in real_val_candidates if isinstance(x, OntologyTermNode) and x.term_id.split(":")[0] == "UO"]
            print "Found unit nodes: %s" % unit_nodes

            # If there is one real-value ontology term, one numeric token, and one unit node, then create real-value-property node
            edge = DerivesInto("Real-value extraction") # TODO should use a different edge-type
            if len(real_val_term_nodes) == 1:
                prop_term_node = list(real_val_term_nodes)[0]
                for numeric_node in numeric_nodes:
                    if len(unit_nodes) == 1:
                        unit_node = list(unit_nodes)[0]
                        rv_node = RealValuePropertyNode(prop_term_node.term_id, float(numeric_node.token_str), unit_node.term_id)
                        text_mining_graph.add_edge(prop_term_node, rv_node, edge)
                        text_mining_graph.add_edge(numeric_node, rv_node, edge)
                        text_mining_graph.add_edge(unit_node, rv_node, edge)
                    elif len(unit_nodes) == 0:
                        if prop_term_node.term_id in self.default_units:
                            default_unit_id = self.default_units[prop_term_node.term_id]
                        else:
                            default_unit_id = "missing"
                        rv_node = RealValuePropertyNode(prop_term_node.term_id, float(numeric_node.token_str), default_unit_id)
                        text_mining_graph.add_edge(prop_term_node, rv_node, edge)
                        text_mining_graph.add_edge(numeric_node, rv_node, edge)  
                    else:
                        rv_node = RealValuePropertyNode(prop_term_node.term_id, float(numeric_node.token_str), None)
                        text_mining_graph.add_edge(prop_term_node, rv_node, edge)
                        text_mining_graph.add_edge(numeric_node, rv_node, edge)
                    
        return text_mining_graph       


class ParseTimeWithUnit_Stage:
    """Parse artifacts that represent units of time and look 
    something like '48h'. That is, expand '48h' to '48 hour'."""

    def __init__(self):
        self.regex = r'^([0-9]*)\s*(h|hr|mo|d|min)$'
        self.unit_to_expansion = {
            "hr": "hour",
            "h": "hour",
            "mo": "month",
            "d": "day",
            "min": "minute"
        }

    def run(self, text_mining_graph):
        tnode_to_edges = defaultdict(lambda: [])
        unit_t_nodes = Set()
        for t_node in text_mining_graph.token_nodes:
            m = re.search(self.regex, t_node.token_str)
            try:
                value = m.group(1)
                unit = m.group(2)
                value_start = t_node.origin_gram_start
                value_end = t_node.origin_gram_start + len(value)
                unit_start = t_node.origin_gram_end - len(unit)
                unit_end = t_node.origin_gram_end
                value_t_node = TokenNode(value, value_start, value_end)
                unit_t_node = TokenNode(unit, unit_start, unit_end)
                tnode_to_edges[t_node].append(value_t_node)
                tnode_to_edges[t_node].append(unit_t_node)
                unit_t_nodes.add(unit_t_node)
            except AttributeError as e:
                pass
        
        parse_edge = DerivesInto("Parse time and unit")
        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, parse_edge)

        unit_edge = DerivesInto("Parse as unit synonym")
        for unit_t_node in unit_t_nodes:
            expanded_unit = self.unit_to_expansion[unit_t_node.token_str]
            syn_unit_t_node = TokenNode(
                expanded_unit, 
                unit_t_node.origin_gram_start,
                unit_t_node.origin_gram_end)       
            text_mining_graph.add_edge(unit_t_node, syn_unit_t_node, unit_edge)
           
        return text_mining_graph 


#######################################
#   Stages for mapping 'consequent'
#   terms
#######################################

class CustomConsequentTerms_Stage:
    def __init__(self):
        with open(CUST_TERM_TO_CONSEQ_TERMS_JSON, "r") as f:
            self.term_to_consequent = json.load(f)

    def run(self, text_mining_graph):
        node_to_new_edges = defaultdict(lambda: [])
        edge = Inference("Custom consequent term")

        for ont_node in text_mining_graph.ontology_term_nodes:
            if ont_node.term_id in self.term_to_consequent:
                for implied_term_id in self.term_to_consequent[ont_node.term_id]:
                    #new_ont_node = OntologyTermNode(implied_term_id, consequent=True)
                    new_ont_node = OntologyTermNode(implied_term_id)
                    node_to_new_edges[ont_node].append((new_ont_node, edge))

        for node, new_edges in node_to_new_edges.iteritems():
            for e in new_edges:
                text_mining_graph.add_edge(node, e[0], e[1])

        return text_mining_graph
 

class LinkedTermsOfSuperterms_Stage:
    def __init__(self):
        with open(TERM_TO_LINKED_ANCESTOR_JSON, "r") as f:
            self.term_to_implied_terms = json.load(f)

    def run(self, text_mining_graph):

        node_to_new_edges = defaultdict(lambda: [])
        edge = Inference("Linked term of superterm")

        for ont_node in text_mining_graph.ontology_term_nodes:
            if ont_node.term_id in self.term_to_implied_terms:
                for implied_term_id in self.term_to_implied_terms[ont_node.term_id]:
                    #new_ont_node = OntologyTermNode(implied_term_id, consequent=True)
                    new_ont_node = OntologyTermNode(implied_term_id)
                    node_to_new_edges[ont_node].append((new_ont_node, edge))

        for node, new_edges in node_to_new_edges.iteritems():
            for e in new_edges:
                text_mining_graph.add_edge(node, e[0], e[1])

        return text_mining_graph



class ConsequentCulturedCell_Stage:
    
    def run(self, text_mining_graph):

        edge = Inference("Cell culture from cell line")
        new_o_node_cl = OntologyTermNode("CL:0000010")
        new_o_node_bto = OntologyTermNode("EFO_BTO:0000214")
        cell_line_nodes = deque()
        for o_node in text_mining_graph.ontology_term_nodes:
            if o_node.namespace() == "CVCL":
                cell_line_nodes.append(o_node)

        for c_node in cell_line_nodes:
            text_mining_graph.add_edge(c_node, new_o_node_cl, edge)
            text_mining_graph.add_edge(c_node, new_o_node_bto, edge)
 
        return text_mining_graph

class ImpliedDevelopmentalStageFromAge_Stage:
    
    def run(self, text_mining_graph):
        for real_node in text_mining_graph.real_value_nodes:
            if real_node.property_term_id == "EFO:0000246" \
                and real_node.unit_term_id == "UO:0000036" \
                and real_node.value > 18:
                edge = Inference("Infer developmental stage")
                uberon_node = OntologyTermNode("EFO:0001272")
                efo_node = OntologyTermNode("UBERON:0007023")
                text_mining_graph.add_edge(real_node, uberon_node, edge)
                text_mining_graph.add_edge(real_node, efo_node, edge)
        return text_mining_graph


class InferCellLineTerms_Stage:

    def __init__(self):
        with open(CELL_LINE_TERMS_JSON, "r") as f:
            self.cvcl_to_mappings = json.load(f)

    def run(self, text_mining_graph):
        onode_to_edges = defaultdict(lambda: [])
        for o_node in text_mining_graph.ontology_term_nodes:
            if o_node.namespace() == "CVCL" and o_node.term_id in self.cvcl_to_mappings:
                for t_id in self.cvcl_to_mappings[o_node.term_id]["mapped_terms"]:
                    new_o_node = OntologyTermNode(t_id)
                    onode_to_edges[o_node].append(new_o_node) 
                for real_val in self.cvcl_to_mappings[o_node.term_id]["real_value_properties"]:
                    new_rv_node = RealValuePropertyNode(real_val[0], real_val[1], real_val[2])
                    onode_to_edges[o_node].append(new_rv_node)

        edge = Inference("Inferred from cell line data")
        for source_node, target_nodes in onode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph



#####################################
#   Helper methods
#####################################

def is_number(q_str):
    try:
        float(q_str)
        return True
    except ValueError:
        return False

def get_ngrams(text, n):
    words = nltk.word_tokenize(text)
    new_words = []
    for word in words:
        if word == "``":
            new_words.append('"')
        elif word == "''":
            new_words.append('"')
        else:
            new_words.append(word)
    words = new_words
        

    if not words:
        return [], []

    text_i = 0
    curr_word = words[0]
    word_i = 0
    word_char_i = 0

    word_to_indices = defaultdict(lambda: [])
    for text_i in range(len(text)):
        if word_char_i == len(words[word_i]):
            word_i += 1
            word_char_i = 0
        if word_i == len(words):
            break
        
        if text[text_i] ==  words[word_i][word_char_i]:
            word_to_indices[word_i].append(text_i)
            word_char_i += 1
        text_i += 1

    n_grams = []
    intervals = []
    for i in range(0, len(words)-n+1):
        grams = words[i:i+n]
        text_char_begin = word_to_indices[i][0]
        text_char_end = word_to_indices[i+n-1][-1]
        n_gram = text[text_char_begin: text_char_end+1]
        n_grams.append(n_gram)
        intervals.append((text_char_begin, text_char_end+1))       

    return n_grams, intervals    



def nltk_n_grams(in_str, n):
    result_grams = []
    regex = "[0-9a-zA-Z]+.*[0-9a-zA-Z]+"
    n_grams, intervals = get_ngrams(in_str, n)
    for gram in n_grams:
        m = re.search(regex, gram)
        if m:
            match = m.group(0)
            result_grams.append(match)
        else:
            print "Regex failed on %s" % gram
    return result_grams



def all_nltk_n_grams(in_str):
    thresh = 8
    result_grams = Set()
    for n in range(1, thresh):
        n_grams = nltk_n_grams(in_str, n)
        for gram in n_grams: 
            result_grams.add(gram)
    print result_grams
    return list(result_grams)


def edit_below_thresh(str1, str2, thresh):
    len1 = len(str1)
    len2 = len(str2)
    max_len = max([len1, len2])

    # If the length difference between the two strings 
    # is greater than the threshold, we can return false.
    len_diff = abs(len1-len2)
    if len_diff / max_len > thresh:
        return False

    # Compute lower bound edit distance based on character
    # frequency
    c1 = Counter(str1)
    c2 = Counter(str2)
    lowerbound_dist = 0
    for ch, fr in c1.iteritems():
        if ch not in c2:
            lowerbound_dist += (0.5 * fr)
        else:
            lowerbound_dist += (0.5 * abs(c1[ch] - c2[ch]))

    for ch, fr in c2.iteritems():
        if ch not in c1:
            lowerbound_dist += (0.5 * fr)

    if lowerbound_dist / max_len > thresh:
        return False

    print "Attempting to compute edit distance..."
    dist = edit_distance(str1, str2)
    print "Edit distance is %s" % str(dist)
    norm_dist = float(dist)/float(max_len)
    return norm_dist <= thresh


def edit_below_thresh_precomputed(str1, str2, thresh, str1_char_freqs, str_to_char_freq):
    len1 = len(str1)
    len2 = len(str2)
    max_len = max([len1, len2])

    # If the length difference between the two strings 
    # is greater than the threshold, we can return false.
    len_diff = abs(len1-len2)
    if len_diff / max_len > thresh:
        return False

    # Compute lower bound edit distance based on character
    # frequency
    c1 = str1_char_freqs
    c2 = str_to_char_freq[str2]
    lowerbound_dist = 0
    for ch, fr in c1.iteritems():
        if ch not in c2:
            lowerbound_dist += (0.5 * fr)
        else:
            lowerbound_dist += (0.5 * abs(c1[ch] - c2[ch]))

    for ch, fr in c2.iteritems():
        if ch not in c1:
            lowerbound_dist += (0.5 * fr)

    if lowerbound_dist / max_len > thresh:
        return False

    print "Attempting to compute edit distance..."
    dist = edit_distance(str1, str2)
    print "Edit distance is %s" % str(dist)
    norm_dist = float(dist)/float(max_len)
    return norm_dist <= thresh



 
def main():

    #print get_ngrams("Lonza Walkersville, Inc (Walkersville, MD)", 2)
    #return

    #print nltk_n_grams("polysubstance abuse (NO diabetes, hypertension, coronary artery disease, cancer)")


    #tag_to_val = {"BioSampleModel": "Human",
    #    "Fusion": "Negative",
    #    "age": "15",
    #    "biomaterial provider": "Memorial Sloan Kettering Cancer Center",
    #    "disease": "Embryonal rhabdomyosarcoma",
    #    "isolate": "RMS66",
    #    "sample type": "Metastatic (recurrence)",
    #    "sex": "Unknown",
    #    "tissue": "Tumor",
    #    "cell line": "HeLa-S3",
    #    "cell type": "fibroblast"}

    #tag_to_val = {
    #    "disease": "head and neck squamus cell carcinoma"}

    #tag_to_val = {
    #    "blahahaldksfj": "cord blood derived cells",
    #    "lkajsdl;kfjsd": "whole blood"
    #}

    #tag_to_val = {
    #    "cell line": "U2OS"
    #}

    #tag_to_val = {
    #    "AA BB CC": "DD EE FF GG"}

    #tag_to_val = {
    #    "cell line": "HeLa-S3",
    #    "organism": "Homo sapiens",
    #    "treatment": "Amanitin (RNA pol II inhibitor)"}

    #tag_to_val = {
    #    "disease": "Breast Cancers",
    #    "Tissue": "Breast Tumor Tissue"}

    #tag_to_val = {
    #    "Cell type": "Human Umbilical Vein Endothelial Cells"
    #}

    #tag_to_val = {
    #    "cell line": "HeLa"}

    #tag_to_val = {
    #    "source name": "HeLa"}
    
    #tag_to_val = {
    #    "blahaha": "HeLa",
    #    "haha": "HeLa"}

    #tag_to_val = {"haha": "iPSC"}

    #tag_to_val = {
    #    "source": "foreskin fibroblast"}

    #tag_to_val = {
    #    "age": "24 year"
    #}


    #tag_to_val = {
    #    "BioSampleModel": "Generic",
    #    "age at diagnosis": "11",
    #    "deep ulcer": "No",
    #    "diagnosis": "CD",
    #    "gender": "Male",
    #    "histopathology": "Normal",
    #    "paris age at diagnosis": "A1b",
    #    "source_name": "cCD with Normal histopathology in A1b Male"
    #}

    #tag_to_val = {
    #    "blood type": "A+",
    #    "description": "poly(A)+"
    #}

    #tag_to_val = {
    #    "sample": "culture fibroblast"
    #}

    #tag_to_val = {
    #    "age at diagnosis": "11",
    #    "paris age at diagnosis": "A1b"
    #}

    #tag_to_val = {
    #    "cell line": "K562",
    #    "grna target": "globin HS2 enhancer",
    #    "source_name": "Cultured K562 cells",
    #    "transduced gene": "dCas9-KRAB"
    #}

    #tag_to_val = {
    #    "a": "1111 2222 3333 4444 5555 6666 7777 8888 9999 10101010 11111111",
    #}

    #tag_to_val = {
    #    "a": "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ KKKK"
    #}

    #tag_to_val = {
    #    "Cell type": "Human Umbilical Vein Endothelial Cells",
    #    "Cell type abbr.": "HUVEC",
    #    "Donor age": "newborn",
    #    "Donor cause-of-death": "-",
    #    "Donor gender": "male",
    #    "Donor race": "Arabic",
    #    "Internal label": "WadaLab-IHEC30",
    #    "Lot. no.": "03321",
    #    "Manufacture Date": "01-JUN-2013",
    #    "Provider": "Lonza Walkersville, Inc (Walkersville, MD)",
    #    "Sample type": "Cell",
    #    "Source name": "Human Umbilical Vein Endothelial Cells (HUVEC) Lot.03321",
    #    "Total cell count": "380000",
    #    "Total cell count unit": "cell per mili-litter",
    #    "Valid until": "Not applicable when stored below -150 degrees celsius",
    #    "Viability": "93%"}

    #tag_to_val = {
    #    "BioSourceType": "frozen_sample",
    #    "DiseaseLocation": "oropharynx",
    #    "MetastasisFreeSurvivalDelay": "8 month",
    #    "MetastasisFreeSurvivalEvent": "1",
    #    "Organism": "Homo sapiens",
    #    "age": "56 year",
    #    "disease": "head and neck squamous cell carcinoma",
    #    "disease staging": "4",
    #    "organism part": "oropharynx",
    #    "sex": "male"}

    #tag_to_val = {
    #    "disease": "head and neck squamous cell carcinoma"}

    tag_to_val = {
        "timepoint": "48hr"
    }


    key_val_filt = KeyValueFilter_Stage()
    init_tokens_stage = InitKeyValueTokens_Stage()
    hier_ngram = HierarchicalNGram_Stage()
    ngram = NGram_Stage()
    lower_stage = Lowercase_Stage()
    destem = RemoveStem_Stage()
    cvcl_syn = CellosaurusSynonyms_Stage()
    man_at_syn = ManuallyAnnotatedSynonyms_Stage()
    infer_cell_line = InferCellLineTerms_Stage()
    #exact_match = ExactStringMatching_Stage(["13"], query_len_thresh=3)
    prop_spec_syn = PropertySpecificSynonym_Stage()
    downstream = RemoveDownstreamEdgesOfMappedTokenNodes_Stage()
    subphrase = RemoveSubIntervalOfMatched_Stage()   
    filt_match_priority = FilterOntologyMatchesByPriority_Stage()
    match_cust_targs = ExactMatchCustomTargets_Stage()
    cellline_to_implied_disease = CellLineToImpliedDisease_Stage() 
    infer_dev_stage = ImpliedDevelopmentalStageFromAge_Stage() 
    cell_culture = ConsequentCulturedCell_Stage()
    linked_super = LinkedTermsOfSuperterms_Stage()
    cust_conseq = CustomConsequentTerms_Stage() 
    delimit = Delimit_Stage('+')
    delimit2 = Delimit_Stage('-')
    block_cell_line_key = BlockCellLineNonCellLineKey_Stage()
    subphrase_linked = RemoveSubIntervalOfMatchedBlockAncestralLink_Stage()
    acr_to_expan = AcronymToExpansion_Stage()
    time_unit = ParseTimeWithUnit_Stage()

    #efo_og, x, y = load_ontology.load("13") 
    #fuzzy_match_EFO = FuzzyStringMatching_Stage(efo_og, 0.1, query_len_thresh=3)

    #cvcl_og, x, y = load_ontology.load("4")
    #fuzzy_match_CVCL = FuzzyStringMatching_Stage(cvcl_og, 0.1, query_len_thresh=3)
    

    #exact_match = ExactStringMatching_Stage(["1", "2", "4", "5", "7", "8", "9"], query_len_thresh=3)
    fuzzy_match = FuzzyStringMatching_Stage(0.1, query_len_thresh=3)
    real_val = ExtractRealValue_Stage()


    #stages = [init_tokens_stage, hier_ngram, delimit, delimit2]
    #stages = [init_tokens_stage, hier_ngram, exact_match, match_cust_targs, block_cell_line_key]
    #stages = [init_tokens_stage, hier_ngram, exact_match, cellline_to_implied_disease]   
    #stages = [init_tokens_stage, acr_to_expan]
    
    stages = [init_tokens_stage, time_unit, fuzzy_match, real_val]

    p = Pipeline(stages, defaultdict(lambda: 1.0))
    
    mapped_terms, real_val_props =  p.run(tag_to_val)
    result = {"Mapped ontology terms": [x.to_dict() for x in mapped_terms], "Real valued properties":  [x.to_dict() for x in real_val_props]}
    print json.dumps(result, indent=4, separators=(',', ': '))
    
    #print json.dumps([x.to_dict() for x in p.run(tag_to_val)], indent=4, separators=(',', ': '))



if __name__ == "__main__":
    main()




