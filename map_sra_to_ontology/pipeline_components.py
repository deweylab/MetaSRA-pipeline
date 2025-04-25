###########################################################################
#   Components for building metadata mapping pipelines. 
###########################################################################

import json
from collections import defaultdict, deque
import re
import pickle

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance

import pkg_resources as pr
import json
import os
from os.path import join

import load_ontology
from text_reasoning_graph import *
import ball_tree_distance
from load_specialist_lex import SpecialistLexicon

import pybktree
from pybktree import BKTree
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
TERM_ARTIFACT_COMBOS_JSON = pr.resource_filename(resource_package, join("metadata", "term_artifact_combo.json"))

TOKEN_SCORING_STRATEGY = defaultdict(lambda: 1) # TODO We want an explicit score dictionary

VERBOSE = False

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
    def __init__(
        self, 
        property_id, 
        consequent, 
        value, 
        unit_id, 
        orig_key, 
        orig_val, 
        mapping_path
    ):
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
        tokens = set()
        for tag, val in tag_to_val.iteritems():
            kv_node = KeyValueNode(
                tag.encode('utf-8'), 
                val.encode('utf-8')
            )    
            tm_graph.add_node(kv_node)

        # Process stages of pipeline
        for stage in self.stages:
            tm_graph = stage.run(tm_graph)

        if VERBOSE:
            print "\n---------GRAPH----------"
            print tm_graph
            print "------------------------\n"

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

            # Extract path from key-value to node TODO This might not work for all matches
            path = []
            c_node = orig_kv_node
            while c_node != mapped_node:
                path.append((
                    c_node, 
                    prev[c_node][1], 
                    prev[c_node][0]
                ))
                c_node = prev[c_node][0]
        
            if VERBOSE: 
                try:
                    print "Path from ontology node '%s' to closest key-value %s is %s" % (
                        str(mapped_node).encode('utf-8'), 
                        str(orig_kv_node).encode('utf-8'), 
                        str(path).encode('utf-8')
                    )
                except:
                    pass

            return path[0][0].key, path[0][0].value, path

        def is_consequent(mapped_node):
            consequent_edges = set([
                Inference("Custom consequent term"), 
                Inference("Linked term of superterm"), 
                Inference("Cell culture from cell line"),
                Inference("Infer developmental stage"), 
                Inference("Inferred from cell line data")
            ])
            for edge in text_mining_graph.reverse_edges[mapped_node]:
                if edge not in consequent_edges:
                     return False
            return True

        mapped_terms = []
        exclude_ids = set([
            x.property_term_id 
            for x in text_mining_graph.real_value_nodes
        ])
        exclude_nodes = set([
            x 
            for x in text_mining_graph.ontology_term_nodes 
            if x.term_id in exclude_ids
        ])
        for o_node in text_mining_graph.ontology_term_nodes:
            r = extract_mapping(o_node)
            if r:
                consequent = is_consequent(o_node)
                mapped_terms.append(
                    MappedTerm(
                        o_node.term_id, 
                        consequent, 
                        r[0], 
                        r[1], 
                        r[2]
                    )
                )
        
        real_value_properties = []
        for rv_node in text_mining_graph.real_value_nodes:
            r = extract_mapping(rv_node)
            if r:
                consequent = is_consequent(rv_node)
                real_value_properties.append(
                    RealValueProperty(
                        rv_node.property_term_id, 
                        consequent, 
                        rv_node.value, 
                        rv_node.unit_term_id, 
                        r[0], 
                        r[1], 
                        r[2]
                    )
                )

        return mapped_terms, real_value_properties

 

###################################################################################
#   Graph transformation stages
###################################################################################

class InitKeyValueTokens_Stage:
    def run(self, text_mining_graph):
        print "Initializing text reasoning graph..."
        curr_index = 0
        for kv_node in text_mining_graph.key_val_nodes:

            ind_start = curr_index
            ind_end = curr_index + len(kv_node.key)
            key_node = TokenNode(kv_node.key, ind_start, ind_end)
            text_mining_graph.add_edge(kv_node, key_node, DerivesInto("key"))

            curr_index = ind_end
            ind_start = curr_index
            ind_end = curr_index + len(kv_node.value)
            val_node = TokenNode(kv_node.value, ind_start, ind_end)
            text_mining_graph.add_edge(kv_node, val_node, DerivesInto("val"))

            curr_index = ind_end

            if VERBOSE:
                print "Generated key node: %s" % (key_node)
                print "Generated value node: %s" % (val_node)

        return text_mining_graph


class KeyValueFilter_Stage:
    def __init__(
        self, 
        perform_filter_keys=True, 
        perform_filter_values=True
    ):
        with open(FILTER_KEYS_JSON, "r") as f:
            j = json.load(f)
            self.filter_keys = set(j["filter_keys"])
            self.filter_values = set(j["filter_values"])
            self.perform_filter_keys = perform_filter_keys
            self.perform_filter_values = perform_filter_values

    def run(self, text_mining_graph):
        print "Filtering key-value pairs..."
        if self.perform_filter_keys:
            remove_kv_nodes = [
                x 
                for x in text_mining_graph.key_val_nodes 
                if x.key in self.filter_keys
            ]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        if self.perform_filter_values:
            remove_kv_nodes = [
                x 
                for x in text_mining_graph.key_val_nodes 
                if x.value in self.filter_values
            ]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        return text_mining_graph


class TwoCharMappings_Stage():
    def __init__(self):
        with open(TWO_CHAR_MAPPINGS_JSON, "r") as f:
            self.str_to_mappings = json.load(f)
    
    def run(self, text_mining_graph):
        print "Matching specified two-character artifacts..."
        for t_node in text_mining_graph.token_nodes:
            if t_node.token_str in self.str_to_mappings:
                for t_id in self.str_to_mappings[t_node.token_str]:
                    match_node = OntologyTermNode(t_id) 
                    edge = FuzzyStringMatch(
                        t_node.token_str, 
                        t_node.token_str, 
                        "CUSTOM_TWO_CHAR_MATCH",
                        edit_dist=0
                    )
                    text_mining_graph.add_edge(
                        t_node, 
                        match_node, 
                        edge
                    )
        return text_mining_graph
        

class Synonyms_Stage(object):
    def __init__(self, syn_set_name, syn_f):
        self.syn_set_name = syn_set_name
        syn_sets_f = pr.resource_filename(
            resource_package, 
            join("synonym_sets", syn_f)
        )
        with open(syn_sets_f, "r") as f:
            self.syn_sets = [set(x) for x in json.load(f)]

    def run(self, text_mining_graph):
        print "Searching for synonyms..."
        edge = DerivesInto("synonym via %s synonym set" % self.syn_set_name)
        tnode_to_edges = defaultdict(lambda: []) # Buffer to store new edges
        for t_node in text_mining_graph.token_nodes:
            for syn_set in self.syn_sets:
                if t_node.token_str in syn_set:
                    for syn in syn_set:
                        tnode_to_edges[t_node].append(
                            TokenNode(
                                syn, 
                                t_node.origin_gram_start, 
                                t_node.origin_gram_end
                            )
                        )

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
        print "Generating N-grams..."
        edge = DerivesInto("N-Gram") # All edges will be of this type
        tnode_to_edges = defaultdict(lambda: []) # Buffer to store new edges

        # Compute the length of each gram
        for t_node in text_mining_graph.token_nodes:
            grams, intervals = get_ngrams(t_node.token_str, 1)
            max_n = min(self.n_thresh, len(grams))
            for n in range(1, max_n):
                n_gram_strs, intervals = get_ngrams(t_node.token_str, n)
                for i in range(len(n_gram_strs)):
                    g_str = n_gram_strs[i]
                    interval = intervals[i]
                    new_t_node = TokenNode(
                        g_str, 
                        t_node.origin_gram_start + interval[0], 
                        t_node.origin_gram_start + interval[1]
                    )
                    tnode_to_edges[t_node].append(new_t_node)
        
        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)

        return text_mining_graph

      
class Lowercase_Stage:
    def run(self, text_mining_graph):
        print "Generating lower-cased artifacts..."
        edge = DerivesInto("Lowercase")
        tnode_to_edges = defaultdict(lambda: [])
        for t_node in text_mining_graph.token_nodes:
            tnode_to_edges[t_node] = TokenNode(
                t_node.token_str.lower(), 
                t_node.origin_gram_start, 
                t_node.origin_gram_end
            )
 
        for source_node, target_node in tnode_to_edges.iteritems():
            text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph


class PropertySpecificSynonym_Stage:
    def __init__(self):
        with open(PROPERTY_SPECIFIC_SYNONYMS_JSON, 'r') as f:
            self.property_id_to_syn_sets = json.load(f)

    def run(self, text_mining_graph):
        print "Searching for property-specific synonyms..."
        for kv_node in text_mining_graph.key_val_nodes:
            # Find all downstream nodes of the 'key' token-nodes 
            key_term_nodes = set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        key_term_nodes.update(
                            text_mining_graph.downstream_nodes(t_node)
                        )

            key_term_nodes = [
                x 
                for x in key_term_nodes 
                if isinstance(x, OntologyTermNode) 
                and x.term_id in self.property_id_to_syn_sets
            ]

            if len(key_term_nodes) == 0:
                continue

            # Gather all nodes that are children of the this key-value's value
            for key_term_node in key_term_nodes:
                nodes_check_syn = set()
                edge = DerivesInto("Property-specific synonym")
                for edge in text_mining_graph.forward_edges[kv_node]:
                    if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                        for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                            for down_node in text_mining_graph.downstream_nodes(t_node):
                                if isinstance(down_node, TokenNode):
                                    for syn_set in self.property_id_to_syn_sets[key_term_node.term_id]:
                                        #print "Is the issue with %s" % down_node.token_str
                                        #print down_node.token_str
                                        down_node_str = down_node.token_str.decode('utf-8')
                                        if down_node_str in syn_set:
                                            for syn in syn_set:
                                                if syn != down_node_str:
                                                    new_node = TokenNode(syn, down_node.origin_gram_start, down_node.origin_gram_end)
                                                    text_mining_graph.add_edge(down_node, new_node, edge)

        return text_mining_graph


class BlockCellLineNonCellLineKey_Stage:
    def __init__(self):
        self.cell_line_keys = set([
            "EFO:0000322", 
            "EFO:0000324"
        ])
        #self.cell_line_phrases = set(["source_name"])
        self.cell_line_phrases = set()

        cvcl_og, x,y = load_ontology.load("4") 

        # Cell line terms are all CVCL terms and those terms in the EFO 
        # they link to
        self.cell_line_terms = set(cvcl_og.id_to_term.keys())
        with open(TERM_TO_LINKED_ANCESTOR_JSON, 'r') as f:
            term_to_suplinked = json.load(f)
            for t_id in cvcl_og.id_to_term:
                if t_id in term_to_suplinked:
                    self.cell_line_terms.update(term_to_suplinked[t_id])

    def run(self, text_mining_graph):
        print "Checking cell line terms for proper context..."
        kv_nodes_cellline_val = deque()
        for kv_node in text_mining_graph.key_val_nodes:
            # Find children of the key that indicate they encode a cell-line value 
            key_term_nodes = set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        key_term_nodes.update( 
                            text_mining_graph.downstream_nodes(t_node)
                        )

            key_term_nodes = [
                x for x in key_term_nodes 
                if (
                    isinstance(x, OntologyTermNode) 
                    and x.term_id in self.cell_line_keys
                ) 
                or (
                    isinstance(x, CustomMappingTargetNode) 
                    and x.rep_str in self.cell_line_phrases
                )
            ]

            if len(key_term_nodes) > 0:
                kv_nodes_cellline_val.append(kv_node)

    
        remove_nodes = deque()
        for kv_node in text_mining_graph.key_val_nodes:
            if kv_node in kv_nodes_cellline_val:
                continue

            # Gather all nodes that are children of the key-nodes that do not 
            # contain a cell-line value. Remove them if they represent a cell 
            # line.
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        for down_node in text_mining_graph.downstream_nodes(t_node):
                            if isinstance(down_node, OntologyTermNode):
                                if down_node.term_id in self.cell_line_terms:

                                    # Check whether this node has a path to a 
                                    # cell line term node
                                    dist, prev = text_mining_graph.shortest_path(
                                        down_node, 
                                        use_reverse_edges=True
                                    )
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


class PrioritizeExactMatchOverFuzzyMatch:
    def run(self, text_mining_graph):
        remove_edge_args = []

        # This is a list of sets of artifact-nodes. All nodes in each
        # set originate from the same intervals in the raw key-value
        # metadata and all match lexically to a target. 
        matched_t_node_sets = []
        for t_node_1 in text_mining_graph.token_nodes:
            start = t_node_1.origin_gram_start
            end = t_node_1.origin_gram_end
            t_node_set = set([t_node_1])
            for t_node_2 in text_mining_graph.token_nodes:
                if t_node_2.origin_gram_start == start and t_node_2.origin_gram_end == end:
                    t_node_set.add(t_node_2)
            matched_t_node_sets.append(t_node_set)


        # Go through each node set and if one of the nodes matched
        # to a target with edit distance = 0, then remove all matches
        # from other nodes to targets with edit distance > 0
        for t_node_set in matched_t_node_sets:
            found_exact = False
            for t_node in t_node_set:
                for edge in text_mining_graph.forward_edges[t_node]:
                    if isinstance(edge, FuzzyStringMatch) and edge.edit_dist == 0:
                        found_exact = True
                        break
           
#            # TODO REMOVE 
#            if found_exact:
#                print "Found exact match in token node set: %s" % t_node_set
#            # TODO REMOVE

            if found_exact:
                for t_node in t_node_set:
                    for edge, target_nodes in text_mining_graph.forward_edges[t_node].iteritems():
                        if isinstance(edge, FuzzyStringMatch) and edge.edit_dist > 0:
#                            print "Found match w/ edit distance > 0: %s --%s--> %s" % (t_node, edge, target_nodes)
                            remove_edge_args += [
                                (
                                    t_node,
                                    target_node,
                                    edge
                                )
                                for target_node in target_nodes
                            ]

        for remove_edge_arg in remove_edge_args:
            #print "Attempting removal: %s" % str(remove_edge_arg)
            text_mining_graph.delete_edge(*remove_edge_arg)
            #print
        return text_mining_graph


class SPECIALISTLexInflectionalVariants:
    def __init__(self, specialist_lex):
        self.specialist_lex = specialist_lex

    def run(self, text_mining_graph):
        print "Generating inflectional variants..." 
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
                tnode_to_edges[t_node].append(
                    TokenNode(
                        new_str, 
                        t_node.origin_gram_start, 
                        t_node.origin_gram_end
                    )
                )
            
        for source_node, target_nodes in tnode_to_edges.iteritems():
            for target_node in target_nodes:
                text_mining_graph.add_edge(source_node, target_node, edge)
        return text_mining_graph

 
class SPECIALISTSpellingVariants:
    def __init__(self, specialist_lex):
        self.specialist_lex = specialist_lex

    def run(self, text_mining_graph):
        print "Generating spelling variants..."
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


class Delimit_Stage:
    """
    Delimits each artifact by a given regex and sequence
    of delimited substrings are used to generate new
    set of artifacts.
    """
    def __init__(self, delimiter):
        self.delimiter = delimiter

    def run(self, text_mining_graph):
        print "Delimiting artifacts on special character '%s'..." % self.delimiter
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
    """
    If an artifact has an exact string match with multiple ontology
    terms, prioritize the ontology terms for which that matched 
    string is closer semantically to the ontology term's concept.
    For example, prioritize a match to a term name over an inexact
    synonym.
    """    
    def run(self, text_mining_graph):
        def is_edge_direct_match(edge):
            return edge.match_target == "TERM_NAME" \
                or edge.match_target == "EXACT_SYNONYM" \
                or edge.match_target == "ENRICHED_SYNONYM"

        # TODO I believe we need to check Exact matches as well as fuzzy matches? This needs to be debugged...
        def is_edge_to_node_a_match(edge, targ_node, id_space):
            return isinstance(edge, FuzzyStringMatch) \
                and isinstance(targ_node, OntologyTermNode) \
                and targ_node.term_id.split(":")[0] == id_space

        print "Filtering synonym matches by semantic similarity to term..."
        id_spaces = set([
            x.term_id.split(":")[0] 
            for x in text_mining_graph.ontology_term_nodes
        ])
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
    """
    Perform exact-string matching for all artifacts against
    the ontologies. Uses a trie data structure.
    """
    def __init__(
        self, 
        target_og_ids, 
        query_len_thresh=None, 
        match_numeric=False
    ):
          
        #self.mappable_ogs = load_mappable_ontologies(target_og_ids)
        self.query_len_thresh = query_len_thresh
        self.match_numeric = match_numeric
        if VERBOSE:
            print "Building trie..."
        tups = deque()
        self.terms_array = deque()
        curr_i = 0

        ontology_graphs = [
            load_ontology.load(x)[0] 
            for x in target_og_ids
        ]
        for og in ontology_graphs:
            for term in og.get_mappable_terms():
                self.terms_array.append(term)
                tups.append((
                    term.name.decode('utf-8'), 
                    [curr_i]
                ))
                for syn in term.synonyms:
                    try:
                        tups.append((
                            syn.syn_str.decode('utf-8'), 
                            [curr_i]
                        ))
                    except UnicodeEncodeError:
                        if VERBOSE:
                            print "Warning! Unable to decode unicode of a synonym \
                                for term %s" % term.id
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
        print "Performing exact string matching..."
        for t_node in text_mining_graph.token_nodes:
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
                    text_mining_graph.add_edge(
                        t_node, 
                        match_node, 
                        FuzzyStringMatch(
                            t_node.token_str, 
                            term.name, 
                            "TERM_NAME", 
                            edit_dist=0
                        )
                    )
                else:
                    for syn in term.synonyms:
                        if t_node.token_str == syn.syn_str:
                            text_mining_graph.add_edge(
                                t_node, 
                                match_node, 
                                FuzzyStringMatch(
                                    t_node.token_str, 
                                    syn.syn_str, "%s_SYNONYM" % syn.syn_type, 
                                    edit_dist=0
                                )
                            )

        return text_mining_graph


class FuzzyStringMatching_Stage:
    """
    Use a pre-constructed BK-tree to perform fuzzy matching
    for all artifacts against the ontologies.
    """
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

        try:
            within_edit_thresh = self.bk_tree.find(query, 2)
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
        
            if VERBOSE:
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
        print "Performing fuzzy string matching..."
        c = 0
        if VERBOSE:
            print "%d total nodes to be matched" % len(text_mining_graph.token_nodes)
        for t_node in text_mining_graph.token_nodes:
            c += 1
            if VERBOSE:
                print "Searching %dth node in the BK-tree: '%s'" % (c, t_node.token_str)

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
                if VERBOSE:
                    print "Mapping artifact '%s' to term %s" % (matched_str, term_id)
                text_mining_graph.add_edge(
                    t_node, 
                    match_node, 
                    FuzzyStringMatch(
                        t_node.token_str, 
                        matched_str, 
                        match_type, 
                        edit_dist=edit_dist
                    )
                )

        return text_mining_graph


class TermArtifactCombinations_Stage:
    """
    Given a certain ontology term is found in combination with a set 
    of artifact strings, then map to a specific ontology term. For
    example, if a sample maps to the term 'T cell' and we find the
    artifact 'CD4+', then we should map the term 
    'CD4-positive alpha-beta T cell'.  That is, the information for
    a specific ontology term may be distributed over the key value 
    pairs. 
    """   
    def __init__(self):
        with open(TERM_ARTIFACT_COMBOS_JSON, 'r') as f:
            self.term_artifact_combos = json.load(f)
 
    def run(self, text_mining_graph):
        all_artifacts = set([
            a_node.token_str
            for a_node in text_mining_graph.token_nodes
        ])
        all_term_ids = set([
            ont_node.term_id 
            for ont_node in text_mining_graph.ontology_term_nodes
        ])
        edge = Inference("Found co-occuring artifacts")
        add_edges = []
        for combo in self.term_artifact_combos:
            # Check whether the sample mapped to all required 
            # ontology terms
            required_terms = set(combo['required_terms'].keys())
            num_required_terms = len(required_terms)
            if not required_terms <= all_term_ids:            
                continue

            # Check whether we have derived all required artifacts
            num_required_artifacts = len(combo['required_artifacts'])
            num_found_artifacts = 0
            for artifact_set in combo['required_artifacts']:
                if len(set(artifact_set) & all_artifacts) > 0:
                     num_found_artifacts += 1
            if num_found_artifacts < num_required_artifacts:
                continue
       
            # If we've reached this point, then all requirements
            # have been met for mapping to the consequent term 
            conseq_term_id = combo['consequent_term']
            required_ont_nodes = [
                ont_node
                for ont_node in text_mining_graph.ontology_term_nodes
                if ont_node.term_id in required_terms
            ]
            for ont_node in required_ont_nodes:
                add_edges.append((
                    ont_node,
                    OntologyTermNode(conseq_term_id),
                    edge
                ))
        for e in add_edges:
            text_mining_graph.add_edge(e[0], e[1], e[2])
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
        
        print "Blocking subphrases of mapped superphrases..."
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
            superphrase_nodes = set()
            for mapped_t_node in mapped_t_nodes:
                if is_superphrase(mapped_t_node, t_node):
                    superphrase_nodes.add(mapped_t_node)
            if len(superphrase_nodes) == 0:
                continue

            if VERBOSE:
                print "\n\nNode %s has superphrase nodes: %s" % (t_node, superphrase_nodes)

            exclude_edges = set([DerivesInto("N-Gram"),  DerivesInto("Delimiter")])
            superphrase_node_to_reachable = {x:text_mining_graph.downstream_nodes(x, exclude_edges=exclude_edges) for x in superphrase_nodes}
           
            mapped_from_t = [x for x in text_mining_graph.get_children(t_node) if isinstance(x, MappingTargetNode)]
            
            keep_as_mappable = set()
            for mft in mapped_from_t:
                # Check if this mapped node from this token node is also reachable from all superphrase nodes.
                # If so, we want to maintain its reachability from the current token node.
                reachable_from_all_supernodes = True
                for superphrase_node, reachable_from_superphrase in superphrase_node_to_reachable.iteritems():
                    if mft not in reachable_from_superphrase:
                        reachable_from_all_supernodes = False
                        break
                if reachable_from_all_supernodes:
                    if VERBOSE:
                        print "The node %s that is reachable from the token node is reachable from all of the superphrase nodes." % mft
                    keep_as_mappable.add(mft)

            del_edges = deque()
            for edge in text_mining_graph.forward_edges[t_node]:
                for targ_node in text_mining_graph.forward_edges[t_node][edge]:
                    reachable_from_targ_node = set(text_mining_graph.downstream_nodes(targ_node))
                    if len(reachable_from_targ_node.intersection(keep_as_mappable)) == 0:
                        del_edges.append((t_node, targ_node, edge))
                    else:
                        if VERBOSE:
                            print "Target node from the current node, %s, can reach a node that we want to keep as mappable: %s" % (targ_node, reachable_from_targ_node.intersection(keep_as_mappable))

            for d in del_edges:
                #print "This edge did not make the cut! %s --%s--> %s" % (t_node, edge, targ_node)
                text_mining_graph.delete_edge(d[0], d[1], d[2])
            

        return text_mining_graph



class ExactMatchCustomTargets_Stage:
    def __init__(self):
        with open(NOUN_PHRASES_JSON, "r") as f:
            self.noun_phrases = set(json.load(f)) 
     
    def run(self, text_mining_graph):
        print "Matching to custom noun-phrases..."
        for t_node in text_mining_graph.token_nodes:
            if t_node.token_str in self.noun_phrases:
                c_node = CustomMappingTargetNode(t_node.token_str)
                edge = FuzzyStringMatch(
                    t_node.token_str, 
                    t_node.token_str, 
                    "CUSTOM_NOUN_PHRASE", 
                    edit_dist=0
                )
                text_mining_graph.add_edge(t_node, c_node, edge)
        return text_mining_graph 
        

class CellLineToImpliedDisease_Stage:
    def __init__(self):
        with open(CELL_LINE_TO_IMPLIED_DISEASE_JSON, "r") as f:
            self.term_to_implied_terms = json.load(f)

    def run(self, text_mining_graph):
        print "Finding disease terms implied by cell line terms..."
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
    """
    Expand acronyms to their full name. For example,
    'iPSC' will be expanded to 'induced pluripotent
    stem cell'. 
    """
    def __init__(self):
        with open(ACRONYM_TO_EXPANSION_JSON, "r") as f:
            self.acr_to_expansions = json.load(f)

    def run(self, text_mining_graph):
        print "Generating acronym expansions..."
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


###################################################################################
#   Only to be used with ATCC data
###################################################################################

class ATCCKeyValueFilter_Stage:
    def __init__(
        self, 
        perform_filter_keys=True, 
        perform_filter_values=True
    ):
        with open(CELL_LINE_FILTER_KEYS_JSON, "r") as f:
            j = json.load(f)
            self.filter_keys = set(j["filter_keys"])
            self.filter_values = set(j["filter_values"])
            self.perform_filter_keys = perform_filter_keys
            self.perform_filter_values = perform_filter_values

    def run(self, text_mining_graph):
        if self.perform_filter_keys:
            remove_kv_nodes = [
                x 
                for x in text_mining_graph.key_val_nodes 
                if x.key in self.filter_keys
            ]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        if self.perform_filter_values:
            remove_kv_nodes = [
                x 
                for x in text_mining_graph.key_val_nodes 
                if x.value in self.filter_values
            ]
            for kv_node in remove_kv_nodes:
                text_mining_graph.delete_node(kv_node)
        return text_mining_graph


###################################################################################
#   'Real value' extractions
###################################################################################

class ExtractRealValue_Stage:
    """
    Extract numeric properties as tuples of the form:
    (property, value, unit)
   
    This is accomplished by searching the graph emanating
    from the key for a property ontology term. If found, we
    search the graph emanating from the value for a number
    and for a unit ontology term. 
    """

    def __init__(self):
        with open(REAL_VALUE_PROPERTIES, "r") as f:
            j = json.load(f)
            self.real_val_tids = j["property_term_ids"]
            self.default_units = j["default_units"]

    def run(self, text_mining_graph):
        print "Extracting real-value properties..."
    
        for kv_node in text_mining_graph.key_val_nodes:
            if VERBOSE:
                print "Checking whether key-value pair node %s encodes a \
                    real-value property." % kv_node

            # Find ontology-terms that refer to real-valued properties 
            real_val_term_nodes = set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        real_val_term_nodes.update( 
                            text_mining_graph.downstream_nodes(t_node)
                        )

            real_val_term_nodes = [
                x 
                for x in real_val_term_nodes 
                if isinstance(x, OntologyTermNode) 
                and x.term_id in self.real_val_tids
            ]
            if VERBOSE:
                print "Set of real-value properties we are searching \
                    for: %s" % self.real_val_tids
                print "Found property nodes: %s" % real_val_term_nodes
            if len(real_val_term_nodes) == 0:
                continue
        
            # Gather all nodes that are children of the this key-value's value
            real_val_candidates = set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        real_val_candidates.update(
                            text_mining_graph.downstream_nodes(t_node)
                        )    

            if VERBOSE:
                print "The real-value candidates are: %s" % real_val_candidates
            numeric_nodes = [
                x 
                for x in real_val_candidates 
                if isinstance(x, TokenNode) 
                and is_number(x.token_str)
            ]
            if VERBOSE:
                print "Found numeric nodes: %s" % numeric_nodes
            unit_nodes = [
                x 
                for x in real_val_candidates 
                if isinstance(x, OntologyTermNode) 
                and x.term_id.split(":")[0] == "UO"
            ]
            if VERBOSE:
                print "Found unit nodes: %s" % unit_nodes

            # If there is one real-value ontology term, one numeric token, 
            # and one unit node, then create real-value-property node
            edge = DerivesInto("Real-value extraction") # TODO should use a different edge-type
            if len(real_val_term_nodes) == 1:
                prop_term_node = list(real_val_term_nodes)[0]
                for numeric_node in numeric_nodes:
                    if len(unit_nodes) == 1:
                        unit_node = list(unit_nodes)[0]
                        rv_node = RealValuePropertyNode(
                            prop_term_node.term_id, 
                            float(numeric_node.token_str), 
                            unit_node.term_id
                        )
                        text_mining_graph.add_edge(
                            prop_term_node, 
                            rv_node, 
                            edge
                        )
                        text_mining_graph.add_edge(
                            numeric_node, 
                            rv_node, 
                            edge
                        )
                        text_mining_graph.add_edge(
                            unit_node, 
                            rv_node, 
                            edge
                        )
                    elif len(unit_nodes) == 0:
                        if prop_term_node.term_id in self.default_units:
                            default_unit_id = self.default_units[prop_term_node.term_id]
                        else:
                            default_unit_id = "missing"
                        rv_node = RealValuePropertyNode(
                            prop_term_node.term_id, 
                            float(numeric_node.token_str), 
                            default_unit_id
                        )
                        text_mining_graph.add_edge(
                            prop_term_node, 
                            rv_node, 
                            edge
                        )
                        text_mining_graph.add_edge(
                            numeric_node, 
                            rv_node, 
                            edge    
                        )  
                    else:
                        rv_node = RealValuePropertyNode(
                            prop_term_node.term_id, 
                            float(numeric_node.token_str), 
                            None
                        )
                        text_mining_graph.add_edge(
                            prop_term_node, 
                            rv_node, 
                            edge
                        )
                        text_mining_graph.add_edge(
                            numeric_node, 
                            rv_node, 
                            edge
                        )
        return text_mining_graph       


class ParseTimeWithUnit_Stage:
    """
    Parse artifacts that represent units of time and look something like '48h'. 
    That is, expand '48h' to '48 hour'.
    """
    def __init__(self):
        self.regex = r'^([0-9]*)\s*(h|hr|mo|d|min)$'
        self.unit_to_expansion = {
            "hr": "hour",
            "h": "hour",
            "hrs": "hour",
            "mo": "month",
            "d": "day",
            "min": "minute"
        }
        self.time_nodes = set(["EFO:0000721", "EFO:0000724"])

    def run(self, text_mining_graph):
        kv_nodes_time_val = set()
        for kv_node in text_mining_graph.key_val_nodes:
            # Find children of the key that indicate they encode a time-related 
            # real-value
            key_time_nodes = set()
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "key":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        key_time_nodes.update(text_mining_graph.downstream_nodes(t_node))
            key_time_nodes = [
                x 
                for x in key_time_nodes 
                if isinstance(x, OntologyTermNode) 
                and x.term_id in self.time_nodes
            ]
            if len(key_time_nodes) > 0:
                kv_nodes_time_val.add(kv_node)


        parseable_token_nodes = set()
        for kv_node in text_mining_graph.key_val_nodes:
            if kv_node not in kv_nodes_time_val:
                continue

            # Gather all nodes that are children of the key-nodes that do 
            # not contain a cell-line value. Remove them if they represent 
            # a cell line
            for edge in text_mining_graph.forward_edges[kv_node]:
                if isinstance(edge, DerivesInto) and edge.derivation_type == "val":
                    for t_node in text_mining_graph.forward_edges[kv_node][edge]:
                        parseable_token_nodes.update([
                            x 
                            for x in text_mining_graph.downstream_nodes(t_node) 
                            if isinstance(x, TokenNode)
                        ])


        tnode_to_edges = defaultdict(lambda: [])
        unit_t_nodes = set()
        for t_node in parseable_token_nodes:
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


###################################################################################
#   Stages for mapping 'consequent' terms
###################################################################################

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
                    new_ont_node = OntologyTermNode(implied_term_id)
                    node_to_new_edges[ont_node].append((new_ont_node, edge))

        for node, new_edges in node_to_new_edges.iteritems():
            for e in new_edges:
                text_mining_graph.add_edge(node, e[0], e[1])

        return text_mining_graph



class ConsequentCulturedCell_Stage:
    """
    If the sample maps to a Cellosaurus cell line 
    term, then we infer the sample is a cultured 
    cell.
    """ 
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


###################################################################################
#   Helper methods
###################################################################################

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
    text = " ".join(words)

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
            if VERBOSE:
                print "Regex failed on %s" % gram
    return result_grams



 
def main():

    tag_to_val = {
        "cell type": "T cell",
        "marker": "CD4+"
    }

    #tag_to_val = {
    #    "BioSampleModel": "Human",
    #    "age": "freshly isolated umbilical vein endothelial cells",
    #    "biomaterial_provider": "Lonza",
    #    "cell_line": "HUVEC",
    #    "isolate": "freshly isolated umbilical vein endothelial cells",
    #    "sex": "female",
    #    "tissue": "umbilical vein"
    #}
   

    spec_lex = SpecialistLexicon("LEX")
    key_val_filt = KeyValueFilter_Stage()
    init_tokens_stage = InitKeyValueTokens_Stage()
    ngram = NGram_Stage()
    lower_stage = Lowercase_Stage()
    cvcl_syn = CellosaurusSynonyms_Stage()
    man_at_syn = ManuallyAnnotatedSynonyms_Stage()
    infer_cell_line = InferCellLineTerms_Stage()
    #exact_match = ExactStringMatching_Stage(["13"], query_len_thresh=3)
    prop_spec_syn = PropertySpecificSynonym_Stage()
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
    fuzzy_match = FuzzyStringMatching_Stage(0.1, query_len_thresh=3)
    inflec_var = SPECIALISTLexInflectionalVariants(spec_lex)
    spell_var = SPECIALISTSpellingVariants(spec_lex)
    prioritize_exact = PrioritizeExactMatchOverFuzzyMatch()
    art_term_combos = TermArtifactCombinations_Stage()

    #efo_og, x, y = load_ontology.load("13") 
    #fuzzy_match_EFO = FuzzyStringMatching_Stage(efo_og, 0.1, query_len_thresh=3)

    #cvcl_og, x, y = load_ontology.load("4")
    #fuzzy_match_CVCL = FuzzyStringMatching_Stage(cvcl_og, 0.1, query_len_thresh=3)
    
    exact_match = ExactStringMatching_Stage(["1", "2", "4", "5", "7", "8", "9"], query_len_thresh=3)
    #fuzzy_match = FuzzyStringMatching_Stage(0.1, query_len_thresh=3)
    real_val = ExtractRealValue_Stage()


    #stages = [init_tokens_stage, hier_ngram, delimit, delimit2]
    #stages = [init_tokens_stage, hier_ngram, exact_match, match_cust_targs, block_cell_line_key]
    #stages = [init_tokens_stage, hier_ngram, exact_match, cellline_to_implied_disease]   
    #stages = [init_tokens_stage, acr_to_expan]
    #stages = [init_tokens_stage, ngram, exact_match, time_unit, exact_match, real_val]

    stages = [init_tokens_stage, ngram, lower_stage, inflec_var, spell_var, fuzzy_match, prioritize_exact, art_term_combos]

    p = Pipeline(stages, defaultdict(lambda: 1.0))
    
    mapped_terms, real_val_props =  p.run(tag_to_val)
    result = {
        "Mapped ontology terms": [
            x.to_dict() 
            for x in mapped_terms
        ], 
        "Real valued properties": [
            x.to_dict() 
            for x in real_val_props
        ]
    }
    print json.dumps(result, indent=4, separators=(',', ': '))
    
    #print json.dumps([x.to_dict() for x in p.run(tag_to_val)], indent=4, separators=(',', ': '))



if __name__ == "__main__":
    main()




