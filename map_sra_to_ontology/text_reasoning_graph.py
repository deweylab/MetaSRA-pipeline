#########################################################################
#   Implements the Text Reasoning Graph
#########################################################################

from __future__ import print_function
from optparse import OptionParser
from sets import Set
from collections import defaultdict, deque

try:
    import pygraphviz as pgv
except:
    print("Unable to import pygraphviz. Visualization is disabled.")

class EEdge(object):
    def __init__(self, weight):
        self.weight = weight


class Inference(EEdge):
    def __init__(self, inference_type):
        super(Inference, self).__init__(1.0)
        self.inference_type = inference_type

    def __eq__(self, other):
        if isinstance(other, Inference):
            if self.inference_type == other.inference_type:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("INFERENCE", (self.inference_type)))

    def __str__(self):
        return "Inference(weight=%f, inference_type='%s')" % (self.weight, self.inference_type)

    def __repr__(self):
        return self.__str__()

class FuzzyStringMatch(EEdge):
    def __init__(
        self, 
        query_str, 
        matched_str, 
        match_target, 
        edit_dist
    ):
        """
        Args:
            match_target: describes the aspect/field of the target
                that was matched to
            edit_dist: edit distance from query to matched string
        """
        # TODO allows transformation of edit distance to edge-weight
        super(FuzzyStringMatch, self).__init__(1.0)
        self.match_target = match_target
        self.query_str = query_str
        self.matched_str = matched_str
        self.edit_dist = edit_dist

    def __eq__(self, other):
        if isinstance(other, FuzzyStringMatch):
            if self.query_str == other.query_str \
                and self.matched_str == other.matched_str \
                and self.match_target == other.match_target: # weight should be determined by edit-distance 
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((
            "FUZZY_STRING_MATCH", 
            (
                self.query_str, 
                self.matched_str,  
                self.match_target, 
                self.edit_dist
            )
        )) 

    def __str__(self):
        return "FuzzyStringMatch(weight=%f, query_str='%s', matched_str='%s', match_target='%s', edit_dist=%d)" % (
            self.weight, 
            self.query_str.decode('utf-8', 'replace'), 
            self.matched_str, 
            self.match_target, 
            self.edit_dist
        )

    def __repr__(self):
        return self.__str__()

class DerivesInto(EEdge):
    def __init__(self, derivation_type):
        # TODO map derivation type to weight
        super(DerivesInto, self).__init__(1.0)
        self.derivation_type = derivation_type

    def __eq__(self, other):
        if isinstance(other, DerivesInto):
            if self.derivation_type == other.derivation_type: # weight should be determined by derivation type
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("FUZZY_STRING_MATCH", (self.derivation_type)))
           
    def __str__(self):
        return "DerivesInto(weight=%f, derivation_type='%s')" % (self.weight, self.derivation_type)
    
    def __repr__(self):
        return self.__str__()
 

class ENode(object):
    def __init__(self):
        pass

class KeyValueNode(ENode):
    def __init__(self, key, value):
        super(KeyValueNode, self).__init__() 
        self.key = key
        self.value = value

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(("KEY_VALUE_NODE", (self.key, self.value)))

    def __str__(self):
        return "KeyValueNode(key=%s, value=%s)" % (self.key, self.value)

    def __repr__(self):
        return self.__str__()



class TokenNode(ENode):
    def __init__(self, token_str, origin_gram_start, origin_gram_end):
        super(TokenNode, self).__init__()
        self.token_str = token_str
        self.origin_gram_start = origin_gram_start
        self.origin_gram_end = origin_gram_end

    def __eq__(self, other):
        if isinstance(other, TokenNode):
            if self.token_str == other.token_str \
                and self.origin_gram_start == other.origin_gram_start \
                and self.origin_gram_end == other.origin_gram_end:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((
            "TOKEN_NODE", 
            (
                self.token_str, 
                self.origin_gram_start, 
                self.origin_gram_end
            )
        ))

    def __str__(self):
        return "TokenNode(token_str='%s', origin_gram_start=%d, origin_gram_end=%d)" % (
            self.token_str, 
            self.origin_gram_start, 
            self.origin_gram_end
        )

    def __repr__(self):
        return self.__str__()



class MappingTargetNode(ENode):
     def __init__(self):
        super(MappingTargetNode, self).__init__()



class CustomMappingTargetNode(MappingTargetNode):
    def __init__(self, rep_str):
        super(CustomMappingTargetNode, self).__init__()
        self.rep_str = rep_str

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("CUSTOM_MAPPING_TARGET_NODE", (self.rep_str)))

    def __str__(self):
        return "CustomMappingTargetNode(rep_str='%s')" % (self.rep_str)

    def __repr__(self):
        return self.__str__()



class OntologyTermNode(MappingTargetNode):
    def __init__(self, term_id):
        super(OntologyTermNode, self).__init__()
        self.term_id = term_id

    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __ne__(self, other):
        return not self.__eq__(other) 

    def __hash__(self):
        return hash(("ONTOLOGY_TERM_NODE", (self.term_id)))
    
    def __str__(self):
        return "OntologyTermNode(term_id='%s')" % self.term_id
    
    def __repr__(self):
        return self.__str__()
    
    def namespace(self):
        return self.term_id.split(":")[0]


class OntologyTermNode_OLD(MappingTargetNode):
    def __init__(self, term_id, consequent=False):
        super(OntologyTermNode, self).__init__()
        self.term_id = term_id
        self.consequent = consequent

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("ONTOLOGY_TERM_NODE", (self.term_id, self.consequent)))

    def __str__(self):
        return "OntologyTermNode(term_id='%s', consequent=%s)" % (self.term_id, str(self.consequent))

    def __repr__(self):
        return self.__str__()

    def namespace(self):
        return self.term_id.split(":")[0]


class RealValuePropertyNode(ENode):
    def __init__(self, property_term_id, value, unit_term_id):
        super(RealValuePropertyNode, self).__init__()
        self.property_term_id = property_term_id
        self.value = value
        self.unit_term_id = unit_term_id

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(("RealValuePropertyNode", (self.property_term_id, self.value, self.unit_term_id)))

    def __str__(self):
        return "RealValuePropertyNode(property_term_id='%s', value=%f, unit_term_id='%s')" % (
            self.property_term_id, 
            self.value, 
            self.unit_term_id
        )

    def __repr__(self):
        return self.__str__()



class TextReasoningGraph:
    def __init__(self, prohibit_cycles=True):
        self.token_nodes = Set()
        self.key_val_nodes = Set()
        self.ontology_term_nodes = Set()
        self.custom_mapping_target_nodes = Set()
        self.real_value_nodes = Set()
        self.forward_edges = defaultdict(lambda: defaultdict(lambda: Set())) 
        self.reverse_edges = defaultdict(lambda: defaultdict(lambda: Set()))    
        self.prohibit_cycles = prohibit_cycles

    
    @property
    def mapping_target_nodes(self):
        return self.ontology_term_nodes.union(self.custom_mapping_target_nodes)   
 
    def __str__(self):
        try:
            r_str = "key-value nodes:\n"  
            for kv_node in self.key_val_nodes:
                r_str += "%s\n" % kv_node
            r_str += "\ntoken nodes:\n"
            for tok_node in self.token_nodes:
                r_str += "%s\n" % tok_node
            r_str += "\nontology term nodes:\n"    
            for ont_node in self.ontology_term_nodes:
                r_str += "%s\n" % ont_node   
            r_str += "\nreal-value property nodes:\n"
            for rv_node in self.real_value_nodes:
                r_str += "%s\n" % rv_node
            r_str += "\ncustom mapping target nodes:\n"
            for cmt_node in self.custom_mapping_target_nodes:
                r_str += "%s\n" % cmt_node 

            r_str += "\nforward_edges:\n"
            for node_source, etype_to_node in self.forward_edges.iteritems():
                for edge, node_target in etype_to_node.iteritems():
                    r_str += "%s %s %s\n" % (node_source, edge, node_target) 

            r_str += "\nreverse_edges:\n"
            for node_source, etype_to_node in self.reverse_edges.iteritems():
                for edge, node_target in etype_to_node.iteritems():
                    r_str += "%s %s %s\n" % (node_source, edge, node_target)
            return r_str
        except UnicodeDecodeError:
            print("Unicode decode error. Error converting graph to string...")
            return "" 
        
   
    def graphviz(self, root_id=None):
        g = pgv.AGraph(directed='True')               
        
        for o_node in self.ontology_term_nodes:
            g.add_node(o_node.term_id, shape='polygon')

            #g.add_edge(self.id_to_term[curr_id].name, self.id_to_term[sub_id].name)
        return str(g)
 
    def add_node(self, node):
        if isinstance(node, TokenNode):
            if node not in self.token_nodes:
                self.token_nodes.add(node)
        elif isinstance(node, KeyValueNode) and node not in self.key_val_nodes:
            self.key_val_nodes.add(node)
        elif isinstance(node, OntologyTermNode) and node not in self.ontology_term_nodes:
            self.ontology_term_nodes.add(node)
        elif isinstance(node, CustomMappingTargetNode) and node not in self.custom_mapping_target_nodes:
            self.custom_mapping_target_nodes.add(node)
        elif isinstance(node, RealValuePropertyNode) and node not in self.real_value_nodes:
            self.real_value_nodes.add(node)       
 
    def delete_node(self, node):

        if node in self.token_nodes:
            self.token_nodes.remove(node)
        if node in self.key_val_nodes:
            self.key_val_nodes.remove(node)
        if node in self.ontology_term_nodes:
            self.ontology_term_nodes.remove(node)
        if node in self.custom_mapping_target_nodes:
            self.custom_mapping_target_nodes.remove(node)
        if node in self.real_value_nodes:
            self.real_value_nodes.remove(node)

        if node in self.forward_edges:
            # Delete edges from this node
            del_edges = []
            for edge in self.forward_edges[node]:
                for targ_node in self.forward_edges[node][edge]:
                    del_edges.append((node, targ_node, edge))
            for e in del_edges:
                self.delete_edge(e[0], e[1], e[2])

        if node in self.reverse_edges:
            # Delete edges to this node
            del_edges = []
            for edge in self.reverse_edges[node]:
                for source_node in self.reverse_edges[node][edge]:
                    del_edges.append((source_node, node, edge))
            for e in del_edges:
                self.delete_edge(e[0], e[1], e[2])



    def delete_edge(self, node_a, node_b, edge):
        """
        Delete an edge from one node to another.
        Args:
            node_a: the source node
            node_b: the target node
            edge: the edge type between node_a and node_b to be removed
        """
        if node_a in self.forward_edges and edge in self.forward_edges[node_a] and node_b in self.forward_edges[node_a][edge]:
            self.forward_edges[node_a][edge].remove(node_b)
            if len(self.forward_edges[node_a][edge]) == 0:
                del self.forward_edges[node_a][edge]        
            if len(self.forward_edges[node_a]) == 0:  
                del self.forward_edges[node_a]
        
            self.reverse_edges[node_b][edge].remove(node_a)
            if len(self.reverse_edges[node_b][edge]) == 0:
                del self.reverse_edges[node_b][edge]
            if len(self.reverse_edges[node_b]) == 0:
                del self.reverse_edges[node_b]
#        else:
#            print "Warning! Could not delete edge %s -- %s --> %s" % (node_a, edge, node_b)
#            print "node_a=%s in self.forward_edges=? %s" % (node_a, (node_a in self.forward_edges))
#            print "%s in self.forward_edges[node_a]? %s" % (edge, (edge in self.forward_edges[node_a]))
#            print "node_b=%s in self.forward_edges[node_a][edge]? %s" % (node_b, (node_b in self.forward_edges[node_a][edge]))
        
    def add_edge(self, node_a, node_b, edge):
        if self.prohibit_cycles:
            self._add_edge(node_a, node_b, edge)
            if self.is_cycle_present():
                self.delete_edge(node_a, node_b, edge)
                print("Warning! Adding edge %s -- %s --> %s causes a cycle. Edge was not created." % (node_a, edge, node_b))
        else:
            self._add_edge(node_a, node_b, edge)

    def _add_edge(self, node_a, node_b, edge):
        self.add_node(node_a)
        self.add_node(node_b)

        if node_a != node_b: # No self-loops
            self.forward_edges[node_a][edge].add(node_b)
            self.reverse_edges[node_b][edge].add(node_a) 

    def all_nodes(self):
        return Set(self.token_nodes).union(self.key_val_nodes).union(self.ontology_term_nodes).union(self.real_value_nodes).union(self.custom_mapping_target_nodes)

    def get_children(self, node):
        children = Set()
        if not node in self.forward_edges:
            return children
        for edge in self.forward_edges[node]:
            children.update(self.forward_edges[node][edge])
        return children

    def shortest_path(self, source_node, use_reverse_edges=False):
        """
        Find shortest path to all nodes from a source node.
        Implements Dijkstra's algorithm.
        """

        # If using reverse edges, then we are looking at the shortest way of 
        # of getting ~to~ the "source" rather than going from the source.
        if use_reverse_edges:
            cons_edges = self.reverse_edges
        else:
            cons_edges = self.forward_edges

        dist = {}       # maps node to shortest distance to source
        prev = {}       # maps node to (next node, edge type) tuple needed to reach source through shortest path
        queue = []      # queue of nodes with current estimated distance to source
        in_q = Set()    # stores all nodes in the queue


        # Initialization 
        for node  in self.all_nodes():
            if node == source_node:
                dist[node] = 0.0
                queue.append((node, 0.0))
                in_q.add(node)
            else:
                dist[node] = float('inf')
                queue.append((node, float('inf')))
                in_q.add(node)
    
        # Run Dijkstra's algorithm
        while len(queue) > 0:
            queue = sorted(queue, key=lambda x: x[1])
            node = queue[0][0]
            alt = queue[0][1]

            # Remove element from queue
            queue = queue[1:]
            try:
                in_q.remove(node)  
            except KeyError:
                pass # TODO this is bad practice. Should output to a log

            for edge, targ_nodes in cons_edges[node].iteritems():
                for targ_node in targ_nodes:
                    alt = dist[node] + edge.weight
                    if alt < dist[targ_node] and targ_node in in_q:
                        dist[targ_node] = alt
                        prev[targ_node] = (node, edge)
                        queue.append((targ_node, alt))
        
        return dist, prev 

    def downstream_nodes(self, node, depth_first=True, exclude_edges=None):
        """
        Depth first traversal to gather all downstream nodes in 
        the DAG
        Args:
            exclude_edges: a set of edges that we do not want to follow
        """

        downstream = []

        q = deque()

        q.append(node)
        while len(q) > 0:

            if depth_first:
                curr_node = q.pop()
            else:
                curr_node = q.popleft()

            if curr_node in Set(downstream):
                continue
            downstream.append(curr_node)
            if curr_node in self.forward_edges:
                for edge in self.forward_edges[curr_node]:
                    if exclude_edges and edge in exclude_edges:
                        continue
                    for targ_node in self.forward_edges[curr_node][edge]:
                        #if targ_node not in Set(q) and targ_node not in Set(downstream):
                        if targ_node not in Set(downstream):
                            q.append(targ_node)       
 
        return downstream

    def is_cycle_present(self):
        """
        Perform cycle detection on this graph
        """

        def has_incoming_edge(node, removed_nodes, remaining_nodes):
            for edge in self.reverse_edges[node]:
                for in_node in self.reverse_edges[node][edge]:
                    if in_node in remaining_nodes:
                        return True
            return False 

        removed_nodes = Set([x for x in self.all_nodes() if x not in self.reverse_edges])
        remaining_nodes = self.all_nodes().difference(removed_nodes)
        deletion_occurred = True
        while deletion_occurred:

            deletion_occurred = False
            to_remove = Set()
            for node in remaining_nodes:
                has_in_edge = has_incoming_edge(node, removed_nodes, remaining_nodes) 
                if not has_in_edge:
                    to_remove.add(node)

            if len(to_remove) > 0:
                removed_nodes.update(to_remove)
                remaining_nodes = remaining_nodes.difference(to_remove)
                deletion_occurred = True

        if len(remaining_nodes) > 0:
            return True
        else:
            return False

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()
    pass # TODO

if __name__ == "__main__":
    main()
