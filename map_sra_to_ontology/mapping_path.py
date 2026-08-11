from . import text_reasoning_graph as trg

# functions for:
# (1) list -> triples
# (2) triples -> list  
 
def triples_to_list(triples):
    """
    Convert a list of (Node, Edge, Node) triples to a list of nodes and edges.
    """
    nodes_and_edges = []
    for triple in triples:
        to_add = triple[1:] if nodes_and_edges else triple
        nodes_and_edges.extend(to_add)
    return nodes_and_edges

def list_to_triples(nodes_and_edges):
    """
    Convert a list of nodes and edges to a list of (Node, Edge, Node) triples.
    """
    node_indices = range(0, len(nodes_and_edges) - 1, 2)
    return [nodes_and_edges[i: i + 3] for i in node_indices]

def decode(data):
    if not isinstance(data, dict):
        return data

    class_name = data.get("__class__")
    if class_name is None:
        return data

    if class_name == "Inference":
        obj = trg.Inference(data["inference_type"])
        obj.weight = data.get("weight", obj.weight)
        return obj
    if class_name == "FuzzyStringMatch":
        obj = trg.FuzzyStringMatch(
            data["query_str"],
            data["matched_str"],
            data["match_target"],
            data["edit_dist"],
        )
        obj.weight = data.get("weight", obj.weight)
        return obj
    if class_name == "DerivesInto":
        obj = trg.DerivesInto(data["derivation_type"])
        obj.weight = data.get("weight", obj.weight)
        return obj
    if class_name == "KeyValueNode":
        return trg.KeyValueNode(data["key"], data["value"])
    if class_name == "TokenNode":
        if "char_indices" in data and data["char_indices"] is not None:
            return trg.TokenNode(data["token_str"], char_indices=data["char_indices"])
        else:
            return trg.TokenNode(
                data["token_str"],
                origin_gram_start=data["origin_gram_start"],
                origin_gram_end=data["origin_gram_end"],
            )
    if class_name == "CustomMappingTargetNode":
        return trg.CustomMappingTargetNode(data["rep_str"])
    if class_name == "OntologyTermNode":
        return trg.OntologyTermNode(data["term_id"])
    if class_name == "OntologyTermNode_OLD":
        return trg.OntologyTermNode_OLD(data["term_id"], consequent=data.get("consequent", False))
    if class_name == "RealValuePropertyNode":
        return trg.RealValuePropertyNode(data["property_term_id"], data["value"], data["unit_term_id"])

    return data

def encode(obj):
    if isinstance(obj, trg.Inference):
        return {
            "__class__": obj.__class__.__name__,
            "weight": obj.weight,
            "inference_type": obj.inference_type,
        }
    if isinstance(obj, trg.FuzzyStringMatch):
        return {
            "__class__": obj.__class__.__name__,
            "weight": obj.weight,
            "query_str": obj.query_str,
            "matched_str": obj.matched_str,
            "match_target": obj.match_target,
            "edit_dist": obj.edit_dist,
        }
    if isinstance(obj, trg.DerivesInto):
        return {
            "__class__": obj.__class__.__name__,
            "weight": obj.weight,
            "derivation_type": obj.derivation_type,
        }
    if isinstance(obj, trg.KeyValueNode):
        return {
            "__class__": obj.__class__.__name__,
            "key": obj.key,
            "value": obj.value,
        }
    if isinstance(obj, trg.TokenNode):
        obj_dict = {
            "__class__": obj.__class__.__name__,
            "token_str": obj.token_str
        }
        gram_indices = list(range(obj.origin_gram_start, obj.origin_gram_end))
        if obj.char_indices == gram_indices:
            return obj_dict | {
                "origin_gram_start": obj.origin_gram_start,
                "origin_gram_end": obj.origin_gram_end
            }
        else:
            return obj_dict | {
                "char_indices": obj.char_indices
            }

    if isinstance(obj, trg.CustomMappingTargetNode):
        return {
            "__class__": obj.__class__.__name__,
            "rep_str": obj.rep_str,
        }
    if isinstance(obj, trg.OntologyTermNode):
        return {
            "__class__": obj.__class__.__name__,
            "term_id": obj.term_id,
        }
    if isinstance(obj, trg.OntologyTermNode_OLD):
        return {
            "__class__": obj.__class__.__name__,
            "term_id": obj.term_id,
            "consequent": obj.consequent,
        }
    if isinstance(obj, trg.RealValuePropertyNode):
        return {
            "__class__": obj.__class__.__name__,
            "property_term_id": obj.property_term_id,
            "value": obj.value,
            "unit_term_id": obj.unit_term_id,
        }

    raise TypeError("Object of type %s is not JSON serializable" % obj.__class__.__name__)
