import pkg_resources as pr
import json

import ontology_graph

def load(ontology_index):
    resource_package = __name__

    SAMPLE_TAG_VALS = pr.resource_filename(resource_package, "./metadata/sample_to_tag_to_values.json")

    config_f = pr.resource_filename(resource_package, "./ontology_configurations.json")
    with open(config_f, "r") as f:
        j = json.load(f)
    ont_config = j[ontology_index] 

    include_ontologies = ont_config["included_ontology_projects"]
    restrict_to_idspaces = ont_config["id_spaces"]
    is_restrict_roots = ont_config["restrict_to_specific_subgraph"]
    restrict_to_roots = ont_config["subgraph_roots"] if is_restrict_roots else None
    exclude_terms = ont_config["exclude_terms"] 
    
    og = ontology_graph.load_ontology(include_ontologies=include_ontologies,
        restrict_to_idspaces=restrict_to_idspaces, restrict_to_roots=restrict_to_roots, 
        exclude_terms=exclude_terms)

    return og, include_ontologies, restrict_to_roots

def load_mappable_ontologies(ontology_ids):
    ogs = [load(x)[0] for x in ontology_ids]
    return ontology_graph.MappableOntologies(ogs)


def main():
    og, i, r = load("4")
    print og.id_to_term["CVCL:C792"]

if __name__ == "__main__":
    main()
