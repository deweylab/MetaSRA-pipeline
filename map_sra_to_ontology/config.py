from __future__ import print_function
from io import open # Python 2/3 compatibility
import json
from os.path import join
from pathlib import Path
import pkg_resources as pr


class Config:
    def __init__(self, ref_path):
        self.ref_path = ref_path
        self.LEX_DIR = ref_path / "LEX"
        self.OBO_DIR = ref_path / "obo"
        self.PREFIX_TO_FNAME = ref_path / "ont_prefix_to_filename.json"
        self.TERM_TO_LINKED_ANCESTOR_JSON =  ref_path / "term_to_superterm_linked_terms.json"
        self.CELL_LINE_TO_IMPLIED_DISEASE_JSON = ref_path / "cellline_to_disease_implied_terms.json"
    
        # Relative paths to resources
        resource_package = __name__
        self.FILTER_KEYS_JSON = pr.resource_filename(resource_package, join("metadata", "filter_key_val_rules.json"))
        self.CELL_LINE_FILTER_KEYS_JSON = pr.resource_filename(resource_package, join("metadata", "cell_line_filter_key_val_rules.json"))
        self.PROPERTY_SPECIFIC_SYNONYMS_JSON = pr.resource_filename(resource_package, join("metadata", "has_val_syn_term_ids.json"))
        self.NOUN_PHRASES_JSON = pr.resource_filename(resource_package, join("metadata", "noun_phrases.json"))
        self.ACRONYM_TO_EXPANSION_JSON = pr.resource_filename(resource_package, join("metadata", "acronym_to_expansions.json"))
        self.REAL_VALUE_PROPERTIES = pr.resource_filename(resource_package, join("metadata", "real_valued_properties.json"))
        self.CUST_TERM_TO_CONSEQ_TERMS_JSON = pr.resource_filename(resource_package, join("metadata", "custom_term_to_consequent_terms.json"))
        self.CELL_LINE_TERMS_JSON = pr.resource_filename(resource_package, join("metadata", "cvcl_mappings.json"))
        self.TWO_CHAR_MAPPINGS_JSON = pr.resource_filename(resource_package, join("metadata", "two_char_mappings.json"))
        self.TERM_ARTIFACT_COMBOS_JSON = pr.resource_filename(resource_package, join("metadata", "term_artifact_combo.json"))        

        self.SYN_SETS_PATH = Path(pr.resource_filename(resource_package, "synonym_sets"))

    def ontology_name_to_location(self):
        prefix_to_location = {}
        with open(self.PREFIX_TO_FNAME, "r") as f:
            for prefix, fname in json.load(f).items():
                prefix_to_location[prefix] = self.OBO_DIR / fname
        return prefix_to_location
        
    def specialist_lex_location(self):
        return self.LEX_DIR
