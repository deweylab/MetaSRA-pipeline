import pkg_resources as pr
import os
from os.path import join

resource_package = __name__
LEX_DIR = pr.resource_filename(resource_package, "LEX")
OBO_DIR = pr.resource_filename(resource_package, "obo")

# TODO set to name of the Disease Ontology's OBO file
DOID_FNAME = "doid.09-19-16.obo" 

# TODO set to name of the Cell Ontology's OBO file  
CL_FNAME = "cl.09-19-16.obo" 

# TODO set Cellosaurus's OBO file name
CVCL_FNAME = "cellosaurus.2-29-16.obo"

# TODO set to Uberon's OBO file name
UBERON_FNAME = "uberon.09-19-16.obo"

# TODO set to EFO's OBO file name
EFO_FNAME = "efo.05-10-16.obo"

# TODO set to Unit Ontology's OBO file name
UO_FNAME = "uo.07-28-16.obo"

# TODO set to Chemical Ontology's OBO file name
CHEBI_FNAME = "chebi.07-28-16.obo"

# Map ontology prefix to location of the ontology's obo file
OBOS = {
    "DOID" : join(OBO_DIR, DOID_FNAME),
    "CL"   : join(OBO_DIR, CL_FNAME),
    "CVCL" : join(OBO_DIR, CVCL_FNAME),
    "UBERON" : join(OBO_DIR, UBERON_FNAME),
    "EFO" : join(OBO_DIR, EFO_FNAME),
    "UO" : join(OBO_DIR, UO_FNAME),
    "CHEBI" : join(OBO_DIR, CHEBI_FNAME)
    }

def ontology_name_to_location():
    return OBOS

def specialist_lex_location():
    return LEX_DIR
