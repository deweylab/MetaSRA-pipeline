import pkg_resources as pr
import os
from os.path import join

resource_package = __name__
OBO_DIR = pr.resource_filename(resource_package, "obo")

# Map ontology prefix to location of the ontology's obo file
OBOS = {
    "DOID" : join(OBO_DIR, "doid.09-19-16.obo"),
    "CL"   : join(OBO_DIR, "cl.09-19-16.obo"),
    "CLO"  : join(OBO_DIR, "clo.obo"),
    "CVCL" : join(OBO_DIR, "cellosaurus.2-29-16.obo"),
    "UBERON" : join(OBO_DIR, "uberon.09-19-16.obo"),
    "EFO" : join(OBO_DIR, "efo.05-10-16.obo"),
    "UO" : join(OBO_DIR, "uo.07-28-16.obo"),
    "CHEBI" : join(OBO_DIR, "chebi.07-28-16.obo")
    }

def ontology_name_to_location():
    return OBOS
