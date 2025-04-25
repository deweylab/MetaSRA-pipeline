from __future__ import print_function
import dill
import sys
import os
from os.path import join
import pkg_resources as pr

from . import predict_sample_type
from .predict_sample_type.learn_classifier import *

# The dilled objects need the python path to point to the predict_sample_type
# directory
sys.path.append(pr.resource_filename(__name__, "predict_sample_type"))

class SampleTypePredictor:
    # The constructor accepts a CVCL ontology graph to allow for use of more recent
    # versions of the Cellosaurus. The default is None, which will use the
    # Cellosaurus version that was used to train the classifier.
    def __init__(self, cvcl_og=None):
        vectorizer_f = pr.resource_filename(__name__, join("predict_sample_type", "sample_type_vectorizor.dill"))
        classifier_f = pr.resource_filename(__name__, join("predict_sample_type", "sample_type_classifier.dill"))
        with open(vectorizer_f, "rb") as f:
            self.vectorizer = dill.load(f)
        with open(classifier_f, "rb") as f:
            self.model = dill.load(f)
        if cvcl_og is not None:
            self.model.cvcl_og = cvcl_og
    
    def predict(self, tag_to_val, mapped_terms, real_props):
        # Make sample-type prediction
        feat_v = self.vectorizer.convert_to_features(
            get_ngrams_from_tag_to_val(tag_to_val),
            mapped_terms)
        predicted, confidence = self.model.predict(
            feat_v,
            mapped_terms,
            real_props)
        return predicted, confidence

def run_sample_type_prediction(tag_to_val, mapped_terms, real_props):
    predictor = SampleTypePredictor()
    return predictor.predict(tag_to_val, mapped_terms, real_props)
