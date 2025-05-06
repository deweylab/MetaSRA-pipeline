from __future__ import print_function
from io import open # Python 2/3 compatibility
import pickle
import sys
import os
from os.path import join
import pkg_resources as pr

from . import predict_sample_type
from .predict_sample_type.learn_classifier import *

# The pickled objects need the python path to point to the predict_sample_type
# directory
sys.path.append(pr.resource_filename(__name__, "predict_sample_type"))

class SampleTypePredictor:
    # The constructor accepts a CVCL ontology graph to allow for use of more recent
    # versions of the Cellosaurus. The default is None, which will use the
    # Cellosaurus version that was used to train the classifier.
    def __init__(self, cvcl_og=None):
        vectorizer_f = pr.resource_filename(__name__, join("predict_sample_type", "sample_type_vectorizer.pickle"))
        classifier_f = pr.resource_filename(__name__, join("predict_sample_type", "sample_type_classifier.pickle"))
        with open(vectorizer_f, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(classifier_f, "rb") as f:
            if sys.version_info[0] == 2:
                self.model = pickle.load(f)
            else:
                self.model = pickle.load(f, encoding='latin1')
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
