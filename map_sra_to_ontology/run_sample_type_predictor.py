import dill
import sys
import os
from os.path import join
import pkg_resources as pr
resource_package = __name__

import predict_sample_type
from predict_sample_type.learn_classifier import *

# The dilled objects need the python path to point to the predict_sample_type
# directory
sys.path.append(pr.resource_filename(resource_package, "predict_sample_type"))

def run_sample_type_prediction(tag_to_val, mapped_terms, real_props):
    # Load the dilled vectorizer and model
    vectorizer_f = pr.resource_filename(resource_package, join("predict_sample_type", "sample_type_vectorizor.dill"))
    classifier_f = pr.resource_filename(resource_package, join("predict_sample_type", "sample_type_classifier.dill"))
    with open(vectorizer_f, "rb") as f:
        vectorizer = dill.load(f)
    with open(classifier_f, "rb") as f:
        model = dill.load(f)

    # Make sample-type prediction
    feat_v = vectorizer.convert_to_features(
        get_ngrams_from_tag_to_val(tag_to_val),
        mapped_terms)
    predicted, confidence = model.predict(
        feat_v,
        mapped_terms,
        real_props)

    return predicted, confidence

