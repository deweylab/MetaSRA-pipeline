#############################################################################
#   The cell-type classifier. Uses a set of one-vs.-rest binary classifiers
#   to make a set of initial predictions. These predictions are then
#   narrowed down with a set of rules based on domain knowledge.
#############################################################################

from __future__ import print_function
from optparse import OptionParser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy import sparse


def mutual_info_rank_features(feature_vecs, binary_labels):
    """
    Given a set of feature vectors and binary labels, return
    the list of indices of the features ranked by mutual information
    with the binary labels.
    Args:
        feature_vecs: list of feature vectors
        binary_labels: list of binary labels
    """

    # Convert Features to Boolean values
    bin_feature_vecs = []
    for feature_v in feature_vecs:
        
        nfv = []
        for elem in feature_v:
            if elem > 0:
                nfv.append(1)
            else:
                nfv.append(0)
        bin_feature_vecs.append(nfv)

    mutual_infos = []
    num_features = len(bin_feature_vecs[0])
    for i in range(num_features):
        row_i = [x[i] for x in bin_feature_vecs]
        mi = mutual_info_score(row_i, binary_labels)
        mutual_infos.append(mi)

    ranked_indices = [
        index 
        for (mi,index) 
        in sorted(zip(
            mutual_infos,
            [x for x in range(num_features)]
        ))
    ]
    return ranked_indices


class OneVsRestClassifier:

    def __init__(
        self, 
        classif_type, 
        ngram_vec_scaffold, 
        term_vec_scaffold, 
        cvcl_og, 
        num_features_per_class=50, 
        use_predicted_term_rules=True):

        self.classif_type = classif_type
        self.ngram_vec_scaffold = ngram_vec_scaffold
        self.term_vec_scaffold = term_vec_scaffold

        self.class_to_classifier = None
        self.feature_cutoff = -1 * num_features_per_class

        self.use_predicted_term_rules = use_predicted_term_rules
        self.cvcl_og = cvcl_og

    def _features(self, feature_v):
        new_feature_v = []
        for index in self.filt_features:
            new_feature_v.append(feature_v[index])
        return new_feature_v

    def _learn_classifier(self, features, train_labels):
        classif = LogisticRegression(penalty='l1')
        classif.fit(sparse.csr_matrix(features), train_labels)   
        return classif

    def _one_v_rest_class(self, clss, label):
        if label == clss:
            return "CLASS"
        else:
            return "OTHER"

    def fit(self, feature_vecs, labels):

        classes = set(labels)
        self.class_to_classifier = {}

        # Generate features
        self.filt_features = set()
        for clss in classes:
            train_labels = [
                self._one_v_rest_class(clss, label) 
                for label in labels
            ]
            self.filt_features.update(
                mutual_info_rank_features(
                    feature_vecs, 
                    train_labels
                )[self.feature_cutoff:]
            )
        self.filt_features = list(self.filt_features)

        # Train logistic regression classifiers
        selected_feature_vecs = []
        for feature_v in feature_vecs:
            selected_feature_vecs.append(
                self._features(feature_v)
            )
        for clss in classes:
            train_labels = [
                self._one_v_rest_class(clss, label) 
                for label in labels
            ]
            classif = self._learn_classifier(
                selected_feature_vecs, 
                train_labels
            )
            self.class_to_classifier[clss] = classif

        

    def predict(self, q_feature_v, predicted_terms, real_value_props):

        all_types = set([
            "cell_line", 
            "in_vitro_differentiated_cells", 
            "induced_pluripotent_stem_cells", 
            "stem_cells", 
            "tissue", 
            "primary_cells"
        ])
        cellosaurus_subset_to_possible_types = {
            "Induced_pluripotent_stem_cell": [
                "in_vitro_differentiated_cells", 
                "induced_pluripotent_stem_cells"
            ],
            "Cancer_cell_line": [
                "cell_line"
            ],
            "Transformed_cell_line": [
                "cell_line"
            ],
            "Finite_cell_line": [
                "cell_line"
            ], 
            "Spontaneously_cell_line": [
                "cell_line"
            ],
            "Embryonic_stem_cell": [
                "stem_cells", 
                "in_vitro_differentiated_cells"
            ],
            "Telomerase_cell_line": [
                "cell_line"
            ],
            "Conditionally_cell_line": [
                "cell_line"
            ],
            "Hybridoma": [
                "cell_line"
            ]
        }

        class_to_confidence = {}

        for clss, classif in self.class_to_classifier.iteritems():
            new_q_feature_v = self._features(q_feature_v)
            pred_probs = classif.predict_proba(
                sparse.csr_matrix([new_q_feature_v])
            )[0]
            clss_index = list(classif.classes_).index("CLASS")
            class_to_confidence[clss] = pred_probs[clss_index]   


        if self.use_predicted_term_rules:
            print("Using predicted term rules")

            is_xenograft = False
            for pred_term in predicted_terms:
                if pred_term == "EFO:0003942":
                    is_xenograft = True
                    for typ in all_types:
                        if typ != "tissue":
                            class_to_confidence[typ] = 0.0

            # If the cell was passaged then we assert the sample is not a 
            # tissue or primary cell sample
            is_passaged = False
            if not is_xenograft:
                for real_val_prop in real_value_props:
                    if real_val_prop["property_id"] == "EFO:0007061" \
                        and real_val_prop["unit_id"] == "UO:0000189":
                        class_to_confidence["tissue"] = 0.0
                        is_passaged = True
                        if real_val_prop["value"] > 0:
                            class_to_confidence["primary_cells"] = 0.0


            # Find the cell-line type in the Cellosaurus
            found_cell_line_type = False
            if not is_xenograft:
                for pred_term in predicted_terms:
                    if pred_term.split(":")[0] == "CVCL":
                        for subset in self.cvcl_og.id_to_term[pred_term].subsets:
                            if subset in cellosaurus_subset_to_possible_types:
                                #print ( # TODO REMOVE
                                #    "This sample mapped to " + str(pred_term)
                                #    "which is a " + str(subset) + " type of cell line."
                                #) # TODO REMOVE
                                zero_types = all_types.difference(set(
                                    cellosaurus_subset_to_possible_types[subset]
                                ))
                                for typ in zero_types:
                                    class_to_confidence[typ] = 0.0
                                found_cell_line_type = True    
                            
            # If the cell-line type is not found, then rule out possible 
            # categories based on mapped ontology terms
            if not found_cell_line_type and not is_xenograft:
                # If 'stem cell' is mapped, then it can't be an immortalized 
                # cell line, tissue, or primary cell sample
                if "CL:0000034" in predicted_terms:
                    print("Sample mapped to stem cell term CL:0000034") 
                    class_to_confidence["cell_line"] = 0.0
                    class_to_confidence["tissue"] = 0.0
                    class_to_confidence["primary_cells"] = 0.0
                # If a specific cell-type is mapped, then it likely is 
                # not a tissue sample
                elif "CL:0002371" in predicted_terms:
                    print (
                        "Sample mapped to a specific cell-type as "
                        "indicated by mapped term CL:0002371"
                    )
                    class_to_confidence["tissue"] = 0.0
                
                # If 'primary cultured cell'  is mapped, and the 
                # cells have not been passaged, then it is likely
                # not tissue, iPSC, cell line, or in vitro 
                # differentiated cell 
                if "CL:0000001" in predicted_terms and not is_passaged:
                    class_to_confidence["tissue"] = 0.0
                    class_to_confidence["cell_line"] = 0.0
                    class_to_confidence["induced_pluripotent_stem_cells"] = 0.0
                    class_to_confidence["in_vitro_differentiated_cells"] = 0.0 
                     

        sum_conf = sum(class_to_confidence.values())
        print("Class to confidence before normalizing: %s" % class_to_confidence)
        print("Sum before normalizing: %f" % sum_conf)
        if sum_conf > 0:
            class_to_confidence = {
                k:v/sum_conf 
                for k,v in class_to_confidence.iteritems()
            }
        print("Class to confidence: %s" % class_to_confidence)
        return max(
            [
                (k,v) 
                for k,v in class_to_confidence.iteritems()
            ], 
            key=lambda x: x[1]
        )


if __name__ == "__main__":
    main()
