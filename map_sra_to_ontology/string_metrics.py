from __future__ import print_function
from collections import Counter
from typing import Callable

import numpy as np

def bag_dist_multiset(str_a, str_b):
    count_a = Counter(str_a)
    count_b = Counter(str_b)

    a_minus_b = 0
    b_minus_a = 0
    for c in count_a:
        if c in count_b:
            if count_a[c] > count_b[c]:
                a_minus_b += count_a[c] - count_b[c]
        else:
            a_minus_b += count_a[c]

    for c in count_b:
        if c in count_a:
            if count_b[c] > count_a[c]:
                b_minus_a += count_b[c] - count_a[c]
        else:
            b_minus_a += count_b[c]

    if a_minus_b > b_minus_a:
        return a_minus_b
    else:
        return b_minus_a

# def bag_dist_multiset(str_a, str_b):

#     # Notes:
#     # Behavior of Counter.subtract is to compute differences in counts, possibly yielding negative counts
#     # when the latter object has a higher count for an element, including when the former has none.
#     # Behavior of +Counter (resp. -Counter) is to only keep positive (resp. -ve) count elements.

#     count_diff = Counter(str_a)
#     count_diff.subtract(Counter(str_b))

#     return max(sum((+count_diff).values()), sum((-count_diff).values()))


def weighted_bag_dist_multiset(str_a, str_b, weight_fn):

    # Notes:
    # Behavior of Counter.subtract is to compute differences in counts, possibly yielding negative counts
    # when the latter object has a higher count for an element, including when the former has none.
    # Behavior of +Counter (resp. -Counter) is to only keep positive (resp. negative) count elements.

    count_diff = Counter(str_a)
    count_diff.subtract(Counter(str_b))

    count_diff = Counter({elem: count*weight_fn(elem) for elem, count in count_diff.items()})

    return max(sum((+count_diff).values()), sum((-count_diff).values()))

def weighted_edit_dist(str_a, str_b, deletion_weight_fn = None, substitution_weight_fn = None):
    """
    Calculate a weighted variant of Levenshtein edit-distance between two strings.

    The weighted variant allows for scoring different characters differently, e.g. one might want
    to score the edit distance between "cat-dog" and "catdog" as less than the edit distance between
    "cat-dog" and "cat-dig".

    :param str_a: first string
    :param str_b: second string
    :param deletion_weight_fn: Function from character to numeric, indicating the weight of a deletion
        (or insertion) of that character (defaults to 1)
    :param substitution_weight_fn: Function from character x character to numeric, indicating the weight
        of substituting one character with another. Defaults to the max of the individual deletion
        scores of the two characters.
    """

    # Init weight functions if not given
    if deletion_weight_fn is None:
        deletion_weight_fn = lambda _: 1
    
    if substitution_weight_fn is None:
        substitution_weight_fn = lambda x, y: max(deletion_weight_fn(x), deletion_weight_fn(y))

    # Using a dynamic programming approach where we align the two strings. For this we populate a
    # len_a + 1 - by len_b + 1 matrix where element at coordinate i, j is the edit distance between
    # str_a[:i] and str_b[:j].

    len_a = len(str_a)
    len_b = len(str_b)

    lev = np.zeros((len_a + 1, len_b + 1))

    for i in range(len_a + 1):

        if i > 0: c_a = str_a[i - 1]

        for j in range(len_b + 1):

            if i == 0 and j == 0: continue

            skip_a = skip_b = substitute = np.inf

            if i > 0:
                # Skipping a character in string a
                skip_a = lev[i - 1, j] + deletion_weight_fn(c_a)

            if j > 0:
                c_b = str_b[j - 1]

                # Skipping a character in string b
                skip_b = lev[i, j - 1] + deletion_weight_fn(c_b)

                if i > 0:
                    # Substitution cost
                    substitute = lev[i - 1, j - 1] + (0 if c_a == c_b else substitution_weight_fn(c_a, c_b))
            
            lev[i, j] = min(skip_a, skip_b, substitute)
    
    return lev[len_a, len_b]

class CasePermissiveAlnumWeightedEditDistance(Callable):

    __name__ = 'CasePermissiveAlnumWeightedEditDistance'

    def __init__(self, non_alnum_weight: float, case_weight: float) -> None:
        self.non_alnum_weight: float = non_alnum_weight
        self.case_weight: float = case_weight
    
    def __call__(self, str_a: str, str_b: str) -> float:
        return min(
            weighted_edit_dist(str_a, str_b, self._weight_fn),
            weighted_edit_dist(str_a.lower(), str_b.lower(), self._weight_fn) + self.case_weight)

    def _weight_fn(self, char: str):
        return 1 if char.isalnum() else self.non_alnum_weight
    
class CasePermissiveAlnumWeightedBagDistance(Callable):

    __name__ = 'CasePermissiveAlnumWeightedBagDistance'

    def __init__(self, non_alnum_weight: float, case_weight: float) -> None:
        self.non_alnum_weight: float = non_alnum_weight
        self.case_weight: float = case_weight
    
    def __call__(self, str_a: str, str_b: str) -> float:
        return min(
            weighted_bag_dist_multiset(str_a, str_b, self._weight_fn),
            weighted_bag_dist_multiset(str_a.lower(), str_b.lower(), self._weight_fn) + self.case_weight)
    
    def _weight_fn(self, char: str):
        return 1 if char.isalnum() else self.non_alnum_weight