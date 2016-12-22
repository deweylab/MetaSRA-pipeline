from collections import Counter

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

