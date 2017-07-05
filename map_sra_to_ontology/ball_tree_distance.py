from collections import Counter

def bag_dist_multiset_OLD(str_a, str_b):
    count_a = Counter(str_a)
    count_b = Counter(str_b)
    a_minus_b_counts = Counter(str_a)    
    b_minus_a_counts = Counter(str_b)
    a_minus_b_counts.subtract(count_b)
    b_minus_a_counts.subtract(count_a)
    a_minus_b = sum([x for x in a_minus_b_counts.values() if x > 0])
    b_minus_a = sum([x for x in b_minus_a_counts.values() if x > 0])
    if a_minus_b > b_minus_a:
        return a_minus_b
    else:
        return b_minus_a

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

def bag_dist(vec_a,vec_b):
    a_minus_b = 0
    b_minus_a = 0
    for i in range(len(vec_a)):
        a = vec_a[i]
        b = vec_b[i]
        if a > b:
            a_minus_b += a-b
        elif b > a:
            b_minus_a += b-a
    if a_minus_b > b_minus_a:
        return a_minus_b
    else:
        return b_minus_a
    #return max([a_minus_b, b_minus_a])

print bag_dist_multiset("aaab", "aabcc")
