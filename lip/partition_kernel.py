

def partition(h, w, k, s, p):
    if type(k) is not tuple:
        k = (k, k)
    if type(s) is not tuple:
        s = (s, s)
    if type(p) is not tuple:
        p = (p, p)
    r0 = min(k[0], h + 2 * p[0] - k[0] + 1)
    r1 = min(k[1], w + 2 * p[1] - k[1] + 1)
    # print(r0, r1)

    all_classes = []

    init_classes = equivalence_class(0, 0, (r0, r1), s)
    all_classes += init_classes
    t = (k[0] - r0, k[1] - r1)
    for i in range(0, t[0]+1):
        for j in range(0, t[1]+1):
            if i == 0 and j == 0:
                continue
            classes_new = increment(init_classes, i, j)
            all_classes += classes_new
    # print(all_classes)

    idx_set = []
    for classes in all_classes:
        tmp = []
        for c in classes:
            # print(c)
            tmp.append(sub2lin(c, k[0], k[1]))
        idx_set.append(tmp)

    # remove redundancy
    idx_set = sorted(idx_set, key=lambda x:len(x))
    idx_set2 = [(c, set(c)) for c in idx_set]
    for i in range(len(idx_set2)):
        c, sc = idx_set2[i]
        for j in range(i+1, len(idx_set2)):
            c_, sc_ = idx_set2[j]
            if sc <= sc_:
                idx_set.remove(c)
                break

    return idx_set


def equivalence_class(x, y, r, s):
    all_idx = []
    for i in range(x, r[0]):
        for j in range(y, r[1]):
            all_idx.append((i,j))

    classes = []
    while len(all_idx) > 0:
        x0, y0 = all_idx[0]
        tmp = []
        for x in range(x0, r[0], s[0]):
            for y in range(y0, r[1], s[1]):
                tmp.append((x, y))
                all_idx.remove((x, y))
        classes.append(tmp)
    
    return classes

    
def increment(classes, i, j):
    classes_new = []
    for eqv in classes:
        tmp = [(c[0] + i, c[1] + j) for c in eqv]
        classes_new.append(tmp)
    return classes_new
    

def sub2lin(sub, k0, k1):
    return sub[1] * k0 + sub[0]


if __name__ == '__main__':
    h, w, k, s, p = (6, 6, 3, 1, (1,0))
    idx_set = partition(h, w, k, s, p)
    print(idx_set)
