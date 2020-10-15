def commonChars(A):
    if len(A) == 1:
        return list(A[0])

    res = []
    base = A[0]
    candidates = A[1:].copy()
    for c in base:
        remain = []
        for word in candidates:
            idx = word.find(c)
            if idx >= 0:
                word_list = list(word)
                word_list.pop(idx)
                word = ''.join(word_list)
                remain.append(word)
            else:
                break
        if len(remain) == len(candidates):
            res.append(c)
            candidates = remain.copy()
    return res


x = ["bella", "label", "roller"]
commonChars(x)

