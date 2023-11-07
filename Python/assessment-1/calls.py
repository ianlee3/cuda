%%timeit
normalize(greyscales,out=normalized)
weigh(normalized, weights,out=weighted)
SOLUTION = activate(weighted,out=activated)
