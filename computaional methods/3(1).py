import numpy as np


def mlog(x, iter):
    s = 0
    for k in range(iter):
        s += ser(x, k + 1)
    return s


def ser(x, iter):
    return (-1) ** (iter + 1) * (x - 1) ** iter / iter


def lag(x, N):
    return (-1) ** (N + 1) * (x - 1) ** (N + 1) * x ** (-N) / (N * (N + 1))


print(np.log(2))
print(mlog(2., 10) + lag(2., 10))
print(mlog(2., 1000))
