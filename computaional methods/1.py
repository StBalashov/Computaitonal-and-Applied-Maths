import numpy as np

def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)

eps = np.finfo(np.double).eps
print("Machine tochonst", eps)


def f_div_mult(x, d=np.pi, n=52):
    for k in range(n): x = x / d
    for k in range(n): x = x * d
    return x


x0 = np.logspace(-4, 4, 100, dtype=np.double)
x = f_div_mult(x0)
err = relative_error(x0, x)
print("Errors", err[:4], "...")


def f_sqrt_sqr(x, n=52):
    for k in range(n): x = np.sqrt(x)
    for k in range(n): x = x * x
    return x


x = f_sqrt_sqr(x0)
err = relative_error(x0, x)
print("Errors", err[:4], "...")

def f_sqrt_sqr2(x, n=52):
    for k in range(n): x = MyNumber2.sqrt(x)
    for k in range(n): x = x * x
    return x.to_float()


class MyNumber2(object):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return "{}".format(self.to_float())

    def from_float(x):
        return MyNumber2(np.log(x))

    def to_float(self):
        return np.e ** self.data

    def __mul__(self, other):
        return MyNumber2(self.data + other.data)

    def sqrt(self):
        return MyNumber2(self.data * 0.5)

x = f_sqrt_sqr2(MyNumber2.from_float(x0))
err = relative_error(x0, x)
print("Errors log", err[:4], "...")
