import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

def relative_error(x0,x): return np.abs(x0-x)/np.abs(x0)

N = 20
un = np.cos((np.pi * (np.arange(N) + 1 / 2)) / (N + 1))
xu = (1 + 2 * un / 3) / (1 - 2 * un / 3)
yn = np.log(xu)
x = np.linspace(1 / 5, 5, 1000)
y = np.log(x)

L = scipy.interpolate.lagrange(xu, yn)
yl = L(x)
plt.plot(x, y, '-k')
plt.plot(xu, yn, '.b')
plt.plot(x, yl, '-r')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()

plt.semilogy(x, relative_error(y, yl))
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.show()

N = 20
un = np.cos((np.pi * (np.arange(N) + 1 / 2)) / (N + 1))
un2 = np.linspace(-1, 1, 1000)

xu = (1 + 2 * un / 3) / (1 - 2 * un / 3)
xu2 = (1 + 2 * un2 / 3) / (1 - 2 * un2 / 3)

y = np.log(xu)
y2 = np.log(xu2)

L = scipy.interpolate.lagrange(un, y)
yl = L(x)
plt.plot(un2, yl, '--g')
plt.plot(un2, y2, '--r')
plt.plot(un, y, '.b')
plt.show()

plt.semilogy(un2, relative_error(y2, yl))
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.show()


N = 20
un = np.cos((np.pi * (np.arange(N) + 1 / 2)) / (N + 1))
un2 = np.linspace(-1, 1, 1000)

xu = (1 + 2 * un / 3) / (1 - 2 * un / 3)
xu2 = (1 + 2 * un2 / 3) / (1 - 2 * un2 / 3)

y = np.log(xu)
y2 = np.log(xu2)

L = scipy.interpolate.lagrange(un, y)

coef = L.c
coef[1::2] = 0
pol = np.poly1d(coef)


yl = pol(un2)
plt.plot(un2, yl, '--g')
plt.plot(un, y, '.b')
plt.show()
