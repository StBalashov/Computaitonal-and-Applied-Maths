import numpy as np
import matplotlib.pyplot as plt


def EulerIntegrator(h, y0, f):
    return y0 + h * f(y0)


def oneStepErrorPlot(f, y, integrator):
    eps = np.finfo(float).eps
    steps = np.logspace(-10, 0, 50)
    y0 = y(0)
    yPrecise = [y(t) for t in steps]
    yApproximate = [integrator(t, y0, f) for t in steps]
    h = [np.maximum(np.max(np.abs(yp-ya)), eps)
         for yp, ya in zip(yPrecise, yApproximate)]  # погрешности
    plt.loglog(steps, h, '-')
    plt.xlabel('Шаг интегрирования')
    plt.ylabel('Погрешность одного шага')


def firstOrderPlot():
    ax = plt.gca()  # создание объекта оси
    steps = np.asarray(ax.get_xlim())  # границы на оси х
    plt.loglog(steps, steps, '--r')


f = lambda y: y
yExact = lambda t: np.exp(t)


def integrate(N, delta, f, y0, integrator):
    for n in range(N):
        y0 = integrator(delta, y0, f)
    return y0


def intervalErrorPlot(
        f, y, integrator, T=1, maxstepNumber=1000, numberOfPointsOnPlot=16):
    eps = np.finfo(float).eps
    stepNumber = np.logspace(0, np.log10(maxstepNumber), numberOfPointsOnPlot).astype(np.int)
    steps = T/stepNumber
    y0 = y(0)
    yPrecise = y(T)
    yApproximate = [integrate(N, T/N, f, y0, integrator) for N in stepNumber]
    h = [np.maximum(np.max(np.abs(yPrecise-ya)), eps) for ya in yApproximate]
    plt.loglog(steps, h, '.-')
    plt.xlabel('Шаг интегрирования')
    plt.ylabel('Погрешность интегрирования на интервале')






# intervalErrorPlot(f, yExact, EulerIntegrator)
# firstOrderPlot()
# plt.legend(['интегратор', 'первый порядок'], loc=2)
# plt.show()


f = lambda y: 1
yExact = lambda t: t


def NewtonIntegrator(h, y0, f):
    return y0 + h * f[0](y0) + f[0](y0) * f[1](y0) * h * h / 2


f = (lambda y: y, lambda y: 1)
yExact = lambda t: np.exp(t)


def ModifiedEulerIntegrator(h, y0, f):
    yIntermediate = y0 + f(y0) * h / 2
    return y0 + h * f(yIntermediate)


f = lambda y: y
yExact = lambda t: np.exp(t)


def RungeKuttaIntegrator(h, y0, f):
    k1 = f(y0)
    k2 = f(y0 + k1 * h / 2)
    k3 = f(y0 + k2 * h / 2)
    k4 = f(y0 + k3 * h)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6


f = lambda y: y
yExact = lambda t: np.exp(t)



def NewtonMethod(F, x0):
    for i in range(100):
        x = x0 - F[0](x0) / F[1](x0)
        if x == x0: break
        x0 = x
    return x0


def BackwardEulerIntegrator(h, y0, f):
    F = (lambda y: y0 + h * f[0](y) - y, lambda y: h * f[1](y) - 1)
    return NewtonMethod(F, y0)


alpha = -10
f = (lambda y: alpha * y, lambda y: alpha)
yExact = lambda t: np.exp(alpha * t)


def fixIntervalErrorPlot(
        f, y, integrator, maxstepNumber=100, stepSize=10e-1, pointOfStart=0):
    eps = np.finfo(float).eps
    y0 = y(pointOfStart)
    intervals = [stepSize * n for n in range(maxstepNumber)]
    yPricise = [y(pointOfStart + n * stepSize) for n in range(maxstepNumber)]
    yApproximate = [integrate(stepNumber, stepSize, f, y0, integrator)
                    for stepNumber in range(maxstepNumber)]
    h = [np.maximum(np.max(np.abs(yp - ya)), eps) for yp, ya in zip(yPricise, yApproximate)]
    plt.loglog(intervals, h, '.-')
    plt.xlabel('Интервал интегрирования')
    plt.ylabel('Погрешность интегрирования на интервале')


stepSize = 10e-1
stepNumber = 40

fixIntervalErrorPlot(f[0], yExact, EulerIntegrator, stepNumber, stepSize)
fixIntervalErrorPlot(f, yExact, BackwardEulerIntegrator, stepNumber, stepSize)
fixIntervalErrorPlot(f[0], yExact, ModifiedEulerIntegrator, stepNumber, stepSize)
fixIntervalErrorPlot(f[0], yExact, RungeKuttaIntegrator, stepNumber, stepSize)
x = [stepSize * n for n in range(stepNumber)]
y = [np.exp(-2 * arg) for arg in x]
plt.loglog(x, y)
y = [np.exp(0 * arg) for arg in x]
plt.loglog(x, y)
y = [np.exp(2 * arg) for arg in x]
plt.loglog(x, y)
y = [np.exp(4 * arg) for arg in x]
plt.loglog(x, y)
y = [np.exp(6 * arg) for arg in x]
plt.loglog(x, y)
firstOrderPlot()
plt.legend([
    'метод Эйлера',
    'неявный метод Эйлера',
    'модифицированный метод Эйлера',
    'метод Рунге-Кутты',
    'первый порядок'], loc=2)
plt.title('Погрешности на интервалах и экспоненты')
plt.show()
