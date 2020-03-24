import matplotlib.pyplot as plt
import numpy as np
import math


base = 10


def exact_sum(K):
    return 1.


def samples(K):
    parts = [np.full((base ** k,), float(base) ** (-k) / K) for k in range(0, K)]
    samples = np.concatenate(parts)
    return np.random.permutation(samples)


def direct_sum(x):
    s = 0.
    for e in x:
        s += e
    return s


def number_of_samples(K):
    return np.sum([base ** k for k in range(0, K)])


def exact_mean(K):
    return 1. / number_of_samples(K)


def exact_variance(K):
    # варианты значений
    values = np.asarray([float(base) ** (-k) / K for k in range(0, K)], dtype=np.double)
    # количество вхождений для каждого значения
    count = np.asarray([base ** k for k in range(0, K)])
    return np.sum(count * (values - exact_mean(K)) ** 2) / number_of_samples(K)


def relative_error(x0, x):
    return np.abs(x0 - x) / np.abs(x)


def kahan_sum(x):
    s = 0.0
    c = 0.0
    for i in x:
        y = i - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


K = 7
x = samples(K)
print("Размер выборки:", len(x))
print("Минимальное значение:", np.min(x))
print("Максимальное значение:", np.max(x))

exact_sum_for_x = exact_sum(K)
direct_sum_for_x = direct_sum(x)
print("Погрешность прямой суммы:",
      relative_error(exact_sum_for_x, direct_sum_for_x))

sorted_x = x[np.argsort(x)]
sorted_sum_for_x = direct_sum(sorted_x)
print("Погрешность суммы по возрастанию:",
      relative_error(exact_sum_for_x, sorted_sum_for_x))

sorted_x = x[np.argsort(x)[::-1]]
sorted_sum_for_x = direct_sum(sorted_x)
print("Погрешность суммы по убыванию:",
      relative_error(exact_sum_for_x, sorted_sum_for_x))


Kahan_sum_for_x = kahan_sum(x)
print("Погрешность суммы по Кэхэну:",
      relative_error(exact_sum_for_x, Kahan_sum_for_x))


mean = 1e6
delta = 1e-5


def samples(N_over_two):
    x = np.full((2 * N_over_two,), mean, dtype=np.double)
    x[:N_over_two] += delta
    x[N_over_two:] -= delta
    return np.random.permutation(x)


def direct_mean(x):
    return direct_sum(x) / len(x)


def direct_first_var(x):
    return direct_mean((x - direct_mean(x)) ** 2)


def direct_second_var(x):
    return direct_mean(x ** 2) - direct_mean(x) ** 2


def online_second_var(x):
    m = x[0]
    m2 = x[0] ** 2
    for n in range(1, len(x)):
        m = (m * (n-1) + x[n]) / n
        m2 = (m2 * (n-1) + x[n] ** 2) / n
    return m2 - m ** 2


def online_first_var(x):
    m = x[0]
    m2 = 0
    for n in range(1, len(x)):
        m = (m * (n - 1) + x[n]) / n
        m2 = ((n - 1) / n) * m2 + ((n - 1) / n ** 2) * (x[n] - m) ** 2
    return m2


x = samples(1000000)
print("Размер выборки:", len(x))
print("Среднее значение:", exact_mean())
print("Дисперсия:", exact_variance())
print("Погрешность среднего для встроенной функции:",
      relative_error(exact_mean(), np.mean(x)))
print("Погрешность дисперсии для встроенной функции:",
      relative_error(exact_variance(), np.var(x)))
print("Погрешность среднего для последовательного суммирования:",
      relative_error(exact_mean(), direct_mean(x)))

print("Погрешность рекурентной оценки дисперсии для последовательного суммирования:",
      relative_error(exact_variance(), direct_second_var(x)))
print("Погрешность рекурентной оценки дисперсии для однопроходного суммирования вторым способом:",
      relative_error(exact_variance(), online_second_var(x)))
print("Погрешность оценки дисперсии по первой формуле для последовательного суммирования:",
      relative_error(exact_variance(), direct_first_var(x)))
print("Погрешность рекурентной оценки дисперсии для однопроходного суммирования первым способом:",
      relative_error(exact_variance(), online_first_var(x)))


def samples(K):
    return np.random.permutation(
        np.asarray([math.sin(k) for k in range(1, K)])
    )


def exact_sum(K):
    return (math.sin(K) - (math.cos(K) / math.tan(0.5)) + (1 / math.tan(0.5))) / 2


K = 100000
x = samples(K)
exact_sum_for_x = exact_sum(K)
print("Точная сумма:", exact_sum_for_x)

direct_sum_for_x = direct_sum(x)
print("Прямая сумма для sin(k):", direct_sum_for_x)
print("Погрешность прямой суммы для sin(k):",
      relative_error(exact_sum_for_x, direct_sum_for_x))

Kahan_sum_for_x = kahan_sum(x)
print("Погрешность суммы по Кэхэну без порядка:",
      relative_error(exact_sum_for_x, Kahan_sum_for_x))

sorted_x = x[np.argsort(x)]
sorted_sum_for_x = direct_sum(sorted_x)
print("Сумма для sin(k) по возрастанию:", sorted_sum_for_x)
print("Погрешность суммы для sin(k) по возрастанию:",
      relative_error(exact_sum_for_x, sorted_sum_for_x))

Kahan_sum_for_x = kahan_sum(sorted_x)
print("Погрешность суммы по Кэхэну по возрастанию:",
      relative_error(exact_sum_for_x, Kahan_sum_for_x))

sorted_x = x[np.argsort(x)[::-1]]
sorted_sum_for_x = direct_sum(sorted_x)
print("Сумма для sin(k) по убыванию:", sorted_sum_for_x)
print("Погрешность суммы для sin(k) по убыванию:",
      relative_error(exact_sum_for_x, sorted_sum_for_x))

Kahan_sum_for_x = kahan_sum(sorted_x)
print("Погрешность суммы по Кэхэну по убыванию:",
      relative_error(exact_sum_for_x, Kahan_sum_for_x))






