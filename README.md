```python
import random

import matplotlib.pyplot as plt
import numpy as np


def number(x, a):
    if x == a:
        return 1
    return 0


def fuzzy_trapezoidal_number(x, fuzzy_number):
    a, b, c, d = fuzzy_number
    if x < a or x > d:
        return 0
    if b <= x <= c:
        return 1
    if a <= x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzification_number(x_n, da, db):
    a = x_n - da - db
    b = x_n - db
    c = x_n + db
    d = x_n + da + db
    return [a, b, c, d]


def min_two_trapezoidal_number(x, fuzzy_number1, fuzzy_number2):
    mu1 = fuzzy_trapezoidal_number(x, fuzzy_number1)
    mu2 = fuzzy_trapezoidal_number(x, fuzzy_number2)
    return min(mu1, mu2)


def is_crossing_two_segments(b1, c1, b2, c2):
    if b1 > c1:
        b1, c1 = c1, b1
    if b2 > c2:
        b2, c2 = c2, b2
    if b1 > c2:
        b1, b2 = b2, b1
        c1, c2 = c2, c1
    if c1 < b2:
        return 0
    return 1


def max_min_two_trapezoidal_number(fuzzy_number1, fuzzy_number2):
    a1, b1, c1, d1 = fuzzy_number1
    a2, b2, c2, d2 = fuzzy_number2
    b = is_crossing_two_segments(b1, c1, b2, c2)
    if b:
        return 1
    if b1 > c2:
        a1, b1, c1, d1 = a2, b2, c2, d2
    b = is_crossing_two_segments(c1, d1, a2, b2)
    if not b:
        return 0
    x = (a2 * c1 - d1 * b2) / (a2 - b2 + c1 - d1)
    return (d1 - x) / (d1 - c1)



def frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number):
    sred = (a + b + c + d) / 4
    fuzzy_number, m = sred
    return min(fuzzy_trapezoidal_number(x, fuzzy_number), m)


def max_two_trapezoidal_number(x, fuzzy_number1, fuzzy_number2):
    mu1 = fuzzy_trapezoidal_number(x, fuzzy_number1)
    mu2 = fuzzy_trapezoidal_number(x, fuzzy_number2)
    return max(mu1, mu2)


def max_two_frequent_trapezoidal_number(x, frequent_fuzzy_number1, frequent_fuzzy_number2):
    mu1 = frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number1)
    mu2 = frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number2)
    return max(mu1, mu2)


def max_n_frequent_trapezoidal_number(x, list_frequent_fuzzy_numbers):
    maximum = 0
    for frequent_fuzzy_number in list_frequent_fuzzy_numbers:
        maximum = max(maximum, frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number))
    return maximum


def square_max_n_frequent_trapezoidal_number(list_frequent_fuzzy_numbers):
    n = 100
    a = list_frequent_fuzzy_numbers[0][0][0]
    d = list_frequent_fuzzy_numbers[0][0][-1]
    for frequent_fuzzy_number in list_frequent_fuzzy_numbers:
        a = min(a, frequent_fuzzy_number[0][0])
        d = max(d, frequent_fuzzy_number[0][-1])
    h = (d - a) / n
    s = 0
    for i in range(0, n):
        s += max_n_frequent_trapezoidal_number(a + i * h, list_frequent_fuzzy_numbers) * h
    return s


def center_mass_n_frequent_trapezoidal_number(list_frequent_fuzzy_numbers):
    n = 100
    a = list_frequent_fuzzy_numbers[0][0][0]
    d = list_frequent_fuzzy_numbers[0][0][-1]
    for frequent_fuzzy_number in list_frequent_fuzzy_numbers:
        a = min(a, frequent_fuzzy_number[0][0])
        d = max(d, frequent_fuzzy_number[0][-1])
    h = (d - a) / n
    s1 = 0
    s2 = 0
    for i in range(0, n):
        s1 += (a + i * h) * max_n_frequent_trapezoidal_number(a + i * h, list_frequent_fuzzy_numbers) * h
        s2 += max_n_frequent_trapezoidal_number(a + i * h, list_frequent_fuzzy_numbers) * h
    if s2 <= 0.00001:
        return (d - a) / 2 + a
    return s1 / s2


def fuzzy_controller_for_x_v(x_n, v_n, dx_n, dv_n, base):
    # Фазификация наблюдаемых параметров
    fuzzy_x_n = fuzzification_number(x_n, dx_n[0], dx_n[1])
    fuzzy_v_n = fuzzification_number(v_n, dv_n[0], dv_n[1])
    # Обработка базы нечетких правил
    list_frequent_fuzzy_numbers = []
    for rule in base:
        max_min_x = max_min_two_trapezoidal_number(fuzzy_x_n, rule[0])
        max_min_v = max_min_two_trapezoidal_number(fuzzy_v_n, rule[1])
        level_for_frequent = min(max_min_x, max_min_v)
        list_frequent_fuzzy_numbers.append([rule[2], level_for_frequent])
    v_u = center_mass_n_frequent_trapezoidal_number(list_frequent_fuzzy_numbers)
    return v_u


def fuzzy_controller(x_n, dx, base):
    # Фазификация наблюдаемых параметров
    fuzzy_x = [fuzzification_number(x_n[i], dx[i][0], dx[i][1]) for i in range(len(x_n))]
    # Обработка базы нечетких правил
    list_frequent_fuzzy_numbers = []
    for rule in base:
        max_min = [max_min_two_trapezoidal_number(fuzzy_x[i], rule[i]) for i in range(len(x_n))]
        list_frequent_fuzzy_numbers.append([rule[2], min(max_min)])
    return center_mass_n_frequent_trapezoidal_number(list_frequent_fuzzy_numbers)


def model_with_control(x_n_0, v_n_0, t_max, dt, dx_n, dv_n, base):
    x_n = x_n_0
    v_n = v_n_0

    t_n_lst = np.arange(0, t_max, dt)
    x_n_lst = []
    v_n_lst = []
    v_u_lst = []

    for _ in t_n_lst:
        v_u = fuzzy_controller([x_n, v_n], [dx_n, dv_n], base)
        v_n = v_n + v_u
        x_n = x_n + v_n * dt
        v_n_lst.append(v_n)
        x_n_lst.append(x_n)
        v_u_lst.append(v_u)

    return t_n_lst, x_n_lst, v_n_lst, v_u_lst



def model_with_control_for_optimization(lst):
    a, b = 0, 10
    x_n_0 = 5
    v_n_0 = 0.5
    t_max = 60
    dt = 0.5
    x_n = x_n_0
    v_n = v_n_0

    dx_n = [abs(lst[0]), abs(lst[1])]  # Параметры для фазификации x
    dv_n = [abs(lst[2]), abs(lst[3])]  # Параметры для фазификации v
    # База нечетких правил
    rule1 = [sorted(lst[4:8]), sorted(lst[8:12]), sorted(lst[12:16])]
    rule2 = [sorted(lst[16:20]), sorted(lst[20:24]), sorted(lst[24:28])]
    base = [rule1, rule2]

    quality = 0

    for _ in np.arange(0, t_max, dt):
        v_u = fuzzy_controller([x_n, v_n], [dx_n, dv_n], base)
        v_n = v_n + v_u
        x_n = x_n + v_n * dt
        if x_n > b or x_n < a:
            quality -= 1

    return quality





dx_n = [17, 8]  # Параметры для фазификации x
dv_n = [16, 19]  # Параметры для фазификации v
# База нечетких правил
rule1 = [sorted([18.3, -12.7, 12.6, 12.5]), sorted([-18.4, -7, 9.75394657303227, -7.5]), sorted([3.8, 11, -10, -10.5])]
rule2 = [sorted([-17.1, -6.7, 2, -4]), sorted([-3, 12, 17, -12.5]), sorted([5.678, 17, 0.9, 1.3])]


base = [rule1, rule2]

t_n_lst, x_n_lst, v_n_lst, v_u_lst = model_with_control(x_n_0=5, v_n_0=0, t_max=60, dt=0.5, dx_n=dx_n, dv_n=dv_n,
                                                        base=base)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Наблюдаемые параметры тележки')
ax1.plot(t_n_lst, x_n_lst)
plt.xlabel("Ось X")
plt.ylabel("Ось Y")
ax2.plot(t_n_lst, v_n_lst)
ax3.plot(t_n_lst, v_u_lst)

plt.show()

```
