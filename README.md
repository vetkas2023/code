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
    fuzzy_number, m = frequent_fuzzy_number
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


# lst = [1, 2, 0.5, 0.5, 8, 9, 20, 30, -500, -500, 500, 500, -1, -0.5, -0.5, -0.1, -20, -10, 3, 5, -500, -500, 500, 500,
#        0.1, 0.5, 0.5, 1]

# max_quality = None
# max_lst = []
# for i in range(100):
#     lst = []
#     for j in range(28):
#         lst.append(random.uniform(-20, 20))
#     quality = model_with_control_for_optimization(lst)
#     print(quality)
#     if max_quality is None or quality > max_quality:
#         max_quality = quality
#         max_lst = lst
# print(max_quality)
# print(lst)

# t_n_lst, x_n_lst, v_n_lst, v_u_lst = model_with_control(x_n_0=5, v_n_0=0.5, t_max=60, dt=0.5, dx_n=dx_n, dv_n=dv_n,
#                                                         base=base)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Наблюдаемые параметры тележки')
# ax1.plot(t_n_lst, x_n_lst)
# ax2.plot(t_n_lst, v_n_lst)
# ax3.plot(t_n_lst, v_u_lst)
# plt.show()


# a, b = 0, 10
# dx_n = [1, 2]  # Параметры для фазификации x
# dv_n = [0.5, 0.5]  # Параметры для фазификации v
# # База нечетких правил
# rule1 = [[8, 9, 20, 30], [-500,-500, 500, 500], [-1, -0.5, -0.5, -0.1]]
# rule2 = [[-20, -10, 3, 5], [-500,-500, 500, 500], [0.1, 0.5, 0.5, 1]]

dx_n = [17.701209317074973, 8.733023911402423]  # Параметры для фазификации x
dv_n = [16.082253773029237, 19.167126493222316]  # Параметры для фазификации v
# База нечетких правил
rule1 = [sorted([18.183408938832805, -12.602980914906583, 12.608773007533209, 12.27657215426936]), sorted([-18.721150346170973, -7.112715906648951, 9.75394657303227, -7.941706246969829]), sorted([3.9129161532139207, 11.541436702887307, -10.459157156591946, -10.411858100896069])]
rule2 = [sorted([-17.449553588770538, -6.0288156902403145, 2.4131734969112912, -4.067324357749529]), sorted([-3.3869952914961985, 12.651379120779708, 17.35131448595331, -12.116299780226814]), sorted([6.74425081597008, 17.48652917019723, 0.9648018535019958, 1.3426637351435033])]


base = [rule1, rule2]

t_n_lst, x_n_lst, v_n_lst, v_u_lst = model_with_control(x_n_0=5, v_n_0=0, t_max=60, dt=0.5, dx_n=dx_n, dv_n=dv_n,
                                                        base=base)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Наблюдаемые параметры тележки')
ax1.plot(t_n_lst, x_n_lst)
ax2.plot(t_n_lst, v_n_lst)
ax3.plot(t_n_lst, v_u_lst)
plt.show()

# dx_n = [1, 1] # Параметры для фазификации x
# dv_n = [1, 1] # Параметры для фазификации v
# # База нечетких правил
# rule1 = [[1, 3, 7, 9], [-1, 0, 1, 1], [-2, -1, 0, 1]]
# rule2 = [[2, 4, 6, 10], [2, 5, 8, 9], [1, 3, 5, 7]]
# rule3 = [[6, 9, 12, 15], [-5, -4, -3, -2], [6, 7, 8, 9]]
# base = [rule1, rule2, rule3]
# print(fuzzy_controller_for_x_v(1, 1, dx_n, dv_n, base))

# fig, ax = plt.subplots()
# min_x, max_x, h = 0, 15, 0.001
# x_lst = np.arange(min_x, max_x, h)
# # list_frequent_fuzzy_numbers = [[[1, 3, 7, 9], 0.5], [[6, 8, 9, 12], 0.8], [[4, 5, 7, 16], 0.3]]
# list_frequent_fuzzy_numbers = [[[2, 2, 3, 3], 0.5], [[4, 4, 5, 5], 0.5]]
# for frequent_fuzzy_number in list_frequent_fuzzy_numbers:
#     y_lst = [frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number) for x in x_lst]
#     ax.plot(x_lst, y_lst)
# y2_lst = [max_n_frequent_trapezoidal_number(x, list_frequent_fuzzy_numbers) for x in x_lst]
# ax.plot(x_lst, y2_lst)
# # print(square_max_n_frequent_trapezoidal_number(list_frequent_fuzzy_numbers))
# print(center_mass_n_frequent_trapezoidal_number(list_frequent_fuzzy_numbers))
# plt.show()


# min_x, max_x, h = 0, 15, 0.001
# x_lst = np.arange(min_x, max_x, h)
# frequent_fuzzy_number1 = [[1, 3, 7, 9], 0.5]
# frequent_fuzzy_number2 = [[6, 8, 9, 12], 0.8]
# y_lst = [frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number1) for x in x_lst]
# y2_lst = [frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number2) for x in x_lst]
# y3_lst = [max_two_frequent_trapezoidal_number(x, frequent_fuzzy_number1, frequent_fuzzy_number2) for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# ax.plot(x_lst, y2_lst)
# ax.plot(x_lst, y3_lst)
# plt.show()


# min_x, max_x, h = 0, 15, 0.001
# x_lst = np.arange(min_x, max_x, h)
# fuzzy_number1 = [1, 3, 7, 9]
# fuzzy_number2 = [6, 8, 9, 12]
# y_lst = [fuzzy_trapezoidal_number(x, fuzzy_number1) for x in x_lst]
# y2_lst = [fuzzy_trapezoidal_number(x, fuzzy_number2) for x in x_lst]
# y3_lst = [max_two_trapezoidal_number(x, fuzzy_number1, fuzzy_number2) for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# ax.plot(x_lst, y2_lst)
# ax.plot(x_lst, y3_lst)
# plt.show()

# min_x, max_x, h = 0, 15, 0.001
# x_lst = np.arange(min_x, max_x, h)
# fuzzy_number1 = [1, 3, 7, 9]
# m = 0.1
# y_lst = [fuzzy_trapezoidal_number(x, fuzzy_number1) for x in x_lst]
# frequent_fuzzy_number = [fuzzy_number1, m]
# y2_lst = [frequent_fuzzy_trapezoidal_number(x, frequent_fuzzy_number) for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# ax.plot(x_lst, y2_lst)
# plt.show()

# min_x, max_x, h = 0, 15, 0.001
# x_lst = np.arange(min_x, max_x, h)
# fuzzy_number1 = [1, 3, 7, 9]
# fuzzy_number2 = [6, 9, 9, 12]
# y_lst = [fuzzy_trapezoidal_number(x, fuzzy_number1) for x in x_lst]
# y2_lst = [fuzzy_trapezoidal_number(x, fuzzy_number2) for x in x_lst]
# y3_lst = [min_two_trapezoidal_number(x, fuzzy_number1, fuzzy_number2) for x in x_lst]
# maximum = max_min_two_trapezoidal_number(fuzzy_number1, fuzzy_number2)
# y4_lst = [maximum for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# ax.plot(x_lst, y2_lst)
# ax.plot(x_lst, y3_lst)
# ax.plot(x_lst, y4_lst)
# plt.show()

# min_x, max_x, h = 0, 20, 0.001
# x_lst = np.arange(min_x, max_x, h)
# x_n = 10
# fuzzy_number = fuzzification_number(x_n, 1, 1)
# y_lst = [fuzzy_trapezoidal_number(x, fuzzy_number) for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# plt.show()

# min_x, max_x, h = 0, 10, 0.001
# x_lst = np.arange(min_x, max_x, h)
# y_lst = [fuzzy_trapezoidal_number(x, [1, 3, 7, 9]) for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# plt.show()

# min_x, max_x, h = 0, 10, 0.001
# x_lst = np.arange(min_x, max_x, h)
# y_lst = [number(x, 5) for x in x_lst]
# fig, ax = plt.subplots()
# ax.plot(x_lst, y_lst)
# plt.show()
```
