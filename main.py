from math import sqrt
import numpy as np

def seidel(A, b, eps):
    n = len(A)
    x = [.0 for i in range(n)]

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = np.max(np.abs(x_new - x)) <= eps
        x = x_new

    return x

A = np.array([[4, -2, 1],
              [2, -5, 3],
              [1, 1, -6]])

b = np.array([-2, -25, -15])
x = np.array(0)

x = seidel(A, b, x)

#проверка
residual = np.dot(A, x) - b

if np.allclose(residual, 0, atol=0.001):
    print("Решение системы уравнений:")
    print(x)
else:
    print("Ошибка при решении системы уравнений")