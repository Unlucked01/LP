import numpy as np


def balance(supply, demand, cost):
    if np.sum(supply) != np.sum(demand):
        print("Несбалансированная задача о назначениях.")
        difference = abs(np.sum(supply) - np.sum(demand))
        if np.sum(supply) < np.sum(demand):
            new_row = np.zeros((1, cost.shape[1]), dtype=int)
            cost = np.vstack([cost, new_row])
            supply = np.append(supply, difference)
        else:
            demand = np.append(demand, difference)
            new_row = np.zeros((cost.shape[0], 1), dtype=int)
            cost = np.hstack([cost, new_row])
        print("Открытая модель успешно приведена к закрытой.")
    else:
        print("Задача сбалансированна")
    return supply, demand, cost


def northwest_corner(supply, demand, cost):
    r, c = 0, 0
    m, n = cost.shape
    X = np.zeros((m, n))
    Z = 0
    while r != m and c != n:
        if supply[r] <= demand[c]:
            demand[c] -= supply[r]
            X[r][c] = supply[r]
            Z += supply[r] * cost[r][c]
            r += 1
        else:
            supply[r] -= demand[c]
            X[r][c] = demand[c]
            Z += demand[c] * cost[r][c]
            c += 1
    return Z, X


costs = np.array([[10, 7, 4, 1, 4],
                [2, 7, 10, 6, 11],
                [8, 5, 3, 2, 2],
                [11, 8, 12, 16, 13]])

supply = np.array([100, 250, 200, 300])
demand = np.array([200, 200, 100, 100, 250])

print("Вариант 6")
print("Количество поставляемого груза:", *supply)
print("Количество потребляемого груза:", *demand)

supply, demand, cost = balance(supply, demand, costs)
Z, X = northwest_corner(supply, demand, cost)

print(f"Таблица распределения:\n{X}\n\nZ = {Z}")