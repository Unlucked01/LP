import numpy as np
import pandas as pd


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
    basis_indices = []
    while r != m and c != n:
        basis_indices.append((r, c))
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
    return Z, X, basis_indices


def find_potentials(cost, basis_indices):
    m, n = cost.shape
    u, v = [None] * m, [None] * n
    u[0] = 0
    while None in u or None in v:
        for i, j in basis_indices:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i, j] - u[i]
            elif u[i] is None and v[j] is not None:
                u[i] = cost[i, j] - v[j]
    return u, v


def is_optimal(cost, basis_indices, u, v):
    m, n = cost.shape
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis_indices:
                if (u[i] + v[j]) > cost[i, j]:
                    return True
    return False


def choose_start_cell(cost, basis_indices, u, v):
    m, n = cost.shape
    deltas = np.zeros((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis_indices:
                if (u[i] + v[j]) > cost[i, j]:
                    deltas[i, j] = (u[i] + v[j]) - cost[i, j]
    print("Cтоимость:")
    print(pd.DataFrame(cost).to_string(index=False, header=False))
    print("Оценки экономии:\n", pd.DataFrame(deltas, index=u, columns=v),
          "\nМаксимальный элемент = ", np.max(deltas))
    max_delta_cells = np.argwhere(deltas == np.max(deltas))
    min_cost_cells = sorted(max_delta_cells, key=lambda pos: cost[pos[0], pos[1]])
    start_cell = tuple(min_cost_cells[0])

    return start_cell


def find_cycle(basis_indices, start_index):
    visited = set()
    path = []

    def search(curr_cell, prev_cell, is_vertical):
        if curr_cell in visited:
            if curr_cell == start_index and len(path) > 3:
                return True
            return False
        visited.add(curr_cell)
        path.append(curr_cell)
        for next_cell in (basis_indices + [start_index]):
            if next_cell == prev_cell:
                continue
            next_cell_row, next_cell_col = next_cell
            current_cell_row, current_cell_col = curr_cell

            if is_vertical:
                if next_cell_col == current_cell_col:
                    if search(next_cell, curr_cell, False):
                        return True
            elif next_cell_row == current_cell_row:
                if search(next_cell, curr_cell, True):
                    return True

        visited.remove(curr_cell)
        path.pop()
        return False

    if search(start_index, None, False):
        return path
    else:
        return []


def redistribute_load(X, basis_indices, cycle):
    min_weight = 10000000000
    for i, j in cycle[1::2]:
        min_weight = min(min_weight, X[i][j])

    print("Минимальное значение в цикле = ", min_weight)
    min_weight_cells = []
    for i, j in cycle[1::2]:
        if X[i][j] == min_weight:
            min_weight_cells.append((i, j))

    print("Ячейки с минимальным значением = ", min_weight_cells)
    max_cost_cell = sorted(min_weight_cells, key=lambda pos: cost[pos[0]][pos[1]], reverse=True)[0]
    print("Ячейка с максимальной стоимостью перевозки из минимальных = ", max_cost_cell)
    basis_indices.remove(tuple(max_cost_cell))
    basis_indices.append(cycle[0])
    print("Базисные индексы = ", basis_indices)

    for i, j in cycle[::2]:
        X[i][j] += min_weight

    for i, j in cycle[1::2]:
        X[i][j] -= min_weight
    return X


def solve(supply, demand, cost):
    Z, X, basis_indices = northwest_corner(supply, demand, cost)
    while True:
        u, v = find_potentials(cost, basis_indices)
        if not is_optimal(cost, basis_indices, u, v):
            break
        print(f"u = {u}, v = {v}")
        start_cell = choose_start_cell(cost, basis_indices, u, v)
        print("Стартовая ячейка цикла = ", start_cell)
        cycle = find_cycle(basis_indices, start_cell)
        print("Цикл:", cycle)
        print("До преобразований:\n", pd.DataFrame(X, dtype=int).to_string(index=False, header=False))
        X = redistribute_load(X, basis_indices, cycle)
        print("После преобразований:\n", pd.DataFrame(X, dtype=int).to_string(index=False, header=False))
        print()

    new_Z = 0
    for i, j in basis_indices:
        new_Z += cost[i, j] * X[i, j]

    return X, Z, new_Z, basis_indices


def check_basics(basis_indices, supply, demand):
    return len(basis_indices) == len(supply) + len(demand) - 1


def check_zeros(basis_indices, cost):
    return any(cost[i][j] for i, j in basis_indices)


if __name__ == "__main__":
    cost = np.array(
            [[22, 14, 16, 28, 30],
             [19, 17, 26, 36, 36],
             [37, 30, 31, 39, 41]])

    supply = np.array([350, 200, 300])
    demand = np.array([170, 140, 200, 195, 145])

    supply, demand, cost = balance(supply, demand, cost)

    solution, Z, new_Z, basis_indices = solve(supply, demand, cost)

    print("Оптимальное решение:")
    print(pd.DataFrame(solution).to_string(index=False, header=False))
    print("\nЗначение целевой функции до: ", Z)
    print("Значение целевой функции после: ", new_Z)

    if check_basics(basis_indices, supply, demand) and check_zeros(basis_indices, cost):
        print("Решение базисное и невырожденное")
    elif check_basics(basis_indices, supply, demand) and not check_zeros(basis_indices, cost):
        print("Решение базисное и вырожденное")
    elif not check_basics(basis_indices, supply, demand) and check_zeros(basis_indices, cost):
        print("Решение не базисное и вырожденное")
    elif not check_basics(basis_indices, supply, demand) and not check_zeros(basis_indices, cost):
        print("Решение не базисное и невырожденное")

