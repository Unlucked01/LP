import sys
from typing import Literal

import numpy as np
import pandas as pd


class TransportationProblem:
    def __init__(self, supply, demand, costs):
        self.supply = np.array(supply)
        self.demand = np.array(demand)
        self.costs = np.array(costs)
        self.num_consumers = len(self.demand)
        self.num_suppliers = len(self.supply)
        self.u = []
        self.v = []
        self.Z = 0
        self.basic_indexes = [()]
        self.numbers_elem = 0
        self._check_balanced()

    def _check_balanced(self):
        if sum(self.supply) != sum(self.demand):
            print("Несбалансированная задача о назначениях.")
            difference = abs(sum(self.supply) - sum(self.demand))
            if sum(self.supply) < sum(self.demand):
                new_row = np.zeros((1, self.costs.shape[1]), dtype=int)
                self.costs = np.vstack([self.costs, new_row])
                self.supply = np.append(self.supply, difference)
            else:
                self.demand = np.append(self.demand, difference)
                new_row = np.zeros((self.costs.shape[0], 1), dtype=int)
                self.costs = np.hstack([self.costs, new_row])
            print("Открытая модель успешно приведена к закрытой.")

    def form_matrix(self):
        return pd.DataFrame(self.costs, index=self.supply, columns=self.demand)

    def formed_matrix(self, solution):
        return pd.DataFrame(solution, index=self.supply, columns=self.demand)

    def northwest_corner(self):
        self.num_suppliers = len(self.supply)
        self.num_consumers = len(self.demand)
        supply_temp = np.copy(self.supply)
        demand_temp = np.copy(self.demand)
        solution = np.array([["---"] * self.num_consumers] * self.num_suppliers)
        r = 0
        c = 0
        while r < self.num_suppliers and c < self.num_consumers:
            if supply_temp[r] <= demand_temp[c]:
                solution[r, c] = supply_temp[r]
                demand_temp[c] -= supply_temp[r]
                self.Z += supply_temp[r] * self.costs[r][c]
                supply_temp[r] = 0
                r += 1
            else:
                solution[r, c] = demand_temp[c]
                supply_temp[r] -= demand_temp[c]
                self.Z += demand_temp[c] * self.costs[r][c]
                demand_temp[c] = 0
                c += 1
        return solution

    def is_basic(self, dist_table):
        self.basic_indexes = self._find_basic_indices(dist_table)
        self.numbers_elem = self.num_suppliers + self.num_consumers - 1
        return len(self.basic_indexes) == self.numbers_elem

    @staticmethod
    def _find_basic_indices(dist_table):
        basic_indices = []
        for i in range(len(dist_table)):
            for j in range(len(dist_table[0])):
                if dist_table[i][j] != "---":
                    basic_indices.append((i, j))
        return basic_indices

    def potential_calc(self, dist_table):
        solution = np.full(shape=dist_table.shape, fill_value=dist_table)
        while True:
            if not self.is_basic(solution):
                break
            self._find_potentials()
            print("u = ", self.u, " v = ", self.v)
            max_negative_index = self._new_basic_variable()
            if max_negative_index is None:
                print("Достижение оптимального решения.")
                break
            print("Найдена новая базисная переменная:", max_negative_index)
            cycle = self._find_cycle(max_negative_index)
            solution = self._cycle_apply(cycle, dist_table)

    # def _basic_solution(self):
    #     self.u = np.full_like(self.supply, np.min)
    #     self.v = np.full_like(self.demand, np.min)
    #     self.u[0] = 0
    #     print(self.basic_indexes)
    #     print(self.u, self.v)
    #     exit(0)
    #     for i, j in self.basic_indexes:
    #         if self.u[i] == 0 and self.v[j] == 0:
    #             self.v[i] = self.costs[i][j]
    #             print("v[i]=", self.v[i], self.costs[i][j])
    #         elif self.u[i] == 0:
    #             self.u[i] = self.costs[i][j] - self.v[j]
    #             print("u[i] =", self.u[i], self.costs[i][j], " v[j] =", self.v[j])
    #         elif self.v[j] == 0:
    #             self.v[j] = self.costs[i][j] - self.u[i]
    #             print("v[i] =", self.v[i], self.costs[i][j], " u[i] =", self.u[i])

    def _find_potentials(self):
        m, n = self.costs.shape
        self.u = np.full(m, np.nan)
        self.v = np.full(n, np.nan)
        self.u[0] = 0

        while np.isnan(self.u).any() or np.isnan(self.v).any():
            for i, j in self.basic_indexes:
                if np.isnan(self.u[i]) and not np.isnan(self.v[j]):
                    self.u[i] = self.costs[i][j] - self.v[j]
                elif not np.isnan(self.u[i]) and np.isnan(self.v[j]):
                    self.v[j] = self.costs[i][j] - self.u[i]

    def _find_negative_elements(self):
        negative_elements = []
        for i in range(self.num_suppliers):
            for j in range(self.num_consumers):
                if (i, j) not in self.basic_indexes:
                    reduced_cost = self.costs[i][j] - self.u[i] - self.v[j]
                    if reduced_cost < 0:
                        negative_elements.append(((i, j), abs(reduced_cost)))
        return negative_elements

    def _new_basic_variable(self):
        negative_elements = self._find_negative_elements()
        if negative_elements:
            max_negative_index = self._max_index(negative_elements)
            return max_negative_index
        else:
            return None

    @staticmethod
    def _max_index(arr):
        max_value = float('-inf')
        max_index = None
        for index, value in arr:
            if abs(value) > max_value:
                max_value = abs(value)
                max_index = index
        return max_index

    def _find_cycle(self, start_index):
        visited = set()
        path = []

        def search(curr_cell, prev_cell, is_vertical):
            if curr_cell in visited:
                if curr_cell == start_index and len(path) > 3:
                    return True
                return False
            visited.add(curr_cell)
            path.append(curr_cell)
            for next_cell in (self.basic_indexes + [start_index]):
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

    def _cycle_apply(self, cycle_indexes, dist_table):
        values = [int(dist_table[i][j]) for i, j in cycle_indexes[1:]]
        min_value = min(values)
        print("Минимальное значение: ", min_value)

        solution = np.array([["---"] * self.num_consumers] * self.num_suppliers)
        dist_table[cycle_indexes[0][0]][cycle_indexes[0][1]] = 0
        add = True

        for i, j in ([cycle_indexes[0]] + self.basic_indexes):
            if (i, j) in cycle_indexes[::-1]:
                elem = int(dist_table[i][j])
                if add:
                    elem += min_value
                else:
                    elem -= min_value
                add = not add
                solution[i][j] = str(elem) if elem != 0 else "---"
            else:
                if solution[i][j] == "---":
                    solution[i][j] = dist_table[i][j]
        print(solution)
        return solution



costs = [
    [10, 7, 4, 1, 4],
    [2, 7, 10, 6, 11],
    [8, 5, 3, 2, 2],
    [11, 8, 12, 16, 13]
]

supply = [100, 250, 200, 300]
demand = [200, 200, 100, 100, 250]

print("Вариант 6")
print("Количество поставляемого груза:", *supply)
print("Количество потребляемого груза:", *demand)
print()

tp = TransportationProblem(supply, demand, costs)
print(tp.form_matrix())
dist_table = tp.northwest_corner()
print("\nТаблица распределения:")
print(tp.formed_matrix(dist_table))
print(tp.Z)
print("Решение базисное" if tp.is_basic(dist_table) else "Не базисное решение")
tp.potential_calc(dist_table)
