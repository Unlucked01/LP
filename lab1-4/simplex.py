import math
import numpy as np
import pandas as pd


class SimplexSolver:
    def __init__(self, c, A, b, M=1000):
        self.c = c
        self.A = A
        self.b = b
        self.M = M
        self.tableau = self.to_tableau()
        self.iteration = 1
        self.x_values = [f'x{i + 1}' for i in range(len(self.tableau[0]) - 1)]
        self.cols = self.x_values + ['b']
        self.rows = self.x_values[len(self.A):len(self.tableau[0])] + ['Z']

    def simplex_solver(self):
        while self.can_be_improved():
            print(f"Iteration {self.iteration}:")
            pivot_position = self.get_pivot_position()
            self.print_tableau()
            self.rows = self.form_tableau_header(pivot_position)
            if any(x < 0 for x in self.tableau[-1]):
                self.tableau = self.pivot_step(pivot_position)
            else:
                break
            self.iteration += 1
            print()
        print(f"Final tableau after {self.iteration - 1} iterations:")
        self.print_tableau()
        return self.get_solution()

    def simplex_M_solver(self):
        m = len(self.A)  # Number of constraints
        n = len(self.c)  # Number of variables
        self.x_values = [f'x{i + 1}' for i in range(n)]
        self.tableau = self.to_tableau()
        self.cols = self.x_values + [f'y{i + 1}' for i in range(m)] + ['b']
        self.rows = self.cols[n:-1] + ['Z'] + [' ']

        self.tableau.append([0] * (n + m + 1))
        for i in range(n):
            self.tableau[-1][i] = -(self.A[0][i] + self.A[1][i])
        self.simplex_solver()

        if self.iteration == 1:
            print("\nНет решения")
            exit(0)

        print("-------------------step 2-------------------\n")
        self.to_M_tableau(n)
        self.print_tableau()

    def dual_simplex_solver(self):
        self.simplex_solver()


    # def cutting_plane(self):
    #     self.simplex_solver()
    #     print()
    #     self.cutting_plane_solver()
    #     self.print_tableau()

    # def cutting_plane_solver(self):
    #     temp = [self.tableau[i][-1] - math.floor(self.tableau[i][-1]) for i in range(len(self.A))]
    #     print(temp)
    #     res = []
    #     for t in temp:
    #         if 0.01 < t < 0.98:
    #             res.append(t)
    #     result = np.max(res)
    #     print(result)

    def to_tableau(self):
        c_extended = self.c + [0] * len(self.A)
        A_extended = []
        B_extended = self.b + [0]
        for i in range(len(self.A)):
            row = self.A[i] + [1 if j == i else 0 for j in range(len(self.A))]
            A_extended.append(row)
        xb = [row + [bi] for row, bi in zip(A_extended, B_extended)]
        z = c_extended + [0]
        return xb + [z]

    def to_M_tableau(self, n):
        del self.cols[n]
        del self.cols[n]  # After removing 'y1', 'y2' becomes the new n-th element
        self.tableau = np.delete(self.tableau, n, 1)
        self.tableau = np.delete(self.tableau, n, 1)
        self.tableau = self.tableau[:-1]
        self.rows = self.rows[:-1]

    def can_be_improved(self):
        z = self.tableau[-1]
        return any(x < 0 for x in z[:-1])

    def get_pivot_position(self):
        z = self.tableau[-1]
        column = np.argmin(z[:-1])
        restrictions = []
        for eq in self.tableau[:-1]:
            el = eq[column]
            restrictions.append(math.inf if el <= 0 else eq[-1] / el)
        row = restrictions.index(min(restrictions))
        return row, column

    def pivot_step(self, pivot_position):
        new_tableau = [[] for eq in self.tableau]
        i, j = pivot_position
        pivot_value = self.tableau[i][j]
        new_tableau[i] = np.array(self.tableau[i]) / pivot_value
        for eq_i, eq in enumerate(self.tableau):
            if eq_i != i:
                multiplier = np.array(new_tableau[i]) * self.tableau[eq_i][j]
                new_tableau[eq_i] = np.array(self.tableau[eq_i]) - multiplier
        return new_tableau

    @staticmethod
    def is_basic(column):
        return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

    def get_solution(self):
        columns = np.array(self.tableau).T
        solutions = []
        for column in columns[:-1]:
            solution = 0
            if self.is_basic(column):
                one_index = column.tolist().index(1)
                solution = columns[-1][one_index]
            solutions.append(solution)
        solutions.append(columns[-1][-1])
        return solutions

    def form_tableau_header(self, pivot_position):
        i, j = pivot_position
        self.rows[i] = self.cols[j]
        return self.rows

    def print_tableau(self):
        data = []
        for row in self.tableau:
            rounded_row = [round(element, 3) for element in row]
            data.append(rounded_row)
        df = pd.DataFrame(data, columns=self.cols, index=self.rows)
        print(df)

