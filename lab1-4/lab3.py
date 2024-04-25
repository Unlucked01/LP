from simplex import SimplexSolver

c = [5, 3, 4, 1]
A = [
    [1, 3, 2, 2],
    [2, 2, 1, 1],
]
b = [3, 3]

solver = SimplexSolver(c, A, b, 1000)
solver.simplex_M_solver()
