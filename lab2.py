from simplex import SimplexSolver

c = [1, -1, -3]
A = [
    [2, -1, 1],
    [-4, 2, -1],
    [3, 0, 1]
]
b = [1, 2, 5]

solver = SimplexSolver(c, A, b)
solver.simplex_solver()
