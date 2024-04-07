from simplex import SimplexSolver

c = [-7, -3]
A = [
    [5, 2],
    [8, 4],
]
b = [20, 38]

solver = SimplexSolver(c, A, b)
solver.dual_simplex_solver()
