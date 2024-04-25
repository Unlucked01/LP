import pandas as pd
import numpy as np
import TransportationProblem as TP

costs = [[10, 7, 4, 1, 4],
        [2, 7, 10, 6, 11],
        [8, 5, 3, 2, 2],
        [11, 8, 12, 16, 13]]

supply = [100, 250, 200, 300]
demand = [200, 200, 100, 100, 250]

print("Вариант 6")
print("Количество поставляемого груза:", *supply)
print("Количество потребляемого груза:", *demand)
print()

tp = TP.TransportationProblem(supply, demand, costs)
print(tp.form_matrix())
dist_table = tp.northwest_corner()
print("\nТаблица распределения:")
print(tp.formed_matrix(dist_table), "\n", tp.Z)
print(tp.is_basic(dist_table))
print(tp.basic_indexes)
tp.potential_calc()
exit(0)
