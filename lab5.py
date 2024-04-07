import pandas as pd
import numpy as np

grid = [[22, 14, 16, 28, 30],
        [19, 17, 26, 36, 36],
        [37, 30, 31, 39, 41]]

supply = [350, 200, 300]
demand = [350, 140, 200, 195, 195]

print("Вариант 6")
print("Количество поставляемого груза:", *supply)
print("Количество потребляемого груза:", *demand)
print()

if sum(supply) != sum(demand):
    print("Несбалансированная задача о назначениях.")
    difference = abs(sum(supply) - sum(demand))
    if sum(supply) < sum(demand):
        grid.append([0] * len(demand))
        supply.append(difference)
    else:
        demand.append(difference)
        for row in grid:
            row.append(0)

    supply_temp = np.copy(supply)
    demand_temp = np.copy(demand)
    print("Открытая модель успешно приведена к закрытой.")
else:
    supply_temp = np.copy(supply)
    demand_temp = np.copy(demand)

df = pd.DataFrame(grid, index=supply, columns=demand)
print(df)

startR = 0
startC = 0
distribution_table = [["-"] * len(demand_temp) for _ in range(len(supply_temp))]
Z = 0

while startR < len(supply_temp) and startC < len(demand_temp):
    if supply_temp[startR] <= demand_temp[startC]:
        distribution_table[startR][startC] = supply_temp[startR]
        Z += supply_temp[startR] * grid[startR][startC]
        demand_temp[startC] -= supply_temp[startR]
        startR += 1
    else:
        distribution_table[startR][startC] = demand_temp[startC]
        Z += demand_temp[startC] * grid[startR][startC]
        supply_temp[startR] -= demand_temp[startC]
        startC += 1

res_dist_table = pd.DataFrame(distribution_table, index=supply, columns=demand)
print("\nТаблица распределения:")
print(res_dist_table, "\n")

non_zero_elements = [elem for row in distribution_table for elem in row if elem != "-"]
numbers_elem = len(supply_temp) + len(demand_temp) - 1

if len(non_zero_elements) == numbers_elem:
    print(f"Решение базисное, содержит: {len(non_zero_elements)} компонентов")
else:
    print(f"Решение не базисное {len(non_zero_elements)} != {numbers_elem}")

if np.all(non_zero_elements) > 0 and len(non_zero_elements) == numbers_elem:
    print("Решение невырожденное, т.к. все элементы отличные от 0, положительны")
else:
    print("Решение вырожденное!")

print(f"\nНачальное значение целевой функции\nZ = {Z}(ден.ед.)")
