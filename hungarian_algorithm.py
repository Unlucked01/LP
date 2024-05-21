# Lab 7
import numpy as np


def hungarian_algorithm(cost_matrix):
    copy_matrx = np.copy(cost_matrix)
    n = cost_matrix.shape[0]
    # Вычитание минимального элемента в каждой строке
    copy_matrx -= np.min(copy_matrx, axis=1, keepdims=True)
    # Вычитание минимального элемента в каждом столбце
    copy_matrx -= np.min(copy_matrx, axis=0, keepdims=True)
    print("Первое приближение\n", copy_matrx)

    while True:
        min_zeros_row = np.argmin(np.sum(copy_matrx == 0, axis=1))
        m_copy = np.copy(copy_matrx)

        for i in range(min_zeros_row, min_zeros_row + n):
            indices_r = np.where(m_copy[i % n] == 0)[0]
            if indices_r.size == 0:
                continue

            m_copy[i % n, indices_r[1:]] = -1

            indices_c = np.where(m_copy[:, indices_r[0]] == 0)[0]
            indices_c = indices_c[indices_c != i % n]
            m_copy[indices_c, indices_r[0]] = -1

        if np.count_nonzero(m_copy == 0) == n:
            print("Достигнуто оптимальное решение")
            Z = form_result(m_copy, cost_matrix)
            break

        line_rows, line_cols = min_covering_lines(copy_matrx)
        print("\nВычеркиваем строчки: ", *line_rows, "\nВычеркиваем колонки: ", *line_cols)

        mask_rows = np.ones(n, dtype=bool)
        mask_cols = np.ones(n, dtype=bool)
        mask_rows[line_rows] = False
        mask_cols[line_cols] = False
        min_uncovered_value = np.min(copy_matrx[mask_rows][:, mask_cols])

        # Обновление весов
        copy_matrx[np.ix_(mask_rows, mask_cols)] -= min_uncovered_value
        copy_matrx[np.ix_(~mask_rows, ~mask_cols)] += min_uncovered_value
        print("После оптимизации:\n", copy_matrx)
        print()

    # Возвращаем оптимальное решение
    return m_copy, Z


def min_covering_lines(matrix):
    m = np.copy(matrix)

    cols_lines = set()
    rows_lines = set()

    num_zeros_rows = np.sum(m == 0, axis=1)
    num_zeros_cols = np.sum(m == 0, axis=0)
    max_count = max(np.max(num_zeros_rows), np.max(num_zeros_cols))

    max_count_in_r = np.count_nonzero(num_zeros_rows == max_count)
    max_count_in_c = np.count_nonzero(num_zeros_cols == max_count)

    if max_count_in_r >= max_count_in_c:
        rows = np.where(num_zeros_rows == max_count)[0]
        for row in rows:
            rows_lines.add(row)
            m[row][m[row] == 0] = -1
        columns_with_zeros = np.where(np.any(m == 0, axis=0))[0]
        for col in columns_with_zeros:
            cols_lines.add(col)
            m[:, col] = np.where(m[:, col] == 0, -1, m[:, col])
    else:
        cols = np.where(num_zeros_cols == max_count)[0]
        for col in cols:
            cols_lines.add(col)
            m[:, col] = np.where(m[:, col] == 0, -1, m[:, col])

        rows_with_zeros = np.where((m == 0).any(axis=1))[0]
        for row in rows_with_zeros:
            rows_lines.add(row)
            m[row][m[row] == 0] = -1
    return list(rows_lines), list(cols_lines)


def form_result(zeros_matrix, cost_matrix):
    Z = 0
    for index in np.argwhere(zeros_matrix == 0):
        Z += cost_matrix[index[0], index[1]]
    return Z


cost_matrix = np.array([[50, 50, 120, 20],
                        [70, 40, 20, 30],
                        [90, 30, 50, 140],
                        [70, 20, 60, 70]])

# cost_matrix = np.array([[10, 5, 9, 18, 11],
#                         [13, 19, 6, 12, 14],
#                         [3, 2, 4, 4, 5],
#                         [18, 9, 12, 17, 15],
#                         [11, 6, 14, 19, 10]])

assignment, Z = hungarian_algorithm(cost_matrix)
print(f"До преобразований:\n{cost_matrix}\nОптимальное назначение:\n{assignment}\n\nZ = {Z}")