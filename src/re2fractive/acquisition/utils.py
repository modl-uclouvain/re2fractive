import itertools

import numpy as np


def generate_weights(M, wind=0.01):
    window = int(1 / wind)
    weights = []
    for combination in itertools.product(range(window + 1), repeat=M):
        if sum(combination) == window:
            weight = [val * (100 / window) / 100 for val in combination]
            # weight = [round(val, 5) for val in weight]
            weights.append(weight)
    return np.array(weights)


def matmul_max(matrix1, matrix2):
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    # Check if the matrices can be multiplied
    if cols1 != rows2:
        raise ValueError(
            "Number of columns in the first matrix must equal the number of rows in the second matrix."
        )

    # Initialize the result matrix with zeros
    result = [[0] * cols2 for _ in range(rows1)]

    # Transpose the second matrix for better cache utilization
    transposed_matrix2 = list(map(list, zip(*matrix2)))

    # Perform matrix multiplication
    for i in range(rows1):
        for j in range(cols2):
            row1 = matrix1[i]
            col2 = transposed_matrix2[j]
            result[i][j] = max(x * y for x, y in zip(row1, col2))

    return result
