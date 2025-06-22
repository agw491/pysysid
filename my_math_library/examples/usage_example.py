"""
Example usage of the my_math_library package.
"""

from my_math_library import (
    add, multiply, power, add_vectors, dot_product,
    mean, variance, standard_deviation, Matrix,
    create_identity_matrix
)


def main():
    print("=== My Math Library Examples ===\n")

    # Basic operations
    print("Basic Operations:")
    print(f"add(5, 3) = {add(5, 3)}")
    print(f"multiply(4, 7) = {multiply(4, 7)}")
    print(f"power(2, 8) = {power(2, 8)}")
    print()

    # Vector operations
    print("Vector Operations:")
    vec_a = [1.0, 2.0, 3.0, 4.0]
    vec_b = [5.0, 6.0, 7.0, 8.0]
    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"A + B = {add_vectors(vec_a, vec_b)}")
    print(f"A · B = {dot_product(vec_a, vec_b)}")
    print()

    # Statistical functions
    print("Statistical Functions:")
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    print(f"Data: {data}")
    print(f"Mean: {mean(data):.2f}")
    print(f"Variance: {variance(data):.2f}")
    print(f"Standard Deviation: {standard_deviation(data):.2f}")
    print()

    # Matrix operations
    print("Matrix Operations:")

    # Create matrices
    matrix_a = Matrix([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    matrix_b = Matrix([[7.0, 8.0],
                       [9.0, 10.0],
                       [11.0, 12.0]])

    print(f"Matrix A (2x3): {matrix_a}")
    print("Matrix A data:")
    for i in range(matrix_a.rows):
        row = [matrix_a.get(i, j) for j in range(matrix_a.cols)]
        print(f"  {row}")

    print(f"\nMatrix B (3x2): {matrix_b}")
    print("Matrix B data:")
    for i in range(matrix_b.rows):
        row = [matrix_b.get(i, j) for j in range(matrix_b.cols)]
        print(f"  {row}")

    # Matrix multiplication
    result = matrix_a * matrix_b
    print(f"\nA * B = {result}")
    print("Result data:")
    for i in range(result.rows):
        row = [result.get(i, j) for j in range(result.cols)]
        print(f"  {row}")

    # Transpose
    transpose_a = matrix_a.transpose()
    print(f"\nTranspose of A: {transpose_a}")

    # Identity matrix
    identity = create_identity_matrix(3)
    print(f"\n3x3 Identity Matrix: {identity}")
    print("Identity matrix data:")
    identity_list = identity.to_list()
    for row in identity_list:
        print(f"  {row}")


if __name__ == "__main__":
    main()