import pytest
from my_math_library import (
    add, multiply, power, add_vectors, dot_product,
    mean, variance, standard_deviation, Matrix,
    create_identity_matrix, create_zeros_matrix
)


def test_add():
    assert add(2.0, 3.0) == 5.0
    assert add(-1.0, 1.0) == 0.0


def test_multiply():
    assert multiply(4.0, 5.0) == 20.0
    assert multiply(-2.0, 3.0) == -6.0


def test_power():
    assert power(2.0, 3.0) == 8.0
    assert power(5.0, 0.0) == 1.0


def test_add_vectors():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = add_vectors(a, b)
    assert result == [5.0, 7.0, 9.0]


def test_add_vectors_different_sizes():
    a = [1.0, 2.0]
    b = [3.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        add_vectors(a, b)


def test_dot_product():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = dot_product(a, b)
    assert result == 32.0  # 1*4 + 2*5 + 3*6


def test_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mean(values) == 3.0


def test_variance():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = variance(values)
    assert abs(result - 2.5) < 1e-10


def test_standard_deviation():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = standard_deviation(values)
    expected = (2.5) ** 0.5
    assert abs(result - expected) < 1e-10


def test_matrix_creation():
    m = Matrix(2, 3)
    assert m.rows == 2
    assert m.cols == 3


def test_matrix_from_list():
    data = [[1.0, 2.0], [3.0, 4.0]]
    m = Matrix(data)
    assert m.rows == 2
    assert m.cols == 2
    assert m.get(0, 0) == 1.0
    assert m.get(1, 1) == 4.0


def test_matrix_get_set():
    m = Matrix(2, 2)
    m.set(0, 0, 5.0)
    m.set(1, 1, 10.0)
    assert m.get(0, 0) == 5.0
    assert m.get(1, 1) == 10.0


def test_matrix_addition():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = m1 + m2
    assert result.get(0, 0) == 6.0
    assert result.get(1, 1) == 12.0


def test_matrix_multiplication():
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    result = m1 * m2
    assert result.get(0, 0) == 19.0  # 1*5 + 2*7
    assert result.get(1, 1) == 50.0  # 3*6 + 4*8


def test_matrix_transpose():
    m = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t = m.transpose()
    assert t.rows == 3
    assert t.cols == 2
    assert t.get(0, 1) == 4.0
    assert t.get(2, 0) == 3.0


def test_identity_matrix():
    identity = create_identity_matrix(3)
    assert identity.get(0, 0) == 1.0
    assert identity.get(1, 1) == 1.0
    assert identity.get(2, 2) == 1.0
    assert identity.get(0, 1) == 0.0
