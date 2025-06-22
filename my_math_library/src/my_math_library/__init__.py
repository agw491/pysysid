"""
A fast math library with C++ backend using pybind11.
"""

from typing import List, Union

try:
    from ._core import *  # Import all C++ functions
except ImportError as e:
    raise ImportError(
        "Failed to import compiled C++ extension. "
        "Make sure the package is properly installed with: uv pip install -e ."
    ) from e

__version__ = "0.1.0"
__author__ = "Your Name"

# Re-export main functions for better IDE support
__all__ = [
    "add",
    "multiply",
    "power",
    "add_vectors",
    "dot_product",
    "mean",
    "variance",
    "standard_deviation",
    "Matrix",
]


def create_identity_matrix(size: int) -> Matrix:
    """Create an identity matrix of given size."""
    matrix = Matrix(size, size)
    for i in range(size):
        matrix.set(i, i, 1.0)
    return matrix


def create_zeros_matrix(rows: int, cols: int) -> Matrix:
    """Create a matrix filled with zeros."""
    return Matrix(rows, cols)


def create_matrix_from_list(data: List[List[float]]) -> Matrix:
    """Create a matrix from a 2D Python list."""
    return Matrix(data)
