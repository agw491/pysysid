# cython_wrapper.pyx
from libc.stdlib cimport malloc, free

# Declare external C functions
cdef extern from "src/math_ops.h":
    double add_numbers(double a, double b)
    double multiply_numbers(double a, double b)
    int factorial(int n)
    void bubble_sort(double arr[], int n)

# Python wrapper functions
def py_add_numbers(double a, double b):
    "Add two numbers using C function. "
    return add_numbers(a, b)

def py_multiply_numbers(double a, double b):
    "Multiply two numbers using C function."
    return multiply_numbers(a, b)

def py_factorial(int n):
    "Calculate factorial using C function."
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return factorial(n)

def py_bubble_sort(list numbers):
    "Sort a list of numbers using C bubble sort function."
    cdef int n = len(numbers)
    cdef double* arr = <double*>malloc(n * sizeof(double))

    if not arr:
        raise MemoryError("Could not allocate memory")

    try:
        # Copy Python list to C array
        for i in range(n):
            arr[i] = numbers[i]

        # Call C function
        bubble_sort(arr, n)

        # Copy back to Python list
        result = []
        for i in range(n):
            result.append(arr[i])

        return result
    finally:
        free(arr)

# Example of a Cython-optimized function that calls C code
def optimized_sum_of_squares(list numbers):
    "Calculate sum of squares efficiently using Cython and C."
    cdef double total = 0.0
    cdef double num

    for num in numbers:
        total = add_numbers(total, multiply_numbers(num, num))

    return total
