# test_example.py
import cython_math
import time

def test_basic_operations():
    """ Test basic arithmetic operations."""
    print("Testing basic operations:")

    # Test addition
    result = cython_math.py_add_numbers(5.5, 3.2)
    print(f"5.5 + 3.2 = {result}")

    # Test multiplication
    result = cython_math.py_multiply_numbers(4.0, 7.0)
    print(f"4.0 * 7.0 = {result}")

    # Test factorial
    result = cython_math.py_factorial(5)
    print(f"5! = {result}")

def test_sorting():
    """Test the bubble sort function."""
    print("\\nTesting bubble sort:")

    numbers = [64.0, 34.0, 25.0, 12.0, 22.0, 11.0, 90.0]
    print(f"Original: {numbers}")

    sorted_numbers = cython_math.py_bubble_sort(numbers)
    print(f"Sorted: {sorted_numbers}")

def test_performance():
    "Test performance comparison."
    print("\\nTesting performance:")

    numbers = list(range(1000))

    # Python version
    start = time.time()
    python_result = sum(x * x for x in numbers)
    python_time = time.time() - start

    # Cython + C version
    start = time.time()
    cython_result = cython_math.optimized_sum_of_squares(numbers)
    cython_time = time.time() - start

    print(f"Python result: {python_result} (Time: {python_time:.6f}s)")
    print(f"Cython result: {cython_result} (Time: {cython_time:.6f}s)")
    print(f"Speedup: {python_time / cython_time:.2f}x")

if __name__ == "__main__":
    test_basic_operations()
    test_sorting()
    test_performance()
