import numpy as np
import matrix_ops
import platform

def test_accelerate_info():
    """Test Accelerate framework detection"""
    print("Testing Accelerate framework detection...")
    
    info = matrix_ops.get_accelerate_info()
    print(f"Platform: {platform.system()}")
    print(f"Using Accelerate: {info['using_accelerate']}")
    print(f"Backend: {info['backend']}")
    
    if platform.system() == 'Darwin':
        assert info['using_accelerate'] == True, "Should use Accelerate on macOS"
        print("✅ Accelerate framework detected on macOS")
    else:
        assert info['using_accelerate'] == False, "Should not use Accelerate on non-macOS"
        print("✅ Manual implementation on non-macOS platform")
    
    print()
    return True

def test_matrix_vector_multiply():
    """Test matrix-vector multiplication"""
    print("Testing matrix-vector multiplication...")
    
    # Create test data
    matrix = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float64)
    
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    # Expected result: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
    expected = np.array([14.0, 32.0])
    
    # Call C function
    result = matrix_ops.matrix_vector_multiply(matrix, vector)
    
    print(f"Matrix:\n{matrix}")
    print(f"Vector: {vector}")
    print(f"C result: {result}")
    print(f"Expected: {expected}")
    print(f"NumPy result: {matrix @ vector}")
    print(f"Match NumPy: {np.allclose(result, matrix @ vector)}")
    print(f"Test passed: {np.allclose(result, expected)}\n")
    
    return np.allclose(result, expected)

def test_vector_dot_product():
    """Test vector dot product"""
    print("Testing vector dot product...")
    
    # Create test vectors
    vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    vec2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    
    # Expected result: 1*4 + 2*5 + 3*6 = 32
    expected = 32.0
    
    # Call C function
    result = matrix_ops.vector_dot_product(vec1, vec2)
    
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(f"C result: {result}")
    print(f"Expected: {expected}")
    print(f"NumPy result: {np.dot(vec1, vec2)}")
    print(f"Match NumPy: {np.isclose(result, np.dot(vec1, vec2))}")
    print(f"Test passed: {np.isclose(result, expected)}\n")
    
    return np.isclose(result, expected)

def test_accelerate_specific_functions():
    """Test functions only available with Accelerate framework"""
    info = matrix_ops.get_accelerate_info()
    
    if not info['using_accelerate']:
        print("Skipping Accelerate-specific tests (not on macOS)")
        return True
    
    print("Testing Accelerate-specific functions...")
    
    # Test matrix-matrix multiplication
    try:
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order='F')
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64, order='F')
        
        result = matrix_ops.matrix_matrix_multiply(A, B)
        expected = A @ B
        
        print(f"Matrix A:\n{A}")
        print(f"Matrix B:\n{B}")
        print(f"C result:\n{result}")
        print(f"NumPy result:\n{expected}")
        print(f"Matrix multiply test passed: {np.allclose(result, expected)}")
        
        if not np.allclose(result, expected):
            return False
            
    except AttributeError:
        print("Matrix-matrix multiply not available")
        return False
    
    # Test vector scaling
    try:
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        alpha = 2.5
        
        result = matrix_ops.vector_scale(vector, alpha)
        expected = alpha * vector
        
        print(f"Vector: {vector}")
        print(f"Scale factor: {alpha}")
        print(f"C result: {result}")
        print(f"Expected: {expected}")
        print(f"Vector scale test passed: {np.allclose(result, expected)}")
        
        if not np.allclose(result, expected):
            return False
            
    except AttributeError:
        print("Vector scale not available")
        return False
    
    # Test vector axpy (alpha * x + y)
    try:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        alpha = 2.0
        
        result = matrix_ops.vector_axpy(alpha, x, y)
        expected = alpha * x + y
        
        print(f"Vector x: {x}")
        print(f"Vector y: {y}")
        print(f"Alpha: {alpha}")
        print(f"C result: {result}")
        print(f"Expected: {expected}")
        print(f"Vector axpy test passed: {np.allclose(result, expected)}")
        
        if not np.allclose(result, expected):
            return False
            
    except AttributeError:
        print("Vector axpy not available")
        return False
    
    print("✅ All Accelerate-specific tests passed!\n")
    return True

def benchmark_comparison():
    """Compare performance with NumPy"""
    print("Performance comparison...")
    
    # Create larger test data
    sizes = [100, 500, 1000] if platform.system() == 'Darwin' else [100, 500]
    
    for size in sizes:
        print(f"\nMatrix size: {size} x {size}")
        matrix = np.random.rand(size, size).astype(np.float64)
        vector = np.random.rand(size).astype(np.float64)
        
        import time
        
        # Time C implementation
        start = time.time()
        result_c = matrix_ops.matrix_vector_multiply(matrix, vector)
        time_c = time.time() - start
        
        # Time NumPy implementation
        start = time.time()
        result_numpy = matrix @ vector
        time_numpy = time.time() - start
        
        backend = matrix_ops.get_accelerate_info()['backend']
        print(f"Backend: {backend}")
        print(f"C implementation time: {time_c:.4f} seconds")
        print(f"NumPy time: {time_numpy:.4f} seconds")
        print(f"Results match: {np.allclose(result_c, result_numpy)}")
        
        if time_c > 0 and time_numpy > 0:
            ratio = time_c / time_numpy
            print(f"Speed ratio (C/NumPy): {ratio:.2f}")
            if ratio < 1:
                print(f"C implementation is {1/ratio:.1f}x faster")
            else:
                print(f"NumPy is {ratio:.1f}x faster")

if __name__ == "__main__":
    try:
        print("=== Matrix Operations with Accelerate Framework Test ===\n")
        
        # Run tests
        test1_passed = test_accelerate_info()
        test2_passed = test_matrix_vector_multiply()
        test3_passed = test_vector_dot_product()
        test4_passed = test_accelerate_specific_functions()
        
        if all([test1_passed, test2_passed, test3_passed, test4_passed]):
            print("✅ All tests passed!")
            benchmark_comparison()
        else:
            print("❌ Some tests failed!")
            
    except ImportError:
        print("Module not found. Please build the extension first:")
        print("python setup.py build_ext --inplace")
    except Exception as e:
        print(f"Error: {e}")