# Matrix Operations C Extension

This example demonstrates how to perform matrix-vector operations in C and call them from Python using NumPy arrays.

## Files Overview

- `matrix_ops.c` - C implementation with matrix-vector multiply and dot product functions
- `setup.py` - Build configuration using setuptools
- `test_matrix_ops.py` - Python test script demonstrating usage
- `BUILD_INSTRUCTIONS.md` - This file

## Prerequisites

Make sure you have the following installed:
```bash
pip install numpy setuptools
```

You'll also need a C compiler:
- **Linux/macOS**: GCC (usually pre-installed or via package manager)
- **Windows**: Microsoft Visual Studio Build Tools or MinGW

## Building the Extension

1. **Build the extension module:**
   ```bash
   python setup.py build_ext --inplace
   ```

   This creates a shared library file (e.g., `matrix_ops.cpython-39-x86_64-linux-gnu.so` on Linux)

2. **Alternative build method:**
   ```bash
   pip install .
   ```

## Running the Tests

After building, run the test script:
```bash
python test_matrix_ops.py
```

Expected output:
```
Testing matrix-vector multiplication...
Matrix:
[[1. 2. 3.]
 [4. 5. 6.]]
Vector: [1. 2. 3.]
Result: [14. 32.]
Expected: [14. 32.]
Match NumPy: True
Test passed: True

Testing vector dot product...
Vector 1: [1. 2. 3.]
Vector 2: [4. 5. 6.]
Result: 32.0
Expected: 32.0
Match NumPy: True
Test passed: True

All tests passed!
Performance comparison...
Matrix size: 1000 x 1000
C implementation time: 0.0234 seconds
NumPy time: 0.0012 seconds
Results match: True
Speed ratio (NumPy/C): 0.05
```

## Using in Your Code

Once built, you can import and use the functions:

```python
import numpy as np
import matrix_ops

# Matrix-vector multiplication
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
x = np.array([5, 6], dtype=np.float64)
result = matrix_ops.matrix_vector_multiply(A, x)

# Vector dot product  
v1 = np.array([1, 2, 3], dtype=np.float64)
v2 = np.array([4, 5, 6], dtype=np.float64)
dot_result = matrix_ops.vector_dot_product(v1, v2)
```

## Key Features

- **NumPy Integration**: Seamlessly works with NumPy arrays
- **Memory Efficient**: No unnecessary copying of data
- **Error Handling**: Proper dimension checking and error messages
- **Type Safety**: Ensures inputs are valid NumPy arrays
- **Performance**: Direct C implementation (though NumPy's optimized BLAS is usually faster)

## Notes

- Arrays must be of type `np.float64` (double precision)
- The C code assumes row-major order (NumPy default)
- For production use, consider using existing optimized libraries like NumPy/SciPy which use highly optimized BLAS implementations
- This example is educational - NumPy's built-in operations are typically faster due to optimized BLAS libraries

## Troubleshooting

**Import Error**: Make sure the extension was built successfully and the shared library file exists in your directory.

**Compilation Errors**: Ensure you have the correct compiler and NumPy development headers installed.

**Wrong Results**: Verify that your input arrays are `dtype=np.float64` and have the correct dimensions.