# Cython C Integration Example

This project demonstrates how to use Cython to wrap C code and create Python extensions.

## Project Structure

- `src/math_ops.c` - C implementation of mathematical functions
- `src/math_ops.h` - C header file with function declarations  
- `cython_wrapper.pyx` - Cython wrapper that exposes C functions to Python
- `setup.py` - Build configuration
- `test_example.py` - Test script demonstrating usage

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install cython numpy setuptools
   ```

2. Build the extension:
   ```bash
   python setup.py build_ext --inplace
   ```

3. Run tests:
   ```bash
   python test_example.py
   ```

## Alternative Build Methods

### Using pip (development mode):
```bash
pip install -e .
```

### Using cythonize directly:
```bash
cythonize -i cython_wrapper.pyx
```

## Features Demonstrated

- Wrapping C functions with Cython
- Memory management with malloc/free
- Type declarations for performance
- Error handling in Cython
- Performance comparison between Python and C
- Working with arrays and pointers

## Key Cython Concepts

- `cdef extern` - Declare external C functions
- `cdef` - Declare C variables and functions
- Memory management with `malloc`/`free`
- Type annotations for performance
- Compiler directives for optimization

## Troubleshooting

- Ensure all C files are included in `sources` list
- Check include paths for header files
- Use `--verbose` flag for detailed build output
- Clean build artifacts: `python setup.py clean --all`
"""

# ===== Build Instructions =====
"""
To build and use this project:

1. Create the directory structure and save each file
2. Install requirements: pip install cython numpy
3. Build: python setup.py build_ext --inplace
4. Test: python test_example.py

The build process will:
- Compile the C code (math_ops.c)
- Generate C code from Cython (.pyx -> .c)
- Compile everything into a Python extension module
- Create a .so/.pyd file that can be imported in Python
"""