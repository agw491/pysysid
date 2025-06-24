from setuptools import setup, Extension
import numpy
import platform

# Platform-specific configuration
extra_compile_args = ['-O3']# -DACCELERATE_LAPACK_ILP64']
extra_link_args = []
libraries = []
library_dirs = []

# Check if we're on macOS to use Accelerate framework
if platform.system() == 'Darwin':
    # macOS: Use Accelerate framework
    extra_link_args.extend(['-framework', 'Accelerate'])
    extra_compile_args.append('-DUSE_ACCELERATE=1')
    print("Building with Apple Accelerate framework support")
else:
    # Other platforms: could link to OpenBLAS, Intel MKL, etc.
    # This example falls back to manual implementation
    print("Building with manual C implementation (no BLAS)")
    
    # Uncomment below to use OpenBLAS on Linux (if installed)
    # libraries = ['openblas']
    # extra_compile_args.append('-DUSE_OPENBLAS=1')
    
    # Uncomment below to use Intel MKL (if installed)
    # libraries = ['mkl_rt']
    # extra_compile_args.append('-DUSE_MKL=1')

# Define the extension module
matrix_ops_module = Extension(
    'matrix_ops',
    sources=['matrix_ops.c'],
    include_dirs=[numpy.get_include()],
    libraries=libraries,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

setup(
    name='matrix_ops',
    version='1.0',
    description='Matrix operations in C with optimized BLAS support',
    ext_modules=[matrix_ops_module],
    zip_safe=False,
)