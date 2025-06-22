from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "cython_math",  # Name of the compiled module
        sources=[
            "cython_wrapper.pyx",  # Cython source
            "src/math_ops.c"       # C source
        ],
        include_dirs=[
            "src/",                # Include directory for headers
            numpy.get_include()    # NumPy headers (if needed)
        ],
        language="c"
    )
]

setup(
    name="cython-math-example",
    version="0.1.0",
    description="Example of using Cython to wrap C code",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,    # Use Python 3
            'boundscheck': False,   # Disable bounds checking for performance
            'wraparound': False,    # Disable negative index wrapping
        }
    ),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "cython>=0.29",
        "numpy"
    ]
)