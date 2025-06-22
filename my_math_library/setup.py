from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "my_math_library._core",
        [
            "src/cpp/math_operations.cpp",
            "src/cpp/bindings.cpp",
        ],
        include_dirs=[
            "src/cpp",
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=14,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
