#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "math_operations.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fast math operations implemented in C++";

    // Basic functions
    m.def("add", &math_ops::add, "Add two numbers",
          py::arg("a"), py::arg("b"));

    m.def("multiply", &math_ops::multiply, "Multiply two numbers",
          py::arg("a"), py::arg("b"));

    m.def("power", &math_ops::power, "Raise base to exponent",
          py::arg("base"), py::arg("exponent"));

    // Vector operations
    m.def("add_vectors", &math_ops::add_vectors, "Add two vectors element-wise",
          py::arg("a"), py::arg("b"));

    m.def("dot_product", &math_ops::dot_product, "Compute dot product of two vectors",
          py::arg("a"), py::arg("b"));

    // Statistical functions
    m.def("mean", &math_ops::mean, "Compute mean of a vector",
          py::arg("values"));

    m.def("variance", &math_ops::variance, "Compute variance of a vector",
          py::arg("values"));

    m.def("standard_deviation", &math_ops::standard_deviation,
          "Compute standard deviation of a vector",
          py::arg("values"));

    // Matrix class
    py::class_<math_ops::Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>(), "Create matrix with given dimensions",
             py::arg("rows"), py::arg("cols"))
        .def(py::init<const std::vector<std::vector<double>>&>(),
             "Create matrix from 2D list",
             py::arg("data"))
        .def("get", &math_ops::Matrix::get, "Get element at (row, col)",
             py::arg("row"), py::arg("col"))
        .def("set", &math_ops::Matrix::set, "Set element at (row, col)",
             py::arg("row"), py::arg("col"), py::arg("value"))
        .def_property_readonly("rows", &math_ops::Matrix::rows)
        .def_property_readonly("cols", &math_ops::Matrix::cols)
        .def("__add__", &math_ops::Matrix::operator+)
        .def("__mul__", &math_ops::Matrix::operator*)
        .def("transpose", &math_ops::Matrix::transpose)
        .def("to_list", &math_ops::Matrix::to_list, "Convert to 2D Python list")
        .def("__repr__", [](const math_ops::Matrix& m) {
            return "<Matrix " + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ">";
        });
}