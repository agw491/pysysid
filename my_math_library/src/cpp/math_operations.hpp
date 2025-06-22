#ifndef MATH_OPERATIONS_HPP
#define MATH_OPERATIONS_HPP

#include <vector>

namespace math_ops {
    // Basic arithmetic operations
    double add(double a, double b);
    double multiply(double a, double b);
    double power(double base, double exponent);

    // Vector operations
    std::vector<double> add_vectors(const std::vector<double>& a, const std::vector<double>& b);
    double dot_product(const std::vector<double>& a, const std::vector<double>& b);

    // Statistical functions
    double mean(const std::vector<double>& values);
    double variance(const std::vector<double>& values);
    double standard_deviation(const std::vector<double>& values);

    // Matrix class
    class Matrix {
    public:
        Matrix(size_t rows, size_t cols);
        Matrix(const std::vector<std::vector<double>>& data);

        // Accessors
        double get(size_t row, size_t col) const;
        void set(size_t row, size_t col, double value);
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }

        // Operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix transpose() const;

        // Utility
        std::vector<std::vector<double>> to_list() const;

    private:
        size_t rows_, cols_;
        std::vector<double> data_;
    };
}

#endif // MATH_OPERATIONS_HPP