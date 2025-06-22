#include "math_operations.hpp"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace math_ops {

    double add(double a, double b) {
        return a + b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double power(double base, double exponent) {
        return std::pow(base, exponent);
    }

    std::vector<double> add_vectors(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }

    double mean(const std::vector<double>& values) {
        if (values.empty()) {
            throw std::invalid_argument("Cannot compute mean of empty vector");
        }

        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        return sum / values.size();
    }

    double variance(const std::vector<double>& values) {
        if (values.size() < 2) {
            throw std::invalid_argument("Need at least 2 values to compute variance");
        }

        double m = mean(values);
        double sum_sq_diff = 0.0;

        for (double value : values) {
            double diff = value - m;
            sum_sq_diff += diff * diff;
        }

        return sum_sq_diff / (values.size() - 1);
    }

    double standard_deviation(const std::vector<double>& values) {
        return std::sqrt(variance(values));
    }

    // Matrix implementation
    Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_.resize(rows * cols, 0.0);
    }

    Matrix::Matrix(const std::vector<std::vector<double>>& data) {
        if (data.empty() || data[0].empty()) {
            throw std::invalid_argument("Matrix cannot be empty");
        }

        rows_ = data.size();
        cols_ = data[0].size();
        data_.reserve(rows_ * cols_);

        for (const auto& row : data) {
            if (row.size() != cols_) {
                throw std::invalid_argument("All rows must have the same size");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    double Matrix::get(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data_[row * cols_ + col];
    }

    void Matrix::set(size_t row, size_t col, double value) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        data_[row * cols_ + col] = value;
    }

    Matrix Matrix::operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrices must have the same dimensions for addition");
        }

        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Matrix Matrix::operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Invalid dimensions for matrix multiplication");
        }

        Matrix result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    Matrix Matrix::transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.set(j, i, get(i, j));
            }
        }
        return result;
    }

    std::vector<std::vector<double>> Matrix::to_list() const {
        std::vector<std::vector<double>> result(rows_, std::vector<double>(cols_));
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result[i][j] = get(i, j);
            }
        }
        return result;
    }
}