#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// Matriz densa de doubles, almacenada en un unico bloque contiguo (row-major).
// Implementa a mano el procesamiento matricial que pide el enunciado.
class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;  // elemento (i,j) -> data_[i * cols_ + j]

public:
    Matrix(std::size_t rows = 0, std::size_t cols = 0, double init = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init) {}

    static Matrix identity(std::size_t n) {
        Matrix I(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
        return I;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Acceso con verificacion de limites.
    double& operator()(std::size_t i, std::size_t j) {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("Matrix: indice fuera de rango");
        return data_[i * cols_ + j];
    }
    const double& operator()(std::size_t i, std::size_t j) const {
        if (i >= rows_ || j >= cols_) throw std::out_of_range("Matrix: indice fuera de rango");
        return data_[i * cols_ + j];
    }

    std::vector<double> row(std::size_t i) const {
        std::vector<double> r(cols_);
        for (std::size_t j = 0; j < cols_; ++j) r[j] = (*this)(i, j);
        return r;
    }

    std::vector<double> column(std::size_t j) const {
        std::vector<double> c(rows_);
        for (std::size_t i = 0; i < rows_; ++i) c[i] = (*this)(i, j);
        return c;
    }

    Matrix transpose() const {
        Matrix T(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) T(j, i) = (*this)(i, j);
        return T;
    }

    // Producto matricial: (m x n) * (n x p) -> (m x p)
    Matrix multiply(const Matrix& B) const {
        if (cols_ != B.rows_) throw std::invalid_argument("Matrix::multiply: dimensiones incompatibles");
        Matrix C(rows_, B.cols_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k) {
                const double a = (*this)(i, k);
                for (std::size_t j = 0; j < B.cols_; ++j) C(i, j) += a * B(k, j);
            }
        return C;
    }

    // Producto matriz-vector: (m x n) * (n) -> (m)
    std::vector<double> multiply(const std::vector<double>& v) const {
        if (cols_ != v.size()) throw std::invalid_argument("Matrix::multiply: dimensiones incompatibles");
        std::vector<double> r(rows_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) r[i] += (*this)(i, j) * v[j];
        return r;
    }

    // Inversa por Gauss-Jordan con pivoteo parcial. Lanza excepcion si es singular.
    Matrix inverse() const {
        if (rows_ != cols_) throw std::invalid_argument("Matrix::inverse: la matriz debe ser cuadrada");
        const std::size_t n = rows_;
        Matrix a(*this);
        Matrix inv = identity(n);

        // Escala de referencia para decidir (de forma relativa) si un pivote es "casi cero".
        double scale = 0.0;
        for (double x : data_) scale = std::fmax(scale, std::fabs(x));
        const double eps = 1e-12 * (scale > 0.0 ? scale : 1.0);

        for (std::size_t col = 0; col < n; ++col) {
            // 1) Pivoteo parcial: fila con mayor valor absoluto en esta columna.
            std::size_t pivot = col;
            double best = std::fabs(a(col, col));
            for (std::size_t r = col + 1; r < n; ++r)
                if (std::fabs(a(r, col)) > best) { best = std::fabs(a(r, col)); pivot = r; }
            if (best < eps) throw std::runtime_error("Matrix::inverse: matriz singular");

            if (pivot != col)
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(a(col, j), a(pivot, j));
                    std::swap(inv(col, j), inv(pivot, j));
                }

            // 2) Normalizar la fila pivote.
            const double p = a(col, col);
            for (std::size_t j = 0; j < n; ++j) { a(col, j) /= p; inv(col, j) /= p; }

            // 3) Anular el resto de la columna.
            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) continue;
                const double f = a(r, col);
                if (f == 0.0) continue;
                for (std::size_t j = 0; j < n; ++j) {
                    a(r, j) -= f * a(col, j);
                    inv(r, j) -= f * inv(col, j);
                }
            }
        }
        return inv;
    }
};

#endif  // MATRIX_H
