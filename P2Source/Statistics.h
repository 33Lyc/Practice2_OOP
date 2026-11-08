#ifndef STATISTICS_H
#define STATISTICS_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Matrix.h"

// Computo estadistico implementado a mano (sin librerias externas).
namespace Stats {

inline double mean(const std::vector<double>& v) {
    if (v.empty()) throw std::invalid_argument("Stats::mean: vector vacio");
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

// Desviacion estandar poblacional (divide entre n), igual que un StandardScaler.
inline double stddev(const std::vector<double>& v) {
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size()));
}

inline std::vector<double> columnMeans(const Matrix& X) {
    std::vector<double> m(X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j) m[j] = mean(X.column(j));
    return m;
}

inline std::vector<double> columnStds(const Matrix& X) {
    std::vector<double> s(X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j) s[j] = stddev(X.column(j));
    return s;
}

// ---- Metricas de error ----
inline void checkSameSize(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty())
        throw std::invalid_argument("Stats: vectores de distinto tamano o vacios");
}

// Error cuadratico medio
inline double mse(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
    checkSameSize(yTrue, yPred);
    double s = 0.0;
    for (std::size_t i = 0; i < yTrue.size(); ++i) {
        const double e = yTrue[i] - yPred[i];
        s += e * e;
    }
    return s / static_cast<double>(yTrue.size());
}

// Raiz del error cuadratico medio
inline double rmse(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
    return std::sqrt(mse(yTrue, yPred));
}

// Error absoluto medio
inline double mae(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
    checkSameSize(yTrue, yPred);
    double s = 0.0;
    for (std::size_t i = 0; i < yTrue.size(); ++i) s += std::fabs(yTrue[i] - yPred[i]);
    return s / static_cast<double>(yTrue.size());
}

// Coeficiente de determinacion R^2 = 1 - SS_res / SS_tot
inline double r2(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
    checkSameSize(yTrue, yPred);
    const double m = mean(yTrue);
    double ssRes = 0.0, ssTot = 0.0;
    for (std::size_t i = 0; i < yTrue.size(); ++i) {
        ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
        ssTot += (yTrue[i] - m) * (yTrue[i] - m);
    }
    return (ssTot == 0.0) ? 0.0 : 1.0 - ssRes / ssTot;
}

}  // namespace Stats

#endif  // STATISTICS_H
