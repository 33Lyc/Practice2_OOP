#ifndef CSV_READER_H
#define CSV_READER_H

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Matrix.h"

// Conjunto de datos: X (caracteristicas), y (objetivo) y nombres de columnas.
struct Dataset {
    std::vector<std::string> headers;  // nombres de todas las columnas (la ultima es y)
    Matrix X;
    std::vector<double> y;
};

inline std::string trimString(const std::string& s) {
    const char* ws = " \t\r\n";
    const std::size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    const std::size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

inline std::vector<std::string> splitLine(const std::string& line, char delim = ',') {
    std::vector<std::string> out;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, delim)) out.push_back(trimString(field));
    return out;
}

// Lee un CSV con encabezado. Todas las columnas menos la ultima son X; la ultima es y.
inline Dataset loadCsv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("No se pudo abrir el archivo: " + path);

    Dataset ds;
    std::string line;

    if (!std::getline(file, line)) throw std::runtime_error("Archivo vacio: " + path);
    // Quitar BOM UTF-8 si existe.
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF)
        line = line.substr(3);
    ds.headers = splitLine(line);
    const std::size_t nCols = ds.headers.size();
    if (nCols < 2) throw std::runtime_error("Se necesitan al menos 2 columnas en: " + path);

    std::vector<std::vector<double>> rows;
    std::size_t lineNo = 1;
    while (std::getline(file, line)) {
        ++lineNo;
        if (trimString(line).empty()) continue;
        const std::vector<std::string> f = splitLine(line);
        if (f.size() != nCols)
            throw std::runtime_error("Linea " + std::to_string(lineNo) + ": numero de columnas incorrecto");
        std::vector<double> r(nCols);
        for (std::size_t j = 0; j < nCols; ++j) {
            try {
                r[j] = std::stod(f[j]);
            } catch (const std::exception&) {
                throw std::runtime_error("Linea " + std::to_string(lineNo) + ": valor no numerico '" + f[j] + "'");
            }
        }
        rows.push_back(r);
    }
    if (rows.empty()) throw std::runtime_error("El archivo no tiene datos: " + path);

    ds.X = Matrix(rows.size(), nCols - 1);
    ds.y.resize(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t j = 0; j + 1 < nCols; ++j) ds.X(i, j) = rows[i][j];
        ds.y[i] = rows[i][nCols - 1];
    }
    return ds;
}

// Separa en entrenamiento y prueba mezclando las filas con una semilla fija
// (resultados reproducibles). Mezclar es necesario: algunos CSV vienen ordenados.
inline std::pair<Dataset, Dataset> trainTestSplit(const Dataset& d, double trainRatio, unsigned seed = 42) {
    const std::size_t n = d.X.rows();
    if (trainRatio <= 0.0 || trainRatio >= 1.0) throw std::invalid_argument("trainRatio debe estar en (0,1)");
    std::size_t nTrain = static_cast<std::size_t>(static_cast<double>(n) * trainRatio);
    if (nTrain < 1 || nTrain >= n) throw std::invalid_argument("Muy pocos datos para particionar");

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    Dataset train, test;
    train.headers = test.headers = d.headers;
    train.X = Matrix(nTrain, d.X.cols());
    test.X = Matrix(n - nTrain, d.X.cols());
    train.y.resize(nTrain);
    test.y.resize(n - nTrain);

    for (std::size_t k = 0; k < n; ++k) {
        Dataset& dst = (k < nTrain) ? train : test;
        const std::size_t r = (k < nTrain) ? k : k - nTrain;
        for (std::size_t j = 0; j < d.X.cols(); ++j) dst.X(r, j) = d.X(idx[k], j);
        dst.y[r] = d.y[idx[k]];
    }
    return {train, test};
}

#endif  // CSV_READER_H
