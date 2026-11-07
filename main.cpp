// Practica 2 - Lenguajes y Paradigmas de Programacion (EAFIT)
// Framework orientado a objetos para regresion lineal multiple en C++.
//
// Uso:  ./practica2 [ruta_ice_cream.csv] [ruta_student_exam_scores.csv]
// Por defecto busca los CSV en el directorio actual.

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "CsvReader.h"
#include "LinearRegression.h"
#include "MemoryAnalysis.h"
#include "Matrix.h"

// Objeto en memoria GLOBAL (lo usa el analisis de memoria).
LinearRegression globalModel;

// Imprime los pesos con el nombre de su caracteristica y el bias.
void printModel(const LinearRegression& m, const std::vector<std::string>& headers) {
    const std::vector<double>& w = m.getWeights();
    std::cout << "  weights:\n";
    for (std::size_t j = 0; j < w.size(); ++j)
        std::cout << "    w[" << j << "] (" << headers[j] << ") = " << std::setw(12) << w[j] << "\n";
    std::cout << "  bias: " << m.getBias() << "\n";
}

// Ejecuta el flujo del pseudocodigo del enunciado sobre un dataset.
// Con 1 caracteristica es regresion simple; con varias, multiple.
void runExperiment(const std::string& title, const Dataset& data) {
    std::cout << std::string(70, '=') << "\n" << title << "\n" << std::string(70, '=') << "\n";
    std::cout << "Filas: " << data.X.rows() << " | Caracteristicas: " << data.X.cols()
              << " | Objetivo: " << data.headers.back() << "\n";

    // Particion 80% entrenamiento / 20% prueba
    std::pair<Dataset, Dataset> split = trainTestSplit(data, 0.8, 42);
    const Dataset& train = split.first;
    const Dataset& test = split.second;
    std::cout << "Entrenamiento: " << train.X.rows() << " filas | Prueba: " << test.X.rows() << " filas\n\n";

    // X_train, y_train -> fit
    LinearRegression mlr;
    mlr.fit(train.X, train.y);
    std::cout << "[Descenso del gradiente] iteraciones: " << mlr.getIterations()
              << " | MSE de entrenamiento: " << mlr.getFinalLoss() << "\n";
    printModel(mlr, data.headers);

    // X_test -> predict
    std::vector<double> yHat = mlr.predict(test.X);
    std::cout << "\n  Predicciones sobre el conjunto de prueba:\n";
    for (std::size_t i = 0; i < yHat.size(); ++i)
        std::cout << "    y_hat: " << std::setw(10) << yHat[i] << "   (y real: " << std::setw(10) << test.y[i]
                  << ")\n";

    // score
    std::cout << "\n  score (MSE) : " << mlr.score(test.X, test.y) << "\n"
              << "  RMSE        : " << mlr.rmse(test.X, test.y) << "\n"
              << "  MAE         : " << mlr.mae(test.X, test.y) << "\n"
              << "  R^2         : " << mlr.r2Score(test.X, test.y) << "\n";

    // Verificacion con la solucion exacta (ecuaciones normales)
    LinearRegression exact;
    exact.fitClosedForm(train.X, train.y);
    std::cout << "\n[Ecuaciones normales - solucion exacta, para verificar]\n";
    printModel(exact, data.headers);
    std::cout << "  score (MSE) : " << exact.score(test.X, test.y) << "\n\n";
}

int main(int argc, char* argv[]) {
    const std::string icePath = (argc > 1) ? argv[1] : "Ice_cream_selling_data.csv";
    const std::string studentPath = (argc > 2) ? argv[2] : "student_exam_scores.csv";

    std::cout << std::fixed << std::setprecision(6);

    try {
        Dataset ice = loadCsv(icePath);
        Dataset students = loadCsv(studentPath);

        runExperiment("REGRESION LINEAL SIMPLE - Ice cream selling data", ice);
        runExperiment("REGRESION LINEAL MULTIPLE - Student exam scores", students);

        // El analisis de memoria entrena los modelos con el dataset simple.
        runMemoryAnalysis(globalModel, ice.X, ice.y);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
