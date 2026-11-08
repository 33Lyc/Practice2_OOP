#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Matrix.h"
#include "Statistics.h"

// Regresion lineal multiple:  y_hat = bias + w1*x1 + w2*x2 + ... + wp*xp
// (con p = 1 es la regresion lineal simple).
class LinearRegression {
private:
    // ---- Atributos pedidos por el enunciado ----
    std::vector<double> weights;  // un peso por caracteristica (en la escala ORIGINAL de los datos)
    double bias;                  // termino independiente

    // ---- Parametros del escalado (los llena data_scaling) ----
    std::vector<double> featureMeans;
    std::vector<double> featureStds;

    // ---- Hiperparametros del descenso del gradiente y estado del entrenamiento ----
    double learningRate;
    double tolerance;     // se detiene si el cambio de la perdida es menor a este valor
    double finalLoss;     // ultimo MSE de entrenamiento
    int epochs;           // maximo de iteraciones
    int iterationsRun;    // iteraciones realmente ejecutadas
    bool trained;

    void validate(const Matrix& X, const std::vector<double>& y) const {
        if (X.rows() == 0 || X.cols() == 0) throw std::invalid_argument("LinearRegression: X vacia");
        if (X.rows() != y.size()) throw std::invalid_argument("LinearRegression: X e y tienen distinto numero de filas");
    }

    void requireTrained() const {
        if (!trained) throw std::logic_error("LinearRegression: el modelo no ha sido entrenado (llame fit primero)");
    }

public:
    LinearRegression(double learningRate = 0.05, int epochs = 10000, double tolerance = 1e-12)
        : weights(), bias(0.0), featureMeans(), featureStds(),
          learningRate(learningRate), tolerance(tolerance), finalLoss(0.0),
          epochs(epochs), iterationsRun(0), trained(false) {}

    // ------------------------------------------------------------------
    // data_scaling(): estandariza cada columna de X (z-score): (x - media) / desviacion.
    // Calcula y GUARDA la media y desviacion de cada columna, y devuelve la copia escalada.
    // ------------------------------------------------------------------
    Matrix data_scaling(const Matrix& X) {
        featureMeans = Stats::columnMeans(X);
        featureStds = Stats::columnStds(X);
        for (double& s : featureStds)
            if (s < 1e-12) s = 1.0;  // columna constante: evita dividir entre cero

        Matrix Xs(X.rows(), X.cols());
        for (std::size_t i = 0; i < X.rows(); ++i)
            for (std::size_t j = 0; j < X.cols(); ++j)
                Xs(i, j) = (X(i, j) - featureMeans[j]) / featureStds[j];
        return Xs;
    }

    // ------------------------------------------------------------------
    // fit(): entrena con descenso del gradiente sobre el MSE.
    // Los datos se escalan para que converja bien; al final los pesos se convierten
    // a la escala original, de modo que predict() recibe datos sin escalar.
    // ------------------------------------------------------------------
    void fit(const Matrix& X, const std::vector<double>& y) {
        validate(X, y);
        const std::size_t n = X.rows();
        const std::size_t p = X.cols();
        const double nd = static_cast<double>(n);

        Matrix Xs = data_scaling(X);
        std::vector<double> w(p, 0.0);
        double b = 0.0;

        double prevLoss = INFINITY;
        iterationsRun = 0;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::vector<double> gradW(p, 0.0);
            double gradB = 0.0, loss = 0.0;

            for (std::size_t i = 0; i < n; ++i) {
                double pred = b;
                for (std::size_t j = 0; j < p; ++j) pred += w[j] * Xs(i, j);
                const double err = pred - y[i];
                loss += err * err;
                gradB += err;
                for (std::size_t j = 0; j < p; ++j) gradW[j] += err * Xs(i, j);
            }
            loss /= nd;
            if (!std::isfinite(loss))
                throw std::runtime_error("LinearRegression::fit: el entrenamiento diverge, reduzca learningRate");

            for (std::size_t j = 0; j < p; ++j) w[j] -= learningRate * (2.0 / nd) * gradW[j];
            b -= learningRate * (2.0 / nd) * gradB;

            iterationsRun = epoch;
            finalLoss = loss;
            if (std::fabs(prevLoss - loss) < tolerance) break;
            prevLoss = loss;
        }

        // Volver a la escala original:
        //   y = b + sum w_j * (x_j - mu_j)/sd_j  =>  w_j' = w_j/sd_j ,  b' = b - sum w_j*mu_j/sd_j
        weights.assign(p, 0.0);
        bias = b;
        for (std::size_t j = 0; j < p; ++j) {
            weights[j] = w[j] / featureStds[j];
            bias -= w[j] * featureMeans[j] / featureStds[j];
        }
        trained = true;
    }

    // ------------------------------------------------------------------
    // fitClosedForm(): solucion exacta por ecuaciones normales, theta = (A^T A)^-1 A^T y,
    // con A = [1 | X]. Sirve para verificar el resultado del descenso del gradiente.
    // ------------------------------------------------------------------
    void fitClosedForm(const Matrix& X, const std::vector<double>& y) {
        validate(X, y);
        const std::size_t n = X.rows();
        const std::size_t p = X.cols();

        Matrix A(n, p + 1);
        for (std::size_t i = 0; i < n; ++i) {
            A(i, 0) = 1.0;
            for (std::size_t j = 0; j < p; ++j) A(i, j + 1) = X(i, j);
        }
        const Matrix At = A.transpose();
        const std::vector<double> theta = At.multiply(A).inverse().multiply(At.multiply(y));

        bias = theta[0];
        weights.assign(theta.begin() + 1, theta.end());
        iterationsRun = 0;
        trained = true;
    }

    // predict(): y_hat para cada fila de X (datos en la escala original).
    std::vector<double> predict(const Matrix& X) const {
        requireTrained();
        if (X.cols() != weights.size())
            throw std::invalid_argument("LinearRegression::predict: numero de caracteristicas incorrecto");
        std::vector<double> yHat(X.rows());
        for (std::size_t i = 0; i < X.rows(); ++i) {
            double s = bias;
            for (std::size_t j = 0; j < X.cols(); ++j) s += weights[j] * X(i, j);
            yHat[i] = s;
        }
        return yHat;
    }

    // score(): error de las predicciones, medido como MSE.
    double score(const Matrix& X, const std::vector<double>& y) const {
        return Stats::mse(y, predict(X));
    }

    // Otras metricas de error / ajuste
    double rmse(const Matrix& X, const std::vector<double>& y) const { return Stats::rmse(y, predict(X)); }
    double mae(const Matrix& X, const std::vector<double>& y) const { return Stats::mae(y, predict(X)); }
    double r2Score(const Matrix& X, const std::vector<double>& y) const { return Stats::r2(y, predict(X)); }

    // ---- Getters ----
    const std::vector<double>& getWeights() const { return weights; }
    double getBias() const { return bias; }
    bool isTrained() const { return trained; }
    int getIterations() const { return iterationsRun; }
    double getFinalLoss() const { return finalLoss; }

    // Suma de los tamanos de los atributos declarados (sin relleno/padding).
    // Se usa en el analisis de memoria. Mantener sincronizada con los atributos de arriba.
    static constexpr std::size_t declaredMembersSize() {
        return 3 * sizeof(std::vector<double>)  // weights, featureMeans, featureStds
             + 4 * sizeof(double)               // bias, learningRate, tolerance, finalLoss
             + 2 * sizeof(int)                  // epochs, iterationsRun
             + sizeof(bool);                    // trained
    }
};

#endif  // LINEAR_REGRESSION_H
