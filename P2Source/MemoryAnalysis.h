#ifndef MEMORY_ANALYSIS_H
#define MEMORY_ANALYSIS_H

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "LinearRegression.h"
#include "Matrix.h"

// Imprime direccion, tamano y datos de un objeto LinearRegression.
inline void printModelMemory(const std::string& label, const std::string& region, const LinearRegression& m) {
    std::cout << "  [" << label << "] zona de memoria: " << region << "\n"
              << "    direccion del objeto (&obj)         : " << static_cast<const void*>(&m) << "\n"
              << "    sizeof(obj)                          : " << sizeof(m) << " bytes\n"
              << "    direccion del buffer de weights      : " << static_cast<const void*>(m.getWeights().data())
              << "  (siempre en el HEAP)\n"
              << "    bytes en heap usados por weights     : " << m.getWeights().capacity() * sizeof(double)
              << " bytes (NO cuentan en sizeof)\n";
}

// globalModel: objeto declarado en ambito global (main.cpp), recibido por referencia
// (la referencia conserva la direccion original del objeto global).
inline void runMemoryAnalysis(LinearRegression& globalModel, const Matrix& X, const std::vector<double>& y) {
    std::cout << std::string(70, '=') << "\n"
              << "ANALISIS DE MEMORIA\n"
              << std::string(70, '=') << "\n";

    // ---------------------------------------------------------------
    // 1) De donde sale sizeof(LinearRegression)
    // ---------------------------------------------------------------
    std::cout << "\n1) Tamano de la clase y de sus componentes (arquitectura de "
              << sizeof(void*) * 8 << " bits)\n"
              << "    sizeof(std::vector<double>) : " << sizeof(std::vector<double>)
              << "  (3 punteros: inicio, fin, fin de capacidad; los datos viven en el heap)\n"
              << "    sizeof(double)              : " << sizeof(double) << "\n"
              << "    sizeof(int)                 : " << sizeof(int) << "\n"
              << "    sizeof(bool)                : " << sizeof(bool) << "\n"
              << "    atributos: 3 vectores (weights, featureMeans, featureStds) + 4 double + 2 int + 1 bool\n"
              << "    suma de atributos           : " << LinearRegression::declaredMembersSize() << " bytes\n"
              << "    sizeof(LinearRegression)    : " << sizeof(LinearRegression) << " bytes\n"
              << "    relleno (padding)           : "
              << sizeof(LinearRegression) - LinearRegression::declaredMembersSize()
              << " bytes (alineacion a multiplos de " << alignof(LinearRegression) << ")\n";

    // ---------------------------------------------------------------
    // 2) Objeto global, local y dinamico
    // ---------------------------------------------------------------
    std::cout << "\n2) Objetos en memoria global, local (stack) y dinamica (heap)\n";

    LinearRegression localModel;                        // memoria local: STACK de esta funcion
    LinearRegression* dynamicModel = new LinearRegression();  // memoria dinamica: HEAP

    globalModel.fit(X, y);   // fit reserva el buffer de weights en el heap
    localModel.fit(X, y);
    dynamicModel->fit(X, y);

    printModelMemory("GLOBAL ", "segmento de datos (static storage)", globalModel);
    printModelMemory("LOCAL  ", "stack (pila)", localModel);
    printModelMemory("DINAMICO", "heap (new)", *dynamicModel);
    std::cout << "    El puntero dynamicModel es una variable LOCAL:\n"
              << "      direccion del puntero (&dynamicModel): " << static_cast<const void*>(&dynamicModel)
              << " (stack)\n"
              << "      valor del puntero (dynamicModel)     : " << static_cast<const void*>(dynamicModel)
              << " (heap)\n"
              << "      sizeof(dynamicModel)                 : " << sizeof(dynamicModel)
              << " bytes (solo el puntero)\n"
              << "      sizeof(*dynamicModel)                : " << sizeof(*dynamicModel) << " bytes (el objeto)\n";

    delete dynamicModel;
    dynamicModel = nullptr;

    // ---------------------------------------------------------------
    // 3) Vector de LinearRegression en memoria dinamica
    // ---------------------------------------------------------------
    const std::size_t N = 5;
    std::cout << "\n3) Vector de " << N << " objetos LinearRegression en memoria dinamica\n";

    // 3a) std::vector reservado con new
    std::vector<LinearRegression>* vec = new std::vector<LinearRegression>(N);
    std::cout << "  [std::vector<LinearRegression>* creado con new]\n"
              << "    direccion del objeto vector (heap)  : " << static_cast<const void*>(vec) << "\n"
              << "    sizeof(*vec)                         : " << sizeof(*vec)
              << " bytes (solo 3 punteros, NO incluye los elementos)\n"
              << "    direccion del buffer de elementos    : " << static_cast<const void*>(vec->data()) << "\n"
              << "    size() / capacity()                  : " << vec->size() << " / " << vec->capacity() << "\n"
              << "    bytes del buffer = capacity * sizeof : " << vec->capacity() << " * "
              << sizeof(LinearRegression) << " = " << vec->capacity() * sizeof(LinearRegression) << " bytes\n"
              << "    total en heap (objeto + buffer)      : "
              << sizeof(*vec) + vec->capacity() * sizeof(LinearRegression) << " bytes\n";
    for (std::size_t i = 0; i < vec->size(); ++i) {
        std::cout << "      elemento [" << i << "] en " << static_cast<const void*>(&(*vec)[i]);
        if (i > 0)
            std::cout << "  (+" << reinterpret_cast<const char*>(&(*vec)[i]) -
                                    reinterpret_cast<const char*>(&(*vec)[i - 1])
                      << " bytes respecto al anterior)";
        std::cout << "\n";
    }
    delete vec;

    // 3b) Arreglo dinamico estilo C: new[]
    LinearRegression* arr = new LinearRegression[N];
    std::cout << "  [LinearRegression* arr = new LinearRegression[" << N << "]]\n"
              << "    sizeof(arr)                          : " << sizeof(arr) << " bytes (solo el puntero)\n"
              << "    direccion del primer elemento        : " << static_cast<const void*>(arr) << "\n"
              << "    bytes de los elementos = N * sizeof  : " << N << " * " << sizeof(LinearRegression)
              << " = " << N * sizeof(LinearRegression) << " bytes\n"
              << "    (sizeof(arr) NO da el tamano del arreglo: es un puntero, no un arreglo)\n";
    delete[] arr;

    std::cout << "\n  Memoria dinamica liberada (delete / delete[]).\n";
}

#endif  // MEMORY_ANALYSIS_H
