# Práctica No. 2 — Programación Orientada a Objetos

Curso: Programming Languages and Paradigms · Universidad EAFIT

## Datos

| Nombre | ID | Correo institucional |
| --- | --- | --- |
| Sebastián Escudero | 1013343369 | jsescudero@eafit.edu.co |

Video: <https://go.screenpal.com/watch/cOXjYxn4iBy>

## De qué trata

Es un programa en C++ que predice un número usando otros datos. Por ejemplo, la nota de un examen a partir de las horas de estudio, las horas de sueño, la asistencia y las notas anteriores. Está hecho con clases en archivos `.h` y todo se programó a mano, sin librerías externas.

El modelo guarda dos cosas: los `weights` (cuánto influye cada dato en el resultado) y el `bias` (el valor base de la predicción).

Para compilar y correr:

```
g++ -std=c++17 main.cpp -o practica2
./practica2 Ice_cream_selling_data.csv student_exam_scores.csv
```

## Métodos: qué hace cada uno

| Archivo | Método | Qué hace |
| --- | --- | --- |
| `LinearRegression.h` | `LinearRegression()` | Crea el modelo, todavía sin entrenar. |
| `LinearRegression.h` | `data_scaling()` | Acomoda los datos para que un número no pese más que otro solo por ser más grande. |
| `LinearRegression.h` | `fit()` | Entrena el modelo: cambia los pesos poquito a poco hasta que el error baje lo más posible. |
| `LinearRegression.h` | `fitClosedForm()` | Entrena de otra forma, con una fórmula directa. Sirve para comprobar que `fit()` esté bien. |
| `LinearRegression.h` | `predict()` | Da la predicción para cada fila de datos. |
| `LinearRegression.h` | `score()` | Dice cuánto se equivoca el modelo (error medio al cuadrado: mientras más bajo, mejor). |
| `LinearRegression.h` | `rmse()` | El error típico, en las mismas unidades del resultado. |
| `LinearRegression.h` | `mae()` | El error promedio, sin importar si fue por arriba o por abajo. |
| `LinearRegression.h` | `r2Score()` | Es como la nota del modelo: 1 es perfecto y 0 es igual a adivinar el promedio. |
| `LinearRegression.h` | `getWeights()` | Entrega los pesos. |
| `LinearRegression.h` | `getBias()` | Entrega el bias. |
| `LinearRegression.h` | `isTrained()` | Dice si el modelo ya fue entrenado. |
| `LinearRegression.h` | `getIterations()` | Dice cuántas vueltas dio al entrenar. |
| `LinearRegression.h` | `getFinalLoss()` | Dice con qué error terminó de entrenar. |
| `LinearRegression.h` | `validate()` | (Interno) Revisa que los datos estén bien antes de entrenar. |
| `LinearRegression.h` | `requireTrained()` | (Interno) Da un error si se intenta predecir sin haber entrenado. |
| `LinearRegression.h` | `declaredMembersSize()` | Suma el tamaño de lo que guarda la clase; se usa en el análisis de memoria. |
| `Matrix.h` | `Matrix()` | Crea una tabla de números con las filas y columnas que se pidan. |
| `Matrix.h` | `identity()` | Crea la tabla identidad (unos en la diagonal y ceros en el resto). |
| `Matrix.h` | `rows()` y `cols()` | Dicen cuántas filas y cuántas columnas hay. |
| `Matrix.h` | `operator()` | Lee o cambia un número según su fila y columna. Avisa si te sales de la tabla. |
| `Matrix.h` | `row()` y `column()` | Copian una fila o una columna completa. |
| `Matrix.h` | `transpose()` | Voltea la tabla: las filas pasan a ser columnas. |
| `Matrix.h` | `multiply()` | Multiplica dos tablas, o una tabla por una lista de números (tiene dos versiones). |
| `Matrix.h` | `inverse()` | Calcula la tabla inversa, algo parecido a «dividir» entre una tabla. Avisa si no existe. |
| `Statistics.h` | `mean()` | Calcula el promedio. |
| `Statistics.h` | `stddev()` | Mide qué tan lejos del promedio están los números. |
| `Statistics.h` | `columnMeans()` | Promedio de cada columna. |
| `Statistics.h` | `columnStds()` | Qué tan dispersa está cada columna. |
| `Statistics.h` | `checkSameSize()` | Revisa que dos listas midan lo mismo. |
| `Statistics.h` | `mse()` | Error medio al cuadrado. |
| `Statistics.h` | `rmse()` | La raíz del error anterior. |
| `Statistics.h` | `mae()` | Error absoluto medio. |
| `Statistics.h` | `r2()` | Calcula el R², la «nota» del modelo. |
| `CsvReader.h` | `trimString()` | Quita los espacios y saltos de línea que sobran en un texto. |
| `CsvReader.h` | `splitLine()` | Separa una línea del archivo en sus valores (corta en cada coma). |
| `CsvReader.h` | `loadCsv()` | Lee un archivo CSV y lo deja listo para usar. La última columna es lo que se quiere predecir. |
| `CsvReader.h` | `trainTestSplit()` | Mezcla los datos y los separa: 80 % para entrenar y 20 % para probar. |
| `MemoryAnalysis.h` | `printModelMemory()` | Muestra dónde está un modelo en la memoria y cuánto ocupa. |
| `MemoryAnalysis.h` | `runMemoryAnalysis()` | Hace todo el análisis de memoria: modelo global, local, dinámico y vector de modelos. |
| `main.cpp` | `printModel()` | Imprime los pesos y el bias de un modelo. |
| `main.cpp` | `runExperiment()` | Hace una prueba completa: separa los datos, entrena, predice y mide el error. |
| `main.cpp` | `main()` | Es donde empieza el programa: lee los archivos, corre las pruebas y el análisis de memoria. |

## Pruebas: entradas y salidas

En las dos pruebas los datos se mezclan y se separan: 80 % para entrenar y 20 % para probar.

### Prueba 1: regresión simple (helados)

**Entrada:** el archivo `Ice_cream_selling_data.csv` tiene 49 filas: 39 para entrenar y 10 para probar. Con la temperatura se predicen las ventas.

**Salida:**

```
weights: w[0] (Temperature (°C)) = -0.782653
bias: 17.244217
score (MSE): 107.755606
RMSE: 10.380540
R^2: -0.240307
```

| Predicción (y\_hat) | Valor real |
| --- | --- |
| 15.03 | 12.57 |
| 16.70 | 4.63 |
| 20.04 | 32.28 |
| 19.68 | 22.64 |
| 17.13 | 1.93 |

El programa imprime las 10 predicciones y aquí se muestran las primeras 5. El R² dio negativo porque los datos de helados forman una «U»: se vende mucho con mucho frío y con mucho calor, y una línea recta no puede seguir esa forma. No es un error del programa, porque la otra forma de entrenar (`fitClosedForm`) da lo mismo.

### Prueba 2: regresión múltiple (notas de estudiantes)

**Entrada:** el archivo `student_exam_scores.csv` tiene 200 filas: 160 para entrenar y 40 para probar. Con las horas de estudio, las horas de sueño, la asistencia y las notas anteriores se predice la nota del examen.

**Salida:**

```
w[0] (hours_studied)      =  1.508401
w[1] (sleep_hours)        =  0.975221
w[2] (attendance_percent) =  0.101889
w[3] (previous_scores)    =  0.187561
bias: -2.278068
score (MSE): 9.173793
RMSE: 3.028827
R^2: 0.780445
```

| Predicción (y\_hat) | Valor real |
| --- | --- |
| 38.85 | 39.50 |
| 30.72 | 33.40 |
| 37.78 | 39.20 |
| 38.20 | 34.40 |
| 28.94 | 30.70 |

El programa imprime las 40 predicciones y aquí se muestran las primeras 5. Este modelo funciona bien: el R² de 0.78 quiere decir que explica cerca del 78 % de los cambios en las notas, y en promedio se equivoca unos 3 puntos.

## Análisis de memoria

Creé el modelo en tres lugares distintos. Las direcciones cambian cada vez que se corre el programa, pero los tamaños no.

| Dónde | Dirección | Tamaño |
| --- | --- | --- |
| Global (zona fija del programa) | 0x5563bf8a3280 | 120 bytes |
| Local (la pila o stack) | 0x7ffd4b992810 | 120 bytes |
| Dinámica (el heap, creada con `new`) | 0x5563c6cb2e90 | 120 bytes |

**¿Por qué 120 bytes?** El modelo guarda 3 listas (`vector`) de 24 bytes cada una (72 en total), 4 números decimales de 8 bytes (32), 2 enteros de 4 bytes (8) y 1 dato de sí o no de 1 byte. Eso suma 113. El compilador agrega 7 bytes de relleno para que el total sea múltiplo de 8, porque así el computador lee la memoria más rápido. El tamaño es el mismo en los tres lugares porque depende del tipo de objeto, no de dónde esté guardado.

**¿Y los pesos?** Los números de la lista de pesos se guardan aparte, en el heap, y no cuentan dentro de los 120 bytes.

**¿En qué se diferencian los tres lugares?**

- Global: existe todo el tiempo que el programa esté corriendo.
- Local: se borra solo cuando termina la función.
- Dinámica: se crea con `new` y hay que borrarla a mano con `delete`. El puntero que la apunta es local (8 bytes, en 0x7ffd4b992800) y el objeto está en el heap.

**Vector de 5 modelos en memoria dinámica:** el vector en sí mide 24 bytes, porque solo guarda dónde están sus datos. Los 5 modelos ocupan 5 × 120 = 600 bytes, uno junto al otro y a 120 bytes de distancia (0x5563c6cb14b0, 0x5563c6cb1528, 0x5563c6cb15a0, 0x5563c6cb1618 y 0x5563c6cb1690). En total son 24 + 600 = 624 bytes en el heap.

## Mis fallos con la memoria dinámica

Mientras aprendía `new` y `delete` cometí los errores típicos de quien empieza. Probé cada uno con la clase del proyecto y lo ejecuté de dos formas: normal, y con AddressSanitizer, una herramienta que detecta errores de memoria.

### Fallo 1: pedí memoria y no la devolví (fuga de memoria)

**Qué hice:**

```
for (int i = 0; i < 3; i++) {
    LinearRegression* m = new LinearRegression();
}
```

**Qué pasó:** al correrlo normal no vi nada: ni mensaje ni error, y pensé que estaba bien. Con AddressSanitizer sí avisó: `360 byte(s) leaked in 3 allocation(s)`, o sea 3 modelos de 120 bytes que nunca devolví. En un programa que corre por días, la memoria se iría llenando poco a poco.

**Cómo lo arreglé:** poner un `delete m;` por cada `new`. Todavía mejor: no usar `new` y crear el modelo normal (`LinearRegression m;`), que se borra solo.

### Fallo 2: seguí usando algo que ya había borrado

**Qué hice:**

```
LinearRegression* m = new LinearRegression();
delete m;
std::cout << m->isTrained();
```

**Qué pasó:** al correrlo normal imprimió `0` y terminó sin problema, así que creí que funcionaba. Fue pura suerte. AddressSanitizer dijo `heap-use-after-free`: yo estaba leyendo dentro de un modelo de 120 bytes que ya no era mío.

**Cómo lo arreglé:** después de `delete m;` no volver a usar `m` (y ponerlo en `m = nullptr;` para no olvidarlo). Otra opción es usar `unique_ptr`, que borra solo.

### Fallo 3: borré un arreglo con `delete` en vez de `delete[]`

**Qué hice:**

```
LinearRegression* modelos = new LinearRegression[5];
delete modelos;
```

**Qué pasó:** el programa se cayó con `Segmentation fault` (código de salida 139). AddressSanitizer explicó por qué: intenté liberar una dirección que no era el inicio real de la reserva, que tenía 608 bytes (los 600 de los 5 modelos más 8 que el sistema guarda para recordar cuántos objetos hay).

**Cómo lo arreglé:** lo que se crea con `new[]` se borra con `delete[]`: `delete[] modelos;`.

### Fallo 4: escribí fuera de los límites de un vector

**Qué hice:**

```
std::vector<double> pesos(4);
pesos[4] = 1.0;
```

**Qué pasó:** al correrlo normal no pasó nada, pero estaba mal: un vector de 4 posiciones usa las posiciones 0, 1, 2 y 3, y yo escribí en la 4. AddressSanitizer dijo `heap-buffer-overflow`: escribí 8 bytes justo después de los 32 bytes del vector, en memoria que no era mía.

**Cómo lo arreglé:** usar `pesos.at(4)` en lugar de `pesos[4]`. Con `at()` el programa avisa con un error claro (`vector::_M_range_check: __n (which is 4) >= this->size() (which is 4)`) en vez de dejar pasar el problema. La clase `Matrix` del proyecto también revisa los límites.

## Problemas y soluciones del desarrollo

| Problema | Cómo lo resolví |
| --- | --- |
| El archivo de helados viene ordenado por temperatura, y si separaba los datos en orden probaba con casos muy distintos a los del entrenamiento. | Mezclé los datos antes de separarlos, con una semilla fija para que el resultado se repita siempre igual. |
| Los datos tienen escalas muy distintas (asistencia de 40 a 100 y horas de sueño de 4 a 9) y el modelo aprendía mal. | Los acomodé con `data_scaling()` y después devolví los pesos a la escala original, para poder predecir con datos normales. |
| No sabía si mi entrenamiento estaba bien, solo porque el programa corría. | Programé una segunda forma de entrenar (`fitClosedForm()`) y comprobé que ambas dan casi lo mismo. |
| En los helados el R² salió negativo y pensé que el código estaba mal. | Miré los datos: forman una «U» y una línea recta no sirve. El código estaba bien. |
| No entendía por qué el modelo ocupa 120 bytes si guarda tantas cosas. | Sumé lo que guarda la clase (113 bytes) y vi que faltaban 7 de relleno. También entendí que los pesos viven aparte, en el heap. |

## Conclusiones

1. **S****eparar los datos en el orden en que vienen.** Si el archivo está ordenado, se entrena con unos casos y se prueba con otros muy distintos.&#32;
2. **U****sar los números tal cual.** Si unos van de 4 a 9 y otros de 40 a 100, el modelo aprende mal. Aprendí que hay que ponerlos en una escala parecida.
3. **C****reer que el programa está bien solo porque corre.** Un resultado raro puede venir de los datos y no del código, como el R² negativo de los helados. Aprendí a comprobar con una segunda forma de calcularlo y a mirar los datos.
4. **N****o devolver la memoria que se pide.** Cada `new` necesita su `delete`, o mejor aun, usar `vector` o crear los objetos normales para que se borren solos. Mi fuga de 3 modelos (360 bytes) no mostró ningún error al correr.
5. **U****sar un objeto después de borrarlo.** Pasó sin ningún aviso y el programa parecía bien. Después de un `delete` hay que dejar de usar el puntero.
6. **P****asarse de las posiciones de una lista.** Un vector de 4 posiciones llega hasta la 3. Usar `.at()` hace que el programa avise en vez de escribir donde no debe.
7. **C****reer que `sizeof` mide todo.** En mi prueba, `sizeof` de un puntero a 5 modelos dio 8 bytes y no los 600 que esperaba. Un modelo mide 120 bytes y un `vector` 24, sin contar los datos que guardan en el heap. Y sobre todo aprendí que un programa que no se cae puede tener errores de memoria.
