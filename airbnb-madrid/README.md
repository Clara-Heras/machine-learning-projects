# Airbnb Madrid

Análisis de 13.000 ofertas de alojamiento de Airbnb en Madrid (2017) con dos objetivos: predecir el tipo de alojamiento (piso entero, habitación privada o habitación compartida) y estimar el precio por noche.

## Notebooks

- `01_classic_classifiers.ipynb`: análisis exploratorio, limpieza de datos y comparación de árbol de decisión, KNN y Naive Bayes para predecir el tipo de alojamiento.
- `02_svm_classification_regression.ipynb`: SVM para clasificación (LinearSVC y SVC) y regresión del precio (LinearSVR y SVR).

## Proceso

Las variables numéricas estaban muy sesgadas, así que apliqué transformaciones logarítmicas y eliminé precios y estancias mínimas poco realistas. La clase de habitación compartida es muy minoritaria (menos del 2 % de los datos), lo que condiciona bastante los resultados.

Dividí los datos en entrenamiento y test (80/20, estratificado), comparé los modelos con validación cruzada y ajusté hiperparámetros con GridSearchCV.

## Resultados

Tipo de alojamiento (accuracy en test):

| Modelo | Accuracy |
|---|---|
| Naive Bayes | 81,8 % |
| Árbol de decisión | 83,1 % |
| LinearSVC | 87,3 % |
| KNN (k = 21) | 87,9 % |
| SVC con kernel RBF | 89,0 % |

Precio por noche (escala logarítmica, test):

| Modelo | R² | RMSE |
|---|---|---|
| LinearSVR | 0,496 | 0,495 |
| SVR con kernel RBF | 0,520 | 0,483 |

Los modelos no lineales dieron los mejores resultados en los dos casos. Todos tienen problemas con las habitaciones compartidas, que suelen confundirse con habitaciones privadas.

## Cómo ejecutarlo

Con `airbnb.csv` en la carpeta `data/`:

    pip install -r requirements.txt
    jupyter notebook

