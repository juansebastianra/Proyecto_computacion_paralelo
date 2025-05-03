# Proyecto Computacional de Óptica Visual

Este proyecto desarrolla un sistema de simulación de imágenes de sensores de frente de onda Hartmann-Shack, con el objetivo de generar datasets simulados a partir de coeficientes de Zernike. Se integra un flujo de trabajo paralelizable utilizando `mpi4py`, lo que permite acelerar la creación masiva de imágenes necesarias para el entrenamiento de modelos de inteligencia artificial, especialmente redes neuronales convolucionales.

## Estructura del proyecto

PROYECTO_COMPUTACIONAL_PARALELO/
│
├── dataset/ # Imágenes simuladas generadas (caso 72 imágenes)
├── dataset_by_zernikes/ # Imágenes generadas por distintos NZern (serie o paralelo)
├── dataset_parallel_test/ # Datos de pruebas paralelas por núcleos
├── plots/ # Gráficas generadas con los resultados
├── results/ # CSVs de tiempos de ejecución por NZern y estrategia
│
├── generate_72_images_parallel.py # Generación paralela de 72 imágenes (con MPI)
├── generate_dataset_parallel.py # Comparación de tiempos paralelos por NZern
├── generate_dataset_serial.py # Comparación de tiempos en serie por NZern
│
├── simulation_hs_image.py # Función principal de simulación de imágenes
├── ZernikePolynomials.py # Construcción de polinomios y derivadas de Zernike
├── prueba.ipynb # Cuaderno Jupyter para análisis adicional
├── README.md # Este archivo


## Dependencias principales:

`numpy`

`Pillow`

`mpi4py`

`pandas`

`matplotlib` (para análisis gráfico)

## Ejecución de scripts

### 1. Generación en serie de imágenes (referencia base)

python generate_dataset_serial.py

Genera 24 imágenes por cada NZern en [7, 9, ..., 21] y mide tiempos. Archivos guardados en `dataset_by_zernikes`.

---

### 2. Generación paralela por NZern

mpiexec -n 4 python generate_dataset_parallel.py

Distribuye la generación por procesos MPI. Mismos coeficientes NZern para comparación directa con el caso en serie.

---

### 3. Generación paralela de 72 imágenes con NZern fijo

mpiexec -n 4 python generate_72_images_parallel.py

Crea 72 imágenes con NZern = 21 de manera distribuida. Resultados guardados en `dataset/`.

---

## Análisis y Resultados

Los tiempos de ejecución (serie y paralelo) se guardan como CSV en la carpeta `results/`. Las gráficas se generan en el notebook `prueba.ipynb`, incluyendo:

- Comparación de tiempos totales por NZern.
- Tiempos promedio por imagen.
- Escalabilidad con número de núcleos.
- Extrapolación a datasets de 24,000 imágenes.