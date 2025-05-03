"""
generate_72_images_parallel.py

Este script genera un conjunto fijo de 72 imágenes simuladas de Hartmann-Shack utilizando
paralelización mediante MPI (mpi4py). Todos los procesos cooperan para construir un dataset
de imágenes con 21 coeficientes de Zernike aleatorios por imagen.

Cada proceso genera una porción del total de imágenes (división por bloques), aprovechando
la naturaleza inherentemente paralela del problema. Los resultados se guardan en una carpeta
específica, donde cada imagen es acompañada por un archivo de texto con los coeficientes usados.

Este script debe ejecutarse mediante:
    mpiexec -n <num_procesos> python generate_72_images_parallel.py

Requiere:
- numpy, mpi4py
- simulation_hs_image desde simulation_hs_image.py
"""

from mpi4py import MPI
import numpy as np
import os
from simulation_hs_image import simulation_hs_image
from time import time

# Inicialización MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parámetros de simulación
output_dir = "dataset"
n_images_total = 72
NZern = 21

# Semilla diferente para cada proceso
np.random.seed(42 + rank)

# Proceso maestro marca tiempo inicial
if rank == 0:
    t_start = time()

# Crear carpeta de salida solo una vez
if rank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
comm.Barrier()

# División de trabajo entre procesos
indices = list(range(n_images_total))
my_indices = indices[rank::size]

start = time()
for idx in my_indices:
    czern = np.random.uniform(-1, 1, NZern)
    filename_img = os.path.join(output_dir, f"img_{idx}.png")
    filename_coef = os.path.join(output_dir, f"coef_{idx}.txt")
    simulation_hs_image(filename_img, filename_coef, NZern, czern, parallel=False)
end = time()

print(f"Proceso {rank} terminó {len(my_indices)} imágenes en {end - start:.2f} s.")

comm.Barrier()

# Tiempo total reportado por proceso maestro
if rank == 0:
    t_end = time()
    print(f"\nTiempo total de ejecución con {size} procesos: {t_end - t_start:.2f} segundos")
