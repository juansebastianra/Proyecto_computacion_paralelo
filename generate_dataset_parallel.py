"""
generate_dataset_parallel.py

Este script genera imágenes simuladas de Hartmann-Shack utilizando paralelización distribuida
mediante el estándar MPI (Message Passing Interface), implementado a través de la librería mpi4py.

Para cada cantidad de coeficientes de Zernike (NZern = 7, 9, ..., 21), se generan 24 imágenes con
coeficientes aleatorios, distribuyendo el trabajo entre los distintos procesos disponibles.
Cada proceso se encarga de una porción del total de imágenes, permitiendo acelerar el proceso
de generación masiva de datos para entrenamiento de modelos de aprendizaje profundo.

Este script debe ejecutarse con `mpiexec` o `mpirun`, por ejemplo:
    mpiexec -n 4 python generate_dataset_parallel.py

Requiere:
- numpy, pandas, mpi4py
- simulation_hs_image importada desde simulation_hs_image.py
"""

from mpi4py import MPI
import numpy as np
import os
from time import time
import pandas as pd
from simulation_hs_image import simulation_hs_image

# Inicialización MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuración del dataset
output_dir = "dataset_by_zernikes"
n_images_per_nzern = 24
zernike_range = list(range(7, 22, 2))  # NZern: 7, 9, ..., 21

# Crear carpeta de salida una sola vez
if rank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
comm.Barrier()

local_results = []

# Generación distribuida por cada NZern
for NZern in zernike_range:
    indices = list(range(n_images_per_nzern))
    my_indices = indices[rank::size]
    np.random.seed(42 + rank + NZern * 100)
    times = []

    for idx in my_indices:
        czern = np.random.uniform(-1, 1, NZern)
        filename_img = os.path.join(output_dir, f"img_nz{NZern}_{idx}.png")
        filename_coef = os.path.join(output_dir, f"coef_nz{NZern}_{idx}.txt")

        t0 = time()
        simulation_hs_image(filename_img, filename_coef, NZern, czern, parallel=False)
        t1 = time()
        times.append(t1 - t0)
    
    avg_time = np.mean(times) if times else 0.0
    print(f"[Rank {rank}] Tiempo promedio para NZern={NZern}: {avg_time:.4f} s")
    local_results.append((NZern, avg_time))

# Recolección de resultados por el proceso maestro
all_results = comm.gather(local_results, root=0)

if rank == 0:
    combined = {}
    for proc_results in all_results:
        for nz, t in proc_results:
            combined.setdefault(nz, []).append(t)

    averaged_results = [(nz, np.mean(times)) for nz, times in sorted(combined.items())]
    df = pd.DataFrame(averaged_results, columns=["NZern", "Avg_Time_s"])
    df.to_csv("zernike_parallel_timing.csv", index=False)

    print("\nResumen tiempos promedio paralelizados:")
    print(df)
