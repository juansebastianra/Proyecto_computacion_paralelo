"""
generate_dataset_serial.py

Este script genera imágenes simuladas de Hartmann-Shack en modo secuencial (sin paralelización), 
sirviendo como referencia base para comparar el desempeño con versiones paralelas del mismo proceso. 
Para cada cantidad de coeficientes de Zernike (desde 7 hasta 21 en pasos de 2), se generan 24 imágenes
con coeficientes aleatorios.

El tiempo de ejecución promedio y total por configuración de NZern se registra y guarda en archivos .csv
para su posterior análisis.

Requiere:
- numpy, pandas
- simulation_hs_image importada desde simulation_hs_image.py
"""

import numpy as np
import os
from time import time
import pandas as pd
from simulation_hs_image import simulation_hs_image

# Configuración del dataset
output_dir = "dataset_by_zernikes"
n_images_per_nzern = 24
zernike_range = list(range(7, 22, 2))  # NZern: 7, 9, 11, ..., 21

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

local_results_avg = []
local_results_total = []

# Generación secuencial de imágenes para cada NZern
for NZern in zernike_range:
    np.random.seed(42 + NZern * 100)
    times = []

    for idx in range(n_images_per_nzern):
        czern = np.random.uniform(-1, 1, NZern)
        filename_img = os.path.join(output_dir, f"img_nz{NZern}_{idx}.png")
        filename_coef = os.path.join(output_dir, f"coef_nz{NZern}_{idx}.txt")

        t0 = time()
        simulation_hs_image(filename_img, filename_coef, NZern, czern, parallel=False)
        t1 = time()
        times.append(t1 - t0)

    avg_time = np.mean(times) if times else 0.0
    total_time = np.sum(times) if times else 0.0

    print(f"Tiempo promedio para NZern={NZern}: {avg_time:.4f} s")
    print(f"Tiempo total para NZern={NZern}: {total_time:.4f} s")

    local_results_avg.append((NZern, avg_time))
    local_results_total.append((NZern, total_time))

# Guardar resultados en archivos CSV
df_avg = pd.DataFrame(local_results_avg, columns=["NZern", "Avg_Time_s"])
df_total = pd.DataFrame(local_results_total, columns=["NZern", "Total_Time_s"])

df_avg.to_csv("zernike_serial_timing.csv", index=False)
df_total.to_csv("zernike_serial_timing_total.csv", index=False)

print("\nResumen de tiempos promedio:")
print(df_avg)
