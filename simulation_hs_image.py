# simulation_hs_image.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time

from ZernikePolynomials import zernike_polynomials


def simulation_hs_image(filename_img, filename_coef, NZern, czern,
                         pupil_radius=4.15/2, pitch_mla=0.2, f_mla=7.3878,
                         dx_ccd=5.2, dy_ccd=5.2, n_ccd_x=1280, n_ccd_y=1024,
                         lambda_um=0.532, illum_max=200, noise=False):
    """
    Simula una imagen Hartmann-Shack (HS) a partir de una aberración dada por coeficientes de Zernike 
    y parámetros físicos del sistema óptico.

    Esta función calcula el desplazamiento de los spots focales producidos por una matriz de microlentes
    (MLA) debido a un frente de onda aberrado expandido en polinomios de Zernike. Luego construye la 
    imagen en el plano CCD como una superposición de funciones gaussianas que representan los spots, 
    y guarda tanto la imagen resultante como los coeficientes de Zernike utilizados.

    Parameters
    ----------
    filename_img : str
        Ruta de salida para guardar la imagen simulada (formato PNG).
    filename_coef : str
        Ruta de salida para guardar los coeficientes de Zernike utilizados (formato .txt).
    NZern : int
        Número de coeficientes de Zernike a considerar.
    czern : np.ndarray
        Vector de coeficientes de Zernike (longitud NZern).
    pupil_radius : float, optional
        Radio de la pupila en milímetros. Valor por defecto: 2.075 mm.
    pitch_mla : float, optional
        Tamaño del paso entre microlentes (mm). Por defecto: 0.2 mm.
    f_mla : float, optional
        Longitud focal de la microlente (mm). Por defecto: 7.3878 mm.
    dx_ccd : float, optional
        Tamaño de píxel del sensor CCD en micras (eje x). Por defecto: 5.2 µm.
    dy_ccd : float, optional
        Tamaño de píxel del sensor CCD en micras (eje y). Por defecto: 5.2 µm.
    n_ccd_x : int, optional
        Número de píxeles del CCD en la dirección x. Por defecto: 1280.
    n_ccd_y : int, optional
        Número de píxeles del CCD en la dirección y. Por defecto: 1024.
    lambda_um : float, optional
        Longitud de onda incidente en micras. Por defecto: 0.532 µm.
    illum_max : float, optional
        Valor máximo de iluminación para normalización de la imagen. Por defecto: 200.
    noise : bool, optional
        Si True, se añade ruido gaussiano a la imagen. Por defecto: False.

    Returns
    -------
    None
        La función guarda la imagen HS simulada como archivo PNG y los coeficientes como archivo .txt.
    """
     
    # CCD grid (en micras)
    xs = np.arange(n_ccd_x) * dx_ccd + dx_ccd / 2
    ys = np.arange(n_ccd_y) * dy_ccd + dy_ccd / 2
    Xs, Ys = np.meshgrid(xs, ys)

    # configuracion de matriz de microlentes
    SD_M = pupil_radius * 0.60464  # aumento de la pupila correspondiente al sistema
    xm = np.arange(-SD_M, SD_M + pitch_mla, pitch_mla)
    ym = np.arange(-SD_M, SD_M + pitch_mla, pitch_mla)
    XM, YM = np.meshgrid(xm, ym)
    mask = XM**2 + YM**2 <= SD_M**2
    xm = XM[mask].flatten()
    ym = YM[mask].flatten()

    # Derivadas de Zernikes
    ZDx = zernike_polynomials(xm / SD_M, ym / SD_M, NZern, deriv='dx')
    ZDy = zernike_polynomials(xm / SD_M, ym / SD_M, NZern, deriv='dy')

    # Combinar con coeficientes czern
    zdx = np.sum(ZDx * czern[np.newaxis, np.newaxis, :], axis=2)
    zdy = np.sum(ZDy * czern[np.newaxis, np.newaxis, :], axis=2)

    dx = f_mla * zdx / SD_M  # micras
    dy = f_mla * zdy / SD_M

    # focal spots (aumento)
    mag_mla2ccd = 1.0
    dx *= mag_mla2ccd
    dy *= mag_mla2ccd

    xs0, ys0 = xs.max() / 2, ys.max() / 2
    xmsp0 = mag_mla2ccd * 1000 * xm + xs0
    ymsp0 = mag_mla2ccd * 1000 * ym + ys0

    xmsp = xmsp0 + dx
    ymsp = ymsp0 + dy

    # Generacion de imagenes con spots gaussianos
    Is = np.zeros_like(Xs)
    NA = 0.9 * pitch_mla / 2 / np.sqrt((pitch_mla / 2)**2 + f_mla**2)
    rairy = 1.5 * lambda_um / NA
    sigma = rairy / np.sqrt(dx_ccd * dy_ccd)
    rr2 = 4 * (rairy**2)

    xmsp = np.ravel(xmsp)
    ymsp = np.ravel(ymsp)
    
    for x_c, y_c in zip(xmsp, ymsp):
        dist2 = (Xs - x_c)**2 + (Ys - y_c)**2
        mask = dist2 <= rr2
        Is[mask] += np.exp(-((Xs[mask] - x_c)**2 + (Ys[mask] - y_c)**2) / (2 * sigma**2))

    Is = illum_max * Is / Is.max()
    if noise:
        Is += np.random.normal(loc=0, scale=0.1 * illum_max, size=Is.shape)

    #  Reescalado y Guardado
    img = np.uint8(np.clip(Is, 0, 255))
    Image.fromarray(img).save(filename_img)
    np.savetxt(filename_coef, czern[3:], delimiter=';')

    print(f"Imagen y coeficientes guardados: {filename_img}, {filename_coef}")
