import numpy as np

def zernike_polynomials(x, y, N, deriv=None):
    """
    Calcula los primeros N polinomios de Zernike (hasta 66) o sus derivadas parciales en coordenadas cartesianas.

    Args:
        x (np.ndarray): Coordenadas x normalizadas dentro del círculo unitario (puede ser 1D o 2D).
        y (np.ndarray): Coordenadas y normalizadas dentro del círculo unitario (puede ser 1D o 2D).
        N (int): Número total de modos de Zernike a calcular (máximo 66).
        deriv (str | None): 'dx', 'dy' o None para evaluar derivada parcial o altura.

    Returns:
        np.ndarray: Tensor (..., N) con los valores de los polinomios o sus derivadas.
    """
    if N > 41:
        raise ValueError("Este script solo soporta hasta el modo 41 de Zernike")

    if isinstance(deriv, np.ndarray):
        raise ValueError("El parámetro 'deriv' debe ser un string: None, 'dx' o 'dy'")

    if deriv not in (None, 'dx', 'dy'):
        raise ValueError("'deriv' debe ser None, 'dx' o 'dy'")

    # Precalcular potencias de x e y
    powers = {f"x{i}": x**i for i in range(2, 11)}
    powers.update({f"y{i}": y**i for i in range(2, 11)})

    # Inicializar tensor de salida adaptado a 1D o 2D
    if x.ndim == 2:
        Z = np.zeros((x.shape[0], x.shape[1], N))
    elif x.ndim == 1:
        Z = np.zeros((x.shape[0], N))
    else:
        raise ValueError("x debe ser un arreglo 1D o 2D")

    # Función de acceso rápido a sqrt(n)
    sq = lambda n: np.sqrt(n)

    # Poblar según derivada
    if deriv is None:
        populate_zernike_modes_h(Z, x, y, powers, sq, N)
    elif deriv == 'dx':
        populate_zernike_modes_dx(Z, x, y, powers, sq, N)
    elif deriv == 'dy':
        populate_zernike_modes_dy(Z, x, y, powers, sq, N)

    return Z



def populate_zernike_modes_h(Z, x, y, powers, sq, N=None):

    """
    Llena un arreglo `Z` con los primeros modos de Zernike horizontales (ordenados según Noll)
    evaluados en las coordenadas cartesianas (x, y).


    Parameters
    ----------
    Z : np.ndarray
        Arreglo destino donde se almacenarán los modos de Zernike. 
    x : np.ndarray
        Coordenadas `x` normalizadas sobre el disco unitario.
    y : np.ndarray
        Coordenadas `y` normalizadas sobre el disco unitario.
    powers : dict
        Diccionario con las potencias precomputadas de `x` y `y` desde 2 hasta 10. 
        Las claves deben ser del tipo `'x2'`, `'x3'`, ..., `'y10'`.
    sq : function
        Función que toma un entero `n` y retorna el factor de normalización correspondiente `sqrt(n)`.
    N : int, optional
        Número máximo de modos a poblar en `Z`. Si no se especifica, se poblarán todos los definidos en la función.

    Returns
    -------
    None
        La función modifica `Z` en el lugar (in-place).
    """

    x2, x3, x4, x5, x6, x7, x8, x9, x10 = (powers[f"x{i}"] for i in range(2, 11))
    y2, y3, y4, y5, y6, y7, y8, y9, y10 = (powers[f"y{i}"] for i in range(2, 11))

    slice_func = (
        (lambda arr, i: arr[:, :, i]) if Z.ndim == 3
        else (lambda arr, i: arr[:, i])
    )

    def set_mode(index, expression):
        if N is None or index < N:
            slice_func(Z, index)[...] = expression

    set_mode(0, 1.0)
    set_mode(1, sq(4) * y)
    set_mode(2, sq(4) * x)
    set_mode(3, sq(6) * 2 * x * y)
    set_mode(4, sq(3) * (2 * x2 + 2 * y2 - 1))
    set_mode(5, sq(6) * (x2 - y2))
    set_mode(6, sq(8) * (3 * x2 * y - y3))
    set_mode(7, sq(8) * (3 * x2 * y + 3 * y3 - 2 * y))
    set_mode(8, sq(8) * (3 * x3 + 3 * x * y2 - 2 * x))
    set_mode(9, sq(8) * (x3 - 3 * x * y2))
    set_mode(10, sq(10) * (4 * x3 * y - 4 * x * y3))
    set_mode(11, sq(10) * (8 * x3 * y + 8 * x * y3 - 6 * x * y))
    set_mode(12, sq(5) * (6 * x4 + 12 * x2 * y2 + 6 * y4 - 6 * x2 - 6 * y2 + 1))
    set_mode(13, sq(10) * (4 * x4 + 4 * x2 * y2 - 3 * x2 - 4 * x2 * y2 - 4 * y4 + 3 * y2))
    set_mode(14, sq(10) * (x4 - 6 * x2 * y2 + y4))
    set_mode(15, sq(12) * (5 * x4 * y - 10 * x2 * y3 + y5))
    set_mode(16, sq(12) * (15 * x4 * y + 15 * x2 * y3 - 12 * x2 * y - 5 * x2 * y3 - 5 * y5 + 4 * y3))
    set_mode(17, sq(12) * (10 * x4 * y + 20 * x2 * y3 + 10 * y5 - 12 * x2 * y - 12 * y3 + 3 * y))
    set_mode(18, sq(12) * (10 * x5 + 20 * x3 * y2 + 10 * x * y4 - 12 * x3 - 12 * x * y2 + 3 * x))
    set_mode(19, sq(12) * (5 * x5 + 5 * x3 * y2 - 4 * x3 - 15 * x3 * y2 - 15 * x * y4 + 12 * x * y2))
    set_mode(20, sq(12) * (x5 - 10 * x3 * y2 + 5 * x * y4))
    set_mode(21, sq(14) * (6 * x5 * y - 20 * x3 * y3 + 6 * x * y5))
    set_mode(22, sq(14) * (24 * x5 * y + 24 * x3 * y3 - 20 * x3 * y - 24 * x3 * y3 - 24 * x * y5 + 20 * x * y3))
    set_mode(23, sq(14) * (30 * x5 * y + 60 * x3 * y3 + 30 * x * y5 - 40 * x3 * y - 40 * x * y3 + 12 * x * y))
    set_mode(24, sq(7) * (20 * x6 + 60 * x4 * y2 + 60 * x2 * y4 + 20 * y6 - 30 * x4 - 60 * x2 * y2 - 30 * y4 + 12 * x2 + 12 * y2 - 1))
    set_mode(25, sq(14) * (15 * x6 + 30 * x4 * y2 + 15 * x2 * y4 - 20 * x4 - 20 * x2 * y2 + 6 * x2 - 15 * x4 * y2 - 30 * x2 * y4 - 15 * y6 + 20 * x2 * y2 + 20 * y4 - 6 * y2))
    set_mode(26, sq(14) * (6 * x6 + 6 * x4 * y2 - 5 * x4 - 36 * x4 * y2 - 36 * x2 * y4 + 30 * x2 * y2 + 6 * x2 * y4 + 6 * y6 - 5 * y4))
    set_mode(27, sq(14) * (x6 - 15 * x4 * y2 + 15 * x2 * y4 - y6))
    set_mode(28, sq(16) * (7 * x6 * y - 35 * x4 * y3 + 21 * x2 * y5 - y7))
    set_mode(29, sq(16) * (35 * x6 * y + 35 * x4 * y3 - 30 * x4 * y - 70 * x4 * y3 - 70 * x2 * y5 + 60 * x2 * y3 + 7 * x2 * y5 + 7 * y7 - 6 * y5))
    set_mode(30, sq(16) * (63 * x6 * y + 126 * x4 * y3 + 63 * x2 * y5 - 90 * x4 * y - 90 * x2 * y3 + 30 * x2 * y - 21 * x4 * y3 - 42 * x2 * y5 - 21 * y7 + 30 * x2 * y3 + 30 * y5 - 10 * y3))
    set_mode(31, sq(16) * (35 * x6 * y + 105 * x4 * y3 + 105 * x2 * y5 + 35 * y7 - 60 * x4 * y - 120 * x2 * y3 - 60 * y5 + 30 * x2 * y + 30 * y3 - 4 * y))
    set_mode(32, sq(9) * (70 * x7 + 210 * x5 * y2 + 210 * x3 * y4 + 70 * x * y6 - 140 * x5 - 210 * x3 * y2 - 140 * x * y4 + 90 * x3 + 90 * x * y2 - 20 * x))
    set_mode(33, sq(18) * (56 * x7 + 126 * x5 * y2 + 84 * x3 * y4 + 14 * x * y6 - 105 * x5 - 105 * x3 * y2 - 15 * x * y4 + 60 * x3 + 30 * x * y2 - 10 * x))
    set_mode(34, sq(18) * (28 * x7 + 28 * x5 * y2 - 21 * x5 - 84 * x5 * y2 - 126 * x3 * y4 + 105 * x3 * y2 + 21 * x3 * y4 + 21 * x * y6 - 18 * x * y4))
    set_mode(35, sq(18) * (8 * x7 - 56 * x5 * y2 + 70 * x3 * y4 - 28 * x * y6))
    set_mode(36, sq(20) * (9 * x7 * y - 84 * x5 * y3 + 126 * x3 * y5 - 36 * x * y7))
    set_mode(37, sq(20) * (63 * x7 * y + 63 * x5 * y3 - 56 * x5 * y - 315 * x5 * y3 - 315 * x3 * y5 + 280 * x3 * y3 + 189 * x3 * y5 + 189 * x * y7 - 168 * x * y5 - 9 * x * y7 - 9 * y7 + 8 * y5))
    set_mode(38, sq(20) * (180 * x7 * y + 360 * x5 * y3 + 180 * x3 * y5 - 280 * x5 * y - 280 * x3 * y3 + 105 * x3 * y - 360 * x5 * y3 - 720 * x3 * y5 - 360 * x * y7 + 560 * x3 * y3 + 560 * x * y5 - 210 * x * y3 + 36 * x3 * y5 + 72 * x * y7 + 36 * y7 - 56 * x * y5 - 56 * y5 + 21 * y3))
    set_mode(39, sq(20) * (252 * x7 * y + 756 * x5 * y3 + 756 * x3 * y5 + 252 * x * y7 - 504 * x5 * y - 1008 * x3 * y3 - 504 * x * y5 + 315 * x3 * y + 315 * x * y3 - 60 * x * y - 84 * x5 * y3 - 252 * x3 * y5 - 252 * x * y7 - 84 * y7 + 168 * x3 * y3 + 336 * x * y5 + 168 * y5 - 105 * x * y3 - 105 * y3 + 20 * y))
    set_mode(40, sq(20) * (126 * x8 + 504 * x6 * y2 + 756 * x4 * y4 + 504 * x2 * y6 + 126 * y8 - 280 * x6 - 840 * x4 * y2 - 840 * x2 * y4 - 280 * y6 + 210 * x4 + 420 * x2 * y2 + 210 * y4 - 60 * x2 - 60 * y2 + 5))
    return Z

def populate_zernike_modes_dx(Z, x, y, powers, sq, N=None):

    """
    Llena un arreglo `Z` con las derivadas parciales con respecto a `x` de los modos de Zernike,
    evaluadas en coordenadas cartesianas normalizadas `(x, y)`.

    Parameters
    ----------
    Z : np.ndarray
        Arreglo multidimensional donde se almacenarán las derivadas de los modos de Zernike. 
    x : np.ndarray
        Coordenadas `x` normalizadas sobre el disco unitario.
    y : np.ndarray
        Coordenadas `y` normalizadas sobre el disco unitario.
    powers : dict
        Diccionario que contiene potencias precalculadas de `x` y `y` desde 2 hasta 10.
        Las claves deben ser del tipo `'x2'`, `'x3'`, ..., `'y10'`.
    sq : function
        Función que retorna el factor de normalización `sqrt(n)` para cada modo.
    N : int, optional
        Número máximo de derivadas a poblar. Si se omite, se calcularán todas las definidas.

    Returns
    -------
    None
        Esta función modifica el arreglo `Z` directamente (in-place).
    """
    
    slice_func = (
        (lambda arr, i: arr[:, :, i]) if Z.ndim == 3
        else (lambda arr, i: arr[:, i])
    )

    def set_mode(index, expression):
        if N is None or index < N:
            slice_func(Z, index)[...] = expression

    x2, x3, x4, x5, x6, x7, x8, x9, x10 = [powers[f"x{i}"] for i in range(2, 11)]
    y2, y3, y4, y5, y6, y7, y8, y9, y10 = [powers[f"y{i}"] for i in range(2, 11)]

    set_mode(0, 0)
    set_mode(1, 0)
    set_mode(2, sq(4))
    set_mode(3, sq(6) * 2 * y)
    set_mode(4, sq(3) * 4 * x)
    set_mode(5, sq(6) * 2 * x)
    set_mode(6, sq(8) * 6 * x * y)
    set_mode(7, sq(8) * 6 * x * y)
    set_mode(8, sq(8) * (9 * x2 + 3 * y2 - 2))
    set_mode(9, sq(8) * (3 * x2 - 3 * y2))
    set_mode(10, sq(10) * (12 * x2 * y - 4 * y3))
    set_mode(11, sq(10) * (24 * x2 * y + 8 * y3 - 6 * y))
    set_mode(12, sq(5) * (24 * x3 + 24 * x * y2 - 12 * x))
    set_mode(13, sq(10) * (16 * x3 + 8 * x * y2 - 6 * x - 8 * x * y2))
    set_mode(14, sq(10) * (4 * x3 - 12 * x * y2))
    set_mode(15, sq(12) * (20 * x3 * y - 20 * x * y3))
    set_mode(16, sq(12) * (60 * x3 * y + 30 * x * y3 - 24 * x * y - 10 * x * y3))
    set_mode(17, sq(12) * (40 * x3 * y + 40 * x * y3 - 24 * x * y))
    set_mode(18, sq(12) * (50 * x4 + 60 * x2 * y2 + 10 * y4 - 36 * x2 - 12 * y2 + 3))
    set_mode(19, sq(12) * (25 * x4 + 15 * x2 * y2 - 12 * x2 - 45 * x2 * y2 - 15 * y4 + 12 * y2))
    set_mode(20, sq(12) * (5 * x4 - 30 * x2 * y2 + 5 * y4))
    set_mode(21, sq(14) * (30 * x4 * y - 60 * x2 * y3 + 6 * y5))
    set_mode(22, sq(14) * (120 * x4 * y + 72 * x2 * y3 - 60 * x2 * y - 72 * x2 * y3 - 24 * y5 + 20 * y3))
    set_mode(23, sq(14) * (150 * x4 * y + 180 * x2 * y3 + 30 * y5 - 120 * x2 * y - 40 * y3 + 12 * y))
    set_mode(24, sq(7) * (120 * x5 + 240 * x3 * y2 + 120 * x * y4 - 120 * x3 - 120 * x * y2 + 24 * x))
    set_mode(25, sq(14) * (90 * x5 + 120 * x3 * y2 + 30 * x * y4 - 80 * x3 - 40 * x * y2 + 12 * x - 60 * x3 * y2 - 60 * x * y4 + 40 * x * y2))
    set_mode(26, sq(14) * (36 * x5 + 24 * x3 * y2 - 20 * x3 - 144 * x3 * y2 - 72 * x * y4 + 60 * x * y2 + 12 * x * y4))
    set_mode(27, sq(14) * (6 * x5 - 60 * x3 * y2 + 30 * x * y4))
    set_mode(28, sq(16) * (42 * x5 * y - 140 * x3 * y3 + 42 * x * y5))
    set_mode(29, sq(16) * (210 * x5 * y + 140 * x3 * y3 - 120 * x3 * y - 280 * x3 * y3 - 140 * x * y5 + 120 * x * y3 + 14 * x * y5))
    set_mode(30, sq(16) * (378 * x5 * y + 504 * x3 * y3 + 126 * x * y5 - 360 * x3 * y - 180 * x * y3 + 60 * x * y - 84 * x3 * y3 - 84 * x * y5 + 60 * x * y3))
    set_mode(31, sq(16) * (210 * x5 * y + 420 * x3 * y3 + 210 * x * y5 - 240 * x3 * y - 240 * x * y3 + 60 * x * y))
    set_mode(32, sq(16) * (245 * x6 + 525 * x4 * y2 + 315 * x2 * y4 + 35 * y6 - 300 * x4 - 360 * x2 * y2 - 60 * y4 + 90 * x2 + 30 * y2 - 4))
    set_mode(33, sq(16) * (147 * x6 + 210 * x4 * y2 + 63 * x2 * y4 - 150 * x4 - 90 * x2 * y2 + 30 * x2 - 315 * x4 * y2 - 378 * x2 * y4 - 63 * y6 + 270 * x2 * y2 + 90 * y4 - 30 * y2))
    set_mode(34, sq(16) * (49 * x6 + 35 * x4 * y2 - 30 * x4 - 350 * x4 * y2 - 210 * x2 * y4 + 180 * x2 * y2 + 105 * x2 * y4 + 35 * y6 - 30 * y4))
    set_mode(35, sq(16) * (7 * x6 - 105 * x4 * y2 + 105 * x2 * y4 - 7 * y6))
    set_mode(36, sq(18) * (56 * x6 * y - 280 * x4 * y3 + 168 * x2 * y5 - 8 * y7))
    set_mode(37, sq(18) * (336 * x6 * y + 240 * x4 * y3 - 210 * x4 * y - 800 * x4 * y3 - 480 * x2 * y5 + 420 * x2 * y3 + 144 * x2 * y5 + 48 * y7 - 42 * y5))
    set_mode(38, sq(18) * (784 * x6 * y + 1120 * x4 * y3 + 336 * x2 * y5 - 840 * x4 * y - 504 * x2 * y3 + 180 * x2 * y - 560 * x4 * y3 - 672 * x2 * y5 - 112 * y7 + 504 * x2 * y3 + 168 * y5 - 60 * y3))
    set_mode(39, sq(18) * (784 * x6 * y + 1680 * x4 * y3 + 1008 * x2 * y5 + 112 * y7 - 1050 * x4 * y - 1260 * x2 * y3 - 210 * y5 + 360 * x2 * y + 120 * y3 - 20 * y))
    set_mode(40, sq(9) * (560 * x7 + 1680 * x5 * y2 + 1680 * x3 * y4 + 560 * x * y6 - 840 * x5 - 1680 * x3 * y2 - 840 * x * y4 + 360 * x3 + 360 * x * y2 - 40 * x))
    return Z

def populate_zernike_modes_dy(Z, x, y, powers, sq, N=None):

    """
    Llena un arreglo `Z` con las derivadas parciales con respecto a `y` de los modos de Zernike,
    evaluadas en coordenadas cartesianas normalizadas `(x, y)`.

    Parameters
    ----------
    Z : np.ndarray
        Arreglo multidimensional donde se almacenarán las derivadas de los modos de Zernike. 
    x : np.ndarray
        Coordenadas `x` normalizadas sobre el disco unitario.
    y : np.ndarray
        Coordenadas `y` normalizadas sobre el disco unitario.
    powers : dict
        Diccionario con potencias precalculadas de `x` y `y`, desde `x2` hasta `x10`, e `y2` hasta `y10`.
    sq : function
        Función que devuelve el factor de normalización `sqrt(n)` correspondiente a cada modo de Zernike.
    N : int, optional
        Número máximo de modos a calcular. Si se omite, se calcularán todos los definidos explícitamente.

    Returns
    -------
    None
        Esta función modifica el arreglo `Z` directamente (in-place), sin retorno explícito.
    """

    slice_func = (
        (lambda arr, i: arr[:, :, i]) if Z.ndim == 3
        else (lambda arr, i: arr[:, i])
    )

    def set_mode(index, expression):
        if N is None or index < N:
            slice_func(Z, index)[...] = expression

    x2, x3, x4, x5, x6, x7, x8, x9, x10 = [powers[f"x{i}"] for i in range(2, 11)]
    y2, y3, y4, y5, y6, y7, y8, y9, y10 = [powers[f"y{i}"] for i in range(2, 11)]

    set_mode(0, 0)
    set_mode(1, sq(4))
    set_mode(2, 0)
    set_mode(3, sq(6) * 2 * x)
    set_mode(4, sq(3) * 4 * y)
    set_mode(5, sq(6) * -2 * y)
    set_mode(6, sq(8) * (3 * x2 - 3 * y2))
    set_mode(7, sq(8) * (3 * x2 + 9 * y2 - 2))
    set_mode(8, sq(8) * 6 * x * y)
    set_mode(9, sq(8) * -6 * x * y)
    set_mode(10, sq(10) * (4 * x3 - 12 * x * y2))
    set_mode(11, sq(10) * (8 * x3 + 24 * x * y2 - 6 * x))
    set_mode(12, sq(5) * (24 * x2 * y + 24 * y3 - 12 * y))
    set_mode(13, sq(10) * (-16 * y3 + 6 * y))
    set_mode(14, sq(10) * (-12 * x2 * y + 4 * y3))
    set_mode(15, sq(12) * (5 * x4 - 30 * x2 * y2 + 5 * y4))
    set_mode(16, sq(12) * (15 * x4 + 45 * x2 * y2 - 12 * x2 - 15 * x2 * y2 - 25 * y4 + 12 * y2))
    set_mode(17, sq(12) * (10 * x4 + 60 * x2 * y2 + 50 * y4 - 12 * x2 - 36 * y2 + 3))
    set_mode(18, sq(12) * (40 * x3 * y + 40 * x * y3 - 24 * x * y))
    set_mode(19, sq(12) * (-20 * x3 * y + 20 * x * y3))
    set_mode(20, sq(12) * (-20 * x3 * y + 20 * x * y3))
    set_mode(21, sq(14) * (6 * x5 - 60 * x3 * y2 + 30 * x * y4))
    set_mode(22, sq(14) * (24 * x5 + 72 * x3 * y2 - 20 * x3 - 72 * x3 * y2 - 120 * x * y4 + 60 * x * y2))
    set_mode(23, sq(14) * (30 * x5 + 180 * x3 * y2 + 150 * x * y4 - 40 * x3 - 120 * x * y2 + 12 * x))
    set_mode(24, sq(7) * (120 * x4 * y + 240 * x2 * y3 + 120 * y5 - 120 * x2 * y - 120 * y3 + 24 * y))
    set_mode(25, sq(14) * (60 * x4 * y + 60 * x2 * y3 - 30 * x4 * y - 120 * x2 * y3 - 90 * y5 + 80 * y3 - 12 * y))
    set_mode(26, sq(14) * (12 * x4 * y - 72 * x4 * y - 144 * x2 * y3 + 60 * x2 * y + 24 * x2 * y3 + 36 * y5 - 20 * y3))
    set_mode(27, sq(14) * (-30 * x4 * y + 60 * x2 * y3 - 6 * y5))
    set_mode(28, sq(16) * (7 * x6 - 105 * x4 * y2 + 105 * x2 * y4 - 7 * y6))
    set_mode(29, sq(16) * (35 * x6 + 105 * x4 * y2 - 30 * x4 - 210 * x4 * y2 - 350 * x2 * y4 + 180 * x2 * y2 + 35 * x2 * y4 + 49 * y6 - 30 * y4))
    set_mode(30, sq(16) * (63 * x6 + 378 * x4 * y2 + 315 * x2 * y4 - 90 * x4 - 270 * x2 * y2 + 30 * x2 - 63 * x4 * y2 - 210 * x2 * y4 - 147 * y6 + 90 * x2 * y2 + 150 * y4 - 30 * y2))
    set_mode(31, sq(16) * (35 * x6 + 315 * x4 * y2 + 525 * x2 * y4 + 245 * y6 - 60 * x4 - 360 * x2 * y2 - 300 * y4 + 30 * x2 + 90 * y2 - 4))
    set_mode(32, sq(16) * (210 * x5 * y + 420 * x3 * y3 + 210 * x * y5 - 240 * x3 * y - 240 * x * y3 + 60 * x * y))
    set_mode(33, sq(16) * (84 * x5 * y + 84 * x3 * y3 - 60 * x3 * y - 126 * x5 * y - 504 * x3 * y3 - 378 * x * y5 + 180 * x3 * y + 360 * x * y3 - 60 * x * y))
    set_mode(34, sq(16) * (14 * x5 * y - 140 * x5 * y - 280 * x3 * y3 + 120 * x3 * y + 140 * x3 * y3 + 210 * x * y5 - 120 * x * y3))
    set_mode(35, sq(16) * (-42 * x5 * y + 140 * x3 * y3 - 42 * x * y5))
    set_mode(36, sq(18) * (8 * x7 - 168 * x5 * y2 + 280 * x3 * y4 - 56 * x * y6))
    set_mode(37, sq(18) * (48 * x7 + 144 * x5 * y2 - 42 * x5 - 480 * x5 * y2 - 800 * x3 * y4 + 420 * x3 * y2 + 240 * x3 * y4 + 336 * x * y6 - 210 * x * y4))
    set_mode(38, sq(18) * (112 * x7 + 672 * x5 * y2 + 560 * x3 * y4 - 168 * x5 - 504 * x3 * y2 + 60 * x3 - 336 * x5 * y2 - 1120 * x3 * y4 - 784 * x * y6 + 504 * x3 * y2 + 840 * x * y4 - 180 * x * y2))
    set_mode(39, sq(18) * (112 * x7 + 1008 * x5 * y2 + 1680 * x3 * y4 + 784 * x * y6 - 210 * x5 - 1260 * x3 * y2 - 1050 * x * y4 + 120 * x3 + 360 * x * y2 - 20 * x))
    set_mode(40, sq(18) * (560 * x6 * y + 1680 * x4 * y3 + 1680 * x2 * y5 + 560 * y7 - 840 * x4 * y - 1680 * x2 * y3 - 840 * y5 + 360 * x2 * y + 360 * y3 - 40 * y))
    return Z
