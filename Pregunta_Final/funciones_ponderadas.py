"""
Funciones de apoyo que usan los 3 scripts de análisis.

¿Por qué "ponderadas"? Cada persona de la encuesta representa a un número
distinto de mexicanos (columna `factor`). Para que un promedio o un porcentaje
represente a México y no solo a la muestra, cada persona pesa según su factor.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CARPETA = Path(__file__).resolve().parent
BASE = CARPETA / "data" / "clean" / "emovi2023_final.csv"

ORDEN_TONO = ["Oscuro (A-E)", "Medio (F-G)", "Claro (H-K)"]
ORDEN_EDUC = ["Primaria o menos", "Secundaria", "Preparatoria", "Profesional"]
ORDEN_SEXO = ["Hombre", "Mujer"]


def cargar_base():
    df = pd.read_csv(BASE)
    df["upm_cod"] = pd.factorize(df["upm"])[0]   # conglomerado en número (para errores estándar)
    return df


def media_p(x, w):
    m = x.notna()
    return float(np.average(x[m], weights=w[m]))


def desv_p(x, w):
    m = x.notna()
    mu = np.average(x[m], weights=w[m])
    return float(np.sqrt(np.average((x[m] - mu) ** 2, weights=w[m])))


def mediana_p(x, w):
    d = pd.DataFrame({"x": x, "w": w}).dropna().sort_values("x")
    acum = d["w"].cumsum() / d["w"].sum()
    return float(d.loc[acum >= 0.5, "x"].iloc[0])


def pct_p(condicion, w):
    """Porcentaje ponderado de personas que cumplen una condición (True/False)."""
    return 100 * float(np.average(condicion.astype(float), weights=w))


def corr_p(x, y, w):
    m = x.notna() & y.notna()
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def regresion(formula, datos):
    """Mínimos cuadrados ponderados por `factor`, con errores estándar
    agrupados por conglomerado (upm), como pide el diseño de la encuesta."""
    columnas = [c for c in datos.columns if c in formula] + ["factor", "upm_cod"]
    d = datos[list(dict.fromkeys(columnas))].dropna()
    modelo = smf.wls(formula, data=d, weights=d["factor"])
    return modelo.fit(cov_type="cluster", cov_kwds={"groups": d["upm_cod"]})


def estrellas(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
