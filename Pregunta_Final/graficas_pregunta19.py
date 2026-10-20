"""
Gráficas - Pregunta 19 (ESRU-EMOVI 2023)
-----------------------------------------
Hace 4 gráficas con la base limpia:
  1. Tono de piel vs. escolaridad
  2. Tono de piel vs. nivel socioeconómico
  3. Tono de piel vs. escolaridad, por sexo
  4. Tono de piel vs. nivel socioeconómico, por sexo

Medidas:
  - Escolaridad = años de escuela promedio.
  - Nivel socioeconómico = ingreso mensual por persona del hogar (mediana,
    porque unos pocos ingresos muy altos jalarían el promedio hacia arriba).
Todo se calcula con el factor de expansión, para que represente a México.

Cómo correrlo (desde la carpeta Pregunta_Final):
    pip install pandas numpy matplotlib
    python graficas_pregunta19.py
Las imágenes quedan en la carpeta graficas/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Rutas y datos
# ---------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
df = pd.read_csv(CARPETA / "data" / "clean" / "emovi2023_limpia.csv")
SALIDA = CARPETA / "graficas"
SALIDA.mkdir(exist_ok=True)

# Los tonos de los extremos tienen muy pocas personas (A tiene 27, K tiene 58),
# así que se juntan A-C e I-K para que cada punto tenga al menos 100 personas.
GRUPOS_TONO = {"A": "A–C", "B": "A–C", "C": "A–C", "D": "D", "E": "E", "F": "F",
               "G": "G", "H": "H", "I": "I–K", "J": "I–K", "K": "I–K"}
df["tono"] = df["tono_piel_letra"].map(GRUPOS_TONO)
LETRAS = ["A–C", "D", "E", "F", "G", "H", "I–K"]
MINIMO = 100   # si un punto tiene menos de 100 personas se dibuja hueco

# Colores y estilo
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, GRIS = "#1f1f1e", "#8a8a85"
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA,
    "ytick.color": TINTA,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# 1. Funciones para calcular promedios y medianas ponderadas
# ---------------------------------------------------------------------------
def promedio_ponderado(g, columna):
    d = g[[columna, "factor"]].dropna()
    return np.average(d[columna], weights=d["factor"])


def mediana_ponderada(g, columna):
    d = g[[columna, "factor"]].dropna().sort_values(columna)
    acumulado = d["factor"].cumsum() / d["factor"].sum()
    return d.loc[acumulado >= 0.5, columna].iloc[0]


def resumen(datos, columna, funcion, por_sexo=False):
    """Tabla con el valor por letra de tono de piel (y por sexo si se pide)."""
    grupos = ["tono", "sexo"] if por_sexo else ["tono"]
    tabla = datos.groupby(grupos).apply(
        lambda g: pd.Series({"valor": funcion(g, columna), "n": len(g)}),
        include_groups=False,
    ).reset_index()
    return tabla


# ---------------------------------------------------------------------------
# 2. Función que dibuja una línea por grupo
# ---------------------------------------------------------------------------
def dibujar_linea(ax, tabla, color, etiqueta=None):
    tabla = tabla.set_index("tono").reindex(LETRAS)
    x = np.arange(len(LETRAS))
    ax.plot(x, tabla["valor"], color=color, linewidth=2, label=etiqueta, zorder=2)
    pocos = tabla["n"] < MINIMO
    # puntos llenos = suficientes personas; huecos = pocas personas
    ax.scatter(x[~pocos], tabla["valor"][~pocos], s=55, color=color,
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter(x[pocos], tabla["valor"][pocos], s=55, facecolor="white",
               edgecolor=color, linewidth=1.8, zorder=3)


def formato_eje_x(ax):
    ax.set_xticks(range(len(LETRAS)))
    ax.set_xticklabels(LETRAS)
    ax.set_xlabel("Tono de piel (paleta de color)   A = más oscuro  →  K = más claro")
    ax.set_xlim(-0.4, len(LETRAS) - 0.6)


def guardar(fig, ax, titulo, subtitulo, archivo):
    ax.set_title(titulo, loc="left", fontsize=14, fontweight="bold", color=TINTA, pad=26)
    ax.text(0, 1.03, subtitulo, transform=ax.transAxes, fontsize=10, color=GRIS)
    fig.text(0.01, 0.01, "A–C e I–K se agrupan porque tienen pocas personas. "
             "Fuente: ESRU-EMOVI 2023, datos ponderados con el factor de expansión.",
             fontsize=8.5, color=GRIS)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SALIDA / archivo, dpi=200)
    plt.close(fig)
    print(f"Guardada: graficas/{archivo}")


# ---------------------------------------------------------------------------
# 3. Las 4 gráficas
# ---------------------------------------------------------------------------
TEXTO_ESC = "Años de escolaridad (promedio)"
TEXTO_ING = "Ingreso mensual por persona (mediana, pesos)"

# Gráfica 1: tono de piel vs escolaridad
esc = resumen(df, "anios_esc", promedio_ponderado)
fig, ax = plt.subplots(figsize=(9, 5.5))
dibujar_linea(ax, esc, AZUL)
formato_eje_x(ax)
ax.set_ylabel(TEXTO_ESC)
guardar(fig, ax, "Tono de piel y escolaridad",
        "Años de escuela promedio según el tono de piel", "1_tono_escolaridad.png")

# Gráfica 2: tono de piel vs nivel socioeconómico
ing = resumen(df, "ingreso_pc", mediana_ponderada)
fig, ax = plt.subplots(figsize=(9, 5.5))
dibujar_linea(ax, ing, AZUL)
formato_eje_x(ax)
ax.set_ylabel(TEXTO_ING)
ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
guardar(fig, ax, "Tono de piel y nivel socioeconómico",
        "Ingreso mensual por persona del hogar según el tono de piel",
        "2_tono_nivel_socioeconomico.png")

# Gráfica 3: tono de piel vs escolaridad, por sexo
esc_sexo = resumen(df, "anios_esc", promedio_ponderado, por_sexo=True)
fig, ax = plt.subplots(figsize=(9, 5.5))
dibujar_linea(ax, esc_sexo[esc_sexo["sexo"] == "Hombre"], AZUL, "Hombres")
dibujar_linea(ax, esc_sexo[esc_sexo["sexo"] == "Mujer"], NARANJA, "Mujeres")
formato_eje_x(ax)
ax.set_ylabel(TEXTO_ESC)
ax.legend(loc="upper left", frameon=False)
guardar(fig, ax, "Tono de piel, escolaridad y sexo",
        "Años de escuela promedio según el tono de piel, hombres vs. mujeres",
        "3_tono_escolaridad_sexo.png")

# Gráfica 4: tono de piel vs nivel socioeconómico, por sexo
ing_sexo = resumen(df, "ingreso_pc", mediana_ponderada, por_sexo=True)
fig, ax = plt.subplots(figsize=(9, 5.5))
dibujar_linea(ax, ing_sexo[ing_sexo["sexo"] == "Hombre"], AZUL, "Hombres")
dibujar_linea(ax, ing_sexo[ing_sexo["sexo"] == "Mujer"], NARANJA, "Mujeres")
formato_eje_x(ax)
ax.set_ylabel(TEXTO_ING)
ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
ax.legend(loc="upper left", frameon=False)
guardar(fig, ax, "Tono de piel, nivel socioeconómico y sexo",
        "Ingreso mensual por persona del hogar según el tono de piel, hombres vs. mujeres",
        "4_tono_nivel_socioeconomico_sexo.png")

# ---------------------------------------------------------------------------
# 4. Tabla resumen (para citar números en el reporte)
# ---------------------------------------------------------------------------
tabla = (esc_sexo.rename(columns={"valor": "anios_esc"})
         .merge(ing_sexo.rename(columns={"valor": "ingreso_mediano"}).drop(columns="n"),
                on=["tono", "sexo"]))
tabla.to_csv(SALIDA / "tabla_resumen.csv", index=False, encoding="utf-8-sig")
print("\n", tabla.round(1).to_string(index=False))
