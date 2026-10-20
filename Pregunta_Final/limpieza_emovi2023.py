"""
Limpieza de la base ESRU-EMOVI 2023 (entrevistado_2023.dta)
------------------------------------------------------------
Pregunta 19: ¿Existe una asociación entre el tono de piel (escala/paleta de
color) y el nivel socioeconómico actual o la escolaridad alcanzada de los
entrevistados, y difiere esta relación entre hombres y mujeres?

Este script:
  1. Lee la base original del entrevistado.
  2. Se queda solo con las variables útiles para la pregunta.
  3. Les pone nombres claros y etiquetas en español.
  4. Construye variables nuevas: tono de piel numérico y en grupos,
     años de escolaridad, quintiles de ingreso y un índice de riqueza.
  5. Detecta y quita outliers con la regla del rango intercuartil (IQR).
  6. Guarda la base limpia en CSV (y aparte las filas eliminadas).

Cómo correrlo (desde la carpeta del repositorio):
    pip install pandas numpy pyreadstat
    python limpieza_emovi2023.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Rutas
# ---------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
ZIP = CARPETA / "data" / "raw" / "BASES_DE_DATOS_EMOVI2023.zip"
ENTRADA = CARPETA / "data" / "raw" / "entrevistado_2023.dta"
SALIDA = CARPETA / "data" / "clean"
SALIDA.mkdir(parents=True, exist_ok=True)

# El .dta pesa 41 MB (GitHub web solo acepta 25 MB por archivo), por eso
# en el repo va el .zip original y aquí se descomprime si hace falta.
if not ENTRADA.exists():
    import zipfile
    with zipfile.ZipFile(ZIP) as z:
        ENTRADA.write_bytes(z.read("Data/entrevistado_2023.dta"))

# ---------------------------------------------------------------------------
# 1. Leer la base original
# ---------------------------------------------------------------------------
raw = pd.read_stata(ENTRADA, convert_categoricals=False)
print(f"Base original: {raw.shape[0]:,} filas x {raw.shape[1]} columnas")

# ---------------------------------------------------------------------------
# 2. Funciones de apoyo
# ---------------------------------------------------------------------------
def si_no(serie):
    """Convierte 1=Sí / 2=No  ->  1 / 0."""
    return serie.map({1: 1, 2: 0})


def quintil_ponderado(valores, pesos):
    """Asigna quintiles (1 = 20% más bajo, 5 = 20% más alto) usando el
    factor de expansión, para que cada quintil represente 20% de la
    población y no solo 20% de la muestra."""
    orden = valores.sort_values().index
    peso_acum = pesos.loc[orden].cumsum() / pesos.loc[orden].sum()
    q = np.ceil(peso_acum * 5).clip(1, 5).astype(int)
    return q.reindex(valores.index)


# ---------------------------------------------------------------------------
# 3. Variables de identificación y diseño muestral
# ---------------------------------------------------------------------------
df = pd.DataFrame(index=raw.index)
df["id"] = raw["index"]
df["factor"] = raw["factor"]                 # ponderador (usar en promedios)
df["estrato"] = raw["est"]                   # estrato de diseño
df["upm"] = raw["upm_muestra"].astype(str)   # unidad primaria de muestreo

# ---------------------------------------------------------------------------
# 4. Sexo y características demográficas
# ---------------------------------------------------------------------------
df["sexo"] = raw["sexo"].map({1: "Hombre", 2: "Mujer"})
df["mujer"] = (raw["sexo"] == 2).astype(int)
df["edad"] = raw["edad"]
df["cohorte"] = raw["cohorte"].map({1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64"})
df["region"] = raw["region"].map({1: "Norte", 2: "Norte-occidente",
                                  3: "Centro-norte", 4: "Centro", 5: "Sur"})
df["urbano"] = (raw["rururb"] == 1).astype(int)
df["entidad"] = raw["entidad"]
df["autoadscripcion"] = raw["p110"].map({1: "Negra", 2: "Indígena", 3: "Blanca",
                                         4: "Mestiza", 5: "Ninguna"})
df["lengua_indigena"] = si_no(raw["p111"])

# ---------------------------------------------------------------------------
# 5. Tono de piel
# ---------------------------------------------------------------------------
# p112: tono según la paleta de color, letras A-K.
# Verificado con el colorímetro: A = más oscuro, K = más claro.
letras = list("ABCDEFGHIJK")
df["tono_piel_letra"] = raw["p112"].str.strip().str.upper()
df["tono_piel"] = df["tono_piel_letra"].map({l: i + 1 for i, l in enumerate(letras)})

# Tres grupos para tablas y gráficas (los extremos tienen muy pocas personas)
df["tono_piel_grupo"] = pd.cut(df["tono_piel"], bins=[0, 5, 7, 11],
                               labels=["Oscuro (A-E)", "Medio (F-G)", "Claro (H-K)"])

# Medición objetiva con colorímetro (CIELab). L = luminosidad (0 negro, 100 blanco)
df["L_dorso"] = raw["p113dL"]
df["a_dorso"] = raw["p113da"]
df["b_dorso"] = raw["p113db"]
df["L_muneca"] = raw["p113mL"]
df["a_muneca"] = raw["p113ma"]
df["b_muneca"] = raw["p113mb"]
df["luminosidad_piel"] = df[["L_dorso", "L_muneca"]].mean(axis=1)  # promedio de las dos

# ---------------------------------------------------------------------------
# 6. Escolaridad
# ---------------------------------------------------------------------------
# educ: variable oficial de ESRU en 4 niveles
niveles_educ = {1: "Primaria o menos", 2: "Secundaria", 3: "Preparatoria", 4: "Profesional"}
df["nivel_educ_num"] = raw["educ"]
df["nivel_educ"] = raw["educ"].map(niveles_educ)
df["universidad"] = np.where(raw["educ"].isna(), np.nan, (raw["educ"] == 4).astype(float))

# Años de escolaridad a partir de p9 (último nivel) y p10 (último grado)
# años = años previos al nivel + grado alcanzado en ese nivel
anios_previos = {
    1: 0,   # Preescolar (no suma años de primaria)
    2: 0,   # Primaria
    3: 6,   # Secundaria técnica
    4: 6,   # Secundaria general
    5: 9,   # Preparatoria técnica
    6: 9,   # Preparatoria general
    7: 6,   # Técnica con primaria
    8: 9,   # Técnica con secundaria
    9: 12,  # Técnica con preparatoria
    10: 9,  # Normal básica
    11: 12, # Normal de licenciatura
    12: 12, # Profesional
    13: 16, # Posgrado
}
grado = pd.to_numeric(raw["p10"], errors="coerce").replace(9, np.nan)  # 9 = No sabe
df["anios_esc"] = raw["p9"].map(anios_previos) + grado
df.loc[raw["p9"] == 1, "anios_esc"] = 0    # Preescolar
df.loc[raw["p9"] == 97, "anios_esc"] = 0   # Nunca fue a la escuela

# Escolaridad de los padres (útil como control de origen)
df["educ_padre"] = raw["educp"].map(niveles_educ)
df["educ_madre"] = raw["educm"].map(niveles_educ)

# ---------------------------------------------------------------------------
# 7. Nivel socioeconómico actual (tres medidas)
# ---------------------------------------------------------------------------
# 7a. Ingreso per cápita mensual del hogar (imputado por ESRU, sin faltantes)
df["ingreso_pc"] = raw["ingc_pc"]
df["log_ingreso_pc"] = np.log(raw["ingc_pc"])
df["quintil_ingreso"] = np.nan   # se calcula después de quitar outliers (paso 8)

# 7b. Escalera subjetiva: dónde se pone el hogar del 1 (más pobre) al 10 (más rico)
df["escalera_actual"] = raw["p103"]
df["escalera_14"] = raw["p104"]   # el hogar donde vivía a los 14 años

# 7c. Índice de riqueza del hogar (primer componente principal de bienes y servicios)
bienes = pd.DataFrame(index=raw.index)
bienes["piso_tierra"] = (raw["p88"] == 1).astype(int)
bienes["piso_recubrimiento"] = (raw["p88"] == 3).astype(int)
for c in [f"p94{x}" for x in "abcdefg"] + [f"p95{x}" for x in "abcde"] \
        + [f"p96{x}" for x in "abcdefghijklmnopqr"] + [f"p97{x}" for x in "abcdefghijklmn"]:
    bienes[c] = si_no(raw[c])
bienes["autos"] = raw["p99"].clip(upper=3)                   # topado en 3
bienes["personas_x_cuarto"] = raw["tamhog"] / raw["p89"].clip(lower=1)

z = (bienes - bienes.mean()) / bienes.std()
z = z.loc[:, bienes.std() > 0]
valores_propios, vectores = np.linalg.eigh(np.cov(z.T))
pc1 = z.values @ vectores[:, -1]                             # componente más grande
if np.corrcoef(pc1, df["log_ingreso_pc"])[0, 1] < 0:         # que "más alto = más rico"
    pc1 = -pc1
df["indice_riqueza"] = (pc1 - pc1.mean()) / pc1.std()        # media 0, desviación 1
df["quintil_riqueza"] = np.nan   # se calcula después de quitar outliers (paso 8)
print(f"Índice de riqueza: el 1er componente explica "
      f"{valores_propios[-1] / valores_propios.sum():.0%} de la varianza")

# Clase ocupacional actual (ESRU, 6 clases). Vacía si no trabaja.
df["clase_ocup"] = raw["clase"]

# ---------------------------------------------------------------------------
# 8. Outliers (valores atípicos)
# ---------------------------------------------------------------------------
# Regla del rango intercuartil (IQR): un valor es outlier si está
#   por debajo de Q1 - 1.5*IQR  o  por arriba de Q3 + 1.5*IQR
# donde Q1 = percentil 25, Q3 = percentil 75 e IQR = Q3 - Q1.
#
# Variables donde SÍ se buscan outliers:
#   - log_ingreso_pc: el ingreso se revisa en logaritmo porque el ingreso en
#     pesos es muy sesgado (en pesos la regla marcaría a 1,336 personas solo
#     por tener ingresos altos, aunque sean ingresos reales).
#   - indice_riqueza
#   - luminosidad_piel (medición del colorímetro)
#
# Variables donde NO se buscan (y por qué):
#   - edad: ya viene acotada a 25-64 años por diseño de la encuesta.
#   - escalera_actual / escalera_14: escala fija de 1 a 10.
#   - anios_esc: la regla marcaría a 3,010 personas (0-4 años de escuela o
#     posgrado), que son casos reales y justo los que importan para la pregunta.
#   - tono_piel: escala fija A-K, sin valores imposibles.
#   - factor: es el ponderador de la encuesta, no se debe modificar.
def limites_iqr(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


VARIABLES_OUTLIERS = ["log_ingreso_pc", "indice_riqueza", "luminosidad_piel"]
es_outlier = pd.Series(False, index=df.index)
print("\nOutliers por variable (regla IQR 1.5):")
for var in VARIABLES_OUTLIERS:
    bajo, alto = limites_iqr(df[var].dropna())
    marca = (df[var] < bajo) | (df[var] > alto)       # NaN no cuenta como outlier
    df[f"outlier_{var}"] = marca.astype(int)
    es_outlier |= marca
    print(f"  {var:18s} rango aceptado [{bajo:7.2f}, {alto:7.2f}]  ->  {marca.sum():4d} personas")

outliers = df[es_outlier]
outliers.to_csv(SALIDA / "outliers_eliminados.csv", index=False, encoding="utf-8-sig")
df = df[~es_outlier].drop(columns=[f"outlier_{v}" for v in VARIABLES_OUTLIERS])
print(f"  Total eliminadas: {len(outliers):,} personas "
      f"({len(outliers) / (len(df) + len(outliers)):.1%} de la muestra)")

# Quintiles calculados ya sin outliers
df["quintil_ingreso"] = quintil_ponderado(df["ingreso_pc"], df["factor"])
df["quintil_riqueza"] = quintil_ponderado(df["indice_riqueza"], df["factor"])

# ---------------------------------------------------------------------------
# 9. Revisiones rápidas
# ---------------------------------------------------------------------------
assert df["id"].is_unique, "Hay folios repetidos"
assert df["tono_piel"].between(1, 11).all(), "Tono de piel fuera de rango"
print("\nPromedio de luminosidad (colorímetro) por letra de la paleta:")
print(df.groupby("tono_piel_letra")["luminosidad_piel"].mean().round(1).to_string())

print("\nValores faltantes por variable (solo las que tienen alguno):")
faltan = df.isna().sum()
print(faltan[faltan > 0].to_string())

# ---------------------------------------------------------------------------
# 10. Guardar
# ---------------------------------------------------------------------------
df.to_csv(SALIDA / "emovi2023_limpia.csv", index=False, encoding="utf-8-sig")
print(f"\nBase limpia guardada: {df.shape[0]:,} filas x {df.shape[1]} columnas")
