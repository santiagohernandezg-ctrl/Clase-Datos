"""
Limpieza de la base ESRU-EMOVI 2023 (entrevistado_2023.dta)
------------------------------------------------------------
Pregunta 19: ¿Existe una asociación entre el tono de piel (escala/paleta de
color) y el nivel socioeconómico actual o la escolaridad alcanzada de los
entrevistados, y difiere esta relación entre hombres y mujeres?

El script sigue el orden de la "Guía paso a paso: Limpieza y procesamiento de
microdatos" del curso:

    1. EDA  ->  2. Errores  ->  3. Duplicados  ->  4. Nulos
    ->  5. Outliers  ->  6. Consistencias  ->  7. Encoding

En cada paso se anota en una bitácora cuántas observaciones se afectan
(resultados/bitacora_limpieza.csv).

Cómo correrlo (desde la carpeta Pregunta_Final):
    pip install pandas numpy pyreadstat matplotlib
    python limpieza_emovi2023.py

Salidas:
    data/clean/emovi2023_final.csv        base final (solo variables usadas)
    data/clean/outliers_eliminados.csv    filas quitadas por ser atípicas
    resultados/0_perfilado_variables.csv  perfil de cada variable (paso 1)
    resultados/bitacora_limpieza.csv      qué se hizo y a cuántos afectó
    graficas/eda/                         histogramas y boxplots del EDA
"""

from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Rutas y lectura
# ---------------------------------------------------------------------------
CARPETA = Path(__file__).resolve().parent
ZIP = CARPETA / "data" / "raw" / "BASES_DE_DATOS_EMOVI2023.zip"
ENTRADA = CARPETA / "data" / "raw" / "entrevistado_2023.dta"
SALIDA = CARPETA / "data" / "clean"
RESULTADOS = CARPETA / "resultados"
EDA = CARPETA / "graficas" / "eda"
for carpeta in (SALIDA, RESULTADOS, EDA):
    carpeta.mkdir(parents=True, exist_ok=True)

# El .dta pesa 41 MB (GitHub web acepta 25 MB por archivo): en el repo va el
# .zip original y aquí se descomprime si hace falta.
if not ENTRADA.exists():
    with zipfile.ZipFile(ZIP) as z:
        ENTRADA.write_bytes(z.read("Data/entrevistado_2023.dta"))

raw = pd.read_stata(ENTRADA, convert_categoricals=False)
N_ORIGINAL = len(raw)

bitacora = []


def anotar(paso, accion, afectadas, filas_restantes, decision):
    """Guarda una línea en la bitácora de limpieza y la imprime."""
    bitacora.append({"paso": paso, "accion": accion, "observaciones_afectadas": int(afectadas),
                     "filas_restantes": int(filas_restantes), "decision": decision})
    print(f"  [{paso}] {accion}: {int(afectadas):,} obs. -> {decision}")


def si_no(serie):
    """1 = Sí / 2 = No  ->  1 / 0."""
    return serie.map({1: 1, 2: 0})


def limites_iqr(serie):
    """Límites de la regla del rango intercuartil: Q1 - 1.5*IQR y Q3 + 1.5*IQR."""
    q1, q3 = serie.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def quintil_ponderado(valores, pesos):
    """Quintiles con el factor de expansión: cada quintil = 20% de la población."""
    orden = valores.sort_values().index
    acumulado = pesos.loc[orden].cumsum() / pesos.loc[orden].sum()
    return np.ceil(acumulado * 5).clip(1, 5).astype(int).reindex(valores.index)


# Variables originales que usa el análisis (el resto de las 296 no se necesita)
VARS_ORIGINALES = {
    "index": "Folio", "factor": "Factor de expansión", "est": "Estrato de diseño",
    "upm_muestra": "Unidad primaria de muestreo", "sexo": "Sexo", "edad": "Edad",
    "region": "Región", "rururb": "Rural/urbano",
    "p9": "Último nivel escolar", "p10": "Último grado aprobado", "educ": "Nivel educativo ESRU",
    "educp": "Nivel educativo del padre", "educm": "Nivel educativo de la madre",
    "p59": "Trabajó la semana pasada", "clase": "Clase ocupacional (ESRU)",
    "p103": "Escalera hoy (1-10)", "p104": "Escalera a los 14 (1-10)",
    "p110": "Autoadscripción", "p111": "Habla lengua indígena",
    "p112": "Tono de piel (paleta A-K)", "p113dL": "Colorímetro L dorso",
    "p113mL": "Colorímetro L muñeca", "ingc_pc": "Ingreso per cápita del hogar",
    "tamhog": "Tamaño del hogar", "p88": "Material del piso", "p89": "Cuartos", "p99": "Autos",
}

# ===========================================================================
# PASO 1. EDA (análisis exploratorio): estructura, tipos, rangos y nulos
# ===========================================================================
print("\nPASO 1. EDA")
print(f"  Base original: {raw.shape[0]:,} filas x {raw.shape[1]} columnas")
print("  Unidad de análisis: una fila = una persona entrevistada (25 a 64 años)")

perfil = []
for var, nombre in VARS_ORIGINALES.items():
    s = raw[var]
    numerica = pd.api.types.is_numeric_dtype(s)
    perfil.append({
        "variable": var, "descripcion": nombre, "tipo": str(s.dtype),
        "n_categorias": s.nunique(), "pct_nulos": round(100 * s.isna().mean(), 2),
        "min": s.min() if numerica else s.dropna().min(),
        "max": s.max() if numerica else s.dropna().max(),
        "media": round(s.mean(), 2) if numerica else None,
        "mediana": s.median() if numerica else None,
    })
perfil = pd.DataFrame(perfil)
perfil.to_csv(RESULTADOS / "0_perfilado_variables.csv", index=False, encoding="utf-8-sig")
print(perfil[["variable", "tipo", "n_categorias", "pct_nulos", "min", "max"]].to_string(index=False))
anotar("1. EDA", f"Perfilado de {len(VARS_ORIGINALES)} variables originales", 0, len(raw),
       "Sin cambios; se detecta que el ingreso está muy sesgado y que p10 y el colorímetro tienen nulos")

# Visualización rápida: ingreso en pesos vs. en logaritmo, y boxplots
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].hist(raw["ingc_pc"], bins=60, color="#2a78d6")
ax[0].set_title("Ingreso per cápita (pesos): muy sesgado")
ax[1].hist(np.log(raw["ingc_pc"]), bins=60, color="#2a78d6")
ax[1].set_title("Ingreso per cápita (logaritmo): casi simétrico")
ax[2].boxplot([raw["p113dL"].dropna(), raw["p113mL"].dropna()], tick_labels=["Dorso", "Muñeca"])
ax[2].set_title("Colorímetro: luminosidad L")
fig.tight_layout()
fig.savefig(EDA / "eda_distribuciones.png", dpi=150)
plt.close(fig)

# ===========================================================================
# PASO 2. ERRORES: códigos especiales del diccionario y reglas de rango
# ===========================================================================
print("\nPASO 2. Errores")
# 2a. Comparación con el diccionario: p10 = 9 significa "No sabe" -> NA
no_sabe = (raw["p10"] == 9).sum()
raw["p10"] = raw["p10"].replace(9, np.nan)
anotar("2. Errores", "p10 = 9 ('No sabe' el último grado) recodificado a NA", no_sabe, len(raw),
       "Código especial del diccionario, no es un grado real")
# p9 = 97 ('Ninguno') NO es error: personas sin escolaridad -> 0 años (paso 7)
anotar("2. Errores", "p9 = 97 ('Ninguno') revisado", (raw["p9"] == 97).sum(), len(raw),
       "Se conserva: es una respuesta válida (sin escolaridad = 0 años)")

# 2b. Validación por reglas: rangos válidos según el diseño de la encuesta
reglas = {
    "edad entre 25 y 64": raw["edad"].between(25, 64),
    "tono de piel en A-K": raw["p112"].str.strip().str.upper().isin(list("ABCDEFGHIJK")),
    "escalera hoy entre 1 y 10": raw["p103"].between(1, 10),
    "escalera a los 14 entre 1 y 10": raw["p104"].between(1, 10),
    "ingreso per cápita > 0": raw["ingc_pc"] > 0,
    "factor de expansión > 0": raw["factor"] > 0,
    "colorímetro L entre 0 y 100": raw[["p113dL", "p113mL"]].apply(
        lambda c: c.isna() | c.between(0, 100)).all(axis=1),
}
for regla, ok in reglas.items():
    anotar("2. Errores", f"Regla: {regla}", (~ok).sum(), len(raw),
           "Todas cumplen" if ok.all() else "Revisar")

# ===========================================================================
# PASO 3. DUPLICADOS
# ===========================================================================
print("\nPASO 3. Duplicados")
dup_exactos = raw.duplicated().sum()
dup_folio = raw["index"].duplicated().sum()
raw = raw.drop_duplicates().drop_duplicates(subset="index")
anotar("3. Duplicados", "Filas idénticas en todas las variables", dup_exactos, len(raw),
       "Ninguna; no se elimina nada")
anotar("3. Duplicados", "Folios repetidos (misma persona dos veces)", dup_folio, len(raw),
       "Ninguno; cada folio es único")

# ===========================================================================
# Construcción de variables limpias (nombres claros y en español)
# ===========================================================================
df = pd.DataFrame(index=raw.index)
df["id"] = raw["index"]
df["factor"] = raw["factor"]
df["sexo"] = raw["sexo"].map({1: "Hombre", 2: "Mujer"})
df["estrato"] = raw["est"]                      # diseño muestral: para errores
df["upm"] = raw["upm_muestra"]                    # estándar correctos (por conglomerado)
df["edad"] = raw["edad"]
df["region"] = raw["region"].map({1: "Norte", 2: "Norte-occidente", 3: "Centro-norte",
                                  4: "Centro", 5: "Sur"})
df["autoadscripcion"] = raw["p110"].map({1: "Negra", 2: "Indígena", 3: "Blanca",
                                         4: "Mestiza", 5: "Ninguna"})
df["tono_piel_letra"] = raw["p112"].str.strip().str.upper()   # estandarización de texto
df["luminosidad_piel"] = raw[["p113dL", "p113mL"]].mean(axis=1)
df["ingreso_pc"] = raw["ingc_pc"]
df["escalera_actual"] = raw["p103"]
df["escalera_14"] = raw["p104"]

# Años de escolaridad = años previos al nivel + grado aprobado en ese nivel
ANIOS_PREVIOS = {1: 0, 2: 0, 3: 6, 4: 6, 5: 9, 6: 9, 7: 6, 8: 9, 9: 12, 10: 9,
                 11: 12, 12: 12, 13: 16}
df["anios_esc"] = raw["p9"].map(ANIOS_PREVIOS) + raw["p10"]
df.loc[raw["p9"].isin([1, 97]), "anios_esc"] = 0          # preescolar o ninguno
df["nivel_educ_num"] = raw["educ"]

# Ocupación actual: la ocupación es una de las 3 medidas clásicas del nivel
# socioeconómico (ingreso, educación y ocupación). Quien no trabajó la semana
# pasada (p59 = 6) no tiene clase: eso NO es un dato faltante, es la categoría
# "No trabaja".
CLASES = {1: "Agrícola", 2: "Manual baja calificación", 3: "Manual alta calificación",
          4: "Comercio", 5: "No manual baja calificación", 6: "No manual alta calificación"}
df["clase_ocup"] = raw["clase"].map(CLASES)
df.loc[raw["p59"] == 6, "clase_ocup"] = "No trabaja"

# Educación de los padres: el nivel MÁS ALTO entre padre y madre. Es un control
# de origen familiar: se relaciona con el tono (r = 0.12) y fuertemente con la
# escolaridad (r = 0.47) y el ingreso (r = 0.38) de la persona.
NIVELES = {1: "Primaria o menos", 2: "Secundaria", 3: "Preparatoria", 4: "Profesional"}
df["educ_padres"] = np.fmax(raw["educp"], raw["educm"]).map(NIVELES)

# ===========================================================================
# PASO 4. NULOS
# ===========================================================================
print("\nPASO 4. Nulos")
falta_esc = df["anios_esc"].isna()
# ¿Son aleatorios? Comparar a quienes les falta el dato contra el resto
print(f"  Sin grado escolar: edad media {df.loc[falta_esc, 'edad'].mean():.1f} vs "
      f"{df.loc[~falta_esc, 'edad'].mean():.1f}; ingreso mediano "
      f"{df.loc[falta_esc, 'ingreso_pc'].median():,.0f} vs {df.loc[~falta_esc, 'ingreso_pc'].median():,.0f}")
# NO son aleatorios: son más jóvenes, con más ingreso y casi todos de prepa o
# universidad. Borrarlos sesgaría la base hacia gente menos educada.
# Decisión: imputación simple DENTRO DEL MISMO NIVEL declarado (p9), que sí se conoce:
#   - años: mediana de años de quienes reportaron ese mismo nivel
#   - nivel ESRU: moda del nivel ESRU de quienes reportaron ese mismo nivel
df["esc_imputada"] = falta_esc.astype(int)
mediana_nivel = df.groupby(raw["p9"])["anios_esc"].transform("median")
df.loc[falta_esc, "anios_esc"] = mediana_nivel[falta_esc]
falta_niv = df["nivel_educ_num"].isna()
moda_nivel = df.groupby(raw["p9"])["nivel_educ_num"].transform(
    lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
df.loc[falta_niv, "nivel_educ_num"] = moda_nivel[falta_niv]
df.loc[falta_niv, "esc_imputada"] = 1
anotar("4. Nulos", "Escolaridad sin grado ('No sabe'): imputada con mediana/moda de su nivel",
       df["esc_imputada"].sum(), len(df),
       "Imputación simple: 1.8%, nulos NO aleatorios (más jóvenes y educados); se marca en esc_imputada")

falta_lum = df["luminosidad_piel"].isna()
print(f"  Sin colorímetro: tono medio en paleta {raw.loc[falta_lum, 'p112'].map(dict(zip('ABCDEFGHIJK', range(1, 12)))).mean():.2f}"
      f" vs {raw.loc[~falta_lum, 'p112'].map(dict(zip('ABCDEFGHIJK', range(1, 12)))).mean():.2f}")
anotar("4. Nulos", "Colorímetro sin medición en dorso y muñeca", falta_lum.sum(), len(df),
       "Se deja NA (2.6%): variable secundaria de validación; la paleta (variable principal) está completa")
falta_ocup = df["clase_ocup"].isna()
anotar("4. Nulos", "Trabaja pero sin clase ocupacional asignada", falta_ocup.sum(), len(df),
       "Se deja NA (0.3%): muy pocos casos; se excluyen solo de los análisis de ocupación")
falta_padres = df["educ_padres"].isna()
df["educ_padres"] = df["educ_padres"].fillna("No sabe")
anotar("4. Nulos", "No sabe la educación de ninguno de sus padres", falta_padres.sum(), len(df),
       "Categoría 'No sabe' (no se imputa: no hay información de dónde inferirla)")
anotar("4. Nulos", "Tono de piel, sexo, edad, ingreso, escaleras y factor", 0, len(df),
       "Sin nulos")

# ===========================================================================
# PASO 5. OUTLIERS
# ===========================================================================
print("\nPASO 5. Outliers (regla IQR x 1.5 como detección)")
df["log_ingreso_pc"] = np.log(df["ingreso_pc"])

# Índice de riqueza: primer componente principal de bienes y servicios del hogar
bienes = pd.DataFrame(index=raw.index)
bienes["piso_tierra"] = (raw["p88"] == 1).astype(int)
bienes["piso_recubrimiento"] = (raw["p88"] == 3).astype(int)
for c in ([f"p94{x}" for x in "abcdefg"] + [f"p95{x}" for x in "abcde"]
          + [f"p96{x}" for x in "abcdefghijklmnopqr"] + [f"p97{x}" for x in "abcdefghijklmn"]):
    bienes[c] = si_no(raw[c])
bienes["autos"] = raw["p99"].clip(upper=3)                 # tope: 3 o más autos
bienes["personas_x_cuarto"] = raw["tamhog"] / raw["p89"].clip(lower=1)
z = (bienes - bienes.mean()) / bienes.std()
z = z.loc[:, bienes.std() > 0]
valores, vectores = np.linalg.eigh(np.cov(z.T))
pc1 = z.values @ vectores[:, -1]
if np.corrcoef(pc1, df["log_ingreso_pc"])[0, 1] < 0:       # más alto = más rico
    pc1 = -pc1
df["indice_riqueza"] = (pc1 - pc1.mean()) / pc1.std()

# 5a. Ingreso: la regla IQR en PESOS marcaría a muchos solo por ganar bien
#     (variable muy sesgada), por eso se aplica en LOGARITMO.
bajo, alto = limites_iqr(df["ingreso_pc"])
print(f"  (Referencia) IQR en pesos marcaría {((df['ingreso_pc'] < bajo) | (df['ingreso_pc'] > alto)).sum():,} personas")
bajo, alto = limites_iqr(df["log_ingreso_pc"])
out_ing = (df["log_ingreso_pc"] < bajo) | (df["log_ingreso_pc"] > alto)
anotar("5. Outliers", f"Ingreso per cápita fuera de ${np.exp(bajo):,.0f}-${np.exp(alto):,.0f} (IQR en log)",
       out_ing.sum(), len(df),
       "Se eliminan: es ingreso IMPUTADO por ESRU y los extremos de un modelo son poco confiables y jalan los promedios")

bajo, alto = limites_iqr(df["indice_riqueza"])
out_riq = (df["indice_riqueza"] < bajo) | (df["indice_riqueza"] > alto)
anotar("5. Outliers", "Índice de riqueza fuera del rango IQR", out_riq.sum(), len(df),
       "Se eliminan: combinaciones de bienes extremas que distorsionan el índice")

# 5b. Colorímetro: un valor extremo es un error de MEDICIÓN del aparato, no de la
#     persona. No se borra a la persona: solo se deja vacía esa medición.
bajo, alto = limites_iqr(df["luminosidad_piel"].dropna())
out_lum = (df["luminosidad_piel"] < bajo) | (df["luminosidad_piel"] > alto)
df.loc[out_lum, "luminosidad_piel"] = np.nan
anotar("5. Outliers", "Luminosidad del colorímetro fuera del rango IQR", out_lum.sum(), len(df),
       "Medición a NA, se conserva la persona (error del aparato; la paleta sigue siendo válida)")

# 5c. Variables donde NO se buscan outliers (y por qué)
anotar("5. Outliers", "Edad, escaleras, tono y años de escuela", 0, len(df),
       "No aplica: escalas acotadas por diseño; en escolaridad los extremos (0 años, posgrado) son reales")

es_outlier = out_ing | out_riq
df[es_outlier].to_csv(SALIDA / "outliers_eliminados.csv", index=False, encoding="utf-8-sig")
df = df[~es_outlier].copy()
anotar("5. Outliers", "Total de personas eliminadas por outliers", es_outlier.sum(), len(df),
       f"{es_outlier.mean():.1%} de la muestra")

# ===========================================================================
# PASO 6. CONSISTENCIAS (validación cruzada entre variables)
# ===========================================================================
print("\nPASO 6. Consistencias")
anios_vs_edad = df["anios_esc"] > df["edad"] - 5
anotar("6. Consistencias", "Años de escuela > edad - 5 (imposible)", anios_vs_edad.sum(), len(df),
       "Ninguno")

# El nivel oficial de ESRU (completado) nunca debe superar al nivel declarado en p9
MAX_NIVEL = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 2, 8: 2, 9: 4, 10: 3, 11: 4, 12: 4, 13: 4, 97: 1}
p9 = raw.loc[df.index, "p9"]
nivel_mayor = df["nivel_educ_num"] > p9.map(MAX_NIVEL)
anotar("6. Consistencias", "Nivel ESRU mayor que el nivel declarado en p9", nivel_mayor.sum(), len(df),
       "Ninguno")

# Quien dice que no trabajó no debe tener clase ocupacional
no_trab = raw.loc[df.index, "p59"] == 6
incons_trab = (no_trab & raw.loc[df.index, "clase"].notna()).sum()
anotar("6. Consistencias", "No trabajó (p59 = 6) pero tiene clase ocupacional", incons_trab, len(df),
       "Ninguno")

# Paleta (lo que ve el encuestador) vs colorímetro (aparato): deben ir juntos
letra_num = df["tono_piel_letra"].map(dict(zip("ABCDEFGHIJK", range(1, 12))))
r = df[["luminosidad_piel"]].assign(t=letra_num).corr().iloc[0, 1]
z_letra = df.groupby("tono_piel_letra")["luminosidad_piel"].transform(lambda s: (s - s.mean()) / s.std())
incongruente = z_letra.abs() > 3
df.loc[incongruente, "luminosidad_piel"] = np.nan
anotar("6. Consistencias", f"Paleta vs colorímetro: correlación r = {r:.2f} (positiva, como se espera)",
       incongruente.sum(), len(df),
       "Casos con colorímetro a >3 desv. est. de su letra: medición a NA")
print("  Luminosidad promedio por letra (debe subir de A a K):")
print("  " + df.groupby("tono_piel_letra")["luminosidad_piel"].mean().round(1).to_string().replace("\n", "\n  "))

# ===========================================================================
# PASO 7. ENCODING E INGENIERÍA DE VARIABLES
# ===========================================================================
print("\nPASO 7. Encoding")
# Ordinal: tono de piel de letra a número (A = 1 más oscuro ... K = 11 más claro)
df["tono_piel"] = letra_num
df["tono_piel_grupo"] = pd.cut(df["tono_piel"], bins=[0, 5, 7, 11],
                               labels=["Oscuro (A-E)", "Medio (F-G)", "Claro (H-K)"])
# Categórica con etiquetas
df["nivel_educ"] = df["nivel_educ_num"].map({1: "Primaria o menos", 2: "Secundaria",
                                             3: "Preparatoria", 4: "Profesional"})
# Binarias (0/1)
df["mujer"] = (df["sexo"] == "Mujer").astype(int)
df["universidad"] = (df["nivel_educ_num"] == 4).astype(int)
df["urbano"] = (raw.loc[df.index, "rururb"] == 1).astype(int)
df["lengua_indigena"] = si_no(raw.loc[df.index, "p111"])
# Derivadas: quintiles ponderados (log_ingreso_pc e indice_riqueza ya se crearon)
df["quintil_ingreso"] = quintil_ponderado(df["ingreso_pc"], df["factor"])
anotar("7. Encoding", "Tono ordinal (1-11) y en 3 grupos; binarias mujer/universidad/urbano/lengua; "
       "log del ingreso; quintiles ponderados; índice de riqueza (PCA)", 0, len(df),
       "Variables listas para tablas, pruebas y regresiones")

# ===========================================================================
# Base final: solo las variables que usa el análisis
# ===========================================================================
VARIABLES_FINALES = [
    "id", "factor", "estrato", "upm",                                  # identificación y diseño muestral
    "tono_piel_letra", "tono_piel", "tono_piel_grupo", "luminosidad_piel",   # tono de piel
    "anios_esc", "nivel_educ", "universidad", "esc_imputada",          # escolaridad
    "ingreso_pc", "log_ingreso_pc", "quintil_ingreso", "indice_riqueza",  # nivel socioeconómico
    "escalera_actual", "clase_ocup",
    "sexo", "mujer",                                                   # sexo
    "edad", "region", "urbano", "autoadscripcion", "lengua_indigena",  # controles
    "educ_padres", "escalera_14",                                      # origen familiar
]
final = df[VARIABLES_FINALES]
assert final["id"].is_unique and final["tono_piel"].between(1, 11).all()
final.to_csv(SALIDA / "emovi2023_final.csv", index=False, encoding="utf-8-sig")

bitacora = pd.DataFrame(bitacora)
bitacora.to_csv(RESULTADOS / "bitacora_limpieza.csv", index=False, encoding="utf-8-sig")
print(f"\nBase final: {final.shape[0]:,} personas x {final.shape[1]} variables "
      f"(de {N_ORIGINAL:,} x 296 originales; se conserva el {len(final) / N_ORIGINAL:.1%})")
