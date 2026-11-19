"""
PASO 2 - Patrones adicionales (análisis exploratorio)
=======================================================
El Paso 1 respondió la pregunta directa. Aquí buscamos patrones que la
explican o la complican, conectados con teoría económica:

  2A. Correlaciones        -> mapa general de cómo se relaciona todo
  2B. Origen familiar      -> ¿la brecha se hereda? (movilidad social)
  2C. Rendimiento de la educación -> con la MISMA escolaridad, ¿ganan
                              distinto? (capital humano vs. discriminación)
  2D. Ocupación            -> ¿a qué trabajos llega cada tono y sexo?
  2E. Percepción vs. realidad -> ¿la gente se percibe como lo que tiene?
  2F. Región y localidad   -> ¿la brecha es igual en todo el país?
  2G. Identidad étnica     -> ¿es el tono o es ser indígena?
  2H. Robustez             -> ¿el resultado se sostiene con el colorímetro
                              y sin los datos imputados?

Cómo correrlo (desde Pregunta_Final):  python 2_patrones_exploratorios.py
Resultados: resultados/2*_*.csv
"""

import numpy as np
import pandas as pd

from funciones_ponderadas import (CARPETA, ORDEN_EDUC, ORDEN_TONO, cargar_base, corr_p, estrellas,
                                  media_p, mediana_p, pct_p, regresion)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
SALIDA = CARPETA / "resultados"
df = cargar_base()
W = df["factor"]
CTRL = "edad + I(edad**2) + urbano + C(region)"
ORIGEN = "C(educ_padres) + escalera_14"


def titulo(texto):
    print("\n" + "=" * 78 + "\n" + texto + "\n" + "=" * 78)


def guardar(tabla, nombre, index=True):
    tabla.to_csv(SALIDA / nombre, index=index, encoding="utf-8-sig")


def efecto(r, var="tono_piel"):
    return f"{r.params[var]:+.3f}{estrellas(r.pvalues[var])} (p={r.pvalues[var]:.3f})"


# ===========================================================================
# 2A. MATRIZ DE CORRELACIONES (ponderada)
# ===========================================================================
VARS = ["tono_piel", "luminosidad_piel", "anios_esc", "log_ingreso_pc", "indice_riqueza",
        "escalera_actual", "escalera_14", "edad", "urbano", "lengua_indigena", "mujer"]
matriz = pd.DataFrame({a: {b: corr_p(df[a], df[b], W) for b in VARS} for a in VARS})
guardar(matriz.round(3), "2a_matriz_correlaciones.csv")
titulo("2A. CORRELACIÓN DEL TONO DE PIEL CON CADA VARIABLE (ponderada)")
print(matriz["tono_piel"].drop("tono_piel").round(3).to_string())


# ===========================================================================
# 2B. ORIGEN FAMILIAR: ¿la brecha se hereda?
# ===========================================================================
filas = []
for tono in ORDEN_TONO:
    g = df[df["tono_piel_grupo"] == tono]
    filas.append({"tono": tono, "n": len(g),
                  "escalera_a_los_14": media_p(g["escalera_14"], g["factor"]),
                  "% padres con primaria o menos": pct_p(g["educ_padres"] == "Primaria o menos", g["factor"]),
                  "% padres con universidad": pct_p(g["educ_padres"] == "Profesional", g["factor"])})
t2b = pd.DataFrame(filas)

# ¿Qué parte del efecto del tono desaparece al controlar por el origen familiar?
filas = []
for var, nombre in [("anios_esc", "Años de escolaridad"), ("universidad", "Universidad"),
                    ("log_ingreso_pc", "Log ingreso"), ("indice_riqueza", "Índice de riqueza")]:
    a = regresion(f"{var} ~ tono_piel + mujer + {CTRL}", df).params["tono_piel"]
    b = regresion(f"{var} ~ tono_piel + mujer + {CTRL} + {ORIGEN}", df)
    filas.append({"resultado": nombre, "efecto_sin_origen": a, "efecto_con_origen": b.params["tono_piel"],
                  "p_con_origen": b.pvalues["tono_piel"], "% explicado por origen": 100 * (1 - b.params["tono_piel"] / a)})
t2b_dec = pd.DataFrame(filas)
guardar(t2b, "2b_origen_por_tono.csv", index=False)
guardar(t2b_dec, "2b_cuanto_explica_el_origen.csv", index=False)
titulo("2B. ORIGEN FAMILIAR POR TONO")
print(t2b.round(2).to_string(index=False))
print("\n¿Cuánto del efecto del tono se explica por la familia de origen?")
print(t2b_dec.round(3).to_string(index=False))


# ===========================================================================
# 2C. RENDIMIENTO DE LA EDUCACIÓN: misma escolaridad, ¿mismo ingreso?
# ===========================================================================
filas = []
for sexo in ["Hombre", "Mujer"]:
    for educ in ORDEN_EDUC:
        fila = {"sexo": sexo, "nivel_educ": educ}
        for tono in ORDEN_TONO:
            g = df[(df["sexo"] == sexo) & (df["nivel_educ"] == educ) & (df["tono_piel_grupo"] == tono)]
            fila[f"mediana_{tono}"] = mediana_p(g["ingreso_pc"], g["factor"])
            fila[f"n_{tono}"] = len(g)
        fila["brecha_% Claro vs Oscuro"] = 100 * (fila["mediana_Claro (H-K)"] / fila["mediana_Oscuro (A-E)"] - 1)
        filas.append(fila)
t2c = pd.DataFrame(filas)
guardar(t2c, "2c_ingreso_por_educacion_y_tono.csv", index=False)

titulo("2C. INGRESO MEDIANO POR NIVEL EDUCATIVO Y TONO (misma escolaridad)")
print(t2c[["sexo", "nivel_educ", "mediana_Oscuro (A-E)", "mediana_Claro (H-K)", "n_Oscuro (A-E)",
           "n_Claro (H-K)", "brecha_% Claro vs Oscuro"]].round(0).to_string(index=False))

filas = []
for sexo in ["Hombre", "Mujer"]:
    d = df[df["sexo"] == sexo]
    sin = regresion(f"log_ingreso_pc ~ tono_piel + {CTRL}", d)
    con_esc = regresion(f"log_ingreso_pc ~ tono_piel + anios_esc + {CTRL}", d)
    con_todo = regresion(f"log_ingreso_pc ~ tono_piel + anios_esc + {CTRL} + {ORIGEN}", d)
    filas.append({"sexo": sexo, "n": int(sin.nobs),
                  "efecto_tono_sin_escolaridad": sin.params["tono_piel"],
                  "efecto_tono_con_escolaridad": con_esc.params["tono_piel"],
                  "p_con_escolaridad": con_esc.pvalues["tono_piel"],
                  "efecto_tono_con_escolaridad_y_origen": con_todo.params["tono_piel"],
                  "p_con_todo": con_todo.pvalues["tono_piel"],
                  "rendimiento_1_anio_escuela": con_esc.params["anios_esc"]})
t2c_reg = pd.DataFrame(filas)
guardar(t2c_reg, "2c_efecto_tono_en_ingreso_controlando_educacion.csv", index=False)
print("\nEfecto de una letra más clara sobre log ingreso (x100 ≈ %):")
print(t2c_reg.round(3).to_string(index=False))


# ===========================================================================
# 2D. OCUPACIÓN: ¿a qué trabajos llega cada tono?
# ===========================================================================
CLASES = ["Agrícola", "Manual baja calificación", "Manual alta calificación", "Comercio",
          "No manual baja calificación", "No manual alta calificación", "No trabaja"]
filas = []
for sexo in ["Hombre", "Mujer"]:
    for tono in ORDEN_TONO:
        g = df[(df["sexo"] == sexo) & (df["tono_piel_grupo"] == tono) & df["clase_ocup"].notna()]
        fila = {"sexo": sexo, "tono": tono, "n": len(g)}
        for c in CLASES:
            fila[c] = pct_p(g["clase_ocup"] == c, g["factor"])
        filas.append(fila)
t2d = pd.DataFrame(filas)
guardar(t2d, "2d_ocupacion_por_tono_y_sexo.csv", index=False)
titulo("2D. CLASE OCUPACIONAL (% ponderado) POR TONO Y SEXO")
print(t2d.round(1).to_string(index=False))

trab = df[df["clase_ocup"].notna() & (df["clase_ocup"] != "No trabaja")].copy()
trab["no_manual_alta"] = (trab["clase_ocup"] == "No manual alta calificación").astype(int)
r = regresion(f"no_manual_alta ~ tono_piel * mujer + anios_esc + {CTRL}", trab)
print(f"\nEntre quienes trabajan (n = {int(r.nobs):,}), con la MISMA escolaridad, una letra más clara cambia la")
print(f"probabilidad de un puesto no manual de alta calificación en: hombres {efecto(r)};"
      f" diferencia mujeres {efecto(r, 'tono_piel:mujer')}")


# ===========================================================================
# 2E. PERCEPCIÓN VS. REALIDAD
# ===========================================================================
filas = []
for q in range(1, 6):
    fila = {"quintil_ingreso": q}
    for tono in ORDEN_TONO:
        g = df[(df["quintil_ingreso"] == q) & (df["tono_piel_grupo"] == tono)]
        fila[f"escalera_{tono}"] = media_p(g["escalera_actual"], g["factor"])
        fila[f"n_{tono}"] = len(g)
    filas.append(fila)
t2e = pd.DataFrame(filas)
guardar(t2e, "2e_percepcion_vs_realidad.csv", index=False)
titulo("2E. PERCEPCIÓN (escalera hoy 1-10) DENTRO DE CADA QUINTIL DE INGRESO")
print(t2e.round(2).to_string(index=False))
r = regresion(f"escalera_actual ~ tono_piel + log_ingreso_pc + indice_riqueza + anios_esc + mujer + {CTRL}", df)
print(f"\nCon el mismo ingreso, riqueza y escolaridad, una letra más clara cambia la escalera en {efecto(r)}")
print(f"Correlación ponderada escalera hoy vs. índice de riqueza: {corr_p(df['escalera_actual'], df['indice_riqueza'], W):.3f}")


# ===========================================================================
# 2F. REGIÓN Y LOCALIDAD
# ===========================================================================
filas = []
for col, valores in [("region", ["Norte", "Norte-occidente", "Centro-norte", "Centro", "Sur"]), ("urbano", [1, 0])]:
    for v in valores:
        d = df[df[col] == v]
        ctrl = "edad + I(edad**2) + urbano" if col == "region" else "edad + I(edad**2) + C(region)"
        r_esc = regresion(f"anios_esc ~ tono_piel + mujer + {ctrl}", d)
        r_ing = regresion(f"log_ingreso_pc ~ tono_piel + mujer + {ctrl}", d)
        filas.append({"grupo": f"{col} = {v}" if col == "urbano" else v, "n": len(d),
                      "tono_promedio": media_p(d["tono_piel"], d["factor"]),
                      "% tono oscuro (A-E)": pct_p(d["tono_piel_grupo"] == ORDEN_TONO[0], d["factor"]),
                      "efecto_tono_escolaridad": r_esc.params["tono_piel"], "p_esc": r_esc.pvalues["tono_piel"],
                      "efecto_tono_log_ingreso": r_ing.params["tono_piel"], "p_ing": r_ing.pvalues["tono_piel"]})
t2f = pd.DataFrame(filas).replace({"urbano = 1": "Urbano", "urbano = 0": "Rural"})
guardar(t2f, "2f_brecha_por_region.csv", index=False)
titulo("2F. COMPOSICIÓN Y EFECTO DEL TONO POR REGIÓN Y LOCALIDAD")
print(t2f.round(3).to_string(index=False))


# ===========================================================================
# 2G. IDENTIDAD ÉTNICA: ¿es el tono o es ser indígena?
# ===========================================================================
filas = []
for a in ["Indígena", "Negra", "Mestiza", "Blanca", "Ninguna"]:
    g = df[df["autoadscripcion"] == a]
    filas.append({"autoadscripcion": a, "n": len(g), "tono_promedio": media_p(g["tono_piel"], g["factor"]),
                  "% oscuro (A-E)": pct_p(g["tono_piel_grupo"] == ORDEN_TONO[0], g["factor"]),
                  "% claro (H-K)": pct_p(g["tono_piel_grupo"] == ORDEN_TONO[2], g["factor"]),
                  "% habla lengua indígena": pct_p(g["lengua_indigena"] == 1, g["factor"])})
t2g = pd.DataFrame(filas)
guardar(t2g, "2g_tono_por_autoadscripcion.csv", index=False)
titulo("2G. TONO DE PIEL SEGÚN CÓMO SE IDENTIFICA LA PERSONA")
print(t2g.round(1).to_string(index=False))
filas = []
for var in ["anios_esc", "log_ingreso_pc"]:
    a = regresion(f"{var} ~ tono_piel + mujer + {CTRL}", df)
    b = regresion(f"{var} ~ tono_piel + mujer + {CTRL} + lengua_indigena + C(autoadscripcion)", df)
    filas.append({"resultado": var, "efecto_tono_sin_etnicidad": a.params["tono_piel"],
                  "efecto_tono_con_etnicidad": b.params["tono_piel"], "p_con_etnicidad": b.pvalues["tono_piel"],
                  "efecto_lengua_indigena": b.params["lengua_indigena"], "p_lengua": b.pvalues["lengua_indigena"]})
t2g_reg = pd.DataFrame(filas)
guardar(t2g_reg, "2g_tono_vs_etnicidad.csv", index=False)
print("\n¿El efecto del tono sobrevive al controlar por lengua indígena y autoadscripción?")
print(t2g_reg.round(3).to_string(index=False))


# ===========================================================================
# 2H. ROBUSTEZ
# ===========================================================================
df["lum_z"] = (df["luminosidad_piel"] - df["luminosidad_piel"].mean()) / df["luminosidad_piel"].std()
df["tono_z"] = (df["tono_piel"] - df["tono_piel"].mean()) / df["tono_piel"].std()
filas = []
for var in ["anios_esc", "log_ingreso_pc", "indice_riqueza"]:
    for nombre, medida, datos in [("Paleta (interpretado por entrevistador)", "tono_z", df),
                                  ("Colorímetro (máquina)", "lum_z", df),
                                  ("Paleta, sin escolaridad imputada", "tono_z", df[df["esc_imputada"] == 0])]:
        r = regresion(f"{var} ~ {medida} * mujer + {CTRL}", datos)
        filas.append({"resultado": var, "medida": nombre, "n": int(r.nobs),
                      "efecto_1_desv_est_hombres": r.params[medida], "p": r.pvalues[medida],
                      "diferencia_mujeres": r.params[f"{medida}:mujer"], "p_dif": r.pvalues[f"{medida}:mujer"]})
t2h = pd.DataFrame(filas)
guardar(t2h, "2h_robustez.csv", index=False)
titulo("2H. ROBUSTEZ: efecto de 1 desviación estándar más clara, con distintas medidas")
print(t2h.round(3).to_string(index=False))
print("\nListo. Tablas guardadas en resultados/2a ... 2h")
