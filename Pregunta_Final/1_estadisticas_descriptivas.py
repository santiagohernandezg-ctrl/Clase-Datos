"""
PASO 1 - Estadísticas descriptivas que responden la pregunta
==============================================================
Pregunta 19: ¿Existe una asociación entre el tono de piel y el nivel
socioeconómico actual o la escolaridad alcanzada, y difiere entre hombres
y mujeres?

Sigue la "Guía paso a paso: Estadísticas descriptivas, comparación por
grupos y visualización":
  1A. Descriptiva general  -> variables continuas y categóricas
  1B. Separación por grupo -> Hombres vs. Mujeres (con diferencia y n)
  1C. Separación por tono  -> Oscuro / Medio / Claro, para cada sexo
  1D. Pruebas de diferencia -> ANOVA, Kruskal-Wallis, chi-cuadrada,
                               t de Welch, Mann-Whitney, z de proporciones
  1E. Regresiones          -> ¿el efecto del tono es distinto en mujeres?

Todos los promedios y porcentajes usan el factor de expansión.
Cómo correrlo (desde Pregunta_Final):  python 1_estadisticas_descriptivas.py
Resultados: resultados/1*_*.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from funciones_ponderadas import (CARPETA, ORDEN_EDUC, ORDEN_SEXO, ORDEN_TONO, cargar_base,
                                  desv_p, estrellas, media_p, mediana_p, pct_p, regresion)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
SALIDA = CARPETA / "resultados"
SALIDA.mkdir(exist_ok=True)
df = cargar_base()
W = df["factor"]


def guardar(tabla, nombre, index=True):
    tabla.to_csv(SALIDA / nombre, index=index, encoding="utf-8-sig")


# ===========================================================================
# 1A. DESCRIPTIVA GENERAL
# ===========================================================================
CONTINUAS = {
    "tono_piel": "Tono de piel (1 = A más oscuro ... 11 = K más claro)",
    "luminosidad_piel": "Luminosidad de la piel, colorímetro (más alto = más claro)",
    "anios_esc": "Años de escolaridad",
    "ingreso_pc": "Ingreso mensual por persona del hogar (pesos)",
    "indice_riqueza": "Índice de riqueza del hogar (media 0)",
    "escalera_actual": "Escalera subjetiva hoy (1-10)",
    "escalera_14": "Escalera subjetiva a los 14 años (1-10)",
    "edad": "Edad (años)",
}
filas = []
for var, desc in CONTINUAS.items():
    x = df[var]
    filas.append({"variable": var, "descripcion": desc, "n": int(x.notna().sum()),
                  "media": media_p(x, W), "mediana": mediana_p(x, W), "desv_est": desv_p(x, W),
                  "min": x.min(), "max": x.max()})
t1a = pd.DataFrame(filas)
guardar(t1a, "1a_descriptivos_continuas.csv", index=False)

CATEGORICAS = {
    "tono_piel_grupo": ORDEN_TONO,
    "nivel_educ": ORDEN_EDUC,
    "sexo": ORDEN_SEXO,
    "quintil_ingreso": [1, 2, 3, 4, 5],
    "clase_ocup": ["Agrícola", "Manual baja calificación", "Manual alta calificación", "Comercio",
                   "No manual baja calificación", "No manual alta calificación", "No trabaja"],
    "region": ["Norte", "Norte-occidente", "Centro-norte", "Centro", "Sur"],
    "urbano": [0, 1],
    "autoadscripcion": ["Indígena", "Negra", "Mestiza", "Blanca", "Ninguna"],
    "lengua_indigena": [0, 1],
    "educ_padres": ORDEN_EDUC + ["No sabe"],
}
filas = []
for var, categorias in CATEGORICAS.items():
    for cat in categorias:
        es = df[var] == cat
        filas.append({"variable": var, "categoria": cat, "n": int(es.sum()),
                      "pct_muestra": 100 * es.mean(), "pct_ponderado": pct_p(es, W)})
t1a_cat = pd.DataFrame(filas)
guardar(t1a_cat, "1a_descriptivos_categoricas.csv", index=False)

print("=" * 78)
print("1A. DESCRIPTIVA GENERAL (ponderada, n = %s personas)" % f"{len(df):,}")
print("=" * 78)
print(t1a.drop(columns="descripcion").round(2).to_string(index=False))
print("\nCategóricas (solo tono y educación; el resto en el CSV):")
print(t1a_cat[t1a_cat["variable"].isin(["tono_piel_grupo", "nivel_educ"])].round(1).to_string(index=False))


# ===========================================================================
# 1B. HOMBRES VS. MUJERES (con columna de diferencia y n por grupo)
# ===========================================================================
INDICADORES = {
    "Tono de piel (1-11)": ("tono_piel", "media"),
    "Años de escolaridad": ("anios_esc", "media"),
    "% con universidad": ("universidad", "pct"),
    "Ingreso por persona, mediana ($)": ("ingreso_pc", "mediana"),
    "Ingreso por persona, media ($)": ("ingreso_pc", "media"),
    "Índice de riqueza": ("indice_riqueza", "media"),
    "Escalera hoy (1-10)": ("escalera_actual", "media"),
    "% que no trabaja": ("clase_ocup", "no_trabaja"),
}


def calcular(g, var, tipo):
    w = g["factor"]
    if tipo == "media":
        return media_p(g[var], w)
    if tipo == "mediana":
        return mediana_p(g[var], w)
    if tipo == "pct":
        return pct_p(g[var] == 1, w)
    return pct_p(g[var] == "No trabaja", w)


h, m = df[df["sexo"] == "Hombre"], df[df["sexo"] == "Mujer"]
filas = []
for nombre, (var, tipo) in INDICADORES.items():
    vh, vm = calcular(h, var, tipo), calcular(m, var, tipo)
    filas.append({"indicador": nombre, "Hombres": vh, "Mujeres": vm, "Diferencia (M-H)": vm - vh})
t1b = pd.DataFrame(filas)
t1b.loc[len(t1b)] = ["n (personas en la muestra)", len(h), len(m), len(m) - len(h)]
guardar(t1b, "1b_hombres_vs_mujeres.csv", index=False)
print("\n" + "=" * 78)
print("1B. HOMBRES VS. MUJERES")
print("=" * 78)
print(t1b.round(2).to_string(index=False))


# ===========================================================================
# 1C. POR TONO DE PIEL, PARA CADA SEXO (la tabla que responde la pregunta)
# ===========================================================================
filas = []
for sexo, g_sexo in [("Total", df), ("Hombre", h), ("Mujer", m)]:
    for tono in ORDEN_TONO:
        g = g_sexo[g_sexo["tono_piel_grupo"] == tono]
        fila = {"sexo": sexo, "tono": tono, "n": len(g)}
        for nombre, (var, tipo) in INDICADORES.items():
            if var != "tono_piel":
                fila[nombre] = calcular(g, var, tipo)
        filas.append(fila)
t1c = pd.DataFrame(filas)

# Brecha Claro - Oscuro dentro de cada sexo
brechas = []
for sexo in ["Total", "Hombre", "Mujer"]:
    s = t1c[t1c["sexo"] == sexo].set_index("tono")
    fila = {"sexo": sexo, "tono": "BRECHA Claro - Oscuro", "n": np.nan}
    for col in s.columns.drop(["sexo", "n"]):
        fila[col] = s.loc["Claro (H-K)", col] - s.loc["Oscuro (A-E)", col]
    brechas.append(fila)
t1c = pd.concat([t1c, pd.DataFrame(brechas)], ignore_index=True)
guardar(t1c, "1c_por_tono_y_sexo.csv", index=False)
print("\n" + "=" * 78)
print("1C. RESULTADOS POR TONO DE PIEL Y SEXO (+ brecha Claro - Oscuro)")
print("=" * 78)
cols = ["sexo", "tono", "n", "Años de escolaridad", "% con universidad",
        "Ingreso por persona, mediana ($)", "Índice de riqueza", "Escalera hoy (1-10)"]
print(t1c[cols].round(2).to_string(index=False))


# ===========================================================================
# 1D. PRUEBAS DE DIFERENCIA (estadístico, p-valor, n)
# ===========================================================================
# Nota honesta: estas pruebas clásicas (scipy) usan la muestra SIN ponderar.
# La prueba ponderada y con diseño muestral es la regresión de la sección 1E.
pruebas = []


def agregar(pregunta, grupo, prueba, estadistico, p, n, nota=""):
    pruebas.append({"pregunta": pregunta, "grupo": grupo, "prueba": prueba,
                    "estadistico": estadistico, "p_valor": p, "n": n, "nota": nota})


for sexo, g in [("Total", df), ("Hombre", h), ("Mujer", m)]:
    grupos_esc = [g.loc[g["tono_piel_grupo"] == t, "anios_esc"] for t in ORDEN_TONO]
    grupos_ing = [g.loc[g["tono_piel_grupo"] == t, "ingreso_pc"] for t in ORDEN_TONO]
    grupos_riq = [g.loc[g["tono_piel_grupo"] == t, "indice_riqueza"] for t in ORDEN_TONO]

    # Supuesto de ANOVA: varianzas iguales -> prueba de Levene
    lev = stats.levene(*grupos_esc)
    agregar("Supuesto: ¿varianzas iguales en escolaridad entre tonos?", sexo, "Levene",
            lev.statistic, lev.pvalue, len(g), "p<0.05 = varianzas distintas -> se usa también Kruskal-Wallis")
    an = stats.f_oneway(*grupos_esc)
    agregar("¿Escolaridad promedio distinta entre los 3 tonos?", sexo, "ANOVA de un factor",
            an.statistic, an.pvalue, len(g))
    kw = stats.kruskal(*grupos_esc)
    agregar("¿Escolaridad distinta entre los 3 tonos? (sin suponer normalidad)", sexo, "Kruskal-Wallis",
            kw.statistic, kw.pvalue, len(g))
    kw = stats.kruskal(*grupos_ing)
    agregar("¿Ingreso distinto entre los 3 tonos?", sexo, "Kruskal-Wallis",
            kw.statistic, kw.pvalue, len(g), "ingreso muy sesgado -> no paramétrica")
    an = stats.f_oneway(*grupos_riq)
    agregar("¿Índice de riqueza distinto entre los 3 tonos?", sexo, "ANOVA de un factor",
            an.statistic, an.pvalue, len(g))

    for var, nombre in [("nivel_educ", "nivel educativo"), ("quintil_ingreso", "quintil de ingreso")]:
        tabla = pd.crosstab(g["tono_piel_grupo"], g[var])
        chi = stats.chi2_contingency(tabla)
        v_cramer = np.sqrt(chi.statistic / (tabla.values.sum() * (min(tabla.shape) - 1)))
        agregar(f"¿El {nombre} depende del tono?", sexo, "Chi-cuadrada de independencia",
                chi.statistic, chi.pvalue, int(tabla.values.sum()),
                f"V de Cramér = {v_cramer:.3f} (0 = nada, 0.1 = débil, 0.3 = moderada)")

    # z de proporciones: % con universidad, Claro vs. Oscuro
    cl, os_ = g[g["tono_piel_grupo"] == ORDEN_TONO[2]], g[g["tono_piel_grupo"] == ORDEN_TONO[0]]
    z, p = proportions_ztest([cl["universidad"].sum(), os_["universidad"].sum()], [len(cl), len(os_)])
    agregar("¿% con universidad distinto entre Claro y Oscuro?", sexo, "z de dos proporciones",
            z, p, len(cl) + len(os_),
            f"Claro {100 * cl['universidad'].mean():.1f}% vs Oscuro {100 * os_['universidad'].mean():.1f}% (sin ponderar)")

# Hombres vs. mujeres
t = stats.ttest_ind(m["anios_esc"], h["anios_esc"], equal_var=False)
agregar("¿Escolaridad distinta entre hombres y mujeres?", "Total", "t de Welch", t.statistic, t.pvalue, len(df))
u = stats.mannwhitneyu(m["ingreso_pc"], h["ingreso_pc"])
agregar("¿Ingreso distinto entre hombres y mujeres?", "Total", "Mann-Whitney U", u.statistic, u.pvalue, len(df))
t = stats.ttest_ind(m["tono_piel"], h["tono_piel"], equal_var=False)
agregar("¿Las mujeres fueron clasificadas en tonos distintos a los hombres?", "Total", "t de Welch",
        t.statistic, t.pvalue, len(df), "si sí, hay que comparar dentro de cada sexo")

# Supuesto de normalidad (con n tan grande casi todo sale 'no normal'; se reporta el sesgo)
for var in ["anios_esc", "ingreso_pc", "log_ingreso_pc"]:
    agregar(f"Supuesto: ¿{var} es simétrica?", "Total", "Sesgo (skewness)",
            stats.skew(df[var]), np.nan, len(df), "0 = simétrica; >1 = cola larga a la derecha")

t1d = pd.DataFrame(pruebas)
t1d["significativo_5pct"] = np.where(t1d["p_valor"] < 0.05, "Sí", np.where(t1d["p_valor"].isna(), "", "No"))
guardar(t1d, "1d_pruebas_de_diferencia.csv", index=False)
print("\n" + "=" * 78)
print("1D. PRUEBAS DE DIFERENCIA")
print("=" * 78)
print(t1d[["pregunta", "grupo", "prueba", "estadistico", "p_valor", "n"]]
      .assign(p_valor=lambda d: d["p_valor"].map(lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")))
      .round(2).to_string(index=False))


# ===========================================================================
# 1E. REGRESIONES: ¿el efecto del tono es distinto en hombres y mujeres?
# ===========================================================================
# resultado = b0 + b1*tono + b2*mujer + b3*(tono x mujer) + controles
#   b1       = cambio por cada letra más clara, en HOMBRES
#   b1 + b3  = cambio por cada letra más clara, en MUJERES
#   b3       = diferencia entre ambos (si p > 0.05 -> no hay evidencia de que difiera)
# Modelo A: controles básicos (edad, urbano, región)
# Modelo B: + origen familiar (educación de los padres y escalera a los 14 años),
#           para ver cuánto del efecto del tono viene de la familia de origen.
RESULTADOS = {
    "anios_esc": "Años de escolaridad",
    "universidad": "Prob. de tener universidad (x100 = puntos %)",
    "log_ingreso_pc": "Log ingreso por persona (x100 ≈ % de cambio)",
    "indice_riqueza": "Índice de riqueza (desv. est.)",
}
CONTROLES = {
    "A: edad, urbano, región": "edad + I(edad**2) + urbano + C(region)",
    "B: A + origen familiar": "edad + I(edad**2) + urbano + C(region) + C(educ_padres) + escalera_14",
}
filas = []
for var, desc in RESULTADOS.items():
    for nombre_mod, controles in CONTROLES.items():
        r = regresion(f"{var} ~ tono_piel * mujer + {controles}", df)
        efecto_m = r.t_test("tono_piel + tono_piel:mujer = 0")
        filas.append({
            "resultado": desc, "modelo": nombre_mod, "n": int(r.nobs),
            "efecto_hombres": r.params["tono_piel"], "p_hombres": r.pvalues["tono_piel"],
            "efecto_mujeres": float(efecto_m.effect[0]), "p_mujeres": float(efecto_m.pvalue),
            "diferencia_M_H": r.params["tono_piel:mujer"], "p_diferencia": r.pvalues["tono_piel:mujer"],
            "R2": r.rsquared,
        })
t1e = pd.DataFrame(filas)
guardar(t1e, "1e_regresiones_tono_por_sexo.csv", index=False)

print("\n" + "=" * 78)
print("1E. REGRESIONES: efecto de UNA LETRA MÁS CLARA (ponderado, EE por conglomerado)")
print("=" * 78)
for _, f in t1e.iterrows():
    print(f"\n{f['resultado']}  |  Modelo {f['modelo']}  |  n = {f['n']:,}  |  R² = {f['R2']:.3f}")
    print(f"   Hombres: {f['efecto_hombres']:+.3f}{estrellas(f['p_hombres']):<3}   "
          f"Mujeres: {f['efecto_mujeres']:+.3f}{estrellas(f['p_mujeres']):<3}   "
          f"Diferencia M-H: {f['diferencia_M_H']:+.3f}{estrellas(f['p_diferencia'])} (p = {f['p_diferencia']:.3f})")
print("\n*** p<0.01   ** p<0.05   * p<0.10")
print("\nListo. Tablas guardadas en resultados/1a ... 1e")
