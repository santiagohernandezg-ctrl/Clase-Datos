"""
PASO 3 - Visualizaciones estratégicas
=======================================
Cada gráfica tiene UN mensaje (va en el título), ejes con unidades, n y fuente,
como pide la guía. Reglas que seguimos:
  - Barras siempre empiezan en 0 (no se trunca el eje Y para exagerar).
  - Color con propósito: azul = hombres, naranja = mujeres, en TODAS las gráficas.
  - Gris = referencia o contexto.
  - Paleta revisada para daltonismo.

Requiere correr antes los pasos 1 y 2 (lee sus tablas de resultados/).
Cómo correrlo (desde Pregunta_Final):  python 3_visualizaciones.py
Resultados: graficas/fig1 ... fig8 (.png)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funciones_ponderadas import CARPETA, ORDEN_EDUC, ORDEN_TONO, cargar_base, pct_p, regresion

RES = CARPETA / "resultados"
SALIDA = CARPETA / "graficas"
SALIDA.mkdir(exist_ok=True)
df = cargar_base()

# ---------------------------------------------------------------------------
# Estilo común
# ---------------------------------------------------------------------------
AZUL, NARANJA = "#2a78d6", "#eb6834"          # hombres, mujeres
COLOR_SEXO = {"Hombre": AZUL, "Mujer": NARANJA}
TINTA, TINTA2, GRIS, REJILLA, EJE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONDO = "#fcfcfb"
TONO_CORTO = {"Oscuro (A-E)": "Oscuro\n(A-E)", "Medio (F-G)": "Medio\n(F-G)", "Claro (H-K)": "Claro\n(H-K)"}
FUENTE = "Fuente: ESRU-EMOVI 2023, personas de 25 a 64 años. Ponderado con factor de expansión."
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": REJILLA, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "ytick.left": False, "legend.frameon": False,
})


def encabezado(fig, titulo, subtitulo):
    fig.text(0.04, 0.965, titulo, fontsize=15, fontweight="bold", color=TINTA, va="top")
    fig.text(0.04, 0.905, subtitulo, fontsize=11, color=TINTA2, va="top")


def pie(fig, nota=""):
    fig.text(0.04, 0.02, (nota + "\n" if nota else "") + FUENTE, fontsize=8.5, color=GRIS, va="bottom")


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=200, facecolor=FONDO)
    plt.close(fig)
    print("  ✓", nombre)


def barras_por_sexo(ax, tabla, columna, formato):
    """Barras agrupadas: 3 grupos de tono x 2 sexos, con etiqueta de valor."""
    x = np.arange(len(ORDEN_TONO))
    ancho = 0.36
    for i, sexo in enumerate(["Hombre", "Mujer"]):
        vals = tabla[tabla["sexo"] == sexo].set_index("tono").loc[ORDEN_TONO, columna]
        pos = x + (i - 0.5) * (ancho + 0.03)
        ax.bar(pos, vals, ancho, color=COLOR_SEXO[sexo], label={"Hombre": "Hombres", "Mujer": "Mujeres"}[sexo], zorder=2)
        for p, v in zip(pos, vals):
            ax.text(p, v, formato(v), ha="center", va="bottom", fontsize=10, color=TINTA, fontweight="bold")
    ns = tabla[tabla["sexo"].isin(["Hombre", "Mujer"])].pivot(index="tono", columns="sexo", values="n")
    ax.set_xticks(x, [f"{TONO_CORTO[t]}\nn = {int(ns.loc[t, 'Hombre']):,} H · {int(ns.loc[t, 'Mujer']):,} M"
                      for t in ORDEN_TONO])
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", ncol=2, fontsize=10)


print("Generando gráficas en graficas/ ...")
t1c = pd.read_csv(RES / "1c_por_tono_y_sexo.csv")
t1c = t1c[t1c["tono"].isin(ORDEN_TONO)]

# ===========================================================================
# FIG 1. Universidad por tono y sexo
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6.2))
fig.subplots_adjust(top=0.80, bottom=0.22, left=0.08, right=0.97)
barras_por_sexo(ax, t1c, "% con universidad", lambda v: f"{v:.0f}%")
ax.set_ylim(0, 40)
ax.set_ylabel("% con estudios profesionales terminados")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
encabezado(fig, "Con piel más clara, el doble de probabilidad de tener universidad",
           "% con nivel profesional terminado, por tono de piel (paleta PERLA) y sexo")
pie(fig, "Diferencias Claro vs. Oscuro significativas en ambos sexos (z de proporciones, p < 0.001).")
guardar(fig, "fig1_universidad_por_tono_y_sexo.png")

# ===========================================================================
# FIG 2. Ingreso por tono y sexo
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6.2))
fig.subplots_adjust(top=0.80, bottom=0.22, left=0.10, right=0.97)
barras_por_sexo(ax, t1c, "Ingreso por persona, mediana ($)", lambda v: f"${v:,.0f}")
ax.set_ylim(0, 8000)
ax.set_ylabel("Pesos al mes por persona del hogar (mediana)")
ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
encabezado(fig, "Los hogares de personas de piel clara tienen más ingreso",
           "Ingreso mensual por persona del hogar (mediana), por tono de piel y sexo")
pie(fig, "Mediana porque el ingreso tiene cola larga a la derecha. Diferencias significativas (Kruskal-Wallis, p < 0.001).")
guardar(fig, "fig2_ingreso_por_tono_y_sexo.png")

# ===========================================================================
# FIG 3. ¿El efecto del tono difiere entre hombres y mujeres? (regresiones)
# ===========================================================================
CTRL = "edad + I(edad**2) + urbano + C(region)"
PANELES = [("anios_esc", "Años de escolaridad", 1, "años"),
           ("universidad", "Tener universidad", 100, "puntos %"),
           ("log_ingreso_pc", "Ingreso por persona", 100, "% aprox."),
           ("indice_riqueza", "Índice de riqueza", 1, "desv. est.")]
fig, axes = plt.subplots(1, 4, figsize=(13, 6))
fig.subplots_adjust(top=0.74, bottom=0.27, left=0.04, right=0.98, wspace=0.35)
for ax, (var, nombre, escala, unidad) in zip(axes, PANELES):
    r = regresion(f"{var} ~ tono_piel * mujer + {CTRL}", df)
    efectos = {"Hombre": r.t_test("tono_piel = 0"), "Mujer": r.t_test("tono_piel + tono_piel:mujer = 0")}
    for i, (sexo, tt) in enumerate(efectos.items()):
        est = float(tt.effect[0]) * escala
        lo, hi = (np.asarray(tt.conf_int()).ravel() * escala)
        ax.plot([i, i], [lo, hi], color=COLOR_SEXO[sexo], lw=2.5, solid_capstyle="round", zorder=2)
        ax.plot(i, est, "o", ms=10, color=COLOR_SEXO[sexo], mec=FONDO, mew=2, zorder=3)
        ax.text(i + 0.12, est, f"{est:+.2f}", va="center", fontsize=10, color=TINTA, fontweight="bold")
    ax.axhline(0, color=EJE, lw=1.2, zorder=1)
    ax.set_xlim(-0.5, 1.7)
    ax.set_xticks([0, 1], ["Hombres", "Mujeres"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.1)
    p = r.pvalues["tono_piel:mujer"]
    ax.set_title(f"{nombre}\n({unidad})", fontsize=11, color=TINTA, loc="left")
    ax.text(0.0, -0.13, f"¿Diferente? p = {p:.2f}\n" + ("Sí (al 10%)" if p < 0.10 else "No"),
            transform=ax.transAxes, fontsize=9.5, color=TINTA2, va="top")
encabezado(fig, "El tono pesa casi igual en hombres y mujeres; en ingreso, algo más en ellos (evidencia débil)",
           "Cambio por cada letra más clara en la paleta (punto) e intervalo de confianza al 95% (línea)")
pie(fig, "Regresión ponderada con interacción tono × mujer; controla edad, urbano/rural y región. Errores estándar por conglomerado.")
guardar(fig, "fig3_efecto_del_tono_por_sexo.png")

# ===========================================================================
# FIG 4. ¿Cuánto se hereda? (origen familiar)
# ===========================================================================
t2b = pd.read_csv(RES / "2b_cuanto_explica_el_origen.csv").replace({"Log ingreso": "Ingreso por persona"})
fig, ax = plt.subplots(figsize=(10, 5.2))
fig.subplots_adjust(top=0.76, bottom=0.17, left=0.20, right=0.95)
y = np.arange(len(t2b))[::-1]
expl = t2b["% explicado por origen"].values
ax.barh(y, expl, 0.55, color=GRIS, zorder=2, label="Se explica por la familia de origen")
ax.barh(y, 100 - expl, 0.55, left=expl + 0.6, color=AZUL, zorder=2, label="Persiste con el mismo origen")
for yi, e in zip(y, expl):
    ax.text(e / 2, yi, f"{e:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(e + (100 - e) / 2, yi, f"{100 - e:.0f}%", ha="center", va="center", color="white", fontweight="bold")
ax.set_yticks(y, t2b["resultado"])
ax.set_xlim(0, 101)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10)
encabezado(fig, "Cerca de la mitad de la ventaja de la piel clara viene de la familia de origen",
           "Parte del efecto del tono que desaparece al controlar educación de los padres y escalera a los 14 años")
pie(fig, "Efecto con origen sigue siendo significativo en los 4 casos (p < 0.01). Mismos controles que la Fig. 3.")
guardar(fig, "fig4_cuanto_viene_del_origen.png")

# ===========================================================================
# FIG 5. Misma escolaridad, ¿mismo ingreso?
# ===========================================================================
t2c = pd.read_csv(RES / "2c_ingreso_por_educacion_y_tono.csv")
fig, ax = plt.subplots(figsize=(11.5, 6.2))
fig.subplots_adjust(top=0.78, bottom=0.2, left=0.09, right=0.97)
x = np.arange(len(ORDEN_EDUC))
for i, sexo in enumerate(["Hombre", "Mujer"]):
    vals = t2c[t2c["sexo"] == sexo].set_index("nivel_educ").loc[ORDEN_EDUC, "brecha_% Claro vs Oscuro"]
    pos = x + (i - 0.5) * 0.39
    ax.bar(pos, vals, 0.36, color=COLOR_SEXO[sexo], label={"Hombre": "Hombres", "Mujer": "Mujeres"}[sexo], zorder=2)
    for p_, v in zip(pos, vals):
        ax.text(p_, v + (0.4 if v >= 0 else -0.4), f"{v:+.0f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10, color=TINTA, fontweight="bold")
ax.axhline(0, color=TINTA2, lw=1.2, zorder=3)
ax.set_xticks(x, ORDEN_EDUC)
ax.tick_params(axis="x", length=0)
ax.set_ylim(-10, 20)
ax.set_ylabel("Ingreso de Claros vs. Oscuros (%)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
ax.legend(loc="upper left", ncol=2, fontsize=10)
encabezado(fig, "Con igual escolaridad, los hombres claros siempre ganan más; en mujeres no hay patrón",
           "Diferencia en ingreso mediano por persona: piel clara (H-K) vs. oscura (A-E), dentro de cada nivel educativo")
pie(fig, "Regresión con escolaridad: una letra más clara = +3.0% de ingreso en hombres (p < 0.001) y +0.4% en mujeres (p = 0.60).\n"
         "Ojo: el ingreso es del hogar; 50% de las mujeres no trabaja, así que su ingreso depende también de su pareja.")
guardar(fig, "fig5_mismo_estudio_distinto_ingreso.png")

# ===========================================================================
# FIG 6. Ocupación
# ===========================================================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.8))
fig.subplots_adjust(top=0.76, bottom=0.2, left=0.06, right=0.98, wspace=0.25)
d = df[df["clase_ocup"].notna()]
for ax, titulo, base_fn, cond_fn, lim in [
        (a1, "Puestos no manuales de alta calificación\n(% de quienes trabajan)",
         lambda g: g[g["clase_ocup"] != "No trabaja"], lambda g: g["clase_ocup"] == "No manual alta calificación", 20),
        (a2, "No trabaja\n(% del total)", lambda g: g, lambda g: g["clase_ocup"] == "No trabaja", 80)]:
    x = np.arange(3)
    for i, sexo in enumerate(["Hombre", "Mujer"]):
        vals = []
        for t in ORDEN_TONO:
            g = base_fn(d[(d["sexo"] == sexo) & (d["tono_piel_grupo"] == t)])
            vals.append(pct_p(cond_fn(g), g["factor"]))
        pos = x + (i - 0.5) * 0.39
        ax.bar(pos, vals, 0.36, color=COLOR_SEXO[sexo], label={"Hombre": "Hombres", "Mujer": "Mujeres"}[sexo], zorder=2)
        for p_, v in zip(pos, vals):
            ax.text(p_, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color=TINTA, fontweight="bold")
    ax.set_xticks(x, [TONO_CORTO[t] for t in ORDEN_TONO])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, lim)
    ax.set_yticks(range(0, lim + 1, 5 if lim <= 20 else 10))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(titulo, fontsize=11, color=TINTA, loc="left")
a1.legend(loc="upper left", fontsize=10)
encabezado(fig, "Los hombres claros llegan más a puestos altos; las mujeres oscuras, más fuera del trabajo",
           "Clase ocupacional actual por tono de piel y sexo")
pie(fig, f"n = {len(d):,} (sin 55 personas que trabajan pero no tienen clase asignada). Profesionistas, directivos y técnicos = no manual alta calificación.")
guardar(fig, "fig6_ocupacion_por_tono_y_sexo.png")

# ===========================================================================
# FIG 7. Percepción vs. realidad
# ===========================================================================
t2e = pd.read_csv(RES / "2e_percepcion_vs_realidad.csv")
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(top=0.76, bottom=0.24, left=0.11, right=0.84)
estilos = {"Oscuro (A-E)": ("#52514e", "o", "-"), "Medio (F-G)": (EJE, "o", "-"), "Claro (H-K)": ("#1baf7a", "o", "-")}
for tono, (color, marca, linea) in estilos.items():
    yv = t2e[f"escalera_{tono}"]
    ax.plot(t2e["quintil_ingreso"], yv, linea, marker=marca, color=color, lw=2, ms=8, mec=FONDO, mew=2, zorder=3)
    desfase = {"Oscuro (A-E)": 0.12, "Medio (F-G)": -0.08, "Claro (H-K)": -0.04}[tono]
    ax.text(5.12, yv.iloc[-1] + desfase, tono, va="center", fontsize=10, color=TINTA, fontweight="bold" if tono != "Medio (F-G)" else "normal")
ax.set_xticks(range(1, 6), ["Q1\n20% más pobre", "Q2", "Q3", "Q4", "Q5\n20% más rico"])
ax.tick_params(axis="x", length=0)
ax.set_ylim(1, 10)
ax.set_yticks(range(1, 11))
ax.set_ylabel("Escalón donde se ubica hoy\n(1 = más pobre, 10 = más rico)")
ax.set_xlabel("Quintil de ingreso real por persona")
encabezado(fig, "Con el mismo ingreso, la gente de piel oscura se percibe igual o más arriba",
           "Escalera subjetiva promedio (eje 1-10) por quintil de ingreso; la excepción es el 20% más pobre")
pie(fig, "Regresión con ingreso, riqueza y escolaridad: una letra más clara = -0.07 escalones (p = 0.01).")
guardar(fig, "fig7_percepcion_vs_realidad.png")

# ===========================================================================
# FIG 8. Brecha educativa por región
# ===========================================================================
REGIONES = ["Norte", "Norte-occidente", "Centro-norte", "Centro", "Sur"]
fig, ax = plt.subplots(figsize=(11.5, 5.6))
fig.subplots_adjust(top=0.78, bottom=0.2, left=0.15, right=0.95)
for i, reg in enumerate(REGIONES[::-1]):
    r = regresion("anios_esc ~ tono_piel + mujer + edad + I(edad**2) + urbano", df[df["region"] == reg])
    lo, hi = r.conf_int().loc["tono_piel"]
    est = r.params["tono_piel"]
    color = AZUL if r.pvalues["tono_piel"] < 0.05 else GRIS
    ax.plot([lo, hi], [i, i], color=color, lw=2.5, solid_capstyle="round", zorder=2)
    ax.plot(est, i, "o", ms=10, color=color, mec=FONDO, mew=2, zorder=3)
    ax.text(hi + 0.02, i, f"{est:+.2f} años" + ("" if r.pvalues["tono_piel"] < 0.05 else "  (no signif.)"),
            va="center", fontsize=10, color=TINTA)
ax.axvline(0, color=TINTA2, lw=1.2)
ax.set_yticks(range(5), REGIONES[::-1])
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True)
ax.set_xlim(-0.3, 0.95)
ax.set_xlabel("Años de escolaridad adicionales por cada letra más clara")
encabezado(fig, "La brecha educativa por tono es mayor en el Sur; en el Centro-norte no aparece",
           "Efecto de una letra más clara sobre años de escolaridad, por región (punto) e IC 95% (línea)")
pie(fig, "Azul = significativo al 5%; gris = no significativo. Regresión ponderada por región con controles de sexo, edad y urbano/rural.")
guardar(fig, "fig8_brecha_educativa_por_region.png")

print("Listo.")
