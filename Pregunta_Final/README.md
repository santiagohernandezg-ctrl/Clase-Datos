# Tono de piel, nivel socioeconómico y escolaridad — ESRU-EMOVI 2023

**Pregunta 19.** ¿Existe una asociación entre el tono de piel (medido con la escala/paleta de color) y el nivel socioeconómico actual o la escolaridad alcanzada de los entrevistados, y difiere esta relación entre hombres y mujeres?

Fuente: Encuesta ESRU de Movilidad Social en México 2023 (Centro de Estudios Espinosa Yglesias). Personas de 25 a 64 años, representativa a nivel nacional.

## Estructura

```
├── limpieza_emovi2023.py            # limpieza en los 7 pasos de la guía del curso
├── data/
│   ├── raw/BASES_DE_DATOS_EMOVI2023.zip   # bases originales (sin modificar)
│   └── clean/
│       ├── emovi2023_final.csv      # BASE FINAL: 17,427 personas x 24 variables
│       ├── diccionario_final.csv    # qué significa cada columna
│       └── outliers_eliminados.csv  # las 416 personas quitadas por ser atípicas
├── resultados/
│   ├── 0_perfilado_variables.csv    # paso 1: tipo, rango y % de nulos de cada variable
│   └── bitacora_limpieza.csv        # qué se hizo en cada paso y a cuántas observaciones afectó
├── graficas/eda/                    # histogramas y boxplots del análisis exploratorio
└── docs/
    ├── Diccionario_ESRU_EMOVI_2023.xlsx   # diccionario oficial
    └── pregunta_19.png
```

## Cómo reproducir

```bash
pip install pandas numpy pyreadstat matplotlib
python limpieza_emovi2023.py
```

El script descomprime `entrevistado_2023.dta` del zip (pesa 41 MB, demasiado para subirlo directo a GitHub) y genera la base final, el perfilado y la bitácora.

## Limpieza: los 7 pasos de la guía

Se usa solo `entrevistado_2023.dta` (una fila = una persona). `hogar_2023` es la lista de integrantes del hogar y no hace falta; `ingreso_2017` es de otra ola de la encuesta.

| Paso | Qué se encontró | Decisión | Obs. afectadas |
|---|---|---|---|
| **1. EDA** | 17,843 personas × 296 variables; ingreso muy sesgado (en logaritmo es casi simétrico); nulos en grado escolar y colorímetro | Perfilado de las 22 variables usadas y gráficas exploratorias | — |
| **2. Errores** | `p10 = 9` = "No sabe" (código del diccionario); `p9 = 97` = "Ninguno" | El 9 pasa a NA; el 97 se conserva como 0 años. 7 reglas de rango (edad 25–64, tono A–K, escaleras 1–10, ingreso > 0…) se cumplen | 314 / 344 |
| **3. Duplicados** | Sin filas idénticas ni folios repetidos | Nada que eliminar | 0 |
| **4. Nulos** | Quienes no saben su grado **no son aleatorios**: más jóvenes (37 vs 42 años) y con más ingreso | Imputación simple con la mediana (años) y moda (nivel) de su mismo nivel declarado; marcados en `esc_imputada`. Colorímetro (2.6%) se deja NA: es variable secundaria | 315 / 468 |
| **5. Outliers** | Regla IQR × 1.5. En pesos marcaría 1,336 personas solo por ganar bien, por eso se aplica al **logaritmo** del ingreso | Se eliminan extremos de ingreso (imputado por ESRU, fuera de $774–$41,208) e índice de riqueza. Extremos del colorímetro: solo la medición pasa a NA (error del aparato) | 416 eliminadas (2.3%) |
| **6. Consistencias** | Años de escuela ≤ edad − 5 ✔; nivel ESRU ≤ nivel declarado ✔; paleta vs colorímetro r = 0.33 | 16 mediciones del colorímetro a > 3 desv. est. de su letra pasan a NA | 16 |
| **7. Encoding** | — | Tono ordinal 1–11 y en 3 grupos; binarias (mujer, universidad, urbano, lengua indígena); log del ingreso; quintiles ponderados; índice de riqueza por componentes principales | — |

## Limitaciones de calidad de los datos

- **Tono de piel:** la paleta la asigna el encuestador. Su correlación con el colorímetro es moderada (r = 0.33): la luminosidad sube claramente de F a I, pero entre A y E casi no cambia. La paleta mide cómo se percibe el tono, no solo el color físico.
- **Ingreso:** es ingreso per cápita del hogar **imputado por ESRU**, no el ingreso individual de la persona.
- **Escolaridad imputada:** 310 personas de la base final (1.8%) tienen años de escuela imputados (`esc_imputada = 1`).
- **Grupos extremos pequeños:** pocas personas en los tonos A–C e I–K.
- **Ponderador:** todos los promedios y porcentajes deben usar `factor` para representar a la población.
