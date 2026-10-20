# Tono de piel, nivel socioeconómico y escolaridad — ESRU-EMOVI 2023

**Pregunta 19.** ¿Existe una asociación entre el tono de piel (medido con la escala/paleta de color) y el nivel socioeconómico actual o la escolaridad alcanzada de los entrevistados, y difiere esta relación entre hombres y mujeres?

Fuente: Encuesta ESRU de Movilidad Social en México 2023 (Centro de Estudios Espinosa Yglesias). Personas de 25 a 64 años.

## Estructura

```
├── limpieza_emovi2023.py          # script de limpieza
├── data/
│   ├── raw/BASES_DE_DATOS_EMOVI2023.zip   # bases originales (sin modificar)
│   └── clean/
│       ├── emovi2023_limpia.csv   # base lista para analizar (17,349 x 37)
│       └── diccionario_limpia.csv # qué significa cada columna
└── docs/
    ├── Diccionario_ESRU_EMOVI_2023.xlsx   # diccionario oficial
    └── pregunta_19.png
```

## Cómo reproducir

```bash
pip install pandas numpy pyreadstat
python limpieza_emovi2023.py
```

El script descomprime `entrevistado_2023.dta` del zip (pesa 41 MB, demasiado para subirlo directo) y genera `data/clean/emovi2023_limpia.csv`.

## Decisiones de limpieza

| Tema | Decisión |
|---|---|
| Base usada | Solo `entrevistado_2023.dta`. `hogar_2023` (integrantes del hogar) no se necesita, e `ingreso_2017` es de otra ola de la encuesta. |
| Tono de piel | p112 (letras A–K) → `tono_piel` 1–11. Se verificó con el colorímetro que **A es el más oscuro y K el más claro**. Grupos: Oscuro A–E, Medio F–G, Claro H–K (A–D tienen muy pocas personas). |
| Escolaridad | `nivel_educ` = variable oficial de ESRU (4 niveles). `anios_esc` = años de escuela construidos con p9 + p10. Si el grado es "No sabe" queda vacío (292–314 casos). |
| Nivel socioeconómico | Tres medidas: (1) ingreso per cápita imputado por ESRU y sus quintiles; (2) escalera subjetiva 1–10; (3) índice de riqueza con componentes principales de 48 bienes y servicios de la vivienda. |
| Quintiles | Calculados con el factor de expansión, para que cada quintil sea 20% de la población. |
| Outliers | Regla del rango intercuartil (IQR × 1.5) en `log_ingreso_pc`, `indice_riqueza` y `luminosidad_piel`. Se eliminan **494 personas (2.8%)**; quedan 17,349. Las filas eliminadas se guardan en `data/clean/outliers_eliminados.csv`. No se buscan en edad (ya acotada 25–64), escaleras (1–10), años de escuela (los extremos son casos reales) ni en el factor de expansión. |
| Ponderador | Usar `factor` en promedios y porcentajes; `estrato` y `upm` si se calculan errores estándar con diseño muestral. |

## Faltantes relevantes

- Colorímetro: ~750 casos sin medición (la paleta `tono_piel` no tiene faltantes).
- Escolaridad: 292 casos (grado no sabe).
- Educación de padres: 3,371 (padre) y 1,845 (madre).
- Clase ocupacional: 5,916 (personas que no trabajan).
