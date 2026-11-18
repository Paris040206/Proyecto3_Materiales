"""
Simulador del diagrama Fe-Fe3C - Proyecto 3 (Ingeniería de materiales)

Cómo correrlo:
    pip install -r requirements.txt
    streamlit run app.py

Qué hace: dado un %C y una temperatura, ubica el punto en el diagrama y calcula
las fases presentes, su composición, su cantidad (regla de la palanca) y dibuja
un esquema de la microestructura en equilibrio (enfriamiento muy lento).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


# ============================================================
# 1. MOTOR: datos del diagrama y cálculos (sin Streamlit)
# ============================================================

# --- INICIO MOTOR ---

# Valores del diagrama que se usa en clase
C_FERRITA = 0.02       # máxima solubilidad de C en la ferrita α (a 727 °C)
C_EUTECTOIDE = 0.8
C_GAMMA_MAX = 2.11     # máxima solubilidad de C en la austenita γ (a 1148 °C)
C_EUTECTICO = 4.3
C_CEMENTITA = 6.67     # composición de la cementita Fe3C
T_EUTECTOIDE = 727
T_EUTECTICO = 1148
T_PERITECTICO = 1495
T_MIN, T_MAX = 20, 1600

NOMBRES_FASE = {
    "L": "Líquido (L)",
    "delta": "Ferrita δ",
    "gamma": "Austenita (γ)",
    "alfa": "Ferrita (α)",
    "Fe3C": "Cementita (Fe₃C)",
}

# Líneas de frontera: puntos (%C, T). Con ellas se leen las composiciones de las fases.
LINEAS = {
    "delta_sol": [(0.0, 1538), (0.09, 1495)],     # δ / δ + L
    "liquidus_d": [(0.0, 1538), (0.53, 1495)],    # L / δ + L
    "delta_dg": [(0.0, 1394), (0.09, 1495)],      # δ / δ + γ
    "gamma_dg": [(0.0, 1394), (0.17, 1495)],      # γ / δ + γ
    "gamma_sol": [(2.11, 1148), (0.17, 1495)],    # γ / γ + L
    "liquidus_g": [(4.3, 1148), (0.53, 1495)],    # L / γ + L
    "liquidus_c": [(4.3, 1148), (6.67, 1227)],    # L / L + Fe3C
    "A3_alfa": [(0.02, 727), (0.0, 912)],         # α / α + γ
    "A3_gamma": [(0.8, 727), (0.0, 912)],         # γ / α + γ  (línea A3)
    "Acm": [(0.8, 727), (2.11, 1148)],            # γ / γ + Fe3C (línea Acm)
    "solv_alfa": [(0.0001, 20), (0.0008, 200), (0.003, 400), (0.007, 600), (0.02, 727)],
    "Fe3C": [(6.67, 20), (6.67, 1227)],           # la cementita es un compuesto: %C fijo
}

# Regiones del diagrama. Las de dos fases traen "tie":
# (fase izquierda, línea izquierda, fase derecha, línea derecha)
REGIONES = [
    {"id": "L", "nombre": "L", "fase": "L", "color": "#FDE7C3",
     "poli": [(0, 1538), (0.53, 1495), (4.3, 1148), (6.67, 1227), (6.67, 1600), (0, 1600)]},
    {"id": "dL", "nombre": "δ + L", "color": "#F6D9B5",
     "poli": [(0, 1538), (0.09, 1495), (0.53, 1495)],
     "tie": ("delta", "delta_sol", "L", "liquidus_d")},
    {"id": "d", "nombre": "δ", "fase": "delta", "color": "#E4D2F0",
     "poli": [(0, 1394), (0.09, 1495), (0, 1538)]},
    {"id": "dg", "nombre": "δ + γ", "color": "#D3DDF5",
     "poli": [(0, 1394), (0.09, 1495), (0.17, 1495)],
     "tie": ("delta", "delta_dg", "gamma", "gamma_dg")},
    {"id": "gL", "nombre": "γ + L", "color": "#CFE6F4",
     "poli": [(0.17, 1495), (0.53, 1495), (4.3, 1148), (2.11, 1148)],
     "tie": ("gamma", "gamma_sol", "L", "liquidus_g")},
    {"id": "g", "nombre": "γ", "fase": "gamma", "color": "#A9D6F5",
     "poli": [(0, 912), (0, 1394), (0.17, 1495), (2.11, 1148), (0.8, 727)]},
    {"id": "ag", "nombre": "α + γ", "color": "#BFE3C6",
     "poli": [(0, 912), (0.8, 727), (0.02, 727)],
     "tie": ("alfa", "A3_alfa", "gamma", "A3_gamma")},
    {"id": "a", "nombre": "α", "fase": "alfa", "color": "#7BC67E",
     "poli": [(0, 20), (0, 912), (0.02, 727), (0.007, 600), (0.003, 400),
              (0.0008, 200), (0.0001, 20)]},
    {"id": "aC", "nombre": "α + Fe₃C", "color": "#ECE6DC",
     "poli": [(0.0001, 20), (0.0008, 200), (0.003, 400), (0.007, 600),
              (0.02, 727), (6.67, 727), (6.67, 20)],
     "tie": ("alfa", "solv_alfa", "Fe3C", "Fe3C")},
    {"id": "gC", "nombre": "γ + Fe₃C", "color": "#D6E4EE",
     "poli": [(0.8, 727), (2.11, 1148), (6.67, 1148), (6.67, 727)],
     "tie": ("gamma", "Acm", "Fe3C", "Fe3C")},
    {"id": "LC", "nombre": "L + Fe₃C", "color": "#F3D5C0",
     "poli": [(4.3, 1148), (6.67, 1227), (6.67, 1148)],
     "tie": ("L", "liquidus_c", "Fe3C", "Fe3C")},
]


def x_linea(nombre, T):
    """%C de una línea de frontera a la temperatura T (interpolación)."""
    puntos = sorted(LINEAS[nombre], key=lambda p: p[1])
    return float(np.interp(T, [p[1] for p in puntos], [p[0] for p in puntos]))


def punto_en_poligono(x, y, poligono):
    """Método del rayo: True si el punto (x, y) está dentro del polígono."""
    dentro = False
    j = len(poligono) - 1
    for i in range(len(poligono)):
        xi, yi = poligono[i]
        xj, yj = poligono[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                dentro = not dentro
        j = i
    return dentro


def buscar_region(x0, T):
    """Región que contiene al punto. Si cae justo sobre una línea, se toma la de abajo."""
    for dx, dT in [(0, -0.01), (0, 0.01), (1e-6, -0.01), (-1e-6, -0.01),
                   (1e-6, 0.01), (-1e-6, 0.01)]:
        for region in REGIONES:
            if punto_en_poligono(x0 + dx, T + dT, region["poli"]):
                return region
    return None


def calcular_fases(x0, T, region, ferrita_fija=True):
    """Fases presentes con su composición y fracción (regla de la palanca)."""
    if "tie" not in region:
        return [{"fase": region["fase"], "comp": x0, "frac": 1.0}], None

    fase_izq, linea_izq, fase_der, linea_der = region["tie"]
    xl = x_linea(linea_izq, T)
    xr = x_linea(linea_der, T)
    if ferrita_fija and linea_izq == "solv_alfa":
        xl = C_FERRITA  # simplificación habitual: α con 0.02 %C debajo de 727 °C

    if xr - xl < 1e-9:  # caso límite: la línea de reparto mide cero (fusión de la cementita)
        w_der = 1.0
    else:
        w_der = min(max((x0 - xl) / (xr - xl), 0.0), 1.0)
    fases = [
        {"fase": fase_izq, "comp": xl, "frac": 1 - w_der},
        {"fase": fase_der, "comp": xr, "frac": w_der},
    ]
    return fases, (xl, xr)


def tipo_aleacion(x0):
    if x0 < C_FERRITA:
        return "Hierro casi puro"
    if x0 < C_EUTECTOIDE - 1e-9:
        return "Acero hipoeutectoide"
    if abs(x0 - C_EUTECTOIDE) < 1e-9:
        return "Acero eutectoide"
    if x0 <= C_GAMMA_MAX:
        return "Acero hipereutectoide"
    if x0 < C_EUTECTICO - 1e-9:
        return "Fundición hipoeutéctica"
    if abs(x0 - C_EUTECTICO) < 1e-9:
        return "Fundición eutéctica"
    return "Fundición hipereutéctica"


def microconstituyentes(x0):
    """Constituyentes al terminar de enfriar lentamente por debajo de 727 °C.
    Cada elemento: (tipo para dibujar, nombre, fracción, fórmula usada)."""
    if x0 < C_FERRITA:
        return [("alfa", "Ferrita (α)", 1.0, "toda la aleación es ferrita")]
    if x0 < C_EUTECTOIDE - 1e-9:
        pro = (C_EUTECTOIDE - x0) / (C_EUTECTOIDE - C_FERRITA)
        return [("alfa", "Ferrita proeutectoide (α)", pro,
                 f"({C_EUTECTOIDE} − {x0:.3f}) / ({C_EUTECTOIDE} − {C_FERRITA})"),
                ("perlita", "Perlita (α + Fe₃C en láminas)", 1 - pro, "1 − ferrita proeutectoide")]
    if abs(x0 - C_EUTECTOIDE) < 1e-9:
        return [("perlita", "Perlita (α + Fe₃C en láminas)", 1.0, "toda la aleación es perlita")]
    if x0 <= C_GAMMA_MAX:
        fc = (x0 - C_EUTECTOIDE) / (C_CEMENTITA - C_EUTECTOIDE)
        return [("cementita_red", "Cementita proeutectoide (Fe₃C)", fc,
                 f"({x0:.3f} − {C_EUTECTOIDE}) / ({C_CEMENTITA} − {C_EUTECTOIDE})"),
                ("perlita", "Perlita (α + Fe₃C en láminas)", 1 - fc, "1 − cementita proeutectoide")]
    if x0 < C_EUTECTICO - 1e-9:
        fp = (C_EUTECTICO - x0) / (C_EUTECTICO - C_GAMMA_MAX)
        return [("perlita", "Perlita (de la austenita primaria)", fp,
                 f"({C_EUTECTICO} − {x0:.3f}) / ({C_EUTECTICO} − {C_GAMMA_MAX})"),
                ("ledeburita_perlita", "Ledeburita transformada", 1 - fp, "1 − perlita")]
    if abs(x0 - C_EUTECTICO) < 1e-9:
        return [("ledeburita_perlita", "Ledeburita transformada", 1.0, "toda la aleación es ledeburita")]
    if x0 < C_CEMENTITA - 1e-9:
        fc = (x0 - C_EUTECTICO) / (C_CEMENTITA - C_EUTECTICO)
        return [("cementita", "Cementita primaria (Fe₃C)", fc,
                 f"({x0:.3f} − {C_EUTECTICO}) / ({C_CEMENTITA} − {C_EUTECTICO})"),
                ("ledeburita_perlita", "Ledeburita transformada", 1 - fc, "1 − cementita primaria")]
    return [("cementita", "Cementita (Fe₃C)", 1.0, "toda la aleación es cementita")]


# De fase a tipo de dibujo (para las regiones sin microconstituyentes especiales)
TIPO_DE_FASE = {"L": "liquido", "delta": "delta", "gamma": "gamma",
                "alfa": "alfa", "Fe3C": "cementita"}


def componentes_dibujo(x0, region, fases, micro):
    """Lista (tipo, fracción de área) que se usa para dibujar la microestructura."""
    if "tie" not in region:
        return [(TIPO_DE_FASE[region["fase"]], 1.0)]

    if region["id"] == "aC":
        return [(tipo, frac) for tipo, _, frac, _ in micro]

    if region["id"] == "gC":
        if x0 <= C_GAMMA_MAX:
            frac_cem = [f["frac"] for f in fases if f["fase"] == "Fe3C"][0]
            return [("gamma", 1.0), ("cementita_red", frac_cem)]
        if x0 < C_EUTECTICO - 1e-9:
            fp = (C_EUTECTICO - x0) / (C_EUTECTICO - C_GAMMA_MAX)
            return [("gamma", fp), ("ledeburita_gamma", 1 - fp)]
        if abs(x0 - C_EUTECTICO) < 1e-9:
            return [("ledeburita_gamma", 1.0)]
        if x0 < C_CEMENTITA - 1e-9:
            fc = (x0 - C_EUTECTICO) / (C_CEMENTITA - C_EUTECTICO)
            return [("cementita", fc), ("ledeburita_gamma", 1 - fc)]
        return [("cementita", 1.0)]

    return [(TIPO_DE_FASE[f["fase"]], f["frac"]) for f in fases]


def detectar_reaccion(x0, T):
    """Mensaje si el punto está sobre la temperatura de una reacción invariante."""
    if abs(T - T_EUTECTOIDE) < 0.5 and C_FERRITA <= x0 <= C_CEMENTITA:
        return ("Estás sobre la línea de la **reacción eutectoide** (727 °C): "
                "γ (0.8 %C) → α (0.02 %C) + Fe₃C (6.67 %C). "
                "El resultado mostrado corresponde a la aleación justo por debajo de la línea.")
    if abs(T - T_EUTECTICO) < 0.5 and C_GAMMA_MAX <= x0 <= C_CEMENTITA:
        return ("Estás sobre la línea de la **reacción eutéctica** (1148 °C): "
                "L (4.3 %C) → γ (2.11 %C) + Fe₃C (6.67 %C). "
                "El resultado mostrado corresponde a la aleación justo por debajo de la línea.")
    if abs(T - T_PERITECTICO) < 0.5 and 0.09 <= x0 <= 0.53:
        return ("Estás sobre la línea de la **reacción peritéctica** (1495 °C): "
                "δ (0.09 %C) + L (0.53 %C) → γ (0.17 %C). "
                "El resultado mostrado corresponde a la aleación justo por debajo de la línea.")
    return None


def analizar(x0, T, ferrita_fija=True):
    region = buscar_region(x0, T)
    if region is None:
        return None
    fases, tie = calcular_fases(x0, T, region, ferrita_fija)
    micro = microconstituyentes(x0) if region["id"] in ("aC", "a") else None
    return {
        "region": region,
        "fases": fases,
        "tie": tie,
        "micro": micro,
        "tipo": tipo_aleacion(x0),
        "componentes": [(t, f) for t, f in componentes_dibujo(x0, region, fases, micro) if f > 1e-9],
        "reaccion": detectar_reaccion(x0, T),
    }


# ---------- Dibujo esquemático de la microestructura ----------

COLORES = {
    "alfa": (168, 213, 162),
    "gamma": (169, 214, 245),
    "delta": (228, 205, 240),
    "liquido": (253, 220, 170),
    "cementita": (70, 70, 70),   # láminas, matriz de ledeburita y límites de grano
}
# Cuando la cementita forma granos propios (cementita primaria) se dibuja más clara
GRIS_CEMENTITA_PRIMARIA = (150, 150, 150)


def generar_micro(componentes, tam=320, n_granos=28, semilla=7):
    """Imagen (tam x tam x 3) con granos tipo Voronoi coloreados según las fases.
    componentes: lista de (tipo, fracción de área)."""
    rng = np.random.default_rng(semilla)
    yy, xx = np.mgrid[0:tam, 0:tam]
    coords = np.c_[xx.ravel(), yy.ravel()]

    # 1) Granos: cada píxel pertenece a la semilla más cercana
    semillas = rng.random((n_granos, 2)) * tam
    _, lab = cKDTree(semillas).query(coords)
    lab = lab.reshape(tam, tam)
    areas = np.bincount(lab.ravel(), minlength=n_granos) / lab.size

    # 2) Reparto de granos entre componentes, según su fracción
    en_granos = [(t, f) for t, f in componentes if t != "cementita_red"]
    frac_red = sum(f for t, f in componentes if t == "cementita_red")
    objetivos = np.cumsum([f for _, f in en_granos]) / sum(f for _, f in en_granos)
    tipo = [None] * n_granos
    acumulado, k = 0.0, 0
    for g in rng.permutation(n_granos):
        tipo[g] = en_granos[k][0]
        acumulado += areas[g]
        if k < len(en_granos) - 1 and acumulado >= objetivos[k]:
            k += 1

    # 3) Manchas redondas para la ledeburita
    dist, _ = cKDTree(rng.random((110, 2)) * tam).query(coords)
    en_mancha = dist.reshape(tam, tam) < tam * 0.044

    # 4) Color de cada grano
    img = np.zeros((tam, tam, 3))
    cem = np.array(COLORES["cementita"])
    alfa = np.array(COLORES["alfa"])
    for g in range(n_granos):
        mascara = lab == g
        t = tipo[g]
        if t == "cementita":
            img[mascara] = GRIS_CEMENTITA_PRIMARIA
        elif t in COLORES:
            img[mascara] = np.array(COLORES[t]) * (1 + rng.uniform(-0.04, 0.04))
        elif t == "perlita":
            ang = rng.uniform(0, np.pi)
            laminas = ((xx * np.cos(ang) + yy * np.sin(ang)) / 9.0) % 1.0 < 0.15
            img[mascara & ~laminas] = alfa
            img[mascara & laminas] = cem
        elif t in ("ledeburita_gamma", "ledeburita_perlita"):
            img[mascara] = cem
            mancha = mascara & en_mancha
            if t == "ledeburita_gamma":
                img[mancha] = COLORES["gamma"]
            else:
                ang = rng.uniform(0, np.pi)
                laminas = ((xx * np.cos(ang) + yy * np.sin(ang)) / 4.0) % 1.0 < 0.2
                img[mancha & ~laminas] = alfa
                img[mancha & laminas] = cem

    # 5) Límites de grano (no entre granos de líquido)
    der = np.roll(lab, -1, axis=1)
    aba = np.roll(lab, -1, axis=0)
    borde = (lab != der) | (lab != aba)
    borde[:, -1] = False
    borde[-1, :] = False
    es_liquido = np.array([t == "liquido" for t in tipo])
    borde &= ~(es_liquido[lab] | es_liquido[der] | es_liquido[aba])
    if frac_red > 0:
        grosor = int(min(9, round(1 + 40 * frac_red)))
        img[maximum_filter(borde, size=grosor)] = cem
    else:
        img[borde] = (90, 90, 90)

    return np.clip(img, 0, 255).astype(np.uint8)

# --- FIN MOTOR ---


# ============================================================
# 2. TEXTOS EXPLICATIVOS
# ============================================================

NOMBRES_TIPO = {
    "alfa": "Ferrita (α)",
    "gamma": "Austenita (γ)",
    "delta": "Ferrita δ",
    "liquido": "Líquido (L)",
    "cementita": "Cementita (Fe₃C)",
    "cementita_red": "Cementita (Fe₃C) en los límites de grano",
    "perlita": "Perlita (láminas de α + Fe₃C)",
    "ledeburita_gamma": "Ledeburita (γ + Fe₃C)",
    "ledeburita_perlita": "Ledeburita transformada (perlita + Fe₃C)",
}

COLOR_LEYENDA = {
    "cementita": "#969696",
    "perlita": "repeating-linear-gradient(45deg,#A8D5A2 0 6px,#464646 6px 8px)",
    "ledeburita_gamma": "linear-gradient(90deg,#464646 50%,#A9D6F5 50%)",
    "ledeburita_perlita": "linear-gradient(90deg,#464646 50%,#A8D5A2 50%)",
    "cementita_red": "#464646",
}

EXPLICA_REGION = {
    "L": "Todo está fundido: todavía no hay sólido.",
    "dL": "Empieza a solidificar ferrita δ dentro del líquido.",
    "d": "Ferrita δ (BCC): solo aparece a temperaturas muy altas y con muy poco carbono.",
    "dg": "La ferrita δ se está transformando en austenita.",
    "gL": "La austenita crece dentro del líquido. Al enfriar, el líquido que queda se enriquece "
          "en carbono, hasta llegar a 4.3 %C a 1148 °C.",
    "g": "Austenita (FCC): disuelve hasta 2.11 %C. Es la fase de partida del tratamiento "
         "térmico de los aceros.",
    "ag": "Al cruzar la línea A3, la ferrita empieza a formarse en los límites de grano de la "
          "austenita. La austenita que queda se va enriqueciendo en carbono hasta 0.8 %C.",
    "a": "Ferrita (BCC): disuelve muy poco carbono (máximo 0.02 %C).",
    "gC": "La austenita coexiste con la cementita: en aceros hipereutectoides, la cementita "
          "precipita en los límites de grano al cruzar la línea Acm; en fundiciones, la "
          "austenita está acompañada de ledeburita.",
    "aC": "Por debajo de 727 °C ya no hay austenita: la que quedaba se transformó por la "
          "reacción eutectoide en perlita.",
    "LC": "En fundiciones hipereutécticas, la cementita primaria crece dentro del líquido.",
}

EXPLICA_ALEACION = {
    "Hierro casi puro": "Casi no hay carbono: solo se ve ferrita.",
    "Acero hipoeutectoide": "Al enfriar cruzó la línea A3: primero se formó ferrita proeutectoide en los "
                            "límites de grano, y a 727 °C la austenita restante (0.8 %C) se transformó en "
                            "perlita.",
    "Acero eutectoide": "Toda la austenita se transformó a 727 °C en perlita, sin fase proeutectoide.",
    "Acero hipereutectoide": "Al enfriar cruzó la línea Acm: primero precipitó cementita proeutectoide en los "
                             "límites de grano, y a 727 °C la austenita restante (0.8 %C) se transformó en "
                             "perlita.",
    "Fundición hipoeutéctica": "Primero solidificó austenita primaria. A 1148 °C el líquido restante hizo la "
                               "reacción eutéctica (ledeburita), y a 727 °C la austenita se transformó en perlita.",
    "Fundición eutéctica": "Todo el líquido solidificó a 1148 °C en ledeburita (γ + Fe₃C). Al pasar 727 °C, "
                           "la austenita de la ledeburita se transformó en perlita.",
    "Fundición hipereutéctica": "Primero creció cementita primaria en el líquido. A 1148 °C el líquido restante "
                                "hizo la reacción eutéctica (ledeburita).",
}


def leyenda_html(tipos):
    filas = []
    for t in tipos:
        if t in COLOR_LEYENDA:
            fondo = COLOR_LEYENDA[t]
        else:
            fondo = "#%02x%02x%02x" % COLORES[t]
        filas.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<span style="display:inline-block;width:26px;height:14px;border:1px solid #888;'
            f'background:{fondo}"></span><span>{NOMBRES_TIPO[t]}</span></div>')
    return "".join(filas)


def pasos_calculo(x0, T, info):
    """Texto con el cálculo paso a paso (regla de la palanca y microconstituyentes)."""
    lineas = []
    if info["tie"] is None:
        lineas.append("Hay una sola fase: toda la aleación es esa fase y su composición es la de la "
                      "aleación. No hace falta la regla de la palanca.")
    else:
        xl, xr = info["tie"]
        izq, der = info["fases"]
        n_izq, n_der = NOMBRES_FASE[izq["fase"]], NOMBRES_FASE[der["fase"]]
        lineas += [
            f"**Fases (regla de la palanca)**",
            f"1. Se traza la línea de reparto horizontal a **{T:.0f} °C**.",
            f"2. Donde corta las fronteras se leen las composiciones: **{n_izq}** = {xl:.3f} %C y "
            f"**{n_der}** = {xr:.3f} %C.",
            f"3. Con %C de la aleación = {x0:.3f}:",
            f"   - {n_izq} = ({xr:.3f} − {x0:.3f}) / ({xr:.3f} − {xl:.3f}) = **{izq['frac'] * 100:.1f} %**",
            f"   - {n_der} = ({x0:.3f} − {xl:.3f}) / ({xr:.3f} − {xl:.3f}) = **{der['frac'] * 100:.1f} %**",
        ]
    if info["micro"] is not None and len(info["micro"]) > 1:
        lineas += ["", "**Constituyentes (después de enfriar lentamente por debajo de 727 °C)**"]
        for _, nombre, frac, formula in info["micro"]:
            lineas.append(f"- {nombre} = {formula} = **{frac * 100:.1f} %**")
    return "\n\n".join(lineas)


# ============================================================
# 3. DIAGRAMA INTERACTIVO
# ============================================================

ETIQUETAS = [  # (%C, T, texto, tamaño)
    (3.5, 1450, "L", 16), (2.0, 1310, "γ + L", 12), (1.0, 1000, "Austenita (γ)", 15),
    (0.28, 820, "α+γ", 10), (4.3, 900, "γ + Fe₃C", 14), (6.12, 1175, "L+Fe₃C", 9),
    (3.4, 650, "α + Fe₃C", 14),
]


def crear_diagrama(x0, T, info, vista):
    fig = go.Figure()

    for region in REGIONES:
        xs = [p[0] for p in region["poli"]] + [region["poli"][0][0]]
        ts = [p[1] for p in region["poli"]] + [region["poli"][0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ts, mode="lines", fill="toself", fillcolor=region["color"],
            line=dict(color="#8a8a8a", width=1), hoveron="fills", text=region["nombre"],
            hoverinfo="text", showlegend=False))

    # Líneas de las reacciones invariantes
    for T_inv, xa, xb, nombre in [(T_EUTECTOIDE, C_FERRITA, C_CEMENTITA, "Eutectoide, 727 °C"),
                                  (T_EUTECTICO, C_GAMMA_MAX, C_CEMENTITA, "Eutéctica, 1148 °C"),
                                  (T_PERITECTICO, 0.09, 0.53, "Peritéctica, 1495 °C")]:
        fig.add_trace(go.Scatter(x=[xa, xb], y=[T_inv, T_inv], mode="lines",
                                 line=dict(color="#c0392b", width=3), text=nombre,
                                 hoverinfo="text", showlegend=False))

    # Verticales que separan las zonas de microconstituyentes
    for x_v, t_top in [(C_EUTECTOIDE, T_EUTECTOIDE), (C_GAMMA_MAX, T_EUTECTICO), (C_EUTECTICO, T_EUTECTICO)]:
        fig.add_trace(go.Scatter(x=[x_v, x_v], y=[T_MIN, t_top], mode="lines",
                                 line=dict(color="#555", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))

    for x, t, texto, tam in ETIQUETAS:
        fig.add_annotation(x=x, y=t, text=texto, showarrow=False, font=dict(size=tam, color="#222"))
    fig.add_annotation(x=0.012, y=450, ax=0.6, ay=450, axref="x", ayref="y", text="α",
                       showarrow=True, arrowhead=2, font=dict(size=14, color="#222"))

    # Zonas de microconstituyentes (debajo de 727 °C)
    for x, texto in [(0.4, "α + perlita"), (1.45, "perlita + Fe₃C"),
                     (3.2, "perlita + Fe₃C<br>+ ledeburita"), (5.5, "Fe₃C + ledeburita")]:
        fig.add_annotation(x=x, y=150, text=texto, showarrow=False, font=dict(size=10, color="#444"))

    # Línea de reparto y punto elegido
    if info["tie"]:
        xl, xr = info["tie"]
        fig.add_trace(go.Scatter(x=[xl, xr], y=[T, T], mode="lines+markers", name="Línea de reparto",
                                 line=dict(color="black", dash="dash", width=2),
                                 marker=dict(color="black", size=8, symbol="square"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x0], y=[T], mode="markers", name="Tu punto",
                             marker=dict(symbol="star", size=20, color="#d62728",
                                         line=dict(color="white", width=1.5)),
                             hovertemplate=f"%C = {x0:.3f}<br>T = {T:.0f} °C<extra></extra>"))

    if vista == "Zona de aceros (0 a 2.2 %C)":
        rango_x, marcas = [0, 2.2], [0, 0.5, 0.8, 1.0, 1.5, 2.0, 2.11]
    else:
        rango_x, marcas = [0, 6.8], [0, 0.8, 2.11, 4.3, 6.67]
    fig.update_xaxes(title="% en peso de carbono (%C)", range=rango_x, tickvals=marcas,
                     ticktext=[f"{m:g}" for m in marcas], showgrid=False)
    fig.update_yaxes(title="Temperatura (°C)", range=[0, 1600], dtick=200, showgrid=False)
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.12), plot_bgcolor="white")
    return fig


# ============================================================
# 4. PÁGINA
# ============================================================

st.set_page_config(page_title="Simulador Fe–Fe₃C", page_icon="🧲", layout="wide")
st.title("🧲 Simulador del diagrama Fe–Fe₃C")
st.caption("Elige una temperatura y un %C: te digo qué fases hay, su composición, cuánto hay de cada una "
           "y cómo se ve la microestructura.")

EJEMPLOS = {
    "— Elige un ejemplo —": None,
    "Tipo tarea: 0.25 %C a 800 °C": (0.25, 800.0),
    "Tipo tarea: 0.25 %C a 200 °C": (0.25, 200.0),
    "Tipo tarea: 1.4 %C a 800 °C": (1.4, 800.0),
    "Tipo tarea: 1.4 %C a 600 °C": (1.4, 600.0),
    "Acero eutectoide: 0.8 %C a 600 °C": (0.8, 600.0),
    "Fundición hipoeutéctica: 3 %C a 600 °C": (3.0, 600.0),
    "Fundición eutéctica: 4.3 %C a 1000 °C": (4.3, 1000.0),
    "Fundición hipereutéctica: 5.5 %C a 600 °C": (5.5, 600.0),
    "Líquido: 1 %C a 1500 °C": (1.0, 1500.0),
}


def cargar_ejemplo():
    ejemplo = EJEMPLOS[st.session_state["ejemplo"]]
    if ejemplo:
        st.session_state["C"], st.session_state["T"] = ejemplo


st.session_state.setdefault("C", 0.4)
st.session_state.setdefault("T", 800.0)

with st.expander("ℹ️ Cómo usar el simulador"):
    st.markdown("""
1. Escribe el **%C** de la aleación y la **temperatura**, o elige un ejemplo. El punto (⭐) aparece en el diagrama.
2. A la derecha verás la **región** del diagrama, las **fases** con su composición y cantidad, y los
   **constituyentes** (perlita, ledeburita, etc.).
3. Abajo se dibuja un **esquema de la microestructura** y se explica **cómo se llegó** a ella.
4. Si abres "Cálculo paso a paso", verás la regla de la palanca con tus números, para que puedas
   comprobarla a mano.

Todo corresponde al diagrama de **equilibrio** (enfriamiento muy lento).
    """)

c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
x0 = c1.number_input("% de carbono (%C)", min_value=0.0, max_value=C_CEMENTITA,
                     step=0.05, format="%.3f", key="C")
T = c2.number_input("Temperatura (°C)", min_value=float(T_MIN), max_value=float(T_MAX),
                    step=10.0, format="%.1f", key="T")
c3.selectbox("Ejemplos", list(EJEMPLOS), key="ejemplo", on_change=cargar_ejemplo)
vista = c4.radio("Vista del diagrama", ["Completo", "Zona de aceros (0 a 2.2 %C)"], horizontal=True)
ferrita_fija = st.checkbox("Usar ferrita con 0.02 %C constante por debajo de 727 °C "
                           "(simplificación habitual en clase)", value=True)

info = analizar(x0, T, ferrita_fija)
if info is None:
    st.error("No se encontró una región para ese punto. Revisa los valores.")
    st.stop()

col_diagrama, col_resultados = st.columns([3, 2])

with col_diagrama:
    st.plotly_chart(crear_diagrama(x0, T, info, vista), use_container_width=True)

with col_resultados:
    st.markdown(f"### {info['tipo']}")
    st.markdown(f"**Región del diagrama:** {info['region']['nombre']}")
    if info["reaccion"]:
        st.warning(info["reaccion"])

    st.markdown("#### Fases presentes")
    st.table([{"Fase": NOMBRES_FASE[f["fase"]],
               "Composición (%C)": f"{f['comp']:.3f}".rstrip("0").rstrip("."),
               "Cantidad (%)": f"{f['frac'] * 100:.1f}"}
              for f in info["fases"] if f["frac"] > 1e-9])

    if info["micro"] is not None:
        st.markdown("#### Constituyentes a temperatura ambiente")
        st.table([{"Constituyente": nombre, "Cantidad (%)": f"{frac * 100:.1f}"}
                  for _, nombre, frac, _ in info["micro"]])

st.markdown("---")
col_img, col_texto = st.columns([1, 2])

with col_img:
    st.markdown("#### 🔬 Microestructura (esquema)")
    st.image(generar_micro(info["componentes"]), width=340,
             caption="Dibujo esquemático, no a escala y no es una fotografía real.")
    tipos = []
    for tipo, frac in info["componentes"]:
        if tipo not in tipos and frac > 0:
            tipos.append(tipo)
    st.markdown(leyenda_html(tipos), unsafe_allow_html=True)

with col_texto:
    st.markdown("#### 📖 ¿Qué está pasando?")
    st.markdown(f"**En esta región:** {EXPLICA_REGION[info['region']['id']]}")
    if info["micro"] is not None:
        st.markdown(f"**Cómo se llegó aquí:** {EXPLICA_ALEACION[info['tipo']]}")
    with st.expander("🧮 Cálculo paso a paso", expanded=False):
        st.markdown(pasos_calculo(x0, T, info))

with st.expander("⚠️ Límites del simulador"):
    st.markdown("""
- Es una **lectura del diagrama de equilibrio** (enfriamiento muy lento), no un cálculo termodinámico.
  Las líneas curvas del diagrama se aproximan con segmentos de recta, así que las composiciones leídas
  son aproximadas.
- Valores clave usados: eutectoide 0.8 %C a 727 °C, eutéctico 4.3 %C a 1148 °C, máxima solubilidad de C
  en γ 2.11 %C, en α 0.02 %C, cementita 6.67 %C y peritéctico a 1495 °C.
- Las microestructuras son **esquemas** (los granos y las láminas se generan al azar), no fotografías.
  En una pieza real influyen la velocidad de enfriamiento y otros elementos de aleación.
- Si el punto está justo sobre una línea de reacción, el resultado corresponde a la aleación inmediatamente
  por debajo de esa línea.
    """)
