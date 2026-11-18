"""
Simulador del diagrama Fe-Fe3C - Proyecto 3 (Ingeniería de materiales)

Cómo correrlo:
    pip install -r requirements.txt
    streamlit run app.py

Qué hace: dado un %C y una temperatura, ubica el punto en el diagrama y calcula
las fases presentes, su composición, su cantidad (regla de la palanca) y dibuja
un esquema de la microestructura en equilibrio (enfriamiento muy lento).
También anima el enfriamiento completo de una aleación, desde el líquido hasta
temperatura ambiente.
"""

import time
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt
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


# De fase a tipo de dibujo
TIPO_DE_FASE = {"L": "liquido", "delta": "delta", "gamma": "gamma",
                "alfa": "alfa", "Fe3C": "cementita"}


def capas_dibujo(x0, region, fases, micro):
    """Capas (tipo, fracción, medida) para dibujar la microestructura.
    La última capa es la matriz (ocupa lo que sobra).
    medida: 'nucleos' = cristales redondos que crecen desde un punto (p. ej. en el líquido);
            'limites' = fase que crece desde los límites de grano."""
    rid = region["id"]
    if "tie" not in region:
        return [(TIPO_DE_FASE[region["fase"]], 1.0, None)]

    fr = {f["fase"]: f["frac"] for f in fases}
    if rid == "dL":
        return [("delta", fr["delta"], "nucleos"), ("liquido", fr["L"], None)]
    if rid == "dg":
        return [("gamma", fr["gamma"], "limites"), ("delta", fr["delta"], None)]
    if rid == "gL":
        return [("gamma", fr["gamma"], "nucleos"), ("liquido", fr["L"], None)]
    if rid == "LC":
        return [("cementita", fr["Fe3C"], "nucleos"), ("liquido", fr["L"], None)]
    if rid == "ag":
        return [("alfa", fr["alfa"], "limites"), ("gamma", fr["gamma"], None)]

    if rid == "gC":
        if x0 <= C_GAMMA_MAX:
            return [("cementita_red", fr["Fe3C"], "limites"), ("gamma", fr["gamma"], None)]
        if x0 < C_EUTECTICO - 1e-9:
            fp = (C_EUTECTICO - x0) / (C_EUTECTICO - C_GAMMA_MAX)
            return [("gamma", fp, "nucleos"), ("ledeburita_gamma", 1 - fp, None)]
        if abs(x0 - C_EUTECTICO) < 1e-9:
            return [("ledeburita_gamma", 1.0, None)]
        if x0 < C_CEMENTITA - 1e-9:
            fc = (x0 - C_EUTECTICO) / (C_CEMENTITA - C_EUTECTICO)
            return [("cementita", fc, "nucleos"), ("ledeburita_gamma", 1 - fc, None)]
        return [("cementita", 1.0, None)]

    # rid == "aC": debajo de 727 °C
    fm = {tipo: frac for tipo, _, frac, _ in micro}
    if "ledeburita_perlita" in fm:
        if "cementita" in fm:
            return [("cementita", fm["cementita"], "nucleos"),
                    ("ledeburita_perlita", fm["ledeburita_perlita"], None)]
        if "perlita" in fm:
            return [("perlita", fm["perlita"], "nucleos"),
                    ("ledeburita_perlita", fm["ledeburita_perlita"], None)]
        return [("ledeburita_perlita", 1.0, None)]
    if "alfa" in fm and "perlita" in fm:
        return [("alfa", fm["alfa"], "limites"), ("perlita", fm["perlita"], None)]
    if "cementita_red" in fm:
        return [("cementita_red", fm["cementita_red"], "limites"), ("perlita", fm["perlita"], None)]
    if "perlita" in fm:
        return [("perlita", 1.0, None)]
    if "alfa" in fm:
        return [("alfa", 1.0, None)]
    return [("cementita", 1.0, None)]


def capas_reaccion(x0, T_inv, p):
    """Capas durante una reacción isotérmica (eutectoide o eutéctica) con avance p (0 a 1).
    Devuelve None si a esa temperatura no hay reacción para esa composición."""
    if abs(T_inv - T_EUTECTOIDE) < 0.5 and C_FERRITA <= x0 <= C_GAMMA_MAX:
        # la austenita restante se transforma en perlita, empezando por los límites de grano
        capas = []
        if x0 < C_EUTECTOIDE - 1e-9:
            f_borde = (C_EUTECTOIDE - x0) / (C_EUTECTOIDE - C_FERRITA)
            capas.append(("alfa", f_borde, "limites"))
        elif x0 > C_EUTECTOIDE + 1e-9:
            f_borde = (x0 - C_EUTECTOIDE) / (C_CEMENTITA - C_EUTECTOIDE)
            capas.append(("cementita_red", f_borde, "limites"))
        else:
            f_borde = 0.0
        capas.append(("perlita", p * (1 - f_borde), "limites"))
        capas.append(("gamma", (1 - p) * (1 - f_borde), None))
        return capas

    if abs(T_inv - T_EUTECTICO) < 0.5 and C_GAMMA_MAX <= x0 < C_CEMENTITA - 1e-9:
        # el líquido restante (4.3 %C) se vuelve ledeburita, creciendo desde los cristales
        capas = []
        if x0 < C_EUTECTICO - 1e-9:
            f_primario = (C_EUTECTICO - x0) / (C_EUTECTICO - C_GAMMA_MAX)
            capas.append(("gamma", f_primario, "nucleos"))
        elif x0 > C_EUTECTICO + 1e-9:
            f_primario = (x0 - C_EUTECTICO) / (C_CEMENTITA - C_EUTECTICO)
            capas.append(("cementita", f_primario, "nucleos"))
        else:
            f_primario = 0.0
        resto = 1 - f_primario
        capas.append(("ledeburita_gamma", p * resto, "nucleos"))
        capas.append(("liquido", (1 - p) * resto, None))
        return capas
    return None


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
    capas = [c for c in capas_dibujo(x0, region, fases, micro) if c[1] > 1e-9]
    return {
        "region": region,
        "fases": fases,
        "tie": tie,
        "micro": micro,
        "tipo": tipo_aleacion(x0),
        "capas": capas,
        "reaccion": detectar_reaccion(x0, T),
    }


# ---------- Enfriamiento: cuadros y eventos ----------

TEXTO_EVENTO = {
    ("L", "dL"): "Empieza a solidificar ferrita δ dentro del líquido.",
    ("L", "gL"): "Empieza a solidificar austenita (γ) dentro del líquido.",
    ("L", "LC"): "Empieza a crecer cementita primaria dentro del líquido.",
    ("L", "gC"): "Reacción eutéctica: todo el líquido (4.3 %C) → ledeburita (γ + Fe₃C).",
    ("dL", "d"): "Termina la solidificación: todo es ferrita δ.",
    ("dL", "dg"): "Reacción peritéctica (δ + L → γ): se acaba el líquido y aparece austenita.",
    ("dL", "gL"): "Reacción peritéctica (δ + L → γ): se acaba la ferrita δ y queda austenita + líquido.",
    ("d", "dg"): "La ferrita δ empieza a transformarse en austenita.",
    ("dg", "g"): "La ferrita δ se transformó por completo en austenita.",
    ("gL", "g"): "Termina la solidificación: todo es austenita.",
    ("gL", "gC"): "Reacción eutéctica: el líquido restante (4.3 %C) → ledeburita (γ + Fe₃C).",
    ("LC", "gC"): "Reacción eutéctica: el líquido restante (4.3 %C) → ledeburita (γ + Fe₃C).",
    ("g", "ag"): "Cruza la línea A3: empieza a formarse ferrita en los límites de grano de la austenita.",
    ("g", "gC"): "Cruza la línea Acm: empieza a precipitar cementita en los límites de grano de la austenita.",
    ("ag", "aC"): "Reacción eutectoide: la austenita restante (0.8 %C) → perlita.",
    ("ag", "a"): "Termina la transformación: toda la austenita se volvió ferrita.",
    ("gC", "aC"): "Reacción eutectoide: la austenita (γ) → perlita (α + Fe₃C).",
    ("g", "aC"): "Reacción eutectoide: toda la austenita (0.8 %C) → perlita.",
    ("a", "aC"): "Baja la solubilidad del carbono en la ferrita: empieza a precipitar un poco de cementita.",
}


def eventos_enfriamiento(x0, T_ini=1600, T_fin=20):
    """Temperaturas a las que cambia la región al enfriar la aleación x0: lista (T, texto)."""
    nombres = {r["id"]: r["nombre"] for r in REGIONES}
    previa = buscar_region(x0, T_ini)["id"]
    eventos = []
    for T in range(T_ini - 1, T_fin - 1, -1):
        actual = buscar_region(x0, T)["id"]
        if actual != previa:
            texto = TEXTO_EVENTO.get((previa, actual), f"Se pasa a la región {nombres[actual]}.")
            eventos.append((T, texto))
            previa = actual
    return eventos


def construir_cuadros(x0, T_ini=1600, T_fin=20, paso=25):
    """Cuadros de la animación al enfriar x0. En las reacciones isotérmicas se agregan
    cuadros intermedios para mostrar cómo avanza la transformación."""
    temps = set(range(T_ini, T_fin - 1, -paso))
    temps.add(T_fin)
    temps.update(t for t in (T_PERITECTICO, T_EUTECTICO, T_EUTECTOIDE) if T_fin <= t <= T_ini)
    cuadros = []
    for T in sorted(temps, reverse=True):
        if capas_reaccion(x0, T, 0.0) is not None:
            arriba = analizar(x0, T + 0.05)
            cuadros.append({"T": T, "info": arriba, "capas": arriba["capas"], "nota": ""})
            for p in (0.25, 0.5, 0.75):
                cuadros.append({"T": T, "info": arriba, "capas": capas_reaccion(x0, T, p),
                                "nota": f"Reacción a temperatura constante en curso: {p * 100:.0f} %"})
        info = analizar(x0, T)
        cuadros.append({"T": T, "info": info, "capas": info["capas"], "nota": ""})
    return cuadros


# ---------- Dibujo esquemático de la microestructura ----------

CODIGOS = {"liquido": 0, "delta": 1, "gamma": 2, "alfa": 3, "cementita": 4,
           "cementita_red": 5, "perlita": 6, "ledeburita_gamma": 7, "ledeburita_perlita": 8}

COLORES = {
    "liquido": (253, 220, 170),
    "delta": (228, 205, 240),
    "gamma": (169, 214, 245),
    "alfa": (168, 213, 162),
    "cementita": (150, 150, 150),       # cementita en granos propios (primaria)
    "cementita_red": (70, 70, 70),      # cementita en límites de grano, láminas y matriz de ledeburita
}
GRIS_OSCURO = np.array(COLORES["cementita_red"], dtype=float)
VERDE_ALFA = np.array(COLORES["alfa"], dtype=float)
AZUL_GAMMA = np.array(COLORES["gamma"], dtype=float)
# tabla de colores por código (los códigos 6, 7 y 8 se pintan aparte)
TABLA_COLORES = np.array([COLORES["liquido"], COLORES["delta"], COLORES["gamma"], COLORES["alfa"],
                          COLORES["cementita"], COLORES["cementita_red"], COLORES["alfa"],
                          COLORES["cementita_red"], COLORES["cementita_red"]], dtype=float)


@lru_cache(maxsize=4)
def geometria(tam=320, n_granos=28, semilla=7):
    """Granos tipo Voronoi y mapas auxiliares. Se calcula una sola vez y se reutiliza,
    así los granos son los mismos en todos los cuadros de la animación."""
    rng = np.random.default_rng(semilla)
    yy, xx = np.mgrid[0:tam, 0:tam]
    coords = np.c_[xx.ravel(), yy.ravel()]

    semillas = rng.random((n_granos, 2)) * tam
    d_semilla, lab = cKDTree(semillas).query(coords)
    lab = lab.reshape(tam, tam)
    d_semilla = d_semilla.reshape(tam, tam)

    # límites de grano (1 píxel) y distancia de cada píxel al límite más cercano
    der = np.roll(lab, -1, axis=1)
    aba = np.roll(lab, -1, axis=0)
    borde = (lab != der) | (lab != aba)
    borde[:, -1] = False
    borde[-1, :] = False
    d_borde = distance_transform_edt(~borde)

    # Orden en que se va llenando la imagen: desde los núcleos o desde los límites de grano
    ruido = rng.random((tam, tam)) * 1e-3
    orden_nucleos = np.argsort((d_semilla + ruido).ravel(), kind="stable")
    orden_limites = np.argsort((d_borde + ruido).ravel(), kind="stable")

    # láminas de perlita (cada grano con su orientación) y manchas de la ledeburita
    angulos = rng.uniform(0, np.pi, n_granos)[lab]
    proyeccion = xx * np.cos(angulos) + yy * np.sin(angulos)
    laminas_gruesas = (proyeccion / 8.0) % 1.0 < 0.22
    laminas_finas = (proyeccion / 4.0) % 1.0 < 0.25
    d_mancha, _ = cKDTree(rng.random((110, 2)) * tam).query(coords)
    manchas = d_mancha.reshape(tam, tam) < tam * 0.044
    brillo = (1 + rng.uniform(-0.04, 0.04, n_granos))[lab]

    return {"borde": borde, "orden_nucleos": orden_nucleos, "orden_limites": orden_limites,
            "gruesas": laminas_gruesas, "finas": laminas_finas, "manchas": manchas,
            "brillo": brillo}


def generar_micro(capas, tam=320, n_granos=28, semilla=7):
    """Imagen (tam x tam x 3) de la microestructura. capas: lista de (tipo, fracción, medida);
    la última capa es la matriz y ocupa lo que sobra."""
    geo = geometria(tam, n_granos, semilla)
    capas = [c for c in capas if c[1] > 1e-9] or capas[-1:]

    medida = next((m for _, _, m in capas if m), "limites")
    orden = geo["orden_" + medida]
    tipos = np.zeros(orden.size, dtype=np.int8)
    puntero = 0
    for tipo, fraccion, _ in capas[:-1]:
        n = int(round(fraccion * orden.size))
        tipos[orden[puntero:puntero + n]] = CODIGOS[tipo]
        puntero += n
    tipos[orden[puntero:]] = CODIGOS[capas[-1][0]]
    tipos = tipos.reshape(tam, tam)

    # colores lisos (con un poco de variación de brillo entre granos)
    factor = np.where(tipos <= 4, geo["brillo"], 1.0)
    img = TABLA_COLORES[tipos] * factor[..., None]

    # perlita: láminas alternadas de ferrita y cementita
    perlita = tipos == 6
    img[perlita & geo["gruesas"]] = GRIS_OSCURO
    img[perlita & ~geo["gruesas"]] = VERDE_ALFA

    # ledeburita: matriz de cementita con manchas redondas de austenita o de perlita
    led_g = tipos == 7
    led_p = tipos == 8
    img[led_g | led_p] = GRIS_OSCURO
    img[led_g & geo["manchas"]] = AZUL_GAMMA
    mancha_p = led_p & geo["manchas"]
    img[mancha_p & ~geo["finas"]] = VERDE_ALFA
    img[mancha_p & geo["finas"]] = GRIS_OSCURO

    # límites de grano (no donde hay líquido ni sobre la cementita de los límites)
    liquido = tipos == 0
    junto_a_liquido = liquido | np.roll(liquido, -1, axis=1) | np.roll(liquido, -1, axis=0)
    img[geo["borde"] & ~junto_a_liquido & (tipos != 5)] = (90, 90, 90)

    return np.clip(img, 0, 255).astype(np.uint8)

# --- FIN MOTOR ---


# ============================================================
# 2. TEXTOS Y ELEMENTOS VISUALES
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
    "cementita_red": "#464646",
    "perlita": "repeating-linear-gradient(45deg,#A8D5A2 0 6px,#464646 6px 8px)",
    "ledeburita_gamma": "linear-gradient(90deg,#464646 50%,#A9D6F5 50%)",
    "ledeburita_perlita": "linear-gradient(90deg,#464646 50%,#A8D5A2 50%)",
}
TIPOS_OSCUROS = ("cementita_red", "ledeburita_gamma", "ledeburita_perlita")

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

GLOSARIO = [
    ("Ferrita (α)", "Hierro con estructura BCC que disuelve muy poco carbono (máximo 0.02 %). Es blanda y dúctil."),
    ("Austenita (γ)", "Hierro con estructura FCC que disuelve hasta 2.11 %C. En los aceros al carbono solo es "
                      "estable por encima de 727 °C."),
    ("Cementita (Fe₃C)", "Compuesto de hierro y carbono con 6.67 %C. Es muy dura y frágil."),
    ("Perlita", "Mezcla de láminas alternadas de ferrita y cementita. Se forma cuando la austenita de 0.8 %C "
                "se transforma a 727 °C (reacción eutectoide)."),
    ("Ledeburita", "Mezcla eutéctica de austenita y cementita que se forma a 1148 °C en las fundiciones. "
                   "Al bajar de 727 °C, su austenita se vuelve perlita."),
    ("Proeutectoide", "Lo que se forma antes de la reacción eutectoide: ferrita en los aceros con menos de "
                      "0.8 %C y cementita en los que tienen más."),
    ("Hipo / hiper", "Menos / más carbono que el punto de la reacción: 0.8 %C para la eutectoide y 4.3 %C "
                     "para la eutéctica."),
    ("Acero y fundición", "Se llama acero a la aleación con hasta 2.11 %C y fundición a la que tiene más "
                          "(hasta 6.67 %C)."),
    ("Fase y constituyente", "Una fase es una porción con composición y estructura uniformes (α, γ, Fe₃C). "
                             "Un constituyente es lo que se ve en el microscopio: la perlita, por ejemplo, "
                             "está formada por dos fases (α + Fe₃C)."),
    ("Línea de reparto", "Línea horizontal, a la temperatura elegida, que atraviesa una región de dos fases. "
                         "Sus extremos dan la composición de cada fase."),
    ("Regla de la palanca", "Con la línea de reparto se calcula cuánto hay de cada fase: la cantidad de una "
                            "fase es la distancia del punto al extremo opuesto, entre la longitud total."),
]


def css_color(tipo):
    if tipo in COLOR_LEYENDA:
        return COLOR_LEYENDA[tipo]
    return "#%02x%02x%02x" % COLORES[tipo]


def tipos_unicos(capas):
    tipos = []
    for tipo, fraccion, _ in capas:
        if tipo not in tipos and fraccion > 0:
            tipos.append(tipo)
    return tipos


def chip(tipo):
    return (f'<span style="display:inline-block;width:26px;height:16px;border:1px solid #8a97a3;'
            f'border-radius:4px;background:{css_color(tipo)}"></span>')


def leyenda_html(tipos):
    filas = [f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;color:#1b2733;'
             f'font-size:14px">{chip(t)}<span>{NOMBRES_TIPO[t]}</span></div>' for t in tipos]
    return "".join(filas)


def barra_html(items):
    """Barra apilada. items: lista de (nombre, fracción, tipo)."""
    segmentos = []
    for nombre, fraccion, tipo in items:
        if fraccion <= 1e-9:
            continue
        color_texto = "#ffffff" if tipo in TIPOS_OSCUROS else "#1b2733"
        etiqueta = f"{fraccion * 100:.0f} %" if fraccion >= 0.1 else ""
        segmentos.append(
            f'<div title="{nombre}: {fraccion * 100:.1f} %" style="width:{fraccion * 100:.2f}%;'
            f'background:{css_color(tipo)};display:flex;align-items:center;justify-content:center;'
            f'font-size:13px;font-weight:700;color:{color_texto}">{etiqueta}</div>')
    return ('<div style="display:flex;height:30px;border-radius:8px;overflow:hidden;'
            'border:1px solid #8a97a3;margin:6px 0 10px">' + "".join(segmentos) + "</div>")


def fila_html(tipo, titulo, detalle, porcentaje):
    return (f'<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;margin:6px 0;'
            f'border:1px solid #d5dde5;border-radius:10px;background:#ffffff;color:#1b2733">'
            f'{chip(tipo)}<div style="flex:1"><b>{titulo}</b>'
            f'<div style="color:#5b6b7a;font-size:13px">{detalle}</div></div>'
            f'<div style="font-size:20px;font-weight:800">{porcentaje:.1f} %</div></div>')


def formato_comp(valor):
    return f"{valor:.3f}".rstrip("0").rstrip(".") if valor else "0"


def fases_html(info):
    fases = [f for f in info["fases"] if f["frac"] > 1e-9]
    barra = barra_html([(NOMBRES_FASE[f["fase"]], f["frac"], TIPO_DE_FASE[f["fase"]]) for f in fases])
    filas = [fila_html(TIPO_DE_FASE[f["fase"]], NOMBRES_FASE[f["fase"]],
                       f"Composición: {formato_comp(f['comp'])} %C", f["frac"] * 100) for f in fases]
    return barra + "".join(filas)


def micro_html(micro):
    barra = barra_html([(nombre, frac, tipo) for tipo, nombre, frac, _ in micro])
    filas = [fila_html(tipo, nombre, "", frac * 100) for tipo, nombre, frac, _ in micro if frac > 1e-9]
    return barra + "".join(filas)


def resumen_html(info, x0, T):
    return ('<div style="background:linear-gradient(135deg,#EAF3FB,#F1F8EE);border:1px solid #cfdde9;'
            'border-radius:12px;padding:12px 16px;color:#1b2733">'
            '<div style="font-size:13px;color:#5b6b7a">Aleación</div>'
            f'<div style="font-size:21px;font-weight:800">{info["tipo"]}</div>'
            f'<div style="margin-top:6px;font-size:14px">%C = <b>{x0:.3f}</b> · T = <b>{T:.0f} °C</b> · '
            f'Región: <b>{info["region"]["nombre"]}</b></div></div>')


def cabecera_cuadro_html(T, info, nota):
    aviso = (f'<div style="margin-top:6px;color:#9a6700;font-weight:700">{nota}</div>') if nota else ""
    return ('<div style="display:flex;align-items:center;gap:16px;background:linear-gradient(135deg,#FDF2E3,#EAF3FB);'
            'border:1px solid #d9e2ea;border-radius:12px;padding:10px 16px;color:#1b2733">'
            f'<div style="font-size:32px;font-weight:800">{T:.0f} °C</div>'
            '<div><div style="font-size:13px;color:#5b6b7a">Región del diagrama</div>'
            f'<div style="font-size:19px;font-weight:700">{info["region"]["nombre"]}</div></div></div>' + aviso)


def eventos_html(eventos):
    filas = [(f'<div style="display:flex;gap:12px;align-items:flex-start;margin:7px 0;color:#1b2733">'
              f'<span style="min-width:84px;background:#1F4E79;color:#ffffff;border-radius:8px;'
              f'padding:3px 8px;text-align:center;font-weight:700">{T} °C</span><span>{texto}</span></div>')
             for T, texto in eventos]
    return "".join(filas)


def pasos_calculo(x0, T, info):
    """Texto con el cálculo paso a paso (regla de la palanca y constituyentes)."""
    lineas = []
    if info["tie"] is None:
        lineas.append("Hay una sola fase: toda la aleación es esa fase y su composición es la de la "
                      "aleación. No hace falta la regla de la palanca.")
    else:
        xl, xr = info["tie"]
        izq, der = info["fases"]
        n_izq, n_der = NOMBRES_FASE[izq["fase"]], NOMBRES_FASE[der["fase"]]
        lineas += [
            "**Fases (regla de la palanca)**",
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


BANNER = """
<div style="background:linear-gradient(120deg,#1F4E79 0%,#2E86C1 55%,#6DBE8C 100%);color:#ffffff;
padding:22px 28px;border-radius:14px;margin-bottom:6px;font-family:'Segoe UI',Roboto,Arial,sans-serif">
<div style="font-size:30px;font-weight:800;letter-spacing:0.3px">🧲 Simulador del diagrama Fe–Fe₃C</div>
<div style="font-size:15px;margin-top:6px;opacity:0.95">Ubica un punto, descubre qué fases hay y cómo se ve la
microestructura, y mira cómo cambia al enfriar la aleación.</div>
</div>
"""


# ============================================================
# 3. DIAGRAMA INTERACTIVO
# ============================================================

ETIQUETAS = [  # (%C, T, texto, tamaño)
    (3.5, 1450, "L", 16), (2.0, 1310, "γ + L", 12), (1.0, 1000, "Austenita (γ)", 15),
    (0.28, 820, "α+γ", 10), (4.3, 900, "γ + Fe₃C", 14), (6.12, 1175, "L+Fe₃C", 9),
    (3.4, 650, "α + Fe₃C", 14),
]


def crear_diagrama(x0, T, info, vista, trayecto_desde=None):
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
    for x, t, texto in [(6.62, T_EUTECTOIDE, "727 °C"), (6.62, T_EUTECTICO, "1148 °C"), (0.6, T_PERITECTICO, "1495 °C")]:
        fig.add_annotation(x=x, y=t, text=texto, showarrow=False, yshift=11, xanchor="right",
                           font=dict(size=10, color="#c0392b"))

    # Zonas de microconstituyentes (debajo de 727 °C)
    for x, texto in [(0.4, "α + perlita"), (1.45, "perlita + Fe₃C"),
                     (3.2, "perlita + Fe₃C<br>+ ledeburita"), (5.5, "Fe₃C + ledeburita")]:
        fig.add_annotation(x=x, y=150, text=texto, showarrow=False, font=dict(size=10, color="#444"))

    # Trayecto de enfriamiento (animación)
    if trayecto_desde is not None:
        fig.add_trace(go.Scatter(x=[x0, x0], y=[trayecto_desde, T], mode="lines",
                                 line=dict(color="#d62728", width=2, dash="dot"),
                                 name="Trayecto de enfriamiento", hoverinfo="skip"))

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


def mostrar_cuadro(cuadro, x0, vista, ph_diagrama, ph_info, clave):
    """Dibuja un cuadro de la animación (diagrama a la izquierda, microestructura a la derecha)."""
    info = cuadro["info"]
    ph_diagrama.plotly_chart(
        crear_diagrama(x0, cuadro["T"], info, vista, trayecto_desde=T_MAX),
        use_container_width=True, key=f"diagrama_{clave}")
    with ph_info.container():
        st.html(cabecera_cuadro_html(cuadro["T"], info, cuadro["nota"]))
        st.image(generar_micro(cuadro["capas"]), width=340)
        st.html(leyenda_html(tipos_unicos(cuadro["capas"])))
        st.html(fases_html(info))


@st.cache_data
def eventos_cache(x0):
    return eventos_enfriamiento(x0)


# ============================================================
# 4. PÁGINA
# ============================================================

st.set_page_config(page_title="Simulador Fe–Fe₃C", page_icon="🧲", layout="wide")
st.html(BANNER)

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
st.session_state.setdefault("C_anim", 0.4)
st.session_state.setdefault("T_anim", T_MAX)
if "T_anim_sig" in st.session_state:  # al terminar la animación, el control queda en la temperatura final
    st.session_state["T_anim"] = st.session_state.pop("T_anim_sig")

tab_sim, tab_anim, tab_guia = st.tabs(["🎯 Simulador de punto", "🎬 Enfriamiento animado",
                                       "📚 Guía y glosario"])

# ---------- Guía y glosario (se escribe primero para que cargue rápido) ----------
with tab_guia:
    st.markdown("### Cómo usar el simulador")
    st.markdown("""
1. En **Simulador de punto**, escribe el **%C** y la **temperatura**, o elige un ejemplo. El punto (⭐)
   aparece en el diagrama y se calculan las fases, su composición, su cantidad y los constituyentes.
2. En **Enfriamiento animado**, elige un %C y pulsa **Reproducir**: verás cómo cambia la microestructura
   desde el líquido hasta temperatura ambiente. También puedes arrastrar la temperatura a mano.
3. Abre **Cálculo paso a paso** para ver la regla de la palanca con tus números y comprobarla a mano.

Todo corresponde al diagrama de **equilibrio** (enfriamiento muy lento).
    """)
    st.markdown("### Glosario")
    columnas = st.columns(2)
    for i, (termino, definicion) in enumerate(GLOSARIO):
        with columnas[i % 2]:
            st.html(f'<div style="border:1px solid #d5dde5;border-radius:10px;padding:10px 14px;margin:6px 0;'
                    f'background:#ffffff;color:#1b2733"><b>{termino}</b><br>'
                    f'<span style="font-size:14px;color:#3b4a58">{definicion}</span></div>')
    st.markdown("### Límites del simulador")
    st.markdown("""
- Es una **lectura del diagrama de equilibrio** (enfriamiento muy lento), no un cálculo termodinámico.
  Las líneas curvas del diagrama se aproximan con segmentos de recta, así que las composiciones leídas
  son aproximadas.
- Valores clave usados: eutectoide 0.8 %C a 727 °C, eutéctico 4.3 %C a 1148 °C, máxima solubilidad de C
  en γ 2.11 %C, en α 0.02 %C, cementita 6.67 %C y peritéctico a 1495 °C.
- Las microestructuras son **esquemas** (los granos y las láminas se generan al azar), no fotografías.
  En una pieza real influyen la velocidad de enfriamiento y otros elementos de aleación.
- En la animación, las reacciones a temperatura constante (eutéctica y eutectoide) se muestran con cuadros
  intermedios para ver cómo avanzan; en el diagrama ocurren a una sola temperatura.
- Si el punto está justo sobre una línea de reacción, el resultado corresponde a la aleación inmediatamente
  por debajo de esa línea.
    """)

# ---------- Simulador de punto ----------
with tab_sim:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
        x0 = c1.number_input("% de carbono (%C)", min_value=0.0, max_value=C_CEMENTITA,
                             step=0.05, format="%.3f", key="C")
        T = c2.number_input("Temperatura (°C)", min_value=float(T_MIN), max_value=float(T_MAX),
                            step=10.0, format="%.1f", key="T")
        c3.selectbox("Ejemplos", list(EJEMPLOS), key="ejemplo", on_change=cargar_ejemplo)
        vista = c4.radio("Vista del diagrama", ["Completo", "Zona de aceros (0 a 2.2 %C)"],
                         horizontal=True, key="vista_sim")
        ferrita_fija = st.checkbox("Usar ferrita con 0.02 %C constante por debajo de 727 °C "
                                   "(simplificación habitual en clase)", value=True)

    info = analizar(x0, T, ferrita_fija)
    if info is None:
        st.error("No se encontró una región para ese punto. Revisa los valores.")
        st.stop()

    col_diagrama, col_resultados = st.columns([3, 2])
    with col_diagrama:
        st.plotly_chart(crear_diagrama(x0, T, info, vista), use_container_width=True, key="diagrama_punto")

    with col_resultados:
        st.html(resumen_html(info, x0, T))
        if info["reaccion"]:
            st.warning(info["reaccion"])
        st.markdown("#### Fases presentes")
        st.html(fases_html(info))
        if info["micro"] is not None:
            st.markdown("#### Constituyentes a temperatura ambiente")
            st.html(micro_html(info["micro"]))

    st.markdown("---")
    col_img, col_texto = st.columns([1, 2])
    with col_img:
        st.markdown("#### 🔬 Microestructura (esquema)")
        st.image(generar_micro(info["capas"]), width=340,
                 caption="Dibujo esquemático, no a escala y no es una fotografía real.")
        st.html(leyenda_html(tipos_unicos(info["capas"])))
    with col_texto:
        with st.container(border=True):
            st.markdown("#### 📖 ¿Qué está pasando?")
            st.markdown(f"**En esta región:** {EXPLICA_REGION[info['region']['id']]}")
            if info["micro"] is not None:
                st.markdown(f"**Cómo se llegó aquí:** {EXPLICA_ALEACION[info['tipo']]}")
        with st.expander("🧮 Cálculo paso a paso", expanded=False):
            st.markdown(pasos_calculo(x0, T, info))

# ---------- Enfriamiento animado (va al final: la animación ocupa el script mientras se reproduce) ----------
with tab_anim:
    st.markdown("### Cómo cambia la microestructura al enfriar")
    with st.container(border=True):
        a1, a2, a3, a4 = st.columns([1, 1, 2, 1])
        x_a = a1.number_input("% de carbono (%C)", min_value=0.0, max_value=C_CEMENTITA,
                              step=0.05, format="%.3f", key="C_anim")
        velocidad = a2.selectbox("Velocidad", ["Lenta", "Media", "Rápida"], index=1)
        vista_a = a3.radio("Vista del diagrama", ["Completo", "Zona de aceros (0 a 2.2 %C)"],
                           horizontal=True, key="vista_anim")
        reproducir = a4.button("▶ Reproducir", type="primary", use_container_width=True)
        T_a = st.slider("Temperatura (°C): arrástrala para ver cualquier momento del enfriamiento",
                        T_MIN, T_MAX, step=5, key="T_anim")
        barra = st.progress(0.0)

    with st.expander("📝 ¿Qué le pasa a esta aleación al enfriarse?", expanded=True):
        st.html(eventos_html(eventos_cache(round(x_a, 4))))

    col_d, col_i = st.columns([3, 2])
    ph_diagrama = col_d.empty()
    ph_info = col_i.empty()

    if reproducir:
        cuadros = construir_cuadros(x_a)
        pausa = {"Lenta": 0.4, "Media": 0.2, "Rápida": 0.08}[velocidad]
        for i, cuadro in enumerate(cuadros):
            mostrar_cuadro(cuadro, x_a, vista_a, ph_diagrama, ph_info, f"play_{i}")
            barra.progress((i + 1) / len(cuadros))
            time.sleep(pausa)
        st.session_state["T_anim_sig"] = T_MIN
        st.rerun()
    else:
        info_a = analizar(x_a, float(T_a))
        cuadro = {"T": T_a, "info": info_a, "capas": info_a["capas"], "nota": ""}
        mostrar_cuadro(cuadro, x_a, vista_a, ph_diagrama, ph_info, "fijo")