"""Green-Walker: Fußwege mit möglichst viel Park- und Grünflächen-Nähe.

Zwei Modi: Route von A nach B und "Ründchen" (grüne Runde mit Wunschlänge).

Start:  pip install -r requirements.txt
        streamlit run app.py

Daten:  OpenStreetMap (Overpass via osmnx, ODbL) für Wegenetz + Grünflächen.
        Ortssuche mit Vorschlägen: Photon (photon.komoot.io, OSM-Daten, fehler-
        tolerant, findet auch Läden/Restaurants), Fallback Nominatim.
Kosten: kosten(kante) = laenge * (1 + LAMBDA * (1 - gruen(kante)))
Grün:   Mittel über 8 Abtastpunkte je Kante; 1.0 im Park, linear fallend
        auf 0 im Abstand "Grün-Reichweite".
"""
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import requests
import shapely
import streamlit as st
from shapely import STRtree
from shapely.geometry import LineString, Point
from streamlit.components.v1 import html as st_html
from streamlit_js_eval import get_geolocation
from streamlit_searchbox import st_searchbox

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 30
ox.settings.overpass_rate_limit = False

# Mehrere Overpass-Server: der Hauptserver ist oft überlastet (502/504).
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
    "https://lz4.overpass-api.de/api",
    "https://z.overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]
LOAD_DEADLINE_S = 150  # nach dieser Zeit gibt die App auf und zeigt eine verständliche Meldung
try:
    from osmnx import errors as ox_errors
except Exception:  # pragma: no cover
    ox_errors = None
NO_DATA = getattr(ox_errors, "InsufficientResponseError", ())  # "keine Treffer" ist kein Serverfehler

_OV_ATTR = "overpass_url" if hasattr(ox.settings, "overpass_url") else "overpass_endpoint"
_OV_SUFFIX = "/interpreter" if str(getattr(ox.settings, _OV_ATTR, "")).rstrip("/").endswith("/interpreter") else ""


def use_mirror(base):
    setattr(ox.settings, _OV_ATTR, base + _OV_SUFFIX)


UA = {"User-Agent": "green-walker-app/1.0 (privates Hobbyprojekt)"}
GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve", "recreation_ground"],
    "landuse": ["grass", "forest", "village_green"],
    "natural": ["wood"],
}
# Naturtöne: Moos, Salbei, Sand, Lehm, Stein
C_MOSS, C_SAGE, C_SAND = "#4E6E45", "#9DB38A", "#E9E1D2"
C_ROUTE, C_PULSE = "#3F6B3A", "#CFE3BE"          # grüne Route
C_SHORT = "#C08A47"                              # kürzeste Route (Ocker statt Rot)
C_STONE = "#8A8577"                              # andere Varianten
C_PARK_FILL, C_PARK_LINE = "#B5CDA0", "#7C9F69"
C_TEXT = "#3B3A33"

THEME_TOML = """[theme]
base = "light"
primaryColor = "#5E7F52"
backgroundColor = "#F4EFE6"
secondaryBackgroundColor = "#E9E1D2"
textColor = "#3B3A33"
font = "sans serif"
"""


def ensure_theme():
    """Legt die Theme-Datei an, falls sie fehlt (greift ggf. erst nach einem Neustart der App)."""
    p = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(p):
        try:
            os.makedirs(".streamlit", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(THEME_TOML)
        except Exception:
            pass


ensure_theme()

KIND_DE = {"footway": "Fußweg", "path": "Pfad", "pedestrian": "Fußgängerzone", "steps": "Treppe",
           "cycleway": "Radweg", "track": "Feldweg", "service": "Zufahrtsweg"}


def edge_label(d):
    """(Straßenname, Wegart) einer Kante, jeweils '' wenn unbekannt."""
    n = d.get("name")
    if isinstance(n, (list, tuple)):
        n = n[0] if n else None
    hw = d.get("highway")
    if isinstance(hw, (list, tuple)):
        hw = hw[0] if hw else None
    return (str(n) if n else ""), KIND_DE.get(hw, "")


SAMPLES = 8
MAX_BUF = 400
MAX_SPAN_M = 9000
M_PER_MIN = 80

LEVELS = {"Schnellste": 0.0, "Leicht grün": 1.0, "Ausgewogen": 3.0,
          "Sehr grün": 6.0, "Maximal grün": 12.0}
LEVEL_HINT = {
    "Schnellste": "Nur die Distanz zählt.",
    "Leicht grün": "Kleine Umwege für spürbar mehr Grün.",
    "Ausgewogen": "Guter Mittelweg aus Weg und Grün.",
    "Sehr grün": "Deutliche Umwege sind okay, wenn es grüner wird.",
    "Maximal grün": "Grün hat klar Vorrang vor der Länge.",
}


# ------------------------------------------------------- Ortssuche
def _icon(p):
    k, v = p.get("osm_key", ""), p.get("osm_value", "")
    if v in ("restaurant", "fast_food", "food_court", "biergarten"):
        return "🍽️"
    if v in ("cafe", "ice_cream"):
        return "☕"
    if v in ("bar", "pub"):
        return "🍺"
    if k == "shop":
        return "🛍️"
    if k == "leisure" or v in ("park", "garden"):
        return "🌳"
    if k == "highway":
        return "🛣️"
    if k in ("railway", "public_transport") or v in ("bus_stop", "station"):
        return "🚉"
    if k == "tourism":
        return "🏛️"
    if k == "amenity":
        return "📌"
    return "📍"


def _label(p):
    name = p.get("name")
    addr = " ".join(x for x in [p.get("street"), p.get("housenumber")] if x)
    title = name or addr or p.get("city") or p.get("state") or "Ort"
    loc = " ".join(x for x in [p.get("postcode"),
                               p.get("city") or p.get("town") or p.get("district") or p.get("county")] if x)
    parts = ([addr] if (name and addr) else []) + ([loc] if loc else [])
    return f"{_icon(p)} {title}" + (" · " + ", ".join(parts) if parts else "")


def _pack(lat, lon, label):
    return f"{lat:.6f}|{lon:.6f}|{label}"


def _unpack(v):
    lat, lon, label = v.split("|", 2)
    return float(lat), float(lon), label


STATION_WORDS = {"bahnhof", "bhf", "bf", "hbf", "hauptbahnhof", "station", "haltestelle"}
STATION_TAGS = ["railway:station", "railway:halt", "public_transport:station"]


@st.cache_data(ttl=3600, show_spinner=False)
def _photon(q, blat, blon, tag=None):
    params = {"q": q, "limit": 8, "lang": "de"}
    if blat is not None:
        params.update(lat=blat, lon=blon)
    if tag:
        params["osm_tag"] = tag
    r = requests.get("https://photon.komoot.io/api/", params=params, headers=UA, timeout=6)
    r.raise_for_status()
    return r.json().get("features", [])


@st.cache_data(ttl=3600, show_spinner=False)
def _nominatim(q):
    r = requests.get("https://nominatim.openstreetmap.org/search", headers=UA, timeout=8,
                     params={"q": q, "format": "jsonv2", "limit": 6, "accept-language": "de"})
    r.raise_for_status()
    out = []
    for x in r.json():
        lab = "📍 " + ", ".join(x["display_name"].split(", ")[:3])
        out.append((lab, _pack(float(x["lat"]), float(x["lon"]), lab)))
    return out


def _add(feats, seen, out):
    for f in feats:
        try:
            lon, lat = f["geometry"]["coordinates"]
            lab = _label(f["properties"])
        except Exception:
            continue
        if lab in seen:
            continue
        seen.add(lab)
        out.append((lab, _pack(lat, lon, lab)))


def search_places(term):
    """Vorschläge während der Eingabe.
    - "Bahnhof Ehrenfeld" findet Bahnhöfe/Haltestellen mit "Ehrenfeld" im Namen
      (OSM nennt sie oft "Köln-Ehrenfeld", ohne das Wort Bahnhof).
    - Ohne Treffer wird die Eingabe gekürzt (ohne letztes/erstes Wort), danach Nominatim."""
    term = (term or "").strip().replace("str.", "straße").replace("Str.", "Straße")
    if len(term) < 3:
        return []
    bias = st.session_state.get("bias")
    blat, blon = (round(bias[0], 2), round(bias[1], 2)) if bias else (None, None)

    def ask(q, tag=None):
        try:
            return _photon(q, blat, blon, tag)
        except Exception:
            return []

    out, seen = [], set()
    words = term.split()
    core = [w for w in words if w.lower().strip(".,") not in STATION_WORDS]
    if core and len(core) < len(words):
        for tag in STATION_TAGS:
            _add(ask(" ".join(core), tag), seen, out)
    _add(ask(term), seen, out)
    if not out and len(words) > 1:
        for q in (" ".join(words[:-1]), " ".join(words[1:])):
            _add(ask(q), seen, out)
            if out:
                break
    if not out:
        try:
            out = _nominatim(term)
        except Exception:
            out = []
    return out[:8]


# ------------------------------------------------------- Daten
def corridor_bbox(p1, p2):
    (lat1, lon1), (lat2, lon2) = p1, p2
    ky = 111_320.0
    kx = ky * math.cos(math.radians((lat1 + lat2) / 2))
    span = max(abs(lon2 - lon1) * kx, abs(lat2 - lat1) * ky)
    pad = max(500.0, 0.25 * span)
    bbox = (round(min(lon1, lon2) - pad / kx, 4), round(min(lat1, lat2) - pad / ky, 4),
            round(max(lon1, lon2) + pad / kx, 4), round(max(lat1, lat2) + pad / ky, 4))
    return bbox, span


@st.cache_resource(show_spinner=False, max_entries=4)
def load_area(bbox):
    def get_graph():
        return ox.graph_from_bbox(bbox, network_type="walk", simplify=True)

    def get_parks():
        try:
            gdf = ox.features_from_bbox(bbox, tags=GREEN_TAGS)
        except NO_DATA:
            return None  # wirklich keine Grünflächen im Gebiet
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not len(gdf):
            return None
        gdf = gdf.reset_index(drop=True)
        if "name" not in gdf:
            gdf["name"] = None
        gdf["name"] = gdf["name"].fillna("Grünfläche").astype(str)
        return gdf[["name", "geometry"]]

    # Server in zufälliger Reihenfolge probieren (der Hauptserver wird nicht immer zuerst belastet),
    # mehrere Runden mit wachsender Pause. Bereits erfolgreich geladene Teile liegen im osmnx-Cache
    # und werden bei einem erneuten Versuch nicht noch einmal abgerufen.
    G0 = parks = last = None
    t_end = time.time() + LOAD_DEADLINE_S
    for rnd in range(3):
        for base in random.sample(OVERPASS_MIRRORS, len(OVERPASS_MIRRORS)):
            if time.time() > t_end:
                break
            use_mirror(base)
            try:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_g, f_p = ex.submit(get_graph), ex.submit(get_parks)
                    G0, parks = f_g.result(), f_p.result()
                break
            except Exception as e:
                last = e
                G0 = parks = None
        if G0 is not None or time.time() > t_end:
            break
        time.sleep(5 * (rnd + 1))
    if G0 is None:
        raise RuntimeError(f"Alle Kartenserver sind gerade überlastet. ({last})")

    G = ox.project_graph(G0)
    crs = G.graph["crs"]
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    best = {}
    for u, v, d in G.edges(data=True):
        if (u, v) not in best or d["length"] < best[(u, v)][0]:
            best[(u, v)] = (d["length"], d.get("geometry") or LineString([pos[u], pos[v]]), *edge_label(d))
    keys = list(best)
    lengths = np.array([best[k][0] for k in keys], dtype=float)
    names = [best[k][2] for k in keys]
    kinds = [best[k][3] for k in keys]
    lines = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        lines[i] = best[k][1]
    H = nx.DiGraph()
    for i, (u, v) in enumerate(keys):
        H.add_edge(u, v, i=i)

    D = np.full((SAMPLES, len(keys)), MAX_BUF, dtype=np.float32)
    parks_json, n_parks = None, 0
    if parks is not None:
        pg = parks.to_crs(crs)
        pg["geometry"] = pg.geometry.simplify(2.0).make_valid()
        pg = pg[pg.geometry.notna() & ~pg.geometry.is_empty].reset_index(drop=True)
        n_parks = len(pg)
        if n_parks:
            tree = STRtree(list(pg.geometry))
            for r, f in enumerate((np.arange(SAMPLES) + 0.5) / SAMPLES):
                pts = shapely.line_interpolate_point(lines, f, normalized=True)
                idx, dist = tree.query_nearest(pts, max_distance=MAX_BUF,
                                               return_distance=True, all_matches=False)
                D[r, idx[0]] = dist
            parks_json = pg.to_crs(4326).to_json()

    node_list = list(pos)
    nidx = {n: i for i, n in enumerate(node_list)}
    return {"crs": crs, "pos": pos, "H": H, "lengths": lengths, "lines": lines, "D": D,
            "node_list": node_list, "xy": np.array([pos[n] for n in node_list]),
            "eu": np.array([nidx[u] for u, v in keys]), "ev": np.array([nidx[v] for u, v in keys]),
            "names": names, "kinds": kinds,
            "parks_json": parks_json, "n_parks": n_parks, "n_edges": len(keys)}


def nearest_node(area, lat, lon):
    x, y = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(area["crs"]).iloc[0].coords[0]
    d = (area["xy"][:, 0] - x) ** 2 + (area["xy"][:, 1] - y) ** 2
    return area["node_list"][int(d.argmin())]


# ------------------------------------------------------- Routing
def solve(area, o, t, lam, buf):
    g = np.clip(1 - area["D"] / buf, 0, 1).mean(axis=0)
    L, H = area["lengths"], area["H"]
    weights = {"short": lambda u, v, d: L[d["i"]],
               "green": lambda u, v, d: L[d["i"]] * (1 + lam * (1 - g[d["i"]]))}
    out = {}
    for name, w in weights.items():
        path = nx.shortest_path(H, o, t, weight=w)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum())
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        out[name] = {"path": path, "dist": dist, "green": share}
    return out


# ------------------------------------------------------- Ründchen
LOOP_BEARINGS = 8      # Richtungen, in die Runden probiert werden
LOOP_TOLERANCE = 0.07  # akzeptierte Längenabweichung beim Anpassen (±7 %)
LOOP_REUSE_PENALTY = 4.0  # Strafe für bereits gelaufene Wegstücke (vermeidet Hin-und-zurück)
LOOP_MAX_KM = 6.0


def dist_m(p1, p2):
    (la1, lo1), (la2, lo2) = p1, p2
    x = math.radians(lo2 - lo1) * math.cos(math.radians((la1 + la2) / 2))
    y = math.radians(la2 - la1)
    return 6_371_000 * math.hypot(x, y)


def loop_bbox(p1, p2, target):
    (lat1, lon1), (lat2, lon2) = p1, p2
    ky = 111_320.0
    clat, clon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    kx = ky * math.cos(math.radians(clat))
    half = 0.38 * target + 400 + dist_m(p1, p2) / 2
    return (round(clon - half / kx, 4), round(clat - half / ky, 4),
            round(clon + half / kx, 4), round(clat + half / ky, 4))


def node_green(area, g):
    n = len(area["node_list"])
    s, c = np.zeros(n), np.zeros(n)
    for ends in (area["eu"], area["ev"]):
        np.add.at(s, ends, g)
        np.add.at(c, ends, 1)
    return s / np.maximum(c, 1)


def pick_waypoint(area, ng, x, y, tol):
    """Knoten nahe (x, y); bevorzugt grüne Umgebung, damit die Runde Parks ansteuert."""
    d = np.hypot(area["xy"][:, 0] - x, area["xy"][:, 1] - y)
    cand = np.where(d <= tol)[0]
    if len(cand) == 0:
        cand = np.array([int(d.argmin())])
    score = ng[cand] - 0.5 * d[cand] / tol
    return area["node_list"][int(cand[score.argmax()])]


def build_loop_path(area, g, ng, lam, o, t, theta, rho):
    """Runde A -> 3 Wegpunkte auf einem Kreis -> B. Jede Teilstrecke wird grün-optimal geroutet;
    schon genutzte Wegstücke werden teurer, damit möglichst keine Doppelstrecken entstehen."""
    H, L = area["H"], area["lengths"]
    ax, ay = area["pos"][o]
    cx, cy = ax + rho * math.cos(theta), ay + rho * math.sin(theta)
    wps = []
    for alpha in (90, 180, 270):
        ang = theta + math.pi + math.radians(alpha)
        wps.append(pick_waypoint(area, ng, cx + rho * math.cos(ang), cy + rho * math.sin(ang), 0.3 * rho))
    seq = [o] + wps + [t]
    used, full = set(), [o]
    for s, e in zip(seq[:-1], seq[1:]):
        if s == e:
            continue

        def w(u, v, d, used=used):
            i = d["i"]
            c = L[i] * (1 + lam * (1 - g[i]))
            return c * LOOP_REUSE_PENALTY if ((u, v) in used or (v, u) in used) else c

        p = nx.shortest_path(H, s, e, weight=w)
        full += p[1:]
        used.update(zip(p[:-1], p[1:]))
    return full


def edge_set(path):
    return frozenset((min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:]))


def loop_stats(area, g, path):
    H, Ls = area["H"], area["lengths"]
    pairs = list(zip(path[:-1], path[1:]))
    if not pairs:
        return None
    idx = [H[a][b]["i"] for a, b in pairs]
    l = Ls[idx]
    dist = float(l.sum())
    if dist <= 0:
        return None
    seen, rep = set(), 0.0
    for (a, b), li in zip(pairs, l):
        k = (min(a, b), max(a, b))
        if k in seen:
            rep += li
        else:
            seen.add(k)
    return dist, float((l * g[idx]).sum() / dist), float(rep / dist)


def find_loops(area, o, t, target, lam, buf, progress=None):
    """Probiert mehrere Richtungen, passt den Kreisradius an die Wunschlänge an und
    liefert bis zu 3 deutlich verschiedene Varianten (beste zuerst)."""
    g = np.clip(1 - area["D"] / buf, 0, 1).mean(axis=0)
    ng = node_green(area, g)
    cands = []
    for k in range(LOOP_BEARINGS):
        theta = 2 * math.pi * k / LOOP_BEARINGS
        rho = target / (2 * math.pi * 1.35)
        for _ in range(3):
            try:
                path = build_loop_path(area, g, ng, lam, o, t, theta, rho)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                break
            st_ = loop_stats(area, g, path)
            if not st_:
                break
            dist, green, rep = st_
            cands.append({"path": path, "dist": dist, "green": green, "rep": rep})
            ratio = dist / target
            if abs(ratio - 1) <= LOOP_TOLERANCE:
                break
            rho /= ratio
        if progress:
            progress((k + 1) / LOOP_BEARINGS)
    for c in cands:
        c["score"] = c["green"] - 0.8 * c["rep"] - 2.0 * abs(c["dist"] / target - 1)
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:
        if abs(c["dist"] / target - 1) > 0.2:
            continue
        es = edge_set(c["path"])
        if all(len(es & p["es"]) / len(es | p["es"]) < 0.6 for p in picked):
            c["es"] = es
            picked.append(c)
        if len(picked) == 3:
            break
    if not picked and cands:
        cands[0]["es"] = edge_set(cands[0]["path"])
        picked = [cands[0]]
    return picked


# ------------------------------------------------------- Navigation
COMPASS = ["Norden", "Nordosten", "Osten", "Südosten", "Süden", "Südwesten", "Westen", "Nordwesten"]


def route_geom(area, path):
    """Routenlinie in Projektion (Meter) + Index jedes Pfadknotens in dieser Linie."""
    coords, node_idx = [], [0]
    for a, b in zip(path[:-1], path[1:]):
        c = list(area["lines"][area["H"][a][b]["i"]].coords)
        if math.dist(c[0], area["pos"][a]) > math.dist(c[-1], area["pos"][a]):
            c.reverse()
        coords += c if not coords else c[1:]
        node_idx.append(len(coords) - 1)
    if len(coords) < 2:
        coords = [area["pos"][path[0]]] * 2
        node_idx = [0] * len(path)
    return coords, node_idx


def to_latlon(area, coords):
    ll = gpd.GeoSeries([LineString(coords)], crs=area["crs"]).to_crs(4326).iloc[0]
    return [(y, x) for x, y in ll.coords]


def _bearing(p, q):
    return math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 360


def _back(cum, idx, dmin=15.0):
    j = idx
    while j > 0 and cum[idx] - cum[j] < dmin:
        j -= 1
    return j


def _fwd(cum, idx, dmin=15.0):
    j = idx
    while j < len(cum) - 1 and cum[j] - cum[idx] < dmin:
        j += 1
    return j


def maneuvers(area, path, coords, node_idx, cum, loop=False):
    """Wegehinweise: Start, Abbiegungen (>= 30 Grad), Straßenwechsel an Kreuzungen, Ziel."""
    H = area["H"]

    def label(a, b):
        i = H[a][b]["i"]
        return area["names"][i], area["kinds"][i]

    def on(a, b):
        n, k = label(a, b)
        return n or k

    steps = []
    f0 = _fwd(cum, 0)
    b0 = _bearing(coords[0], coords[f0]) if f0 > 0 else 0.0
    first = on(path[0], path[1])
    steps.append({"idx": 0, "i": "🚶", "t": f"Los geht’s Richtung {COMPASS[int(round(b0 / 45)) % 8]}"
                                            + (f" auf {first}" if first else "")})
    for n in range(1, len(path) - 1):
        a, c, d = path[n - 1], path[n], path[n + 1]
        idx = node_idx[n]
        bi, fo = _back(cum, idx), _fwd(cum, idx)
        if bi == idx or fo == idx:
            continue
        din = _bearing(coords[bi], coords[idx])
        dout = _bearing(coords[idx], coords[fo])
        delta = ((dout - din + 540) % 360) - 180  # > 0: rechts
        mag, side = abs(delta), ("rechts" if delta > 0 else "links")
        target = on(c, d)
        suffix = f" auf {target}" if target else ""
        if mag < 30:
            na, nb = label(a, c)[0], label(c, d)[0]
            deg = len(set(H.successors(c)) | set(H.predecessors(c)))
            if na and nb and na != nb and deg >= 3:
                steps.append({"idx": idx, "i": "⬆", "t": f"Weiter geradeaus auf {nb}"})
            continue
        if mag < 60:
            text, icon = f"Halb {side} abbiegen{suffix}", ("↗" if delta > 0 else "↖")
        elif mag < 120:
            text, icon = f"{side.capitalize()} abbiegen{suffix}", ("↱" if delta > 0 else "↰")
        elif mag < 165:
            text, icon = f"Scharf {side} abbiegen{suffix}", ("↱" if delta > 0 else "↰")
        else:
            text, icon = "Wenden", "↩"
        steps.append({"idx": idx, "i": icon, "t": text})
    steps.append({"idx": node_idx[-1], "i": "🏁",
                  "t": "Ründchen geschafft, Ziel erreicht" if loop else "Du hast dein Ziel erreicht"})
    return steps


def build_nav(area, path, title, loop=False):
    if len(path) < 2:
        return None
    coords, node_idx = route_geom(area, path)
    cum = [0.0]
    for p, q in zip(coords[:-1], coords[1:]):
        cum.append(cum[-1] + math.dist(p, q))
    if cum[-1] < 5:
        return None
    ll = to_latlon(area, coords)
    steps = maneuvers(area, path, coords, node_idx, cum, loop)
    return {
        "title": title, "total": round(cum[-1]),
        "coords": [[round(la, 6), round(lo, 6)] for la, lo in ll],
        "cum": [round(c, 1) for c in cum],
        "steps": [{"d": round(cum[s["idx"]]), "lat": round(ll[s["idx"]][0], 6),
                   "lon": round(ll[s["idx"]][1], 6), "t": s["t"], "i": s["i"]} for s in steps],
    }


MAP_TEMPLATE = r"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<style>
:root{--top:0px;--moss:#4E6E45;--sage:#9DB38A;--sand:#E9E1D2;--paper:#FBF8F1;--line:#D9CFBC;--text:#3B3A33}
*{box-sizing:border-box}
html,body{margin:0;height:100%;overflow:hidden;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text);background:var(--sand);-webkit-tap-highlight-color:transparent}
#map{position:fixed;top:0;left:0;right:0;bottom:0;z-index:0}
.leaflet-container{background:#E9E1D2}
button{font-family:inherit;cursor:pointer}
#banner{position:fixed;z-index:1000;top:calc(var(--top) + 8px);left:10px;right:10px;max-width:620px;margin:0 auto;display:none;gap:12px;align-items:center;background:var(--moss);color:#FBF7EE;border-radius:16px;padding:10px 14px;box-shadow:0 6px 18px rgba(60,50,30,.3)}
body.nav #banner{display:flex}
#banner.off{background:#B07A35}#banner.done{background:#3F5B3A}
#ico{font-size:36px;line-height:1;width:46px;text-align:center;flex:none}
#dist{font-size:1.5rem;font-weight:700;line-height:1.1}
#ins{font-size:1rem;margin-top:2px}
#side{position:fixed;z-index:1000;right:10px;top:calc(var(--top) + 8px);display:flex;flex-direction:column;gap:8px}
body.nav #side{top:calc(var(--top) + 92px)}
#side button{width:44px;height:44px;border-radius:12px;border:1px solid var(--line);background:var(--paper);font-size:20px;box-shadow:0 3px 10px rgba(60,50,30,.25)}
#sheet{position:fixed;z-index:1000;left:10px;right:10px;bottom:10px;max-width:620px;margin:0 auto;background:rgba(251,248,241,.98);border:1px solid var(--line);border-radius:20px;padding:12px;box-shadow:0 8px 28px rgba(60,50,30,.3)}
#ttl{font-size:.85rem;opacity:.75;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#chips{display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;-webkit-overflow-scrolling:touch}
.chip{flex:none;border:1px solid var(--line);background:#fff;border-radius:14px;padding:8px 12px;text-align:left;min-width:108px;color:var(--text)}
.chip b{display:block;font-size:.92rem}.chip span{font-size:.78rem;opacity:.7}
.chip.on{background:#E7EDDB;border-color:var(--sage)}
#stats{display:flex;gap:8px;margin:2px 0 6px}
#stats div{flex:1;background:#F2EDE0;border-radius:12px;padding:6px 8px;min-width:0}
#stats span{display:block;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;opacity:.65;white-space:nowrap}
#stats b{font-size:1.05rem}
#hint{font-size:.8rem;opacity:.7;min-height:1em}
#navbox{display:none}
body.nav #navbox{display:block}body.nav #pre{display:none}
#rem{font-weight:600;font-size:.98rem}
#prog{height:7px;border-radius:5px;background:#DDD3BF;overflow:hidden;margin-top:6px}
#prog>div{height:100%;width:0;background:linear-gradient(90deg,#9DB38A,#4E6E45)}
#btns{display:flex;gap:8px;margin-top:8px}
#go{flex:1;height:48px;border:0;border-radius:14px;background:#5E7F52;color:#FBF7EE;font-size:1rem;font-weight:700}
.sq{width:48px;height:48px;border-radius:14px;border:1px solid var(--line);background:#fff;font-size:20px}
.sq.on{background:#E7EDDB;border-color:var(--sage)}
#vbtn,#fbtn{display:none}body.nav #vbtn,body.nav #fbtn{display:block}
#list{position:fixed;z-index:1100;top:calc(var(--top) + 8px);left:10px;right:10px;bottom:10px;max-width:620px;margin:0 auto;background:var(--paper);border:1px solid var(--line);border-radius:20px;overflow:auto;padding:12px 14px;display:none;box-shadow:0 8px 28px rgba(60,50,30,.3)}
.row{display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #E4DBC8}
.ri{font-size:22px;width:30px;text-align:center;flex:none}.rt{flex:1}.rd{opacity:.65;font-size:.85rem;white-space:nowrap}
</style></head><body>
<div id="map"></div>
<div id="banner"><div id="ico">🧭</div><div><div id="dist"></div><div id="ins"></div></div></div>
<div id="side"><button id="layers" title="Kartenstil wechseln">🗺️</button><button id="fit" title="Ganze Route zeigen">⤢</button></div>
<div id="sheet">
 <div id="ttl"></div>
 <div id="pre"><div id="chips"></div><div id="stats"></div><div id="hint"></div></div>
 <div id="navbox"><div id="rem"></div><div id="prog"><div></div></div></div>
 <div id="btns"><button id="go">▶ Navigation starten</button><button class="sq on" id="vbtn" title="Sprachansagen">🔊</button><button class="sq on" id="fbtn" title="Karte folgt mir">🎯</button><button class="sq" id="lbtn" title="Alle Anweisungen">📋</button><button class="sq" id="demo" title="Demo ohne Gehen">🧪</button></div>
</div>
<div id="list"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const WALK = D.speed || 80, TOP = D.topPad || 0, ky = 110540;
document.documentElement.style.setProperty('--top', TOP + 'px');

const map = L.map('map', {zoomControl: false, zoomAnimation: false, fadeAnimation: false, markerZoomAnimation: false}).setView(D.p1, 15);

// Kartenhintergründe; bei Ausfall des Anbieters wird automatisch gewechselt
const TILES = [
  {n: 'Karte', u: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', o: {subdomains: 'abcd', maxZoom: 19, attribution: '© OpenStreetMap © CARTO'}},
  {n: 'Gelände', u: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', o: {subdomains: 'abc', maxZoom: 17, attribution: '© OpenStreetMap, SRTM | © OpenTopoMap'}},
  {n: 'Satellit', u: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', o: {maxZoom: 19, attribution: 'Esri, Maxar'}},
  {n: 'Straßen', u: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', o: {maxZoom: 19, attribution: 'Esri'}}
];
const FALLBACK = [0, 3, 1];
let tIdx = 0, tLayer = null, tErr = 0, tOk = 0, fbPos = 0;
function setTiles(i) {
  if (tLayer) map.removeLayer(tLayer);
  tIdx = i; tErr = 0; tOk = 0;
  tLayer = L.tileLayer(TILES[i].u, TILES[i].o).addTo(map);
  tLayer.on('tileload', () => { tOk++; });
  tLayer.on('tileerror', () => {
    tErr++;
    if (tErr >= 6 && tOk === 0 && fbPos < FALLBACK.length - 1) { fbPos++; setTiles(FALLBACK[fbPos]); }
  });
}
setTiles(0);

if (D.parks) {
  L.geoJSON(D.parks, {style: {color: '#7C9F69', weight: 1, fillColor: '#B5CDA0', fillOpacity: .5},
    onEachFeature: (f, l) => { if (f.properties && f.properties.name) l.bindTooltip(f.properties.name, {sticky: true}); }}).addTo(map);
}
const pin = e => L.divIcon({html: '<div style="font-size:30px;line-height:30px;filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))">' + e + '</div>',
  className: '', iconSize: [32, 32], iconAnchor: [16, 28]});
L.marker(D.p1, {icon: pin('🚶')}).addTo(map);
if (D.p1[0] !== D.p2[0] || D.p1[1] !== D.p2[1]) L.marker(D.p2, {icon: pin('🏁')}).addTo(map);
const userDot = L.circleMarker([0, 0], {radius: 9, color: '#fff', weight: 3, fillColor: '#2F6F8F', fillOpacity: 1});
const userAcc = L.circle([0, 0], {radius: 10, color: '#2F6F8F', weight: 1, fillColor: '#2F6F8F', fillOpacity: .12});

let cur = -1, layers = [], coords = [], cum = [], steps = [], total = 1, P = [], lat0 = 0, lon0 = 0, kx = 1, done = null;
let navOn = false, along = 0, lastIdx = 0, offCount = 0, finished = false, follow = true, lastPos = null,
    said = {}, voice = true, watchId = null, sim = null, hadFix = false, wake = null;

const toXY = (la, lo) => [(lo - lon0) * kx, (la - lat0) * ky];
function fmt(m) {
  if (m >= 1000) return (m / 1000).toFixed(1).replace('.', ',') + ' km';
  const r = m > 100 ? 10 : 5; return Math.max(0, Math.round(m / r) * r) + ' m';
}
function eta(rem) { return new Date(Date.now() + rem / WALK * 60000).toLocaleTimeString('de-DE', {hour: '2-digit', minute: '2-digit'}); }
function banner(cls, ico, dist, ins) { $('banner').className = cls; $('ico').textContent = ico; $('dist').textContent = dist; $('ins').textContent = ins; }
function say(t) {
  if (!voice || !('speechSynthesis' in window)) return;
  try { const u = new SpeechSynthesisUtterance(t); u.lang = 'de-DE'; speechSynthesis.cancel(); speechSynthesis.speak(u); } catch (e) {}
}
function fitRoute() {
  if (!coords.length) return;
  map.fitBounds(L.latLngBounds(coords), {paddingTopLeft: [24, TOP + 24], paddingBottomRight: [24, 250], animate: false});
}
function buildList() {
  let h = '<div style="font-weight:700;margin-bottom:6px">Wegehinweise</div>';
  steps.forEach((s, i) => { h += '<div class="row"><span class="ri">' + esc(s.i) + '</span><span class="rt">' + esc(s.t) +
    '</span><span class="rd">' + (i === 0 ? 'Start' : 'bei ' + fmt(s.d)) + '</span></div>'; });
  $('list').innerHTML = h + '<p><button class="sq" id="closelist" style="width:auto;padding:0 18px">Schließen</button></p>';
  $('closelist').addEventListener('click', () => { $('list').style.display = 'none'; });
}
function showOption(i, fit) {
  const o = D.options[i];
  cur = i; coords = o.coords; cum = o.cum; steps = o.steps; total = Math.max(o.total, 1);
  lat0 = coords[0][0]; lon0 = coords[0][1]; kx = Math.cos(lat0 * Math.PI / 180) * 111320;
  P = coords.map(c => toXY(c[0], c[1]));
  layers.forEach(l => map.removeLayer(l)); layers = [];
  const add = l => { l.addTo(map); layers.push(l); return l; };
  const showShort = D.mode === 'ab' && i !== 0;
  if (showShort) add(L.polyline(D.options[0].coords, {color: '#C08A47', weight: 4, opacity: .95, dashArray: '8 8'}));
  add(L.polyline(coords, {color: '#fff', weight: 11, opacity: .9}));
  add(L.polyline(coords, {color: '#3F6B3A', weight: 7, opacity: 1}));
  done = add(L.polyline([], {color: '#B9C9A8', weight: 7, opacity: 1}));
  steps.forEach((s, k) => { if (k > 0 && k < steps.length - 1)
    add(L.circleMarker([s.lat, s.lon], {radius: 5, color: '#fff', weight: 2, fillColor: '#4E6E45', fillOpacity: 1})); });
  document.querySelectorAll('.chip').forEach((c, k) => c.classList.toggle('on', k === i));
  $('stats').innerHTML = o.stats.map(s => '<div><span>' + esc(s[0]) + '</span><b>' + esc(s[1]) + '</b></div>').join('');
  $('hint').textContent = (o.hint || '') + (showShort ? ' · gestrichelt: kürzeste Route' : '');
  $('rem').textContent = 'Gesamt ' + fmt(total) + ' · ca. ' + Math.max(1, Math.round(total / WALK)) + ' min';
  buildList();
  if (fit) fitRoute();
}

function snap(xy, from, windowed) {
  let best = null;
  const lo = windowed ? Math.max(0, from - 5) : 0, hi = windowed ? Math.min(P.length - 1, from + 80) : P.length - 1;
  for (let i = lo; i < hi; i++) {
    const a = P[i], b = P[i + 1], dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    let t = l2 > 0 ? ((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(xy[0] - (a[0] + t * dx), xy[1] - (a[1] + t * dy));
    if (!best || d < best.off) best = {i: i, off: d, along: cum[i] + t * Math.sqrt(l2)};
  }
  return best || {i: 0, off: 0, along: 0};
}
function finish() {
  finished = true; stopWatch();
  banner('done', '🏁', 'Geschafft!', steps[steps.length - 1].t); say(steps[steps.length - 1].t);
  $('rem').textContent = 'Ziel erreicht'; $('prog').firstElementChild.style.width = '100%';
}
function onFix(lat, lon, acc) {
  if (finished || !navOn) return;
  lastPos = [lat, lon];
  const xy = toXY(lat, lon);
  let s = snap(xy, lastIdx, true);
  if (s.off > 60) { const g = snap(xy, 0, false); if (g.off < s.off) s = g; }
  lastIdx = s.i; along = s.along;
  userDot.setLatLng(lastPos); if (!map.hasLayer(userDot)) userDot.addTo(map);
  // Genauigkeitskreis nur bei brauchbarer Genauigkeit (sonst überdeckt er die ganze Karte)
  if (acc && acc <= 120) { userAcc.setLatLng(lastPos); userAcc.setRadius(acc); if (!map.hasLayer(userAcc)) userAcc.addTo(map); }
  else if (map.hasLayer(userAcc)) map.removeLayer(userAcc);
  done.setLatLngs(coords.slice(0, lastIdx + 1));
  const limit = Math.max(35, (acc || 0) * 1.5), far = s.off > 1500;
  if (s.off > limit) offCount = hadFix ? offCount + 1 : 3; else offCount = 0;
  hadFix = true;
  const rem = Math.max(0, total - along);
  $('rem').textContent = far ? 'Gesamt ' + fmt(total) : fmt(rem) + ' · ' + Math.max(1, Math.round(rem / WALK)) + ' min · Ankunft ' + eta(rem);
  $('prog').firstElementChild.style.width = (far ? 0 : Math.min(100, 100 * along / total)) + '%';
  if (!far && rem < 20 && along > total * 0.5) { finish(); return; }
  if (far) {
    banner('off', '📍', fmt(s.off), 'Du bist noch weit von der Route entfernt. Zum Ausprobieren: 🧪');
  } else if (offCount >= 3) {
    if (along < 15) banner('off', '🚶', fmt(Math.hypot(xy[0] - P[0][0], xy[1] - P[0][1])), 'bis zum Startpunkt der Route');
    else {
      banner('off', '⚠️', fmt(s.off), 'Abseits der Route. Bitte zurück zur grünen Linie gehen.');
      if (!said.off) { said.off = 1; say('Du bist von der Route abgekommen.'); }
    }
  } else {
    said.off = 0;
    if (along < 12) {
      banner('', steps[0].i, 'Start', steps[0].t);
      if (!said.s0) { said.s0 = 1; say(steps[0].t); }
    } else {
      let k = steps.findIndex(t => t.d > along + 8); if (k < 0) k = steps.length - 1;
      const st = steps[k], dd = Math.max(0, st.d - along);
      banner('', st.i, fmt(dd), st.t);
      if (dd <= 70 && !said[k + 'a']) { said[k + 'a'] = 1; say('In ' + (Math.round(dd / 10) * 10) + ' Metern: ' + st.t); }
      if (dd <= 18 && !said[k + 'b']) { said[k + 'b'] = 1; say(st.t); }
    }
  }
  if (follow && !far) map.setView(lastPos, Math.max(map.getZoom(), 17), {animate: false});
}
function stopWatch() {
  if (watchId !== null) { navigator.geolocation.clearWatch(watchId); watchId = null; }
  if (sim) { clearInterval(sim); sim = null; $('demo').classList.remove('on'); }
  try { if (wake) wake.release(); } catch (e) {} wake = null;
}
function enterNav() {
  navOn = true; finished = false; said = {}; hadFix = false; lastIdx = 0; along = 0; offCount = 0;
  document.body.classList.add('nav'); $('go').textContent = '⏹ Beenden';
  try { if (navigator.wakeLock) navigator.wakeLock.request('screen').then(l => { wake = l; }).catch(() => {}); } catch (e) {}
  say('Navigation gestartet.');
  banner('', '📡', '…', 'Standort wird gesucht');
}
function exitNav() {
  stopWatch(); navOn = false; finished = false;
  document.body.classList.remove('nav'); $('go').textContent = '▶ Navigation starten';
  [userDot, userAcc].forEach(l => { if (map.hasLayer(l)) map.removeLayer(l); });
  if (done) done.setLatLngs([]);
  showOption(cur, true);
}
function startGps() {
  enterNav();
  if (!navigator.geolocation) { banner('off', '📡', 'Kein GPS', 'Dieses Gerät unterstützt keine Standortabfrage.'); return; }
  watchId = navigator.geolocation.watchPosition(
    p => onFix(p.coords.latitude, p.coords.longitude, p.coords.accuracy),
    err => {
      if (err.code === 1) banner('off', '🔒', 'Standort blockiert', 'Bitte den Standortzugriff für diese Seite erlauben und neu starten.');
      else banner('off', '📡', 'Kein GPS-Signal', 'Warte auf Standort …');
    }, {enableHighAccuracy: true, maximumAge: 1000, timeout: 20000});
}
function pointAt(s) {
  let i = 0; while (i < cum.length - 2 && cum[i + 1] < s) i++;
  const l = (cum[i + 1] - cum[i]) || 1, t = Math.min(1, Math.max(0, (s - cum[i]) / l));
  const a = coords[i], b = coords[i + 1];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}
function startDemo() {
  enterNav(); $('demo').classList.add('on'); follow = true; $('fbtn').classList.add('on');
  let s = 0;
  sim = setInterval(() => {
    s = Math.min(total, s + 10); const p = pointAt(s); onFix(p[0], p[1], 5);
    if (s >= total || finished) { clearInterval(sim); sim = null; $('demo').classList.remove('on'); }
  }, 500);
}

$('chips').innerHTML = D.options.map((o, i) => '<button class="chip" data-i="' + i + '"><b>' + esc(o.label) + '</b><span>' + esc(o.sub) + '</span></button>').join('');
$('chips').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b && !navOn) showOption(+b.dataset.i, true); });
$('ttl').textContent = (D.p1[0] === D.p2[0] && D.p1[1] === D.p2[1]) ? '🔄 Ründchen ab ' + D.t1 : '🚶 ' + D.t1 + '  →  🏁 ' + D.t2;
$('go').addEventListener('click', () => { if (navOn) exitNav(); else startGps(); });
$('demo').addEventListener('click', () => { if (navOn && sim) exitNav(); else { if (navOn) exitNav(); startDemo(); } });
$('lbtn').addEventListener('click', () => { $('list').style.display = 'block'; });
$('vbtn').addEventListener('click', () => {
  voice = !voice; $('vbtn').textContent = voice ? '🔊' : '🔇'; $('vbtn').classList.toggle('on', voice);
  if (!voice && 'speechSynthesis' in window) speechSynthesis.cancel();
});
$('fbtn').addEventListener('click', () => {
  follow = !follow; $('fbtn').classList.toggle('on', follow);
  if (follow && lastPos) map.setView(lastPos, Math.max(map.getZoom(), 17), {animate: false});
});
$('layers').addEventListener('click', () => setTiles((tIdx + 1) % TILES.length));
$('fit').addEventListener('click', fitRoute);
map.on('dragstart', () => { follow = false; $('fbtn').classList.remove('on'); });

function fix() { map.invalidateSize({animate: false}); }
window.addEventListener('resize', fix);
if (window.visualViewport) window.visualViewport.addEventListener('resize', fix);
if (window.ResizeObserver) new ResizeObserver(fix).observe($('map'));
showOption(D.default || 0, false);
setTimeout(() => { fix(); if (!navOn) fitRoute(); }, 150);
setTimeout(() => { fix(); if (!navOn) fitRoute(); }, 900);
</script></body></html>
"""


# ------------------------------------------------------- Karten-Ansicht (Vollbild)
TOP_PAD = 86  # Höhe der Suchleiste (Streamlit) über der Karte, in Pixeln


def fmt_km(m):
    return f"{m / 1000:.1f} km".replace(".", ",") if m >= 1000 else f"{m:.0f} m"


def inside(bbox, p, margin=0.002):
    left, bottom, right, top = bbox
    return left + margin <= p[1] <= right - margin and bottom + margin <= p[0] <= top - margin


def solve_levels(area, o, t, buf):
    """Kürzeste Route + grüne Route je Grünstufe (alle vorab, damit die Karte sofort umschalten kann)."""
    g = np.clip(1 - area["D"] / buf, 0, 1).mean(axis=0)
    L, H = area["lengths"], area["H"]

    def run(w):
        path = nx.shortest_path(H, o, t, weight=w)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum()) if idx else 0.0
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        return {"path": path, "dist": dist, "green": share}

    short = run(lambda u, v, d: L[d["i"]])
    out = []
    for name, lam in LEVELS.items():
        r = short if lam == 0 else run(lambda u, v, d, lam=lam: L[d["i"]] * (1 + lam * (1 - g[d["i"]])))
        out.append((name, r))
    return short, out


def parks_geo(area):
    """Grünflächen als kompaktes GeoJSON-Objekt (Koordinaten gerundet)."""
    if "_geo" in area:
        return area["_geo"]
    geo = None
    if area["parks_json"]:
        def rnd(c):
            return [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], (int, float)) else [rnd(x) for x in c]
        feats = []
        for f in json.loads(area["parks_json"]).get("features", []):
            geom = f.get("geometry") or {}
            if "coordinates" not in geom:
                continue
            geom["coordinates"] = rnd(geom["coordinates"])
            feats.append({"type": "Feature", "geometry": geom,
                          "properties": {"name": (f.get("properties") or {}).get("name", "Grünfläche")}})
        geo = {"type": "FeatureCollection", "features": feats}
    area["_geo"] = geo
    return geo


def map_html(payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return MAP_TEMPLATE.replace("__DATA__", data)


def _option(nav, label, sub, hint, stats):
    return {**nav, "label": label, "sub": sub, "hint": hint, "stats": stats}


def make_ab_view(start_v, dest_v, buf=150, area=None, bbox=None, log=lambda *_: None):
    """Alles für die Vollbild-Karte einer Route A -> B berechnen."""
    lat1, lon1, lab1 = _unpack(start_v)
    lat2, lon2, lab2 = _unpack(dest_v)
    p1, p2 = (lat1, lon1), (lat2, lon2)
    if area is None or bbox is None or not inside(bbox, p2) or not inside(bbox, p1):
        bbox, span = corridor_bbox(p1, p2)
        if span > MAX_SPAN_M:
            raise ValueError(f"Start und Ziel sind {span / 1000:.1f} km auseinander. "
                             f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
        log("🗺️ Wegenetz und Grünflächen laden (nur der Korridor zwischen Start und Ziel)")
        t0 = time.time()
        area = load_area(bbox)
        log(f"✅ {fmt_m(area['n_edges'])} Wegstücke, {area['n_parks']} Grünflächen ({time.time() - t0:.0f} s)")
    o, t = nearest_node(area, *p1), nearest_node(area, *p2)
    log("🧭 Routen und Wegehinweise berechnen")
    try:
        short, levels = solve_levels(area, o, t, buf)
    except nx.NetworkXNoPath:
        raise ValueError("Zwischen Start und Ziel gibt es keinen Fußweg im geladenen Gebiet.")
    t1, t2 = lab1.split(" · ")[0], lab2.split(" · ")[0]
    options = []
    for name, r in levels:
        nav = build_nav(area, r["path"], f"{t1} → {t2}")
        if not nav:
            continue
        detour = r["dist"] / short["dist"] - 1 if short["dist"] else 0.0
        mins = max(1, round(r["dist"] / M_PER_MIN))
        options.append(_option(
            nav, name, f"{fmt_km(r['dist'])} · {r['green']:.0%} grün", LEVEL_HINT[name],
            [["Länge", fmt_km(r["dist"])], ["Dauer", f"{mins} min"],
             ["Grünanteil", f"{r['green']:.0%}"], ["Umweg", f"{detour:+.0%}"]]))
    if not options:
        raise ValueError("Start und Ziel liegen zu nah beieinander.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == "Ausgewogen"), 0)
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=parks_geo(area),
                   options=options, default=default, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": dest_v, "area": area, "bbox": bbox, "buf": buf}


def make_loop_view(start_v, end_v, km, level, buf=150, area=None, bbox_km=None,
                   log=lambda *_: None, progress=None):
    """Alles für die Vollbild-Karte eines Ründchens berechnen."""
    lat1, lon1, lab1 = _unpack(start_v)
    p1 = (lat1, lon1)
    if end_v:
        lat2, lon2, lab2 = _unpack(end_v)
        p2 = (lat2, lon2)
    else:
        p2, lab2 = p1, lab1
    target = km * 1000
    if dist_m(p1, p2) > 0.35 * target:
        raise ValueError("Start und Ende liegen zu weit auseinander für diese Länge.")
    if area is None or bbox_km is None or km > bbox_km:
        log("🗺️ Wegenetz und Grünflächen laden")
        t0 = time.time()
        area = load_area(loop_bbox(p1, p2, target))
        bbox_km = km
        log(f"✅ {fmt_m(area['n_edges'])} Wegstücke, {area['n_parks']} Grünflächen ({time.time() - t0:.0f} s)")
    o, t = nearest_node(area, *p1), nearest_node(area, *p2)
    variants = find_loops(area, o, t, target, LEVELS[level], buf, progress=progress)
    t1, t2 = lab1.split(" · ")[0], lab2.split(" · ")[0]
    options = []
    for i, v in enumerate(variants):
        nav = build_nav(area, v["path"], f"Ründchen {i + 1}", loop=True)
        if not nav:
            continue
        mins = max(1, round(v["dist"] / M_PER_MIN))
        diff = v["dist"] / target - 1
        options.append(_option(
            nav, f"Variante {i + 1}", f"{fmt_km(v['dist'])} · {v['green']:.0%} grün",
            f"Wunschlänge {fmt_km(target)} ({diff:+.0%})",
            [["Länge", fmt_km(v["dist"])], ["Dauer", f"{mins} min"],
             ["Grünanteil", f"{v['green']:.0%}"], ["Ohne Doppelweg", f"{1 - v['rep']:.0%}"]]))
    if not options:
        raise ValueError("Hier konnte ich keine passende Runde finden. Versuche eine andere Länge "
                         "oder einen anderen Startpunkt.")
    payload = dict(mode="loop", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=parks_geo(area),
                   options=options, default=0, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "loop", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "end": end_v, "km": km, "km_ok": km, "level": level, "buf": buf,
            "area": area, "bbox_km": bbox_km}


# ------------------------------------------------------- UI
st.set_page_config(page_title="Green-Walker", page_icon="🌳", layout="centered",
                   initial_sidebar_state="collapsed")

BASE_CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:720px}
h1,h2,h3,h4{font-family:Georgia,"Iowan Old Style","Palatino Linotype",serif;letter-spacing:-.005em}
.hero{border-radius:22px;padding:22px 22px;margin-bottom:16px;color:#FBF7EE;
 background:linear-gradient(135deg,#4E6E45 0%,#7E9A6A 58%,#B9A77C 100%);box-shadow:0 8px 24px rgba(78,110,69,.22)}
.hero h1{margin:0;font-size:1.8rem;line-height:1.1;color:#FBF7EE}
.hero p{margin:6px 0 0;font-size:1rem;color:#FBF7EE}
.st-key-card_route,.st-key-card_loop{background:#FBF8F1;border:1px solid #D9CFBC;border-radius:20px;
 padding:18px 16px 14px;box-shadow:0 4px 18px rgba(60,50,30,.08);margin-top:12px}
.hint{opacity:.75;font-size:.92rem;margin:-4px 0 8px}
.trip{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}
.pill{border-radius:999px;padding:6px 14px;background:#E7EDDB;color:#3B3A33;border:1px solid #9DB38A;font-size:.92rem}
.stButton>button{border-radius:14px;font-weight:600;padding:.6rem 1rem}
.stButton>button[kind="primary"]{background:#5E7F52;border-color:#5E7F52;color:#FBF7EE;height:3rem;font-size:1.05rem}
.stButton>button[kind="primary"]:hover{background:#4E6E45;border-color:#4E6E45;color:#FBF7EE}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#E9E1D2;padding:4px;border-radius:14px}
.stTabs [data-baseweb="tab"]{flex:1;justify-content:center;height:42px;border-radius:10px;font-weight:600;color:#6B675B}
.stTabs [aria-selected="true"]{background:#FBF8F1;color:#3B3A33;box-shadow:0 1px 4px rgba(60,50,30,.18)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
</style>
"""

# Nur in der Kartenansicht: Karte füllt den ganzen Bildschirm, Suchleiste schwebt darüber.
MAP_CSS = """
<style>
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],footer{display:none!important}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main{overflow:hidden!important}
.block-container{padding:0!important;max-width:100%!important}
.st-key-mapfull{position:fixed;top:0;left:0;right:0;bottom:0;z-index:1;margin:0}
.st-key-mapfull iframe{position:fixed!important;top:0;left:0;width:100vw!important;height:100vh!important;
 height:100dvh!important;border:0!important;border-radius:0!important}
[data-stale="true"]{opacity:1!important}
.st-key-gwtop{position:fixed;top:10px;left:10px;right:10px;z-index:1000002;max-width:620px;margin:0 auto;
 background:rgba(251,248,241,.98);border:1px solid #D9CFBC;border-radius:16px;padding:8px 10px;
 box-shadow:0 6px 20px rgba(60,50,30,.25)}
.st-key-gwtop [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;align-items:center;gap:8px}
.st-key-gwtop [data-testid="stColumn"],.st-key-gwtop [data-testid="column"]{min-width:0!important;width:auto!important;flex:1 1 0!important}
.st-key-gwtop [data-testid="stColumn"]:has(button),.st-key-gwtop [data-testid="column"]:has(button){flex:0 0 50px!important;width:50px!important}
.st-key-gwtop .stButton>button{padding:.4rem 0;height:46px}
.st-key-gwtop [data-testid="stVerticalBlock"]{gap:.25rem}
.looptitle{font-size:1rem;white-space:nowrap;text-align:center}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

PLACEHOLDER = "Straße, Adresse, Café, Restaurant, Geschäft …"
SEARCH_TIP = ("Tipp: Tippe mindestens 3 Buchstaben, dann erscheinen Vorschläge. "
              "Tippfehler sind okay, es werden ähnliche Orte angezeigt.")


def place_input(label, key):
    return st_searchbox(search_places, key=key, label=label, placeholder=PLACEHOLDER,
                        clear_on_submit=False)


@st.cache_data(ttl=3600, show_spinner=False)
def reverse_name(lat, lon):
    """Adresse zu Koordinaten (Photon), nur zur Anzeige."""
    try:
        r = requests.get("https://photon.komoot.io/reverse", headers=UA, timeout=6,
                         params={"lat": lat, "lon": lon, "lang": "de"})
        p = r.json()["features"][0]["properties"]
        street = " ".join(x for x in [p.get("street") or p.get("name"), p.get("housenumber")] if x)
        return ", ".join(x for x in [street, p.get("city") or p.get("district")] if x)
    except Exception:
        return ""


def start_input(label, key):
    """Startpunkt: per Suche oder per Standortfreigabe des Browsers (Schalter)."""
    use = st.toggle("📍 Meinen aktuellen Standort verwenden", key=f"{key}_useloc",
                    help="Der Browser fragt nach der Standortfreigabe. Der Standort wird nur für die Suche genutzt.")
    if not use:
        v = place_input(label, key)
        if v:
            st.session_state["bias"] = _unpack(v)[:2]
        return v

    skey = f"{key}_loc"
    loc = st.session_state.get(skey)
    if loc is None:
        raw = get_geolocation(component_key=f"{key}_geo")
        if isinstance(raw, dict) and raw.get("error"):
            st.warning("Der Standort ist nicht verfügbar. Bitte die Freigabe im Browser erlauben "
                       "(Schloss-Symbol in der Adressleiste) und den Schalter erneut setzen, "
                       "oder den Startpunkt über die Suche eingeben.")
            return None
        c = raw.get("coords") if isinstance(raw, dict) else None
        if not c or c.get("latitude") is None:
            st.info("📡 Standort wird ermittelt … bitte im Browser die Freigabe erlauben.")
            return None
        loc = (float(c["latitude"]), float(c["longitude"]), c.get("accuracy"))
        st.session_state[skey] = loc
    lat, lon, acc = loc
    name = reverse_name(round(lat, 5), round(lon, 5))
    lab = "📍 Mein Standort" + (f" · {name}" if name else "")
    st.markdown(f'<div class="trip"><span class="pill">{escape(lab)}</span></div>', unsafe_allow_html=True)
    if acc and acc > 1000:
        st.caption(f"Standort ungenau (ca. {acc:,.0f} m). Am Handy hilft GPS bzw. genaue Ortung.".replace(",", "."))
    if st.button("🔄 Standort neu bestimmen", key=f"{key}_relocate"):
        st.session_state.pop(skey, None)
        st.rerun()
    st.session_state["bias"] = (lat, lon)
    return _pack(lat, lon, lab)


def show_error(e):
    msg = str(e)
    if any(x in msg for x in ("502", "503", "504", "überlastet", "Timeout", "timed out", "capacity")):
        st.error("Die freien Kartenserver (OpenStreetMap/Overpass) sind gerade überlastet. "
                 "Ich habe mehrere Server und mehrere Runden probiert. Bitte in ein paar Minuten noch "
                 "einmal auf den Knopf tippen. Schon geladene Teile sind zwischengespeichert, "
                 "ein neuer Versuch geht deshalb oft schneller. Besonders ruhig sind die Server meist "
                 "vormittags.")
    elif isinstance(e, ValueError):
        st.error(msg)
    else:
        st.error(f"Das hat nicht geklappt: {e}")
    with st.expander("Technische Details"):
        st.code(msg[:600])


def fmt_m(x):
    return f"{x:,.0f}".replace(",", ".")


def sources_text(buf, lam):
    return (
        "- Wegenetz (Fußwege) und Grünflächen: OpenStreetMap via Overpass (ODbL)\n"
        "- Ortssuche: Photon (photon.komoot.io) auf OSM-Basis, Fallback Nominatim\n"
        "- Kartenhintergrund: CARTO / OpenStreetMap, Gelände: OpenTopoMap, Satellit: Esri\n"
        f"- Grünflächen-Tags: `{GREEN_TAGS}`\n"
        f"- Score je Abtastpunkt: 1,0 im Park, linear fallend auf 0 im Abstand von {buf} m\n"
        f"- Kantenscore: Mittel aus {SAMPLES} Abtastpunkten\n"
        "- Kosten: Länge × (1 + λ × (1 − Grünscore)), λ je Grünstufe: "
        + ", ".join(f"{k} {v:g}" for k, v in LEVELS.items())
    )


# ---------- Wechsel zwischen Startseite und Kartenansicht
def open_map(view):
    st.session_state["gw_sid"] = st.session_state.get("gw_sid", 0) + 1
    view["sid"] = st.session_state["gw_sid"]
    st.session_state["gw"] = view
    st.session_state["bias"] = view["p1"]


def close_map():
    st.session_state.pop("gw", None)


def bump_km(delta):
    gw = st.session_state.get("gw")
    if gw:
        gw["km"] = round(min(max(gw["km"] + delta, 1.0), LOOP_MAX_KM), 1)
        gw["recalc"] = True


def render_map_view():
    """Vollbild-Karte mit Suchfeld (Ziel ändern) bzw. Längen-Regler (Ründchen) in der Karte."""
    gw = st.session_state["gw"]
    st.markdown(MAP_CSS, unsafe_allow_html=True)
    st.session_state["bias"] = gw["p1"]
    dest_new = None
    with st.container(key="gwtop"):
        if gw["mode"] == "ab":
            c1, c2 = st.columns([1, 6])
            c1.button("←", key="gw_back", use_container_width=True, on_click=close_map, help="Zurück zur Suche")
            with c2:
                dest_new = st_searchbox(search_places, key=f"gw_dest_{gw['sid']}",
                                        placeholder=f"🏁 {gw['t2']} · Ziel ändern …", clear_on_submit=False)
        else:
            c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
            c1.button("←", key="gw_back", use_container_width=True, on_click=close_map, help="Zurück zur Suche")
            km_txt = str(gw["km"]).replace(".", ",")
            c2.markdown(f'<div class="looptitle">🔄 Ründchen · <b>{km_txt} km</b></div>', unsafe_allow_html=True)
            c3.button("−", key="gw_km_minus", use_container_width=True, on_click=bump_km, args=(-0.5,),
                      help="Kürzer")
            c4.button("+", key="gw_km_plus", use_container_width=True, on_click=bump_km, args=(0.5,),
                      help="Länger")

    new = None
    if gw["mode"] == "ab" and dest_new and dest_new != gw["dest"] and dest_new != gw.get("failed"):
        st.toast("🧭 Berechne neue Route …")
        try:
            new = make_ab_view(gw["start"], dest_new, gw["buf"], gw["area"], gw["bbox"])
        except Exception as e:
            gw["failed"] = dest_new
            st.toast(f"⚠️ {e}"[:220])
    elif gw["mode"] == "loop" and gw.get("recalc"):
        gw["recalc"] = False
        st.toast("🔄 Suche neues Ründchen …")
        try:
            new = make_loop_view(gw["start"], gw["end"], gw["km"], gw["level"], gw["buf"],
                                 area=gw["area"], bbox_km=gw["bbox_km"])
        except Exception as e:
            gw["km"] = gw["km_ok"]
            st.toast(f"⚠️ {e}"[:220])
    if new:
        new["sid"] = gw["sid"]
        st.session_state["gw"] = new
        st.rerun()

    with st.container(key="mapfull"):
        st_html(gw["html"], height=700, scrolling=False)


# ---------- Startseite
def home_route():
    with st.container(key="card_route"):
        start_v = start_input("Start", "start_sb")
        ziel_v = place_input("Ziel", "ziel_sb")
        st.caption(SEARCH_TIP)
        go = st.button("Route finden", type="primary", use_container_width=True, key="go_ab")
    if not go:
        return
    if not (start_v and ziel_v):
        st.warning("Bitte Start und Ziel jeweils aus den Vorschlägen auswählen.")
        return
    view = None
    with st.status("Route wird vorbereitet …", expanded=True) as status:
        try:
            view = make_ab_view(start_v, ziel_v, log=st.write)
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        open_map(view)
        st.rerun()


def home_loop():
    with st.container(key="card_loop"):
        st.markdown("Grüne Runde mit Wunschlänge, die am Startpunkt beginnt und dort (oder an einem "
                    "anderen Punkt) wieder endet.")
        a_v = start_input("Startpunkt", "loop_a_sb")
        same = st.checkbox("Ründchen endet wieder am Startpunkt", value=True, key="loop_same")
        b_v = None if same else place_input("Endpunkt", "loop_b_sb")
        st.caption(SEARCH_TIP)
        km = st.slider("Gewünschte Länge (km)", 1.0, LOOP_MAX_KM, 3.0, 0.5, key="loop_km")
        st.markdown(f'<div class="hint">ca. {max(1, round(km * 1000 / M_PER_MIN))} min zu Fuß</div>',
                    unsafe_allow_html=True)
        level = st.select_slider("Wie grün soll es sein?", options=list(LEVELS), value="Sehr grün",
                                 key="level_loop")
        go = st.button("Ründchen finden", type="primary", use_container_width=True, key="go_loop")
    if not go:
        return
    if not a_v or (not same and not b_v):
        st.warning("Bitte Start" + ("" if same else " und Ende") + " aus den Vorschlägen auswählen.")
        return
    view, holder = None, {}

    def prog(f):
        txt = "🔄 Runden in verschiedene Richtungen probieren …"
        if "bar" not in holder:
            holder["bar"] = st.progress(0.0, text=txt)
        holder["bar"].progress(min(f, 1.0), text=txt)

    with st.status("Ründchen wird gesucht …", expanded=True) as status:
        try:
            view = make_loop_view(a_v, None if same else b_v, km, level, log=st.write, progress=prog)
            if "bar" in holder:
                holder["bar"].empty()
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        open_map(view)
        st.rerun()


def render_home():
    st.markdown('<div class="hero"><h1>🌳 Green-Walker</h1>'
                '<p>Gehe den grünen Weg: vorbei an Parks und Grün statt nur auf kürzester Strecke.</p></div>',
                unsafe_allow_html=True)
    tab_route, tab_loop = st.tabs(["🧭 Route", "🔄 Ründchen"])
    with tab_route:
        home_route()
    with tab_loop:
        home_loop()
    with st.expander("Datenquellen und Gewichtung"):
        st.markdown(sources_text(150, LEVELS["Ausgewogen"]))


if st.session_state.get("gw"):
    render_map_view()
else:
    render_home()
