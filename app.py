"""Grüne Route: Fußwege mit möglichst viel Park- und Grünflächen-Nähe.

Start:  pip install -r requirements.txt
        streamlit run app.py

Daten:  OpenStreetMap (Overpass via osmnx, ODbL) für Wegenetz + Grünflächen,
        Geocoding über Nominatim. Benötigt Internetzugang.
Kosten: kosten(kante) = laenge * (1 + LAMBDA * (1 - gruen(kante)))
Grün:   Mittel über 8 Abtastpunkte je Kante; 1.0 im Park, linear fallend
        auf 0 im Abstand "Grün-Reichweite".

Geschwindigkeit: nur ein Korridor um Start/Ziel wird geladen, Wegenetz und
Parks parallel, Parkabstände per räumlichem Index (STRtree). λ und
Reichweite lassen sich danach live ändern, ohne neu zu laden.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import folium
import networkx as nx
import numpy as np
import osmnx as ox
import shapely
import streamlit as st
from shapely import STRtree
from shapely.geometry import LineString, Point
from streamlit.components.v1 import html as st_html

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 90
ox.settings.overpass_rate_limit = False  # spart die Status-Abfrage vor jedem Request

GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve", "recreation_ground"],
    "landuse": ["grass", "forest", "village_green"],
    "natural": ["wood"],
}
SAMPLES = 8        # Abtastpunkte pro Kante
MAX_BUF = 400      # größte einstellbare Reichweite (m)
MAX_SPAN_M = 9000  # größte erlaubte Ausdehnung des Korridors
M_PER_MIN = 80     # ca. 4,8 km/h

LEVELS = {
    "Schnellste": 0.0,
    "Leicht grün": 1.0,
    "Ausgewogen": 3.0,
    "Sehr grün": 6.0,
    "Maximal grün": 12.0,
}
LEVEL_HINT = {
    "Schnellste": "Nur die Distanz zählt.",
    "Leicht grün": "Kleine Umwege für spürbar mehr Grün.",
    "Ausgewogen": "Guter Mittelweg aus Weg und Grün.",
    "Sehr grün": "Deutliche Umwege sind okay, wenn es grüner wird.",
    "Maximal grün": "Grün hat klar Vorrang vor der Länge.",
}


# ---------------------------------------------------------------- Daten
@st.cache_data(ttl=86400, show_spinner=False)
def geocode(q):
    return ox.geocode(q)


def corridor_bbox(p1, p2):
    (lat1, lon1), (lat2, lon2) = p1, p2
    ky = 111_320.0
    kx = ky * math.cos(math.radians((lat1 + lat2) / 2))
    span = max(abs(lon2 - lon1) * kx, abs(lat2 - lat1) * ky)
    pad = max(350.0, 0.25 * span)
    bbox = (
        round(min(lon1, lon2) - pad / kx, 4),
        round(min(lat1, lat2) - pad / ky, 4),
        round(max(lon1, lon2) + pad / kx, 4),
        round(max(lat1, lat2) + pad / ky, 4),
    )
    return bbox, span


@st.cache_resource(show_spinner=False, max_entries=4)
def load_area(bbox):
    """Wegenetz + Grünflächen laden (parallel) und Kanten-Abstände vorberechnen."""

    def get_graph():
        return ox.graph_from_bbox(bbox, network_type="walk", simplify=True)

    def get_parks():
        try:
            gdf = ox.features_from_bbox(bbox, tags=GREEN_TAGS)
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            return gdf[["geometry"]] if len(gdf) else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_g, f_p = ex.submit(get_graph), ex.submit(get_parks)
        G0, parks = f_g.result(), f_p.result()

    G = ox.project_graph(G0)
    crs = G.graph["crs"]
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    # Pro Knotenpaar nur die kürzeste Kante behalten (einfacher, schneller Graph)
    best = {}
    for u, v, d in G.edges(data=True):
        if (u, v) not in best or d["length"] < best[(u, v)][0]:
            best[(u, v)] = (d["length"], d.get("geometry") or LineString([pos[u], pos[v]]))
    keys = list(best)
    lengths = np.array([best[k][0] for k in keys], dtype=float)
    lines = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        lines[i] = best[k][1]
    H = nx.DiGraph()
    for i, (u, v) in enumerate(keys):
        H.add_edge(u, v, i=i)

    # Abstand der Abtastpunkte zur nächsten Grünfläche (vektorisiert, räumlicher Index)
    D = np.full((SAMPLES, len(keys)), MAX_BUF, dtype=np.float32)
    parks_json, n_parks = None, 0
    if parks is not None:
        geoms = parks.to_crs(crs).geometry.simplify(2.0).make_valid()
        geoms = geoms[geoms.notna() & ~geoms.is_empty]
        n_parks = len(geoms)
        if n_parks:
            tree = STRtree(list(geoms))
            for r, f in enumerate((np.arange(SAMPLES) + 0.5) / SAMPLES):
                pts = shapely.line_interpolate_point(lines, f, normalized=True)
                idx, dist = tree.query_nearest(
                    pts, max_distance=MAX_BUF, return_distance=True, all_matches=False
                )
                D[r, idx[0]] = dist
            parks_json = gpd.GeoSeries(geoms, crs=crs).to_crs(4326).to_json()

    node_list = list(pos)
    return {
        "crs": crs, "pos": pos, "H": H, "lengths": lengths, "lines": lines, "D": D,
        "node_list": node_list, "xy": np.array([pos[n] for n in node_list]),
        "parks_json": parks_json, "n_parks": n_parks, "n_edges": len(keys),
    }


def nearest_node(area, lat, lon):
    x, y = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(area["crs"]).iloc[0].coords[0]
    d = (area["xy"][:, 0] - x) ** 2 + (area["xy"][:, 1] - y) ** 2
    return area["node_list"][int(d.argmin())]


# ---------------------------------------------------------------- Routing
def solve(area, o, t, lam, buf):
    g = np.clip(1 - area["D"] / buf, 0, 1).mean(axis=0)
    L, H = area["lengths"], area["H"]
    out = {}
    weights = {
        "short": lambda u, v, d: L[d["i"]],
        "green": lambda u, v, d: L[d["i"]] * (1 + lam * (1 - g[d["i"]])),
    }
    for name, w in weights.items():
        path = nx.shortest_path(H, o, t, weight=w)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum())
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        out[name] = {"path": path, "dist": dist, "green": share}
    return out


def route_latlon(area, path):
    coords = []
    for a, b in zip(path[:-1], path[1:]):
        c = list(area["lines"][area["H"][a][b]["i"]].coords)
        pa = area["pos"][a]
        if math.dist(c[0], pa) > math.dist(c[-1], pa):
            c.reverse()
        coords += c
    if len(coords) < 2:
        coords = [area["pos"][path[0]]] * 2
    ll = gpd.GeoSeries([LineString(coords)], crs=area["crs"]).to_crs(4326).iloc[0]
    return [(y, x) for x, y in ll.coords]


def build_map(area, res, p1, p2):
    m = folium.Map(tiles="CartoDB positron", control_scale=True, zoom_control=True)
    if area["parks_json"]:
        folium.GeoJson(
            area["parks_json"],
            style_function=lambda _: {"color": "#4f9d57", "fillColor": "#9fd89a",
                                      "fillOpacity": 0.45, "weight": 1},
        ).add_to(m)
    short = route_latlon(area, res["short"]["path"])
    green = route_latlon(area, res["green"]["path"])
    folium.PolyLine(short, color="#e4572e", weight=4, opacity=0.9, dash_array="8 8",
                    tooltip="Kürzeste Route").add_to(m)
    folium.PolyLine(green, color="#0d7a45", weight=7, opacity=0.95,
                    tooltip="Grüne Route").add_to(m)
    folium.CircleMarker(p1, radius=8, color="#fff", weight=3, fill=True,
                        fill_color="#111", fill_opacity=1, tooltip="Start").add_to(m)
    folium.CircleMarker(p2, radius=8, color="#111", weight=3, fill=True,
                        fill_color="#fff", fill_opacity=1, tooltip="Ziel").add_to(m)
    pts = short + green
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                  [max(p[0] for p in pts), max(p[1] for p in pts)]], padding=(30, 30))
    return m


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Grüne Route", page_icon="🌳", layout="centered")
st.markdown(
    """
<style>
.block-container{padding-top:1.6rem;max-width:760px}
.hero{border-radius:20px;padding:26px 24px;margin-bottom:18px;color:#fff;
 background:linear-gradient(135deg,#0d7a45 0%,#2fa866 55%,#7bcf8e 100%);
 box-shadow:0 8px 24px rgba(13,122,69,.25)}
.hero h1{margin:0 0 4px;font-size:2rem;line-height:1.1;color:#fff}
.hero p{margin:0;opacity:.92;font-size:1.02rem}
.card{border:1px solid rgba(127,127,127,.25);background:rgba(127,127,127,.07);
 border-radius:16px;padding:14px 16px;height:100%}
.card.accent{border-color:#2fa866;background:rgba(47,168,102,.12)}
.card .lab{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;opacity:.7}
.card .val{font-size:1.7rem;font-weight:700;line-height:1.2}
.card .sub{font-size:.85rem;opacity:.75}
.bar{height:8px;border-radius:6px;background:rgba(127,127,127,.25);margin-top:8px;overflow:hidden}
.bar>div{height:100%;background:linear-gradient(90deg,#2fa866,#0d7a45);border-radius:6px}
.hint{opacity:.75;margin:-6px 0 10px;font-size:.92rem}
div[data-testid="stForm"]{border:none;padding:0}
.stButton>button,.stFormSubmitButton>button{border-radius:12px;font-weight:600;padding:.6rem 1rem}
</style>
<div class="hero"><h1>🌳 Grüne Route</h1>
<p>Der schönere Weg zu Fuß: vorbei an Parks und Grün statt nur auf kürzester Strecke.</p></div>
""",
    unsafe_allow_html=True,
)

with st.form("ziele"):
    c1, c2 = st.columns(2)
    start = c1.text_input("Start", "Brandenburger Tor, Berlin", placeholder="Adresse oder Ort")
    ziel = c2.text_input("Ziel", "Alexanderplatz, Berlin", placeholder="Adresse oder Ort")
    go = st.form_submit_button("Route finden", type="primary", use_container_width=True)

if go:
    st.session_state.pop("ctx", None)
    with st.status("Route wird vorbereitet …", expanded=True) as status:
        try:
            st.write("📍 Adressen suchen")
            p1, p2 = geocode(start), geocode(ziel)
            bbox, span = corridor_bbox(p1, p2)
            if span > MAX_SPAN_M:
                status.update(label="Strecke zu lang", state="error")
                st.error(f"Start und Ziel sind {span / 1000:.1f} km auseinander. "
                         f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
                st.stop()
            st.write("🗺️ Wegenetz und Grünflächen laden (nur der Korridor zwischen Start und Ziel)")
            t0 = time.time()
            area = load_area(bbox)
            took = time.time() - t0
            st.write(f"✅ {area['n_edges']:,} Wegstücke, {area['n_parks']} Grünflächen "
                     f"({took:.0f} s)".replace(",", "."))
            o, t = nearest_node(area, *p1), nearest_node(area, *p2)
            st.session_state["ctx"] = dict(area=area, p1=p1, p2=p2, o=o, t=t,
                                           start=start, ziel=ziel)
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            st.error(f"Das hat nicht geklappt: {e}")

ctx = st.session_state.get("ctx")
if ctx:
    area = ctx["area"]
    st.markdown("#### Wie grün soll es sein?")
    level = st.select_slider("Grünstufe", options=list(LEVELS), value="Ausgewogen",
                             label_visibility="collapsed")
    st.markdown(f'<div class="hint">{LEVEL_HINT[level]}</div>', unsafe_allow_html=True)
    with st.expander("Feintuning"):
        buf = st.slider("Grün-Reichweite um Parks (m)", 50, MAX_BUF, 150, 25,
                        help="So weit vom Parkrand zählt Umgebung noch als grün (linear abnehmend).")

    try:
        res = solve(area, ctx["o"], ctx["t"], LEVELS[level], buf)
    except nx.NetworkXNoPath:
        st.error("Zwischen Start und Ziel gibt es keinen Fußweg im geladenen Gebiet.")
        st.stop()

    s, g = res["short"], res["green"]
    detour = g["dist"] / s["dist"] - 1 if s["dist"] else 0
    gain = (g["green"] - s["green"]) * 100
    k1, k2, k3 = st.columns(3)
    k1.markdown(
        f'<div class="card accent"><div class="lab">Grüne Route</div>'
        f'<div class="val">{g["dist"]:,.0f} m</div>'
        f'<div class="sub">ca. {max(1, round(g["dist"] / M_PER_MIN))} min zu Fuß</div></div>'.replace(",", "."),
        unsafe_allow_html=True)
    k2.markdown(
        f'<div class="card"><div class="lab">Grünanteil</div>'
        f'<div class="val">{g["green"]:.0%}</div>'
        f'<div class="sub">{gain:+.0f} Prozentpunkte</div>'
        f'<div class="bar"><div style="width:{g["green"] * 100:.0f}%"></div></div></div>',
        unsafe_allow_html=True)
    k3.markdown(
        f'<div class="card"><div class="lab">Umweg</div>'
        f'<div class="val">{detour:+.0%}</div>'
        f'<div class="sub">kürzeste: {s["dist"]:,.0f} m · {s["green"]:.0%} grün</div></div>'.replace(",", "."),
        unsafe_allow_html=True)

    st.write("")
    m = build_map(area, res, ctx["p1"], ctx["p2"])
    st_html(m._repr_html_(), height=540)
    st.caption("🟢 grüne Route  ·  🟠 gestrichelt: kürzeste Route  ·  hellgrün: Grünflächen")

    with st.expander("Datenquellen und Gewichtung"):
        st.markdown(
            "- Wegenetz (Fußwege) und Grünflächen: OpenStreetMap via Overpass (ODbL)\n"
            f"- Grünflächen-Tags: `{GREEN_TAGS}`\n"
            f"- Score je Abtastpunkt: 1,0 im Park, linear fallend auf 0 im Abstand von {buf} m\n"
            f"- Kantenscore: Mittel aus {SAMPLES} Abtastpunkten\n"
            f"- Kosten: Länge × (1 + {LEVELS[level]:g} × (1 − Grünscore))"
        )

