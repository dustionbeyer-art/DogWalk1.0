"""Grüne Route: Fußwege mit möglichst viel Park- und Grünflächen-Nähe.

Start:  pip install -r requirements.txt
        streamlit run app.py

Daten:  OpenStreetMap (Overpass via osmnx, ODbL) für Wegenetz + Grünflächen.
        Ortssuche mit Vorschlägen: Photon (photon.komoot.io, OSM-Daten, fehler-
        tolerant, findet auch Läden/Restaurants), Fallback Nominatim.
Kosten: kosten(kante) = laenge * (1 + LAMBDA * (1 - gruen(kante)))
Grün:   Mittel über 8 Abtastpunkte je Kante; 1.0 im Park, linear fallend
        auf 0 im Abstand "Grün-Reichweite".
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import requests
import shapely
import streamlit as st
from folium.plugins import AntPath, Fullscreen, LocateControl
from shapely import STRtree
from shapely.geometry import LineString, Point
from streamlit.components.v1 import html as st_html
from streamlit_searchbox import st_searchbox

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.requests_timeout = 90
ox.settings.overpass_rate_limit = False

UA = {"User-Agent": "gruene-route-app/1.0 (privates Hobbyprojekt)"}
GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve", "recreation_ground"],
    "landuse": ["grass", "forest", "village_green"],
    "natural": ["wood"],
}
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


@st.cache_data(ttl=3600, show_spinner=False)
def _photon(q, blat, blon):
    params = {"q": q, "limit": 8, "lang": "de"}
    if blat is not None:
        params.update(lat=blat, lon=blon)
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


def search_places(term):
    """Vorschläge während der Eingabe. Fehlertolerant; bei 0 Treffern wird die
    Eingabe schrittweise gekürzt (ähnliche Treffer), dann Nominatim als Fallback."""
    term = (term or "").strip()
    if len(term) < 3:
        return []
    bias = st.session_state.get("bias")
    blat, blon = (round(bias[0], 2), round(bias[1], 2)) if bias else (None, None)
    q = term
    for _ in range(3):
        try:
            feats = _photon(q, blat, blon)
        except Exception:
            feats = []
        out, seen = [], set()
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
        if out:
            return out
        words = q.split()
        if len(words) <= 1:
            break
        q = " ".join(words[:-1])
    try:
        return _nominatim(term)
    except Exception:
        return []


# ------------------------------------------------------- Daten
def corridor_bbox(p1, p2):
    (lat1, lon1), (lat2, lon2) = p1, p2
    ky = 111_320.0
    kx = ky * math.cos(math.radians((lat1 + lat2) / 2))
    span = max(abs(lon2 - lon1) * kx, abs(lat2 - lat1) * ky)
    pad = max(350.0, 0.25 * span)
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
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not len(gdf):
                return None
            gdf = gdf.reset_index(drop=True)
            if "name" not in gdf:
                gdf["name"] = None
            gdf["name"] = gdf["name"].fillna("Grünfläche").astype(str)
            return gdf[["name", "geometry"]]
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_g, f_p = ex.submit(get_graph), ex.submit(get_parks)
        G0, parks = f_g.result(), f_p.result()

    G = ox.project_graph(G0)
    crs = G.graph["crs"]
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

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
    return {"crs": crs, "pos": pos, "H": H, "lengths": lengths, "lines": lines, "D": D,
            "node_list": node_list, "xy": np.array([pos[n] for n in node_list]),
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


def _pin(emoji):
    return folium.DivIcon(
        html=f'<div style="font-size:30px;line-height:30px;text-align:center;'
             f'filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))">{emoji}</div>',
        icon_size=(32, 32), icon_anchor=(16, 28))


def _popup(title, r):
    mins = max(1, round(r["dist"] / M_PER_MIN))
    return folium.Popup(f"<b>{title}</b><br>{r['dist']:,.0f} m · ca. {mins} min<br>"
                        f"{r['green']:.0%} Grünanteil".replace(",", "."), max_width=220)


LEGEND = """
<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.96);
padding:10px 14px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.25);
font:13px/1.7 system-ui,-apple-system,sans-serif;color:#222">
<div><span style="display:inline-block;width:24px;height:5px;background:#0d7a45;border-radius:3px;vertical-align:middle;margin-right:8px"></span>Grüne Route</div>
<div><span style="display:inline-block;width:24px;border-top:3px dashed #e4572e;vertical-align:middle;margin-right:8px"></span>Kürzeste Route</div>
<div><span style="display:inline-block;width:14px;height:14px;background:#9fd89a;border:1px solid #4f9d57;border-radius:4px;vertical-align:middle;margin-right:10px"></span>Park / Grünfläche</div>
</div>"""


def build_map(area, res, p1, p2, t1, t2):
    m = folium.Map(tiles=None, control_scale=True, zoom_control=True)
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
                     name="Karte", attr="© OpenStreetMap-Mitwirkende © CARTO",
                     subdomains="abcd", max_zoom=19).add_to(m)
    folium.TileLayer("CartoDB positron", name="Hell").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     name="Satellit", attr="Esri, Maxar, Earthstar Geographics", max_zoom=19).add_to(m)

    if area["parks_json"]:
        folium.GeoJson(
            area["parks_json"], name="Grünflächen",
            style_function=lambda _: {"color": "#4f9d57", "fillColor": "#8fd18a",
                                      "fillOpacity": 0.45, "weight": 1},
            highlight_function=lambda _: {"fillOpacity": 0.7, "weight": 2},
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["🌳"], labels=True),
        ).add_to(m)

    short = route_latlon(area, res["short"]["path"])
    green = route_latlon(area, res["green"]["path"])
    fs = folium.FeatureGroup(name="Kürzeste Route").add_to(m)
    folium.PolyLine(short, color="#e4572e", weight=4, opacity=0.9, dash_array="8 8",
                    popup=_popup("Kürzeste Route", res["short"])).add_to(fs)
    fg = folium.FeatureGroup(name="Grüne Route").add_to(m)
    folium.PolyLine(green, color="#ffffff", weight=11, opacity=0.9).add_to(fg)  # Kontur
    AntPath(green, color="#0d7a45", pulse_color="#b9f0cb", weight=7, delay=1600,
            dash_array=[18, 28], opacity=1, popup=_popup("Grüne Route", res["green"])).add_to(fg)

    folium.Marker(p1, icon=_pin("🚶"), tooltip=f"Start: {t1}").add_to(m)
    folium.Marker(p2, icon=_pin("🏁"), tooltip=f"Ziel: {t2}").add_to(m)

    Fullscreen(position="topright", title="Vollbild", title_cancel="Beenden").add_to(m)
    LocateControl(auto_start=False, strings={"title": "Mein Standort"}).add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND))

    pts = short + green
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                  [max(p[0] for p in pts), max(p[1] for p in pts)]], padding=(40, 40))
    return m


# ------------------------------------------------------- UI
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
.trip{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 14px}
.pill{border-radius:999px;padding:6px 14px;background:rgba(47,168,102,.14);
 border:1px solid rgba(47,168,102,.45);font-size:.92rem}
.stButton>button{border-radius:12px;font-weight:600;padding:.6rem 1rem}
iframe{border-radius:16px}
</style>
<div class="hero"><h1>🌳 Grüne Route</h1>
<p>Der schönere Weg zu Fuß: vorbei an Parks und Grün statt nur auf kürzester Strecke.</p></div>
""",
    unsafe_allow_html=True,
)

start_v = st_searchbox(search_places, key="start_sb", label="Start",
                       placeholder="Straße, Adresse, Café, Restaurant, Geschäft …",
                       clear_on_submit=False)
if start_v:
    st.session_state["bias"] = _unpack(start_v)[:2]  # Ziel-Vorschläge in der Nähe des Starts
ziel_v = st_searchbox(search_places, key="ziel_sb", label="Ziel",
                      placeholder="Straße, Adresse, Café, Restaurant, Geschäft …",
                      clear_on_submit=False)
st.caption("Tipp: Tippe mindestens 3 Buchstaben, dann erscheinen Vorschläge. "
           "Tippfehler sind okay, es werden ähnliche Orte angezeigt.")

go = st.button("Route finden", type="primary", use_container_width=True)

if go:
    if not (start_v and ziel_v):
        st.warning("Bitte Start und Ziel jeweils aus den Vorschlägen auswählen.")
    else:
        st.session_state.pop("ctx", None)
        lat1, lon1, lab1 = _unpack(start_v)
        lat2, lon2, lab2 = _unpack(ziel_v)
        p1, p2 = (lat1, lon1), (lat2, lon2)
        with st.status("Route wird vorbereitet …", expanded=True) as status:
            try:
                bbox, span = corridor_bbox(p1, p2)
                if span > MAX_SPAN_M:
                    status.update(label="Strecke zu lang", state="error")
                    st.error(f"Start und Ziel sind {span / 1000:.1f} km auseinander. "
                             f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
                    st.stop()
                st.write("🗺️ Wegenetz und Grünflächen laden (nur der Korridor zwischen Start und Ziel)")
                t0 = time.time()
                area = load_area(bbox)
                st.write(f"✅ {area['n_edges']:,} Wegstücke, {area['n_parks']} Grünflächen "
                         f"({time.time() - t0:.0f} s)".replace(",", "."))
                st.session_state["ctx"] = dict(
                    area=area, p1=p1, p2=p2,
                    o=nearest_node(area, *p1), t=nearest_node(area, *p2),
                    t1=lab1.split(" · ")[0], t2=lab2.split(" · ")[0])
                status.update(label="Fertig", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Fehler", state="error")
                st.error(f"Das hat nicht geklappt: {e}")

ctx = st.session_state.get("ctx")
if ctx:
    area = ctx["area"]
    st.markdown(f'<div class="trip"><span class="pill">🚶 {ctx["t1"]}</span>'
                f'<span class="pill">🏁 {ctx["t2"]}</span></div>', unsafe_allow_html=True)
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
    m = build_map(area, res, ctx["p1"], ctx["p2"], ctx["t1"], ctx["t2"])
    st_html(m._repr_html_(), height=580)

    with st.expander("Datenquellen und Gewichtung"):
        st.markdown(
            "- Wegenetz (Fußwege) und Grünflächen: OpenStreetMap via Overpass (ODbL)\n"
            "- Ortssuche: Photon (photon.komoot.io) auf OSM-Basis, Fallback Nominatim\n"
            "- Kartenhintergrund: CARTO / OpenStreetMap, Satellit: Esri\n"
            f"- Grünflächen-Tags: `{GREEN_TAGS}`\n"
            f"- Score je Abtastpunkt: 1,0 im Park, linear fallend auf 0 im Abstand von {buf} m\n"
            f"- Kantenscore: Mittel aus {SAMPLES} Abtastpunkten\n"
            f"- Kosten: Länge × (1 + {LEVELS[level]:g} × (1 − Grünscore))"
        )
