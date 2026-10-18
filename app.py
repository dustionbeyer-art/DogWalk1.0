"""Grünes Routing: Fußwege mit möglichst viel Park- und Grünflächen-Nähe.

Start:  pip install -r requirements.txt
        streamlit run app.py

Daten:  OpenStreetMap (Overpass über osmnx, ODbL) für Wegenetz und Grünflächen,
        Geocoding über Nominatim. Benötigt Internetzugang.
Kosten: kosten(kante) = laenge * (1 + LAMBDA * (1 - gruen(kante)))
Grün:   Mittel über 12 Abtastpunkte je Kante; 1.0 im Park, danach linear
        fallend auf 0 im Abstand PUFFER.
"""
import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
import shapely
import folium
import streamlit as st
from shapely.geometry import LineString
from streamlit.components.v1 import html as st_html

GREEN_TAGS = {
    "leisure": ["park", "garden", "nature_reserve", "recreation_ground"],
    "landuse": ["grass", "forest", "village_green"],
    "natural": ["wood"],
}
SAMPLES = 12  # Abtastpunkte pro Kante


def green_scores(G, parks_union, buffer_m):
    """Grünscore 0..1 je Kante, Reihenfolge wie list(G.edges(keys=True))."""
    edges = list(G.edges(keys=True, data=True))
    lines = np.empty(len(edges), dtype=object)
    for i, (u, v, k, d) in enumerate(edges):
        lines[i] = d.get("geometry") or LineString(
            [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        )
    if parks_union is None or parks_union.is_empty:
        return edges, np.zeros(len(edges))
    fr = (np.arange(SAMPLES) + 0.5) / SAMPLES
    dist = np.stack([
        shapely.distance(shapely.line_interpolate_point(lines, f, normalized=True), parks_union)
        for f in fr
    ])
    g = np.clip(1 - dist / buffer_m, 0, 1).mean(axis=0)
    return edges, g


@st.cache_resource(show_spinner="Lade Wegenetz und Grünflächen …")
def load_area(lat, lon, radius, buffer_m):
    G0 = ox.graph_from_point((lat, lon), dist=radius, network_type="walk")
    G = ox.project_graph(G0)
    try:
        parks = ox.features_from_point((lat, lon), tags=GREEN_TAGS, dist=radius)
        parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        parks = parks[["geometry"]].to_crs(G.graph["crs"])
        union = parks.geometry.union_all() if hasattr(parks.geometry, "union_all") else parks.geometry.unary_union
    except Exception:  # keine Grünflächen im Gebiet
        parks = gpd.GeoDataFrame(geometry=[], crs=G.graph["crs"])
        union = None
    edges, g = green_scores(G, union, buffer_m)
    for (u, v, k, d), gi in zip(edges, g):
        d["green"] = float(gi)
    return G0, G, parks


def route_and_stats(G, o, t, weight):
    path = nx.shortest_path(G, o, t, weight=weight)
    L = G_len = 0.0
    green = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = min(G[a][b].values(), key=lambda e: e[weight])
        L += d["length"]
        green += d["length"] * d["green"]
    return path, L, (green / L if L else 0.0)


def latlons(G, path, weight):
    gdf = ox.routing.route_to_gdf(G, path, weight=weight).to_crs(4326)
    pts = []
    for geom in gdf.geometry:
        pts += [(y, x) for x, y in geom.coords]
    return pts


st.set_page_config(page_title="Grünes Routing", page_icon="🌳", layout="centered")
st.title("Grünes Routing")
st.caption("Fußweg mit möglichst viel Grün statt nur der kürzesten Strecke.")

start = st.text_input("Start", "Brandenburger Tor, Berlin")
ziel = st.text_input("Ziel", "Alexanderplatz, Berlin")
lam = st.slider("Grün-Gewicht λ (0 = kürzeste Route)", 0.0, 10.0, 3.0, 0.5)
buffer_m = st.slider("Nähe-Puffer um Grünflächen (m)", 50, 400, 150, 25)

if st.button("Route berechnen", type="primary"):
    try:
        lat1, lon1 = ox.geocode(start)
        lat2, lon2 = ox.geocode(ziel)
    except Exception as e:
        st.error(f"Adresse nicht gefunden: {e}")
        st.stop()

    mid = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)
    radius = int(min(ox.distance.great_circle(lat1, lon1, lat2, lon2) / 2 + 700, 5000))
    G0, G, parks = load_area(round(mid[0], 5), round(mid[1], 5), radius, buffer_m)

    o = ox.distance.nearest_nodes(G0, lon1, lat1)
    t = ox.distance.nearest_nodes(G0, lon2, lat2)
    for u, v, k, d in G.edges(keys=True, data=True):
        d["cost"] = d["length"] * (1 + lam * (1 - d["green"]))

    p_short, L_s, g_s = route_and_stats(G, o, t, "length")
    p_green, L_g, g_g = route_and_stats(G, o, t, "cost")

    c1, c2 = st.columns(2)
    c1.metric("Kürzeste Route", f"{L_s:,.0f} m", f"Grün {g_s:.0%}", delta_color="off")
    c2.metric("Grüne Route", f"{L_g:,.0f} m", f"Grün {g_g:.0%} · +{(L_g / L_s - 1):.0%} Weg", delta_color="off")

    m = folium.Map(location=mid, zoom_start=14, tiles="OpenStreetMap")
    if len(parks):
        folium.GeoJson(
            parks.to_crs(4326).to_json(),
            style_function=lambda _: {"color": "#5fa35a", "fillColor": "#bfe3b4", "fillOpacity": 0.5, "weight": 1},
        ).add_to(m)
    folium.PolyLine(latlons(G, p_short, "length"), color="#d2461e", weight=4, dash_array="6", tooltip="Kürzeste Route").add_to(m)
    folium.PolyLine(latlons(G, p_green, "cost"), color="#0d6b3a", weight=6, tooltip="Grüne Route").add_to(m)
    folium.Marker((lat1, lon1), tooltip="Start").add_to(m)
    folium.Marker((lat2, lon2), tooltip="Ziel").add_to(m)
    st_html(m._repr_html_(), height=520)

    with st.expander("Datenquellen und Gewichtung"):
        st.markdown(
            f"- Wegenetz (`walk`) und Grünflächen: OpenStreetMap über Overpass (ODbL)\n"
            f"- Grünflächen-Tags: `{GREEN_TAGS}`\n"
            f"- Score: 1,0 im Park, linear fallend auf 0 im Abstand von {buffer_m} m\n"
            f"- Kantenscore: Mittel aus {SAMPLES} Abtastpunkten\n"
            f"- Kosten: Länge × (1 + {lam} × (1 − Grünscore))"
        )
      
