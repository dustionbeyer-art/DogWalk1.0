"""GreenWalker: Fußwege mit möglichst viel Park- und Grünflächen-Nähe.

Zwei Modi: Route von A nach B und "Ründchen" (grüne Runde mit Wunschlänge).

Start:  pip install -r requirements.txt
        streamlit run app.py

Daten:  OpenStreetMap (Overpass, ODbL) für Wegenetz + Grünflächen. Wenn die Overpass-Server
        überlastet sind: "Grün-Scan" mit Vektorkacheln (OpenFreeMap) + FOSSGIS-Routenserver.
        Ortssuche mit Vorschlägen: Photon (photon.komoot.io, OSM-Daten, fehler-
        tolerant, findet auch Läden/Restaurants), Fallback Nominatim.
Kosten: kosten(kante) = laenge * (1 + LAMBDA * (1 - gruen(kante)) + RUHE * hauptstrasse(kante))
Grün:   Mittel über 8 Abtastpunkte je Kante; 1.0 im Park, linear fallend
        auf 0 im Abstand "Grün-Reichweite".
"""
import base64
import gzip
import hashlib
import json
import math
import os
import random
import re
import socket
import struct
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from html import escape

import networkx as nx
import numpy as np
import requests
import streamlit as st
try:
    from scipy import ndimage
except Exception:  # Grün-Scan braucht scipy; ohne bleibt die einfache Fußroute
    ndimage = None
from streamlit.components.v1 import html as st_html
from streamlit_js_eval import get_geolocation
try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:  # ältere Version: Favoriten gelten dann nur für diese Sitzung
    streamlit_js_eval = None
from streamlit_searchbox import st_searchbox



class _Lazy:
    """Schwere Bibliotheken (geopandas, osmnx, shapely) erst beim ersten Gebrauch laden:
    die Startseite erscheint dadurch deutlich schneller, vor allem nach dem Aufwachen der App."""

    def __init__(self, name):
        self._name, self._mod = name, None

    def __getattr__(self, attr):
        if self._mod is None:
            import importlib
            self._mod = importlib.import_module(self._name)
            if self._name == "osmnx":
                self._mod.settings.use_cache = True
                self._mod.settings.log_console = False
                self._mod.settings.requests_timeout = 25
                self._mod.settings.overpass_rate_limit = False
        return getattr(self._mod, attr)


gpd, ox, shapely = _Lazy("geopandas"), _Lazy("osmnx"), _Lazy("shapely")


def STRtree(*a, **k):
    return shapely.STRtree(*a, **k)


def LineString(*a, **k):
    return shapely.geometry.LineString(*a, **k)


def Point(*a, **k):
    return shapely.geometry.Point(*a, **k)


# Mehrere Overpass-Server: der Hauptserver ist oft überlastet (502/504).
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
    "https://lz4.overpass-api.de/api",
    "https://z.overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]
LOAD_DEADLINE_S = 16  # harte Zeitgrenze für Overpass; danach übernimmt der Grün-Scan
OSM_CACHE_DIR = os.path.join(tempfile.gettempdir(), "green_walker_osm")
OSM_CACHE_DAYS = 7
_MIRROR_CACHE = {"t": 0.0, "v": None}


def rank_mirrors():
    """Server nach Auslastung sortiert; das Ergebnis gilt 2 Minuten (die Statusabfrage kostete sonst bei jeder
    Berechnung mehrere Sekunden)."""
    now = time.time()
    if _MIRROR_CACHE["v"] and now - _MIRROR_CACHE["t"] < 120:
        return list(_MIRROR_CACHE["v"])
    v = _rank_mirrors_uncached()
    _MIRROR_CACHE.update(t=time.time(), v=list(v))
    return v


def _rank_mirrors_uncached():
    """Fragt den Status aller Overpass-Server kurz und parallel ab: Server mit freien Plätzen zuerst,
    nicht erreichbare Server werden übersprungen. Spart das lange Warten auf überlastete Server."""
    def probe(base):
        try:
            r = requests.get(base + "/status", headers=UA, timeout=2.5)
            if r.status_code != 200:
                return base, 9.0
            t = r.text
            m = re.search(r"(\d+)\s+slots?\s+available\s+now", t)
            if (m and int(m.group(1)) > 0) or "Rate limit: 0" in t:
                return base, 0.0
            m = re.search(r"available after.*?in\s+(\d+)\s+seconds", t)
            return base, (1.0 + min(int(m.group(1)), 60) / 60.0) if m else 1.5
        except Exception:
            return base, 9.0

    with ThreadPoolExecutor(max_workers=len(OVERPASS_MIRRORS)) as ex:
        res = list(ex.map(probe, random.sample(OVERPASS_MIRRORS, len(OVERPASS_MIRRORS))))
    res.sort(key=lambda x: x[1])
    ok = [b for b, s in res if s < 9]
    return ok or list(OVERPASS_MIRRORS)


UA = {"User-Agent": "green-walker-app/1.0 (privates Hobbyprojekt)"}
WATER_WAYS = ["river", "canal", "stream", "riverbank"]
WATER_BUF = 120  # Meter: bis hierher zählt „am Wasser“ (Uferweg, Blick über den See)
WATER_NEAR = 0.4  # Wasser-Score ab dem eine Kante als „am Wasser“ gilt (= ca. 70 m Abstand)
# Flächen, auf denen Hunde meist nicht erlaubt sind: (OSM-Schlüssel, Wert) -> Anzeigename
DOG_ZONE_TAGS = [("landuse", "cemetery"), ("amenity", "grave_yard"), ("leisure", "playground"), ("tourism", "zoo"),
                 ("leisure", "zoo"), ("dog", "no")]
DOG_ZONE_LABEL = {"cemetery": "Friedhof", "grave_yard": "Friedhof", "playground": "Spielplatz", "zoo": "Zoo"}
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


# Straßenbreite je Kante (0 = schmale Gasse / Fußweg, 1 = breite Hauptstraße): aus Straßenklasse,
# Spurenzahl (lanes) und Breitenangabe (width), soweit OpenStreetMap sie kennt
WIDE_CLASS = {"motorway": 1.0, "motorway_link": 1.0, "trunk": 1.0, "trunk_link": 1.0, "primary": 1.0,
              "primary_link": 0.95, "secondary": 0.85, "secondary_link": 0.8, "tertiary": 0.6,
              "tertiary_link": 0.55, "unclassified": 0.35, "residential": 0.25, "service": 0.1,
              "living_street": 0.05, "track": 0.05, "pedestrian": 0.0, "footway": 0.0, "path": 0.0,
              "cycleway": 0.0, "steps": 0.0, "bridleway": 0.0, "corridor": 0.0}
NARROW = 0.3  # bis hierher zählt eine Kante als kleine Gasse / Nebenweg


def _num(x):
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(x)) if x is not None else None
    return float(m.group(1).replace(",", ".")) if m else None


def edge_wide(d, hw):
    w = WIDE_CLASS.get(hw, 0.25)
    lanes, width = _num(d.get("lanes")), _num(d.get("width"))
    if lanes and lanes > 1:
        w = max(w, min(1.0, 0.25 + 0.15 * (lanes - 1)))
    if width:
        w = 0.5 * w + 0.5 * min(max((width - 3.0) / 9.0, 0.0), 1.0)
    return float(min(max(w, 0.0), 1.0))


SAMPLES = 8
MAX_BUF = 400
MAX_SPAN_M = 10000
M_PER_MIN = 80

LEVELS = {"Schnellste": 0.0, "Leicht grün": 1.5, "Ausgewogen": 4.0,
          "Sehr grün": 8.0, "Maximal grün": 14.0, "Grünster Umweg": 40.0}
# Hauptstraßen-Gewicht und Breiten-Gewicht je Stufe. „Maximal grün“ und „Grünster Umweg“ meiden
# Hauptstraßen und breite Straßen stark: bevorzugt kleine Gassen und Nebenstraßen.
LEVEL_QUIET = {"Maximal grün": (4.0, 6.0), "Grünster Umweg": (6.0, 10.0)}


# Zusatzoption „Extrem grün“ bei Route, Ründchen und Ründchen mit Zwischenziel: bis zu 35 % Umweg (bzw. Mehrlänge) sind
# erlaubt, solange die Strecke zu mehr als 70 % grün ist. Schafft eine Strecke das nicht, wird sie nicht angeboten.
EXTREME_NAME = "Extrem grün"
EXTREME_DETOUR = 0.35
EXTREME_GREEN_MIN = 0.70
EXTREME_QUIET = ((4.0, 6.0), (6.0, 10.0))  # (Hauptstraßen-, Breiten-Gewicht): zwei Stile, der grünere gewinnt


def extreme_hint(detour_txt=""):
    return f"Bis {EXTREME_DETOUR:.0%} Umweg erlaubt, dafür über {EXTREME_GREEN_MIN:.0%} Grün{detour_txt}"


def extreme_note(n_ok):
    """Hinweis an der Standardvariante, wenn keine Extrem-grün-Strecke über 70 % Grün möglich war."""
    return (f" · 🌿 Extrem grün: innerhalb von {EXTREME_DETOUR:.0%} Umweg sind hier keine "
            f"{EXTREME_GREEN_MIN:.0%} Grün erreichbar") if not n_ok else ""


def greenest_within(run, budget):
    """Grünste Route mit höchstens `budget` Meter Länge. run(lam, mu, wq) liefert {"dist", "green", ...}.
    Das Grün-Gewicht lam wird je Stil per Halbierung gesucht (grösstes lam, das noch ins Längenbudget passt)."""
    best = None
    for mu, wq in EXTREME_QUIET:
        cands = []
        r = run(300.0, mu, wq)
        if r["dist"] <= budget:
            cands.append(r)
        else:
            lo, hi = 2.0, 300.0
            for _ in range(8):
                mid = math.sqrt(lo * hi)
                r = run(mid, mu, wq)
                if r["dist"] <= budget:
                    cands.append(r)
                    lo = mid
                else:
                    hi = mid
        for r in cands:
            if best is None or r["green"] > best["green"] + 1e-6 or (
                    abs(r["green"] - best["green"]) <= 1e-6 and r["dist"] < best["dist"]):
                best = r
    return best


def level_quiet(name):
    lam = LEVELS[name]
    return LEVEL_QUIET.get(name, (1.0 if lam > 0 else 0.0, 0.0))


LEVEL_HINT = {
    "Schnellste": "Nur die Distanz zählt.",
    "Leicht grün": "Kleine Umwege für spürbar mehr Grün.",
    "Ausgewogen": "Guter Mittelweg aus Weg und Grün.",
    "Sehr grün": "Deutliche Umwege sind okay, wenn es grüner wird.",
    "Maximal grün": "Grün zuerst, Hauptstraßen und breite Straßen meiden: kleine Gassen und Nebenstraßen.",
    "Grünster Umweg": "Auch große Umwege, solange der Grünanteil steigt; ruhige Gassen bevorzugt.",
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


_PH_CACHE, _PH_FAIL, _PH_LOCK = {}, {}, threading.Lock()


def _photon(q, blat, blon, tag=None, box=False):
    """Photon-Suche mit eigenem, thread-sicherem Zwischenspeicher (1 h). Ein Fehlschlag wird 20 s lang gemerkt,
    damit die Suche nicht mehrfach auf denselben toten Server wartet."""
    key = (q, blat, blon, tag, box)
    now = time.time()
    with _PH_LOCK:
        hit = _PH_CACHE.get(key)
        if hit and now - hit[0] < 3600:
            return hit[1]
        if now - _PH_FAIL.get(key, 0) < 20:
            raise RuntimeError("Photon kurz nicht erreichbar")
    try:
        res = _photon_net(q, blat, blon, tag, box)
    except Exception:
        with _PH_LOCK:
            _PH_FAIL[key] = time.time()
        raise
    with _PH_LOCK:
        if len(_PH_CACHE) > 600:
            for k in sorted(_PH_CACHE, key=lambda k_: _PH_CACHE[k_][0])[:200]:
                _PH_CACHE.pop(k, None)
        _PH_CACHE[key] = (time.time(), res)
    return res


def _photon_net(q, blat, blon, tag=None, box=False):
    params = {"q": q, "limit": 12, "lang": "de"}
    if blat is not None:
        params.update(lat=blat, lon=blon)
        if box:  # nur im Umfeld von ca. 10 km um den Standort/Startpunkt suchen
            dlat = 0.095
            dlon = dlat / max(math.cos(math.radians(blat)), 0.3)
            params["bbox"] = f"{blon - dlon:.4f},{blat - dlat:.4f},{blon + dlon:.4f},{blat + dlat:.4f}"
    if tag:
        params["osm_tag"] = tag
    r = requests.get("https://photon.komoot.io/api/", params=params, headers=UA, timeout=4)
    r.raise_for_status()
    return r.json().get("features", [])


@st.cache_data(ttl=3600, show_spinner=False)
def _nominatim(q, blat=None, blon=None):
    params = {"q": q, "format": "jsonv2", "limit": 6, "accept-language": "de"}
    if blat is not None:
        dlat = 0.095
        dlon = dlat / max(math.cos(math.radians(blat)), 0.3)
        params.update(viewbox=f"{blon - dlon:.4f},{blat + dlat:.4f},{blon + dlon:.4f},{blat - dlat:.4f}", bounded=1)
    r = requests.get("https://nominatim.openstreetmap.org/search", headers=UA, timeout=8, params=params)
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


# ------------------------------------------------------- Schlagwortsuche
# Wer „essen“, „Bier“ oder „Brötchen“ tippt, bekommt passende Orte in der Nähe (nach Entfernung sortiert).
# Grundlage ist ein kuratiertes Wörterbuch; optional hilft die KI (Claude) bei unbekannten Begriffen,
# wenn in den Streamlit-Secrets ein ANTHROPIC_API_KEY hinterlegt ist.
CATS = {
    "restaurant": ("🍽️", "Restaurant", ['["amenity"="restaurant"]']),
    "imbiss": ("🌭", "Imbiss", ['["amenity"="fast_food"]']),
    "cafe": ("☕", "Café", ['["amenity"="cafe"]']),
    "kneipe": ("🍺", "Kneipe", ['["amenity"="pub"]', '["amenity"="bar"]']),
    "biergarten": ("🍻", "Biergarten", ['["amenity"="biergarten"]']),
    "kiosk": ("🥤", "Kiosk", ['["shop"="kiosk"]', '["shop"="newsagent"]', '["shop"="convenience"]']),
    "getraenke": ("🍾", "Getränkemarkt", ['["shop"="beverages"]', '["shop"="alcohol"]']),
    "wein": ("🍷", "Weinhandel", ['["shop"="wine"]']),
    "baecker": ("🥐", "Bäckerei", ['["shop"="bakery"]']),
    "eis": ("🍦", "Eisdiele", ['["amenity"="ice_cream"]', '["shop"="ice_cream"]']),
    "supermarkt": ("🛒", "Supermarkt", ['["shop"="supermarket"]']),
    "metzger": ("🥩", "Metzgerei", ['["shop"="butcher"]']),
    "obst": ("🥬", "Obst & Gemüse", ['["shop"="greengrocer"]', '["shop"="farm"]']),
    "blumen": ("💐", "Blumenladen", ['["shop"="florist"]']),
    "apotheke": ("💊", "Apotheke", ['["amenity"="pharmacy"]']),
    "drogerie": ("🧴", "Drogerie", ['["shop"="chemist"]']),
    "geldautomat": ("🏧", "Geldautomat", ['["amenity"="atm"]', '["amenity"="bank"]']),
    "post": ("📮", "Post / Paketshop", ['["amenity"="post_office"]', '["shop"="parcel_locker"]',
                                       '["amenity"="parcel_locker"]']),
    "toilette": ("🚻", "Toilette", ['["amenity"="toilets"]']),
    "trinkwasser": ("🚰", "Trinkwasser", ['["amenity"="drinking_water"]']),
    "spielplatz": ("🛝", "Spielplatz", ['["leisure"="playground"]']),
    "aussicht": ("🌄", "Aussichtspunkt", ['["tourism"="viewpoint"]']),
    "sehenswert": ("🏛️", "Sehenswürdigkeit", ['["tourism"="attraction"]', '["historic"="monument"]']),
    "bahnhof": ("🚉", "Bahnhof / Haltestelle", ['["railway"="station"]', '["railway"="halt"]']),
    "buch": ("📚", "Buchhandlung / Bibliothek", ['["shop"="books"]', '["amenity"="library"]']),
    "tabak": ("🚬", "Tabak / Zeitung", ['["shop"="tobacco"]', '["shop"="newsagent"]']),
}
KEYWORDS = {
    "essen": ["restaurant", "imbiss"], "restaurant": ["restaurant"], "mittagessen": ["restaurant", "imbiss"],
    "abendessen": ["restaurant"], "hunger": ["imbiss", "baecker", "restaurant"], "dinner": ["restaurant"],
    "lunch": ["restaurant", "imbiss"], "pizza": ["restaurant", "imbiss"], "burger": ["imbiss"],
    "döner": ["imbiss"], "doener": ["imbiss"], "imbiss": ["imbiss"], "pommes": ["imbiss"],
    "currywurst": ["imbiss"], "snack": ["imbiss", "kiosk", "baecker"], "fastfood": ["imbiss"],
    "trinken": ["kiosk", "getraenke", "cafe"], "durst": ["kiosk", "getraenke", "trinkwasser"],
    "getränk": ["kiosk", "getraenke"], "getränke": ["getraenke", "kiosk"], "getränkemarkt": ["getraenke"],
    "bier": ["kneipe", "biergarten", "restaurant", "getraenke"], "kneipe": ["kneipe"], "pub": ["kneipe"], "bar": ["kneipe"],
    "bierchen": ["kneipe", "biergarten", "restaurant"], "kölsch": ["kneipe", "restaurant", "biergarten"],
    "koelsch": ["kneipe", "restaurant", "biergarten"], "pils": ["kneipe", "restaurant", "biergarten"],
    "altbier": ["kneipe", "restaurant"], "brauerei": ["kneipe", "restaurant", "biergarten"],
    "brauhaus": ["restaurant", "kneipe"], "gaststätte": ["restaurant", "kneipe"], "gasthaus": ["restaurant", "kneipe"],
    "drink": ["kneipe", "restaurant"], "drinks": ["kneipe", "restaurant"], "ausgehen": ["kneipe", "restaurant", "cafe"],
    "biergarten": ["biergarten"], "feierabendbier": ["kneipe", "biergarten", "restaurant", "kiosk"], "cocktail": ["kneipe"],
    "wein": ["wein", "kneipe"], "kaffee": ["cafe", "baecker"], "café": ["cafe"], "cafe": ["cafe"],
    "kuchen": ["cafe", "baecker"], "tee": ["cafe"], "kiosk": ["kiosk"], "büdchen": ["kiosk"],
    "buedchen": ["kiosk"], "späti": ["kiosk"], "spaeti": ["kiosk"], "trinkhalle": ["kiosk"],
    "wasserhäuschen": ["kiosk"], "bäcker": ["baecker"], "bäckerei": ["baecker"], "baecker": ["baecker"],
    "brötchen": ["baecker"], "broetchen": ["baecker"], "brot": ["baecker"], "frühstück": ["cafe", "baecker"],
    "fruehstueck": ["cafe", "baecker"], "brunch": ["cafe"], "croissant": ["baecker", "cafe"],
    "eis": ["eis"], "eiscreme": ["eis"], "eisdiele": ["eis"], "gelato": ["eis"],
    "einkaufen": ["supermarkt", "kiosk", "baecker"], "einkauf": ["supermarkt"], "supermarkt": ["supermarkt"],
    "lebensmittel": ["supermarkt", "obst"], "rewe": ["supermarkt"], "edeka": ["supermarkt"],
    "aldi": ["supermarkt"], "lidl": ["supermarkt"], "metzger": ["metzger"], "fleisch": ["metzger"],
    "wurst": ["metzger", "imbiss"], "obst": ["obst"], "gemüse": ["obst"], "gemuese": ["obst"],
    "hofladen": ["obst"], "blumen": ["blumen"], "strauß": ["blumen"], "strauss": ["blumen"],
    "geschenk": ["blumen", "buch"], "apotheke": ["apotheke"], "medikament": ["apotheke"],
    "tabletten": ["apotheke"], "drogerie": ["drogerie"], "shampoo": ["drogerie"],
    "geld": ["geldautomat"], "geldautomat": ["geldautomat"], "atm": ["geldautomat"], "bank": ["geldautomat"],
    "abheben": ["geldautomat"], "post": ["post"], "paket": ["post"], "päckchen": ["post"],
    "brief": ["post"], "briefmarke": ["post"], "dhl": ["post"], "hermes": ["post"], "dpd": ["post"],
    "toilette": ["toilette"], "wc": ["toilette"], "klo": ["toilette"], "trinkwasser": ["trinkwasser"],
    "wasser": ["trinkwasser", "kiosk"], "spielplatz": ["spielplatz"], "kinder": ["spielplatz"],
    "aussicht": ["aussicht"], "aussichtspunkt": ["aussicht"], "panorama": ["aussicht"],
    "sehenswürdigkeit": ["sehenswert"], "sehenswürdigkeiten": ["sehenswert"], "denkmal": ["sehenswert"],
    "sehenswert": ["sehenswert"], "bahnhof": ["bahnhof"], "haltestelle": ["bahnhof"], "buch": ["buch"],
    "bücher": ["buch"], "buchhandlung": ["buch"], "bibliothek": ["buch"], "zeitung": ["tabak", "kiosk"],
    "zigaretten": ["tabak", "kiosk"], "tabak": ["tabak"],
}
_SEL_OK = re.compile(r'^\["(amenity|shop|leisure|tourism|historic|natural|railway)"="[a-z_]{2,30}"\]$')
KW_MAX_CATS = 3
KW_RADIUS_M = 2500


def match_categories(term):
    """Findet Kategorien zu einer Eingabe (exakt, Wortanfang, Tippfehler). Leere Liste: kein Schlagwort."""
    import difflib
    words = [w for w in re.split(r"[\s,;/]+", term.lower()) if w]
    if not words or len(words) > 3:
        return []
    found = []
    for w in words:
        hits = []
        if w in KEYWORDS:
            hits = KEYWORDS[w]
        elif len(w) >= 4:
            pref = [k for k in KEYWORDS if k.startswith(w) or (len(k) >= 4 and w.startswith(k))]
            if pref:
                hits = KEYWORDS[min(pref, key=len)]
            else:
                close = difflib.get_close_matches(w, list(KEYWORDS), n=1, cutoff=0.84)
                if close:
                    hits = KEYWORDS[close[0]]
        for h in hits:
            if h not in found:
                found.append(h)
    return found[:KW_MAX_CATS + 1]


@st.cache_data(ttl=86400, show_spinner=False)
def _ai_categories(term):
    """Optional: KI übersetzt unbekannte Begriffe in OSM-Orte. Nur mit ANTHROPIC_API_KEY in den Secrets."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    if not key:
        return []
    prompt = (
        "Ein Fußgänger sucht in einer Spazier-App einen Zwischenstopp und tippt: \"" + term[:60] + "\".\n"
        "Welche OpenStreetMap-Orte passen? Antworte NUR mit JSON: "
        '{"orte":[{"label":"Deutsch, max 3 Wörter","icon":"1 Emoji","tags":[["amenity","pub"]]}]}\n'
        "Höchstens 2 Orte, je höchstens 3 tags. Erlaubte Schlüssel: amenity, shop, leisure, tourism, historic. "
        "Wenn nichts passt: {\"orte\":[]}.")
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=7,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": "claude-haiku-5-5", "max_tokens": 300,
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json().get("content", []))
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    out = []
    for o in data.get("orte", [])[:2]:
        sels = [f'["{k}"="{v}"]' for k, v in o.get("tags", [])[:3] if isinstance(k, str) and isinstance(v, str)]
        sels = [s for s in sels if _SEL_OK.match(s)]
        if sels:
            out.append((str(o.get("icon", "📍"))[:4], str(o.get("label", term))[:30], sels))
    return out


def _cat_defs(term):
    defs = [CATS[c] for c in match_categories(term)]
    if not defs and len(term) >= 4:
        try:
            defs = _ai_categories(term)
        except Exception:
            defs = []
    return defs[:KW_MAX_CATS + 1]


def _op_post(base, q, timeout):
    r = requests.post(base + "/interpreter", data={"data": q}, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _overpass_race(q, timeout=8.0, mirrors=None):
    """Frage gleichzeitig mehrere Overpass-Server, nimm die erste brauchbare Antwort (ein überlasteter
    Server verzögert die Suche so nicht mehr)."""
    from concurrent.futures import as_completed
    mirrors = mirrors or OVERPASS_MIRRORS[:3]
    ex = ThreadPoolExecutor(max_workers=len(mirrors))
    last = None
    try:
        futs = [ex.submit(_op_post, m, q, timeout) for m in mirrors]
        for f in as_completed(futs, timeout=timeout + 2):
            try:
                return f.result()
            except Exception as e:
                last = e
    except Exception as e:
        last = e
    finally:
        ex.shutdown(wait=False)
    raise last or RuntimeError("Overpass nicht erreichbar")


def _sel_kv(sel):
    m = re.match(r'^\["([a-z_]+)"="([a-z_]+)"\]$', sel)
    return (m.group(1), m.group(2)) if m else (None, None)


@st.cache_data(ttl=600, show_spinner=False)
def _nearby_multi(selectors, lat, lon, radius):
    """Alle Orte, die einer der Auswahlen (Tags) entsprechen, rund um (lat, lon): eine einzige Abfrage.
    Ergebnis: Liste (lat, lon, name, straße, nr, (schlüssel, wert))."""
    body = "".join(f"nwr{s}(around:{radius},{lat},{lon});" for s in selectors)
    js = _overpass_race(f"[out:json][timeout:8];({body});out center 800;")
    kvs = [_sel_kv(x) for x in selectors]
    res = []
    for e in js.get("elements", []):
        c = e.get("center") or e
        if "lat" not in c:
            continue
        t = e.get("tags", {})
        kv = frozenset((k, v) for k, v in kvs if k and t.get(k) == v)  # ein Laden kann mehreren Kategorien angehören
        if not kv:
            continue
        res.append((float(c["lat"]), float(c["lon"]), t.get("name") or t.get("brand") or t.get("operator") or "",
                    t.get("addr:street", ""), t.get("addr:housenumber", ""), kv))
    return res


@st.cache_data(ttl=600, show_spinner=False)
def _nominatim_poi(word, lat, lon, half_deg=0.05):
    """Ersatz, wenn Overpass nicht antwortet: Nominatim kennt Begriffe wie „Restaurant“ oder „Apotheke“."""
    box = f"{lon - half_deg / max(math.cos(math.radians(lat)), 0.3):.4f},{lat + half_deg:.4f}," \
          f"{lon + half_deg / max(math.cos(math.radians(lat)), 0.3):.4f},{lat - half_deg:.4f}"
    r = requests.get("https://nominatim.openstreetmap.org/search", headers=UA, timeout=8,
                     params={"q": word, "format": "jsonv2", "limit": 20, "accept-language": "de",
                             "viewbox": box, "bounded": 1})
    r.raise_for_status()
    return [(float(x["lat"]), float(x["lon"]), x.get("name") or "", "", "", frozenset({("nominatim", word)}))
            for x in r.json()]


_GENERIC_WORDS = {"kiosk", "imbiss", "restaurant", "gaststätte", "gaststaette", "café", "cafe", "bäckerei", "baeckerei",
                  "bar", "pub", "getränkemarkt", "getraenkemarkt", "späti", "spaeti", "büdchen", "buedchen", "trinkhalle",
                  "shop", "markt", "und", "der", "die", "das", "zum", "zur", "am"}


def _norm_name(n):
    """Name zum Vergleichen: klein, ohne Zeichen und ohne Allerweltswörter („Hit Kiosk“ = „Hit“)."""
    words = re.findall(r"[a-z0-9äöüß]+", (n or "").lower().replace("&", " und "))
    core = [w for w in words if w not in _GENERIC_WORDS]
    return "".join(core or words)


def merge_same_places(items, bias=None, radius_m=120.0):
    """Ein Geschäft ist oft mehrfach eingetragen (Kiosk + Gaststätte + Getränkemarkt, mehrere OSM-Objekte).
    Gleicher oder fast gleicher Name und weniger als 120 m Abstand: ein Eintrag, die Kategorien werden vereint."""
    groups = []
    for it in items:
        nn = _norm_name(it[2])
        if not nn:
            continue
        for g in groups:
            same = g["nn"] == nn or (min(len(nn), len(g["nn"])) >= 4 and (nn in g["nn"] or g["nn"] in nn))
            if same and dist_m((g["lat"], g["lon"]), (it[0], it[1])) <= radius_m:
                g["kvs"] |= set(it[5])
                if not g["street"] and it[3]:
                    g["street"], g["nr"] = it[3], it[4]
                break
        else:
            groups.append({"lat": it[0], "lon": it[1], "name": it[2], "street": it[3], "nr": it[4],
                           "kvs": set(it[5]), "nn": nn})
    return groups


def _safe_poi(label, lat, lon):
    try:
        return [it for it in _nominatim_poi(label, lat, lon) if it[2]]
    except Exception:
        return []


def _tile_kw_items(selectors, lat, lon, radius):
    """Orte zu den Auswahlen (Tags) aus den freien Vektorkacheln (OpenFreeMap, POI-Ebene): schnell, lokal
    zwischengespeichert und unabhängig von Overpass. Format wie _nearby_multi."""
    tpl = _ofm_template()
    dlat = radius / 110540.0
    dlon = dlat / max(math.cos(math.radians(lat)), 0.3)
    x0, y1 = _tile_of(lat - dlat, lon - dlon, VT_ZOOM)
    x1, y0 = _tile_of(lat + dlat, lon + dlon, VT_ZOOM)
    tiles = [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1) for y in range(min(y0, y1), max(y0, y1) + 1)]
    if len(tiles) > 36:
        tiles = sorted(tiles, key=lambda t: abs(t[0] - (x0 + x1) / 2) + abs(t[1] - (y0 + y1) / 2))[:36]
    kvs = [_sel_kv(x) for x in selectors]
    vals = {v for k, v in kvs if k}
    with ThreadPoolExecutor(max_workers=6) as ex_:
        datas = list(ex_.map(lambda t: _fetch_tile(tpl, VT_ZOOM, t[0], t[1]), tiles))
    res = []
    for (tx, ty), data in zip(tiles, datas):
        if not data:
            continue
        for extent, gtype, parts, props in mvt_decode(data, {"poi"}).get("poi", []):
            if gtype != 1 or not parts:
                continue
            name = props.get("name") or props.get("name:latin") or ""
            cls, sub = str(props.get("class") or ""), str(props.get("subclass") or "")
            hit = {sub} & vals or {cls} & vals  # subclass ist der OSM-Wert, class nur die grobe Gruppe
            if not name or not hit:
                continue
            kv = frozenset((k, v) for k, v in kvs if k and v in hit)
            for part in parts:
                lons, lats = _tile_lonlat(tx, ty, VT_ZOOM, extent, part[:1])
                la, lo = float(lats[0]), float(lons[0])
                if dist_m((lat, lon), (la, lo)) <= radius * 1.1:
                    res.append((la, lo, str(name), "", "", kv))
    return res


@st.cache_data(ttl=900, show_spinner=False)
def _tile_kw_cached(selectors, lat, lon, radius):
    return _tile_kw_items(selectors, lat, lon, radius)


def keyword_places(term, bias):
    """Schlagwort -> nahe Orte als Vorschläge (nach Entfernung, jedes Geschäft nur einmal).
    Zwei unabhängige Quellen gleichzeitig: Vektorkacheln (schnell, lokal) und Overpass (vollständig, mit Adresse)."""
    if not bias:
        return []
    defs = _cat_defs(term)
    if not defs:
        return []
    lat, lon = round(bias[0], 3), round(bias[1], 3)
    allsel = tuple(sorted({sel for _, _, sels in defs for sel in sels}))

    def near(radius):
        try:
            return [it for it in _nearby_multi(allsel, lat, lon, radius) if it[2]]
        except Exception:
            return None

    def tiles(radius):
        try:
            return [it for it in _tile_kw_cached(allsel, lat, lon, radius) if it[2]]
        except Exception:
            return None

    from concurrent.futures import wait as _wait
    t0 = time.time()
    ex_ = ThreadPoolExecutor(max_workers=4)
    f_t, f_s, f_m = ex_.submit(tiles, 2500), ex_.submit(near, 1000), ex_.submit(near, 2500)
    _wait([f_t], timeout=7.0)  # die Kacheln sind meist in 1-2 s da
    have_t = f_t.done() and f_t.result() is not None and len(f_t.result()) >= 6
    _wait([f_s, f_m], timeout=max(0.5, (2.5 if have_t else 9.0) - (time.time() - t0)))
    results = [f.result() for f in (f_t, f_s, f_m) if f.done() and f.result() is not None]
    items = [it for r_ in results for it in r_]
    n_places = len(merge_same_places(items, bias))
    if n_places < 6:
        # zu wenig im Umkreis: größeren Umkreis aus beiden Quellen
        f_tb, f_b = ex_.submit(tiles, 5500), ex_.submit(near, 6000)
        _wait([f_tb, f_b], timeout=8.0)
        for f in (f_tb, f_b):
            if f.done() and f.result():
                items += f.result()
    ex_.shutdown(wait=False)
    if not items:
        # Ersatz: Nominatim (andere Server, anderer Zugang)
        with ThreadPoolExecutor(max_workers=3) as ex2:
            for part in ex2.map(lambda lb: _safe_poi(lb, lat, lon), [lb for _, lb, _s in defs[:3]]):
                items += part
    items.sort(key=lambda it: (dist_m((bias[0], bias[1]), (it[0], it[1])), 0 if it[3] else 1))
    groups = merge_same_places(items, bias)
    out = []
    for g in groups[:60]:
        cats = []  # passende Kategorien, in der Reihenfolge der Suchbegriff-Treffer
        for icon, label, sels in defs:
            kvs = {_sel_kv(x) for x in sels}
            if (g["kvs"] & kvs or ("nominatim", label) in g["kvs"]) and (icon, label) not in cats:
                cats.append((icon, label))
        if not cats:
            continue
        addr = (g["street"] + (" " + g["nr"] if g["nr"] else "")) if g["street"] else ""
        lab = f"{cats[0][0]} {g['name']} · " + " / ".join(c[1] for c in cats[:2]) + (f" · {addr}" if addr else "")
        out.append((cats[0], (lab, _pack(g["lat"], g["lon"], f"{cats[0][0]} {g['name']}"))))
    # Reihum je Kategorie das Nächste (sonst verdrängen z. B. viele Restaurants alle Kneipen): Bar, Biergarten, ...
    buckets = {}
    for c, item in out:
        buckets.setdefault(c, []).append(item)
    mixed = []
    while any(buckets.values()) and len(mixed) < 8:
        for c in list(buckets):
            if buckets[c] and len(mixed) < 8:
                mixed.append(buckets[c].pop(0))
    mixed.sort(key=lambda it: dist_m((bias[0], bias[1]), (_unpack(it[1])[0], _unpack(it[1])[1])))
    return mixed


def _title_of(label):
    t = re.sub(r"^[^0-9A-Za-zÄÖÜäöüß]+", "", label or "")
    return t.split(" · ")[0].strip()


def dedupe_results(items, radius_m=120.0):
    """Ergebnisliste (Label, Wert): gleiche Orte (Name + Nähe) nur einmal, Reihenfolge bleibt."""
    out = []
    for lab, val in items:
        try:
            la, lo, plab = _unpack(val)
        except Exception:
            out.append((lab, val))
            continue
        nn = _norm_name(_title_of(plab))
        dup = False
        for _l2, v2 in out:
            try:
                la2, lo2, p2 = _unpack(v2)
            except Exception:
                continue
            n2 = _norm_name(_title_of(p2))
            if nn and nn == n2 and dist_m((la, lo), (la2, lo2)) <= radius_m:
                dup = True
                break
        if not dup:
            out.append((lab, val))
    return out


# ------------------------------------------------------- Fehlertolerante Adresssuche
_PLZ = re.compile(r"\b\d{5}\b")
_COORD = re.compile(r"(-?\d{1,2}[.,]\d{3,})\s*[,;/ ]\s*(-?\d{1,3}[.,]\d{3,})")


def normalize_query(term):
    """Eingefügte Texte säubern: Zeilenumbrüche, doppelte Leerzeichen, „Str.“/„strasse“, Land am Ende."""
    t = re.sub(r"[\r\n\t]+", " ", term or "")
    t = re.sub(r"(?i)str\.", "straße", t)
    t = re.sub(r"(?i)(?<=\w)strasse\b", "straße", t)
    t = re.sub(r"(?i)[ ,]+(deutschland|germany)\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip(" ,;")


def _umlaut_variant(t):
    """Schreibweisen ohne Umlaute <-> mit Umlauten (Muenchen/München, Strasse/Straße)."""
    if re.search("[äöüßÄÖÜ]", t):
        v = t.translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}))
    else:
        v = re.sub("ae", "ä", t)
        v = re.sub("oe", "ö", v)
        v = re.sub("ue", "ü", v)
        v = re.sub("ss", "ß", v)
    return v if v != t else None


def query_variants(term):
    """Suchvarianten, vom genauen bis zum toleranten Text: mit PLZ, ohne PLZ (kleine PLZ-Abweichungen),
    Straße+Nr+Ort, nur Straße+Ort, Umlaut-Schreibweisen und gekürzte Wörter (Tippfehler am Wortende)."""
    t = normalize_query(term)
    out = [t]
    no_plz = re.sub(r"\s+", " ", _PLZ.sub(" ", t)).strip(" ,")
    no_plz = re.sub(r"\s*,\s*,+", ",", no_plz)
    if no_plz and no_plz != t:
        out.append(no_plz)
    flat = no_plz.replace(",", " ")
    m = re.match(r"^(\D+?)\s+(\d+\s?[a-zA-Z]?)\b[ ,]*(.*)$", flat.strip())
    if m:
        street, nr, city = m.group(1).strip(), m.group(2).replace(" ", ""), m.group(3).strip()
        out += [f"{street} {nr} {city}".strip(), f"{street} {city}".strip(), street]
    for v in list(out[:3]):
        u = _umlaut_variant(v)
        if u:
            out.append(u)
    words = flat.split()
    # verschriebene Straßenendung („Strase“, „Alee“) auf die richtige Endung zurückführen
    import difflib
    fixed = []
    for w_ in words:
        low = w_.lower()
        best = difflib.get_close_matches(low, ["straße", "allee", "weg", "platz", "gasse", "ufer", "damm", "chaussee"], n=1, cutoff=0.78)
        fixed.append(best[0].capitalize() if (best and len(low) >= 4 and low != best[0]) else w_)
    if fixed != words:
        out.insert(1, " ".join(fixed))
    if words:
        long_i = max(range(len(words)), key=lambda i: len(words[i]))
        w = words[long_i]
        for cut in (1, 2):
            if len(w) - cut >= 4:
                out.append(" ".join(words[:long_i] + [w[:-cut]] + words[long_i + 1:]))
    seen, res = set(), []
    for v in out:
        v = v.strip(" ,")
        if len(v) >= 3 and v.lower() not in seen:
            seen.add(v.lower())
            res.append(v)
    return res


def coord_from_text(term):
    """Koordinaten oder Kartenlinks (Google/Apple/OSM: @lat,lon oder lat, lon) direkt als Ort."""
    m = re.search(r"@(-?\d{1,2}\.\d+),(-?\d{1,3}\.\d+)", term) or _COORD.search(term)
    if not m:
        return None
    try:
        la, lo = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
    except ValueError:
        return None
    if -90 <= la <= 90 and -180 <= lo <= 180 and abs(la) > 0.5:
        return la, lo
    return None


def _search_bias():
    """Bezugspunkt der Suche (für Schlagworte wie „Bier“ und für die Umkreissuche): Startpunkt, sonst geteilter
    Standort, sonst ein bereits gewählter Startpunkt eines Reiters, sonst der zuletzt bekannte Punkt."""
    ss = st.session_state
    for getter in (lambda: ss.get("bias"),
                   lambda: ss.get("loc_shared"),
                   lambda: next((_unpack(v)[:2] for k, v in list(ss.items())
                                 if isinstance(k, str) and k.startswith("_keep_") and k.endswith("_a_sb") and v), None),
                   lambda: next((_unpack(v)[:2] for k, v in list(ss.items())
                                 if isinstance(k, str) and k.startswith("_keep_") and v), None),
                   lambda: ss.get("_bias_last")):
        try:
            b = getter()
            if b and b[0] is not None and b[1] is not None and abs(float(b[0])) > 0.5:
                b = (float(b[0]), float(b[1]))
                ss["_bias_last"] = b
                return b
        except Exception:
            continue
    return None


def search_places(term):
    """Ziel-/Zwischenziel-Suche: zuerst im 10-km-Umfeld des Startpunkts."""
    return _search_places(term, False)


def search_start(term):
    """Start-Suche: überall in Deutschland, ohne Umkreisgrenze (der Standort dient nur als sanfter Hinweis)."""
    return _search_places(term, True)


def _search_places(term, free):
    """Vorschläge während der Eingabe.
    - "Bahnhof Ehrenfeld" findet Bahnhöfe/Haltestellen mit "Ehrenfeld" im Namen
      (OSM nennt sie oft "Köln-Ehrenfeld", ohne das Wort Bahnhof).
    - Ohne Treffer wird die Eingabe gekürzt (ohne letztes/erstes Wort), danach Nominatim."""
    term = normalize_query((term or "").strip())
    if len(term) < 3:
        return []
    ll = coord_from_text(term)
    if ll:
        lab = f"📍 Koordinaten · {ll[0]:.5f}, {ll[1]:.5f}"
        return [(lab, _pack(ll[0], ll[1], lab))]
    bias = _search_bias()
    blat, blon = (round(bias[0], 2), round(bias[1], 2)) if bias else (None, None)

    def ask(q, tag=None, box=False):
        try:
            return _photon(q, blat, blon, tag, box and blat is not None)
        except Exception:
            return []

    out, seen = [], set()
    kwhits = []
    kw_ex = ThreadPoolExecutor(max_workers=1)  # Schlagwortsuche (Overpass) läuft neben der Adresssuche
    kw_f = kw_ex.submit(keyword_places, term, bias)
    words = term.split()
    core = [w for w in words if w.lower().strip(".,") not in STATION_WORDS]
    variants = query_variants(term)
    plz_in = _PLZ.search(term)
    addr_like = bool(plz_in) or bool(re.search(r"\d", term))
    # Alle wahrscheinlichen Abfragen gleichzeitig starten (statt eine nach der anderen): die Schleifen unten finden
    # die Antworten dann schon im Zwischenspeicher.
    try:
        first_box = bool(blat is not None and not free)
        jobs = [(q, None, first_box) for q in variants[:4]]
        if core and len(core) < len(words):
            jobs += [(" ".join(core), tag, True) for tag in STATION_TAGS]
        if jobs:
            with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as ex_:
                list(ex_.map(lambda j: ask(*j), jobs))
    except Exception:
        pass
    try:
        kwhits = kw_f.result(timeout=15)
    except Exception:
        kwhits = []
    kw_ex.shutdown(wait=False)
    if core and len(core) < len(words):
        for tag in STATION_TAGS:
            _add(ask(" ".join(core), tag, True), seen, out)
    # 1) im 10-km-Umfeld (Spaziergänge bleiben meist nah), 2) danach ohne Gebietsgrenze
    for boxed in ((True, False) if (blat is not None and not free) else (False,)):
        for k, q in enumerate(variants[:7]):
            _add(ask(q, None, boxed), seen, out)
            if len(out) >= 5 or (k >= 3 and out):
                break
        if len(out) >= 3:
            break
    if not out and len(words) > 1:
        for q in (" ".join(words[:-1]), " ".join(words[1:])):
            _add(ask(q, None, blat is not None and not free), seen, out)
            if out:
                break
    if not out:
        for q in variants[:3]:
            for bb in ((None, None),) if (free or blat is None) else ((blat, blon), (None, None)):
                try:
                    out = _nominatim(q, *bb)
                except Exception:
                    out = []
                if out:
                    break
            if out:
                break
    if bias and out and free:
        # freie Startsuche: Reihenfolge der Relevanz behalten, Entfernung nur anzeigen, wenn sie überschaubar ist
        def dist_f(item):
            la, lo, _ = _unpack(item[1])
            return dist_m((bias[0], bias[1]), (la, lo))
        out = [(f"{lab}  ·  {_fmt_dist(dist_f((lab, val)))}" if dist_f((lab, val)) <= 50000 else lab, val) for lab, val in out]
    elif bias and out:
        def dist_of(item):
            la, lo, _ = _unpack(item[1])
            return dist_m((bias[0], bias[1]), (la, lo))
        if addr_like:
            # eingefügte/genaue Adresse: Treffer mit gleicher bzw. ähnlicher PLZ zuerst, sonst Reihenfolge der Relevanz
            def plz_gap(item):
                m_ = _PLZ.search(item[0])
                return abs(int(m_.group()) - int(plz_in.group())) if (m_ and plz_in) else 99999
            out = [x for _, x in sorted(enumerate(out), key=lambda t_: (plz_gap(t_[1]) // 50, t_[0]))]
        else:
            # nahe Treffer zuerst: aufsteigend nach Entfernung vom Standort bzw. Startpunkt
            out.sort(key=dist_of)
        out = [(f"{lab}  ·  {_fmt_dist(dist_of((lab, val)))}", val) for lab, val in out]
    if kwhits:
        def dist_k(item):
            la, lo, _ = _unpack(item[1])
            return dist_m((bias[0], bias[1]), (la, lo))
        kwhits.sort(key=dist_k)
        kwhits = [(f"{lab}  ·  {_fmt_dist(dist_k((lab, val)))}", val) for lab, val in kwhits][:8]
        return dedupe_results(kwhits + out)[:12]
    return dedupe_results(out)[:8]


def _fmt_dist(m):
    if m < 950:
        return f"{round(m / 10) * 10:.0f} m" if m >= 100 else f"{max(round(m / 5) * 5, 5):.0f} m"
    return f"{m / 1000:.1f} km".replace(".", ",") if m < 20_000 else f"{m / 1000:.0f} km"


# ------------------------------------------------------- Daten
def corridor_bbox(p1, p2):
    (lat1, lon1), (lat2, lon2) = p1, p2
    ky = 111_320.0
    kx = ky * math.cos(math.radians((lat1 + lat2) / 2))
    span = max(abs(lon2 - lon1) * kx, abs(lat2 - lat1) * ky)
    pad = max(600.0, 0.3 * span)  # Platz für Umwege
    bbox = (round(min(lon1, lon2) - pad / kx, 4), round(min(lat1, lat2) - pad / ky, 4),
            round(max(lon1, lon2) + pad / kx, 4), round(max(lat1, lat2) + pad / ky, 4))
    return bbox, span


# Fußwege wie bei osmnx (network_type="walk")
WALK_EXCLUDE = ("abandoned|bus_guideway|construction|cycleway|motor|no|planned|platform|proposed|"
                "raceway|razed|rest_area|services")
WALK_FILTER = ('["highway"]["area"!~"yes"]["access"!~"private|no|customers|delivery|permit"]'
               f'["highway"!~"{WALK_EXCLUDE}"]["foot"!~"no|private"]["service"!~"private|driveway|parking_aisle|drive-through"]'
               '["sidewalk"!~"separate"]["sidewalk:both"!~"separate"]'
               '["sidewalk:left"!~"separate"]["sidewalk:right"!~"separate"]')
_WALK_EXCLUDE_RE = re.compile(WALK_EXCLUDE)


def overpass_query(bbox):
    """Eine einzige Abfrage für Fußwege und Grünflächen (statt zwei getrennter)."""
    w, s, e, n = bbox
    b = f"({s},{w},{n},{e})"
    parts = [f"way{WALK_FILTER}{b};",
             # Hauptverkehrsstraßen immer mitladen (auch wenn ihre Gehwege getrennt erfasst sind): Grundlage für „laut“
             f'way["highway"~"^(trunk|primary|secondary|tertiary)(_link)?$"]{b};']
    for k, vals in GREEN_TAGS.items():
        rx = "|".join(vals)
        parts += [f'way["{k}"~"^({rx})$"]{b};', f'relation["{k}"~"^({rx})$"]{b};']
    # Gewässer: Seen/Teiche (natural=water) und Flüsse/Bäche/Kanäle
    # Hundemodus: Flächen, auf denen Hunde meist verboten sind (Friedhof, Spielplatz, Zoo) oder per dog=no verboten
    parts += [f'way["{k}"="{v}"]{b};' for k, v in DOG_ZONE_TAGS] + [f'relation["{k}"="{v}"]{b};' for k, v in DOG_ZONE_TAGS]
    parts += [f'way["natural"="water"]{b};', f'relation["natural"="water"]{b};',
              f'way["landuse"~"^(reservoir|basin)$"]{b};', f'relation["landuse"~"^(reservoir|basin)$"]{b};',
              f'way["waterway"~"^({"|".join(WATER_WAYS)})$"]{b};']
    return "[out:xml][timeout:25];\n(\n  " + "\n  ".join(parts) + "\n);\n(._;>;);\nout body qt;"


def _osm_complete(path):
    """Prüft, ob die Datei vollständig ist und Overpass keinen Abbruch gemeldet hat."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 4000))
        tail = f.read().decode("utf-8", "ignore")
    return "</osm>" in tail and "runtime error" not in tail


def _cut(resp):
    """Trennt die Verbindung hart (unterbricht auch ein gerade wartendes Lesen)."""
    socks = []
    for path in (("raw", "_connection", "sock"), ("raw", "_fp", "fp", "raw", "_sock")):
        obj = resp
        try:
            for a in path:
                obj = getattr(obj, a)
            socks.append(obj)
        except Exception:
            pass
    for s in socks:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
    try:
        resp.close()
    except Exception:
        pass


def _download_osm(base, query, deadline, tmp, cancel):
    """Lädt die Daten von einem Server in eine Datei. None = fertig, sonst Fehlertext."""
    host = base.split("/")[2]
    try:
        left = deadline - time.time()
        if left < 3:
            return f"{host}: keine Zeit mehr"
        with requests.post(base + "/interpreter", data={"data": query}, headers=UA,
                           timeout=(4, min(25, left)), stream=True) as r:
            if r.status_code != 200:
                return f"{host}: {r.status_code} {r.reason}"
            # Wächter: kappt die Verbindung an der Zeitgrenze, auch wenn Daten nur tröpfeln
            guard = threading.Timer(max(0.5, deadline - time.time()), _cut, args=(r,))
            guard.daemon = True
            guard.start()
            try:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 14):
                        if cancel.is_set():
                            return f"{host}: abgebrochen"
                        f.write(chunk)
            finally:
                guard.cancel()
            if time.time() > deadline:
                return f"{host}: zu langsam"
        if not _osm_complete(tmp):
            return f"{host}: Abfrage abgebrochen"
        return None
    except Exception as e:
        return f"{host}: {type(e).__name__}"


def fetch_osm(bbox, deadline):
    """Lädt OSM-Daten mit harter Zeitgrenze. Zwei Server gleichzeitig: die schnellere Antwort gewinnt, der andere
    Abruf wird abgebrochen. Überlastete Server (429/50x) werden sofort übersprungen. Ergebnis wird einige Tage
    auf der Festplatte zwischengespeichert."""
    from concurrent.futures import FIRST_COMPLETED, wait
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    path = os.path.join(OSM_CACHE_DIR, hashlib.md5((repr(bbox) + "|v5-hund").encode()).hexdigest() + ".osm")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < OSM_CACHE_DAYS * 86400:
        return path
    query, last = overpass_query(bbox), "keine Antwort"
    mirrors = rank_mirrors()
    for rnd in range(2):
        pending = list(mirrors)
        cancel = threading.Event()
        ex = ThreadPoolExecutor(max_workers=3)
        inflight, tmps, n = {}, {}, 0

        def launch():
            nonlocal n
            b = pending.pop(0)
            n += 1
            tmps[b] = f"{path}.{os.getpid()}.{threading.get_ident()}.{rnd}.{n}.part"
            inflight[ex.submit(_download_osm, b, query, deadline, tmps[b], cancel)] = b

        winner = None
        try:
            while len(inflight) < 2 and pending and deadline - time.time() >= 3:
                launch()
            while inflight and not winner:
                done, _ = wait(list(inflight), timeout=max(1.0, deadline - time.time() + 2), return_when=FIRST_COMPLETED)
                if not done:
                    break
                for f in done:
                    b = inflight.pop(f)
                    err = f.result()
                    if err is None and not winner:
                        winner = b
                    elif err:
                        last = err
                # ein Server ist ausgefallen: sofort den nächsten nachrücken lassen
                while not winner and len(inflight) < 2 and pending and deadline - time.time() >= 3:
                    launch()
        finally:
            cancel.set()
            ex.shutdown(wait=False)
        if winner:
            os.replace(tmps[winner], path)
            for b, t in tmps.items():
                if b != winner:
                    threading.Timer(5.0, lambda t=t: os.path.exists(t) and os.remove(t)).start()
            return path
        for t in tmps.values():
            threading.Timer(5.0, lambda t=t: os.path.exists(t) and os.remove(t)).start()
        if deadline - time.time() < 6:
            break
        time.sleep(1)
    raise RuntimeError(f"Alle Kartenserver sind gerade überlastet ({last}).")


def _walkable(el):
    tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
    hw = tags.get("highway")
    if not hw or tags.get("area") == "yes" or _WALK_EXCLUDE_RE.search(hw):
        return False
    # Privatgrundstücke, Hofeinfahrten und Parkplatzgassen sind keine Spazierwege (Durchgänge zwischen Straßen
    # führen so nicht mehr über fremde Grundstücke)
    if any("separate" in str(tags.get(k, "")) for k in ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right")):
        return False  # Straße mit getrennt erfasstem Gehweg: der Gehweg ist der Weg, nicht die Fahrbahn
    return (tags.get("foot") not in ("no", "private")
            and tags.get("access") not in ("private", "no", "customers", "delivery", "permit")
            and tags.get("service") not in ("private", "driveway", "parking_aisle", "drive-through"))


ROAD_CLASSES = {"trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}


def roads_path(path):
    return path[:-4] + ".roads.json"


def walk_xml(path):
    """Schreibt eine zweite Datei nur mit Knoten und Fußwegen (osmnx macht aus jedem Weg eine Kante,
    Parkränder dürfen also nicht mit hinein). Arbeitet gestreamt, damit der Speicher klein bleibt.
    Nebenbei werden die Mittellinien aller Hauptverkehrsstraßen in eine Zusatzdatei geschrieben (auch die,
    deren Gehwege getrennt erfasst sind und die deshalb kein Fußweg sind)."""
    out = path[:-4] + ".walk2.osm"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path) and os.path.exists(roads_path(path)):
        return out
    tmp = out + f".{os.getpid()}.part"
    depth = 0
    nodes, roads = {}, []
    with open(tmp, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="green-walker">\n')
        for ev, el in ET.iterparse(path, events=("start", "end")):
            if ev == "start":
                depth += 1
                continue
            depth -= 1
            if depth != 1:
                continue
            if el.tag == "node":
                try:
                    nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
                except Exception:
                    pass
            elif el.tag == "way":
                hw = next((t.get("v") for t in el.iter("tag") if t.get("k") == "highway"), None)
                if hw in ROAD_CLASSES:
                    pts = [nodes[n.get("ref")] for n in el.iter("nd") if n.get("ref") in nodes]
                    if len(pts) >= 2:
                        roads.append([hw, [[round(x, 6), round(y, 6)] for x, y in pts]])
            if el.tag == "node" or (el.tag == "way" and _walkable(el)):
                el.tail = "\n"
                f.write(ET.tostring(el, encoding="unicode").encode("utf-8"))
            el.clear()
        f.write(b"</osm>\n")
    try:
        import json as _json
        with open(roads_path(path) + ".part", "w", encoding="utf-8") as g:
            _json.dump(roads, g)
        os.replace(roads_path(path) + ".part", roads_path(path))
    except Exception:
        pass
    os.replace(tmp, out)
    return out


ROAD_NOISE_W = {"trunk": 1.0, "primary": 1.0, "secondary": 0.85, "tertiary": 0.5}


def main_road_lines(path, crs):
    """[(Gewicht, LineString im Projektions-KS)] der Hauptverkehrsstraßen aus der Zusatzdatei."""
    import json as _json
    try:
        with open(roads_path(path), encoding="utf-8") as g:
            raw = _json.load(g)
    except Exception:
        return []
    if not raw:
        return []
    geoms = [LineString(pts) for _, pts in raw]
    ws = [ROAD_NOISE_W.get(hw.replace("_link", ""), 0.5) for hw, _ in raw]
    gs = gpd.GeoSeries(geoms, crs=4326).to_crs(crs)
    return list(zip(ws, gs))


def road_noise(lines, roads, dist):
    """Je Kante 0..1: Anteil der Kante (nach Stichproben), der innerhalb von dist m einer Hauptstraße liegt
    (gewichtet nach Straßenklasse)."""
    noise = np.zeros(len(lines), dtype=np.float32)
    if not roads:
        return noise
    groups = {}
    for w, g in roads:
        groups.setdefault(w, []).append(g)
    trees = [(w, STRtree(gs)) for w, gs in groups.items()]
    for f in (np.arange(SAMPLES) + 0.5) / SAMPLES:
        pts = shapely.line_interpolate_point(lines, f, normalized=True)
        best = np.zeros(len(lines), dtype=np.float32)
        for w, tr in trees:
            idx = tr.query_nearest(pts, max_distance=dist, return_distance=False, all_matches=False)
            best[idx[0]] = np.maximum(best[idx[0]], w)
        noise += best
    return noise / SAMPLES


@st.cache_resource(show_spinner=False, max_entries=3, ttl=3600)
def load_area(bbox):
    try:  # osmnx behält sonst nur wenige Wegtags: für Barrierefreiheit zusätzlich Belag, Steigung usw.
        ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | {"surface", "smoothness", "wheelchair", "incline", "dog", "lit"})
    except Exception:
        pass
    _tm0 = time.time()
    path = fetch_osm(bbox, time.time() + LOAD_DEADLINE_S)
    _tm_dl = time.time()
    def _features():
        parks, waters, dogz = None, None, None
        try:
            tags_all = {"leisure": GREEN_TAGS["leisure"] + ["playground", "zoo"],
                        "landuse": GREEN_TAGS["landuse"] + ["reservoir", "basin", "cemetery"],
                        "natural": GREEN_TAGS["natural"] + ["water"], "waterway": WATER_WAYS,
                        "amenity": ["grave_yard"], "tourism": ["zoo"], "dog": ["no"]}
            try:
                gall = ox.features_from_xml(path, tags=tags_all)
            except Exception:
                # Einzelne fehlerhafte Wasser-Geometrien dürfen die Grünflächen nicht mitreißen: getrennt laden
                parts_ = []
                for tg in ({k: v for k, v in tags_all.items() if k != "waterway" and "water" not in v and "reservoir" not in v},
                           {"natural": ["water"], "waterway": WATER_WAYS, "landuse": ["reservoir", "basin"]}):
                    try:
                        parts_.append(ox.features_from_xml(path, tags=tg))
                    except Exception:
                        pass
                if not parts_:
                    raise
                gall = parts_[0] if len(parts_) == 1 else __import__("pandas").concat(parts_)
            isw = np.zeros(len(gall), dtype=bool)
            if "natural" in gall:
                isw |= (gall["natural"] == "water").to_numpy()
            if "landuse" in gall:
                isw |= gall["landuse"].isin(["reservoir", "basin"]).to_numpy()
            if "waterway" in gall:
                isw |= gall["waterway"].notna().to_numpy()
            gdf = gall[~isw]
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

            def _in(df, col, vals):
                return df[col].isin(vals).to_numpy() if col in df else np.zeros(len(df), dtype=bool)
            dz_mask = (_in(gdf, "landuse", ["cemetery"]) | _in(gdf, "amenity", ["grave_yard"]) | _in(gdf, "leisure", ["playground", "zoo"])
                       | _in(gdf, "tourism", ["zoo"]) | _in(gdf, "dog", ["no"]))
            park_mask = (_in(gdf, "leisure", GREEN_TAGS["leisure"]) | _in(gdf, "landuse", GREEN_TAGS["landuse"])
                         | _in(gdf, "natural", GREEN_TAGS["natural"]))
            try:
                dzf = gdf[dz_mask]
                if len(dzf):
                    lab = np.array(["Hundeverbot"] * len(dzf), dtype=object)
                    for col in ("landuse", "amenity", "leisure", "tourism"):
                        if col in dzf:
                            for val, name in DOG_ZONE_LABEL.items():
                                lab[(dzf[col] == val).to_numpy()] = name
                    dzf = dzf[["geometry"]].reset_index(drop=True)
                    dzf["k"] = list(lab)
                    dogz = dzf
            except Exception:
                dogz = None
            gdf = gdf[park_mask]
            if len(gdf):
                gdf = gdf.reset_index(drop=True)
                if "name" not in gdf:
                    gdf["name"] = None
                gdf["name"] = gdf["name"].fillna("Grünfläche").astype(str)
                parks = gdf[["name", "geometry"]]
            wdf = gall[isw]
            if len(wdf):
                waters = wdf[["geometry"]].reset_index(drop=True)
        except Exception:
            parks = None  # keine Grünflächen im Gebiet: Route geht trotzdem
        return parks, waters, dogz

    _fex = ThreadPoolExecutor(max_workers=1)
    _ffut = _fex.submit(_features)  # Grünflächen und Gewässer laufen parallel zum Wegenetz
    try:
        G0 = ox.graph_from_xml(walk_xml(path), bidirectional=True, simplify=True, retain_all=False)
    except Exception as e:
        raise ValueError(f"In diesem Gebiet habe ich keine Fußwege gefunden ({type(e).__name__}).")
    if G0.number_of_edges() == 0:
        raise ValueError("In diesem Gebiet habe ich keine Fußwege gefunden.")
    try:
        parks, waters, dogz = _ffut.result()
    except Exception:
        parks, waters, dogz = None, None, None
    _fex.shutdown(wait=False)


    G = ox.project_graph(G0)
    crs = G.graph["crs"]
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    best = {}
    for u, v, d in G.edges(data=True):
        if (u, v) not in best or d["length"] < best[(u, v)][0]:
            hw = d.get("highway")
            hw = (hw[0] if hw else "") if isinstance(hw, (list, tuple)) else (hw or "")
            best[(u, v)] = (d["length"], d.get("geometry") or LineString([pos[u], pos[v]]), *edge_label(d), hw,
                            edge_wide(d, hw), edge_access_pen(d, hw), *edge_surface(d, hw), edge_dogtag(d), *edge_lit(d, hw))
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

    # Hauptstraßen-Nähe je Kante (0 = ruhig, 1 = an/auf einer Hauptstraße). Grundlage sind die Mittellinien der
    # Hauptverkehrsstraßen aus den Kartendaten, nicht nur die Fußwege: Gehwege laufen oft getrennt neben der Fahrbahn.
    def _calc_noise():
        noise = np.zeros(len(keys), dtype=np.float32)
        roads_ok = False
        try:
            roads = main_road_lines(path, crs)
            if roads:
                noise = road_noise(lines, roads, NOISE_DIST)
                roads_ok = True
        except Exception:
            roads_ok = False
        if not roads_ok:  # Zusatzdatei fehlt: wie früher aus den Straßen im Wegenetz
            wts = np.array([OSM_ROAD_W.get(best[k][4], 0.0) for k in keys], dtype=np.float32)
            main = np.nonzero(wts > 0)[0]
            if len(main):
                rtree = STRtree(list(lines[main]))
                acc = np.zeros(len(keys), dtype=np.float32)
                for f in (np.arange(SAMPLES) + 0.5) / SAMPLES:
                    pts = shapely.line_interpolate_point(lines, f, normalized=True)
                    idx, _ = rtree.query_nearest(pts, max_distance=NOISE_DIST, return_distance=True, all_matches=False)
                    acc[idx[0]] += wts[main][idx[1]]
                noise = acc / SAMPLES
        return noise

    def _calc_D():
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
        return D, parks_json, n_parks

    def _calc_DW():
        DW = np.full((SAMPLES, len(keys)), MAX_BUF, dtype=np.float32)
        water_json = None
        if waters is not None:
            try:
                wg = waters.to_crs(crs)
                wg["geometry"] = wg.geometry.simplify(2.0)
                wg = wg[wg.geometry.notna() & ~wg.geometry.is_empty].reset_index(drop=True)
                if len(wg):
                    wtree = STRtree(list(wg.geometry))
                    for r, f in enumerate((np.arange(SAMPLES) + 0.5) / SAMPLES):
                        pts = shapely.line_interpolate_point(lines, f, normalized=True)
                        idx, dist = wtree.query_nearest(pts, max_distance=MAX_BUF, return_distance=True, all_matches=False)
                        DW[r, idx[0]] = dist
                    wshow = wg.copy()
                    wshow["geometry"] = wshow.geometry.simplify(6.0)
                    water_json = wshow.iloc[:900].to_crs(4326).to_json()
            except Exception:
                water_json = None
        return DW, water_json

    def _calc_dog():
        """Kanten in Hunde-Verbotszonen (Friedhof, Spielplatz, Zoo, dog=no) + Zonen für die Karte."""
        mask = np.zeros(len(keys), dtype=bool)
        zjson, nz_ = None, 0
        if dogz is not None and len(dogz):
            try:
                zg = dogz.to_crs(crs)
                zg["geometry"] = zg.geometry.make_valid()
                zg = zg[zg.geometry.notna() & ~zg.geometry.is_empty].reset_index(drop=True)
                nz_ = len(zg)
                if nz_:
                    ztree = STRtree(list(zg.geometry))
                    cnt = np.zeros(len(keys), dtype=np.int16)
                    for f in (0.2, 0.5, 0.8):
                        pts = shapely.line_interpolate_point(lines, f, normalized=True)
                        hit = ztree.query(pts, predicate="intersects")
                        cnt += np.bincount(hit[0], minlength=len(keys)).clip(0, 1).astype(np.int16)
                    mask = cnt >= 2  # mindestens zwei von drei Stichpunkten liegen in der Zone
                    zs = zg.copy()
                    zs["geometry"] = zs.geometry.simplify(3.0)
                    zjson = zs[["k", "geometry"]].iloc[:500].to_crs(4326).to_json()
            except Exception:
                mask, zjson, nz_ = np.zeros(len(keys), dtype=bool), None, 0
        return mask, zjson, nz_

    # Hauptstraßen-Nähe, Park-Abstand, Wasser-Abstand und Hundezonen sind voneinander unabhängig: parallel rechnen
    with ThreadPoolExecutor(max_workers=4) as _ex:
        _fn, _fd, _fw, _fz = _ex.submit(_calc_noise), _ex.submit(_calc_D), _ex.submit(_calc_DW), _ex.submit(_calc_dog)
        noise = _fn.result()
        D, parks_json, n_parks = _fd.result()
        DW, water_json = _fw.result()
        zone_mask, dogz_json, n_dogz = _fz.result()
    dogno = zone_mask | (dogtag == 1)
    # Belag: ohne Eintrag in den Kartendaten innerhalb von Parks eher Kies/Erde als Asphalt (Schätzung)
    inside_ = (D <= 0.5).mean(axis=0) >= 0.7
    surf_est = np.where(~surf_known & inside_ & (surf == 0), 2, surf).astype(np.int8)
    hard = surf_est < 2

    wide = np.array([best[k][5] for k in keys], dtype=np.float32)
    accp = np.array([best[k][6] for k in keys], dtype=np.float32)
    surf = np.array([best[k][7] for k in keys], dtype=np.int8)
    surf_known = np.array([best[k][8] for k in keys], dtype=bool)
    dogtag = np.array([best[k][9] for k in keys], dtype=np.int8)
    lit = np.array([best[k][10] for k in keys], dtype=np.int8)
    litpen = np.array([best[k][11] for k in keys], dtype=np.float32)


    # Gewässer-Nähe je Kante (Abstand zum nächsten See, Teich oder Fluss)

    node_list = list(pos)
    nidx = {n: i for i, n in enumerate(node_list)}
    return {"bbox": bbox, "crs": crs, "pos": pos, "H": H, "lengths": lengths, "lines": lines, "D": D, "DW": DW,
            "water_json": water_json,
            "node_list": node_list, "xy": np.array([pos[n] for n in node_list]),
            "eu": np.array([nidx[u] for u, v in keys]), "ev": np.array([nidx[v] for u, v in keys]),
            "names": names, "kinds": kinds, "noise": noise, "wide": wide, "accp": accp,
            "surf": surf_est, "surf_known": surf_known, "hard": hard, "dogno": dogno, "dogtag": dogtag,
            "lit": lit, "litpen": litpen,
            "dogz_json": dogz_json, "n_dogz": n_dogz,
            "parks_json": parks_json, "n_parks": n_parks, "n_edges": len(keys),
            "tm": (round(_tm_dl - _tm0, 1), round(time.time() - _tm_dl, 1))}


def _tm_txt(area):
    """Zeitaufteilung beim Laden eines Gebiets (Download / Aufbereiten) für die Ladeanzeige."""
    try:
        d, v = area["tm"]
        return f": Download {d:.0f} s, Aufbereiten {v:.0f} s"
    except Exception:
        return ""


def nearest_node(area, lat, lon):
    x, y = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(area["crs"]).iloc[0].coords[0]
    d = (area["xy"][:, 0] - x) ** 2 + (area["xy"][:, 1] - y) ** 2
    return area["node_list"][int(d.argmin())]


# ------------------------------------------------------- Wassernähe
def _water():
    """Ist die Option „Wassernähe bevorzugen“ aktiv? (wird beim Berechnen aus dem Formular übernommen)"""
    try:
        return bool(st.session_state.get("_water", False))
    except Exception:
        return False


def _access():
    """Ist „Barrierefreiheit“ aktiv? (Treppen, steile Stücke und schlechte Beläge werden gemieden)"""
    try:
        return bool(st.session_state.get("_access", False))
    except Exception:
        return False


_UNPAVED = {"sand", "mud", "dirt", "ground", "grass", "gravel", "fine_gravel", "pebblestone", "unpaved", "earth",
            "woodchips", "compacted", "grass_paver", "rock", "snow", "ice", "clay"}
_BUMPY = {"cobblestone", "sett", "unhewn_cobblestone", "cobblestone:flattened"}
_BAD_SMOOTH = {"bad", "very_bad", "horrible", "very_horrible"}


def edge_access_pen(d, hw):
    """Zuschlag (auf die Wegkosten) für Rollstuhl/Rollator/Kinderwagen: Treppen und „wheelchair=no“ praktisch gesperrt,
    schlechter Belag und starke Steigung teurer. Nur was in OpenStreetMap erfasst ist, kann berücksichtigt werden."""
    def one(k):
        v = d.get(k)
        return (v[0] if isinstance(v, (list, tuple)) and v else v) or ""
    if hw == "steps" or one("wheelchair") == "no":
        return 60.0
    pen = 0.0
    surf, smooth = str(one("surface")), str(one("smoothness"))
    if smooth == "impassable":
        return 60.0
    if surf in _UNPAVED:
        pen += 1.5
    elif surf in _BUMPY:
        pen += 0.8
    elif not surf and hw in ("path", "track", "bridleway"):
        pen += 0.6
    if smooth in _BAD_SMOOTH:
        pen += 1.5
    m = re.match(r"^\s*-?(\d+(?:[.,]\d+)?)\s*%", str(one("incline")))
    if m:
        pct = float(m.group(1).replace(",", "."))
        pen += 2.0 if pct >= 8 else (0.8 if pct >= 6 else 0.0)
    return min(pen, 60.0)


# ------------------------------------------------------- Hundemodus
HOT_C = 30.0           # ab dieser Außentemperatur: Pfotenwarnung auf Asphalt und Pflaster
DOG_NO_PEN = 40.0      # Kostenzuschlag für Wege in Hunde-Verbotszonen (praktisch gesperrt)
_S_ASPHALT = {"asphalt", "concrete", "paved", "chipseal", "bitumen", "metal", "tartan", "rubber", "concrete:lanes"}
_S_STONE = {"paving_stones", "sett", "cobblestone", "unhewn_cobblestone", "cobblestone:flattened", "concrete:plates",
            "bricks", "brick", "stone", "pebblestone:paved", "cement", "tiles"}
_S_SOFT = _UNPAVED | {"wood", "gravel;ground", "dirt/sand", "salt", "grass_paver", "fine_gravel", "compacted", "gravel"}
SURF_NAMES = ["Asphalt", "Pflaster", "Erde/Kies/Gras"]


def _tag1(d, k):
    v = d.get(k)
    return str((v[0] if isinstance(v, (list, tuple)) and v else v) or "").strip().lower()


def edge_surface(d, hw):
    """(Belagsklasse, bekannt) einer Kante: 0 Asphalt/Beton, 1 Pflaster/Platten, 2 Erde/Kies/Gras.
    Steht kein Belag in den Kartendaten, wird er aus der Wegart geschätzt (bekannt = False)."""
    sf = _tag1(d, "surface").split(";")[0]
    if sf:
        if sf in _S_SOFT:
            return 2, True
        if sf in _S_ASPHALT:
            return 0, True
        if sf in _S_STONE or sf.startswith("paving_stones"):
            return 1, True
    if hw in ("path", "track", "bridleway"):
        return 2, False
    if hw in ("pedestrian", "steps"):
        return 1, False
    return 0, False


def edge_dogtag(d):
    """0 = nichts bekannt / erlaubt, 1 = Hunde verboten (dog=no), 2 = Leinenpflicht (dog=leashed)."""
    v = _tag1(d, "dog")
    if v in ("no", "prohibited", "forbidden"):
        return 1
    if v in ("leashed", "on_leash", "lead"):
        return 2
    return 0


LIT_YES = {"yes", "24/7", "automatic", "sunset-sunrise", "dusk-dawn", "interval", "limited", "operating_hours"}
LIT_PEN_NO, LIT_PEN_UNK_PATH, LIT_PEN_UNK_ROAD = 0.6, 0.3, 0.08


def edge_lit(d, hw):
    """(Code, Zuschlag) für „Beleuchtete Wege“: Code 1 = beleuchtet (lit=yes u. ä.), 2 = unbeleuchtet (lit=no),
    0 = nicht erfasst. Nicht erfasste Wege zählen vorsichtig als eher dunkel: Fuß- und Waldwege stärker als Straßen."""
    v = _tag1(d, "lit")
    if v in LIT_YES:
        return 1, 0.0
    if v in ("no", "disused"):
        return 2, LIT_PEN_NO
    return 0, (LIT_PEN_UNK_PATH if hw in ("footway", "path", "track", "bridleway", "steps", "pedestrian", "cycleway")
               else LIT_PEN_UNK_ROAD)


def _lit():
    """Ist „Beleuchtete Wege“ aktiv? (bevorzugt Wege, für die lit=yes in OpenStreetMap steht)"""
    try:
        return bool(st.session_state.get("_lit", False))
    except Exception:
        return False


def _dog():
    try:
        return bool(st.session_state.get("_dog", False))
    except Exception:
        return False


def _hot():
    """Ist es heiß genug für die Pfotenwarnung? (Temperatur wird beim Berechnen aus dem Wetter übernommen.)"""
    try:
        t = st.session_state.get("_temp")
        return t is not None and float(t) >= HOT_C
    except Exception:
        return False


def area_accp(area):
    """Kostenzuschlag je Kante: Barrierefreiheit (wenn aktiv) und Hundemodus (Verbotszonen praktisch gesperrt,
    harter Belag etwas teurer, bei Hitze deutlich)."""
    n = len(area["lengths"])
    acc, dog, hot, lit = _access(), _dog(), _hot(), _lit()
    cache = area.setdefault("_accp_c", {})
    key = (acc, dog, hot, lit)
    if key in cache:
        return cache[key]
    a = area.get("accp")
    pen = np.array(a, dtype=np.float32) if (a is not None and acc) else np.zeros(n, dtype=np.float32)
    if dog:
        dn = area.get("dogno")
        if dn is not None:
            pen = pen + DOG_NO_PEN * np.asarray(dn, dtype=np.float32)
        hard = area.get("hard")
        if hard is not None:
            pen = pen + (0.45 if hot else 0.06) * np.asarray(hard, dtype=np.float32)
    if lit:
        lp = area.get("litpen")
        if lp is not None:
            pen = pen + np.asarray(lp, dtype=np.float32)
    cache[key] = pen
    return pen

# ---- Wetter (Open-Meteo, ohne Schlüssel): Temperatur für die Pfotenwarnung, Wetter für den Ladehund
_WX_CACHE, _WX_PEND, _WX_LOCK = {}, {}, threading.Lock()


def _wx_fetch(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", headers=UA, timeout=4,
                     params={"latitude": round(lat, 3), "longitude": round(lon, 3), "timezone": "auto",
                             "current": "temperature_2m,weather_code,is_day,wind_speed_10m,wind_gusts_10m"})
    c = r.json()["current"]
    return {"temp": float(c["temperature_2m"]), "code": int(c["weather_code"]), "day": bool(c.get("is_day", 1)),
            "wind": float(c.get("wind_speed_10m") or 0.0), "gust": float(c.get("wind_gusts_10m") or 0.0)}


WIND_BREEZY = 30.0   # km/h: spürbar windig
WIND_STORM = 60.0    # km/h Böen: Sturmböen, Astbruchgefahr


def wind_note(w):
    """Hinweistext zum Wind (leer, wenn es ruhig ist)."""
    if not w:
        return ""
    wind, gust = float(w.get("wind") or 0), float(w.get("gust") or 0)
    if gust >= WIND_STORM:
        return (f"💨 Sturmböen bis {gust:.0f} km/h: Wald und Parks mit alten Bäumen heute lieber meiden (Astbruch). "
                "Hund gut sichern, am besten kurz an der Leine bleiben.")
    if wind >= WIND_BREEZY or gust >= 45:
        return (f"💨 Es ist windig ({wind:.0f} km/h, Böen bis {gust:.0f} km/h). Unter großen Bäumen aufpassen, "
                "und der Hund kann durch aufgewirbelte Dinge schreckhaft sein.")
    return ""


def wx_kind(code, temp, day=True):
    """Grobe Wetterlage aus dem WMO-Code: sun, rain, snow, cloud."""
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if 51 <= code <= 67 or 80 <= code <= 82 or code >= 95:
        return "rain"
    if code in (0, 1) and day:
        return "sun"
    return "cloud"


def get_weather(lat, lon, wait=0.0):
    """Aktuelles Wetter am Ort oder None. Wird 10 Minuten gemerkt; der Abruf läuft im Hintergrund und kann
    abgewartet werden (wait Sekunden). Zum Testen: ?temp=33&wx=sun|rain|snow|cloud in der Adresszeile."""
    ov = {}
    try:
        qp = st.query_params
        if qp.get("temp") or qp.get("wx"):
            codes = {"sun": 0, "cloud": 3, "rain": 63, "snow": 73}
            wd = float(str(qp.get("wind", 10)).replace(",", "."))
            ov = {"temp": float(str(qp.get("temp", 20)).replace(",", ".")), "code": codes.get(str(qp.get("wx", "sun")), 0), "day": True,
                  "wind": wd, "gust": wd * 1.4}
    except Exception:
        ov = {}
    if ov:
        return ov
    key = (round(lat, 1), round(lon, 1))
    now = time.time()
    with _WX_LOCK:
        hit = _WX_CACHE.get(key)
        if hit and now - hit[0] < 600:
            return hit[1]
        th = _WX_PEND.get(key)
        if th is None or not th.is_alive():
            def work(key=key, lat=lat, lon=lon):
                try:
                    w = _wx_fetch(lat, lon)
                    with _WX_LOCK:
                        _WX_CACHE[key] = (time.time(), w)
                except Exception:
                    with _WX_LOCK:
                        _WX_CACHE.setdefault(key, (time.time() - 540, None))  # Fehlschlag: nach ~1 Minute neu versuchen
            th = threading.Thread(target=work, daemon=True)
            _WX_PEND[key] = th
            th.start()
    if wait > 0:
        th.join(wait)
    with _WX_LOCK:
        hit = _WX_CACHE.get(key)
    return hit[1] if hit else None


def dog_prefetch(start_v):
    """Wetterabruf schon anstoßen, bevor die Ladeansicht erscheint (läuft im Hintergrund)."""
    try:
        lat, lon, _ = _unpack(start_v)
        get_weather(lat, lon, wait=0.0)
    except Exception:
        pass


def dog_weather(start_v):
    """Wetterabruf für den Startpunkt anstoßen (läuft im Hintergrund, während Wegenetz und Route berechnet werden).
    Abgewartet wird erst, wenn die Ergebnisse gebraucht werden (_resolve_weather), dann ist er meist längst fertig."""
    try:
        lat, lon, _ = _unpack(start_v)
        get_weather(lat, lon, wait=0.0)
        st.session_state["_wx_pending"] = (lat, lon)
        st.session_state["_temp"] = None
        st.session_state["_windnote"] = ""
    except Exception:
        st.session_state["_wx_pending"] = None
        st.session_state["_temp"] = None
        st.session_state["_windnote"] = ""


def _resolve_weather():
    """Temperatur (Pfotenwarnung, nur Hundemodus) und Windhinweis übernehmen; wartet nur noch kurz, falls nötig."""
    try:
        pend = st.session_state.get("_wx_pending")
        if not pend:
            return
        st.session_state["_wx_pending"] = None
        w = get_weather(pend[0], pend[1], wait=3.5 if _dog() else 1.5)
        st.session_state["_temp"] = None if not w else w["temp"]
        st.session_state["_windnote"] = wind_note(w)
    except Exception:
        pass


def _seg_runs(area, path, mask, limit=80):
    """Zusammenhängende Wegstücke (mask je Kante des Pfades) als Linien in Breite/Länge für die Karte."""
    n = len(mask)
    if not n or not mask.any():
        return []
    coords, node_idx = route_geom(area, path)
    runs, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            seg = coords[node_idx[i]:node_idx[j + 1] + 1]
            if len(seg) >= 2:
                runs.append(seg)
            i = j + 1
        else:
            i += 1
    runs = runs[:limit]
    if not runs:
        return []
    ll = gpd.GeoSeries([LineString(r) for r in runs], crs=area["crs"]).to_crs(4326)
    return [[[round(y, 5), round(x, 5)] for x, y in g.coords] for g in ll]


SURF_COL = ["#6B6B6B", "#9A7B5B", "#7C9F4A"]


def dog_info(area, path, temp=None):
    """Hundemodus-Angaben zu einer Route: Belag, Verbotszonen, Leinenpflicht, Pfotenwarnung (+ Karten-Linien)."""
    H, L = area["H"], area["lengths"]
    idx = np.array([H[a][b]["i"] for a, b in zip(path[:-1], path[1:])], dtype=int)
    if not len(idx) or area.get("surf") is None:
        return None
    li = L[idx]
    tot = float(li.sum())
    if tot <= 0:
        return None
    surf = area["surf"][idx]
    known = area["surf_known"][idx]
    share = [float(li[surf == k].sum() / tot) for k in range(3)]
    est = float(li[~known].sum() / tot)
    forb = np.asarray(area["dogno"])[idx]
    leash = np.asarray(area["dogtag"])[idx] == 2
    hard = surf < 2
    hot = temp is not None and temp >= HOT_C
    hard_m = float(li[hard].sum())
    info = {"surf": [[SURF_NAMES[k], round(share[k] * 100), SURF_COL[k]] for k in range(3) if share[k] >= 0.01],
            "est": round(est * 100), "forb": round(float(li[forb].sum())), "leash": round(float(li[leash].sum())),
            "temp": None if temp is None else round(temp), "paw": "", "wind": st.session_state.get("_windnote", "") or ""}
    if hot:
        if hard_m / tot >= 0.05:
            info["paw"] = (f"🐾 Pfotenwarnung bei {temp:.0f} °C: {hard_m / tot:.0%} der Strecke ({hard_m:.0f} m) sind Asphalt oder Pflaster "
                           "(rot gepunktet). Der Boden wird viel heißer als die Luft und kann die Pfoten verbrennen. "
                           "Handrücken-Test: Hältst du ihn keine 5 Sekunden auf den Boden, ist es für Pfoten zu heiß. "
                           "Lieber früh morgens oder abends gehen.")
        else:
            info["paw"] = (f"🐾 Es hat {temp:.0f} °C. Diese Strecke führt fast nur über weichen Boden, das ist gut für die Pfoten. "
                           "Nimm trotzdem Wasser mit und plane Schattenpausen ein.")
    segs = {}
    try:
        if hot:
            segs["hot"] = _seg_runs(area, path, hard)
        if forb.any():
            segs["forbid"] = _seg_runs(area, path, forb)
    except Exception:
        pass
    return info, segs



def edge_water(area):
    """Wasser-Score je Kante: 1 direkt am See/Fluss, linear fallend auf 0 bei WATER_BUF Metern."""
    if "_w" not in area:
        dw = area.get("DW")
        area["_w"] = np.clip(1 - dw / WATER_BUF, 0, 1).mean(axis=0) if dw is not None else None
    return area["_w"]


def edge_green(area, buf, water=None):
    """Attraktivität je Kante: Grünscore; bei aktiver Wassernähe zählt Wasser genauso (das Bessere gilt)."""
    D = area["D"]
    # im Park (Abstand 0) zählt voll, am Parkrand entlang etwas weniger: Wege führen durch Parks statt außen herum
    g = np.where(D <= 0.5, 1.0, INSIDE_FACTOR * np.clip(1 - D / buf, 0, 1) ** GREEN_FALLOFF).mean(axis=0)
    if _water() if water is None else water:
        w = edge_water(area)
        if w is not None:
            # Ufer vor Parkmitte: Wege am Wasser gelten als bestes „Grün“, Park ohne Wasser etwas weniger
            g = np.maximum(0.9 * g, w)
    return g


def water_share(area, path):
    w = edge_water(area)
    if w is None or len(path) < 2:
        return 0.0
    idx = [area["H"][a][b]["i"] for a, b in zip(path[:-1], path[1:])]
    L = area["lengths"][idx]
    return float((L * (w[idx] > WATER_NEAR)).sum() / max(L.sum(), 1.0))


def water_note(area, path):
    """Hinweistext: Anteil am Wasser, bei 0 % mit Begründung (ehrlich statt nur eine Zahl)."""
    share = water_share(area, path)
    if share >= 0.005:
        return f" · {share:.0%} am Wasser"
    dw = area.get("DW")
    if dw is None or float(dw.min()) >= MAX_BUF - 1:
        return " · kein Gewässer im Gebiet gefunden"
    idx = [area["H"][a][b]["i"] for a, b in zip(path[:-1], path[1:])]
    near = float(dw[:, idx].min()) if idx else MAX_BUF
    return f" · 0 % am Wasser (nächstes Ufer ca. {near:.0f} m entfernt)" if near < MAX_BUF - 1 else \
        " · 0 % am Wasser (Ufer weiter als 400 m)"


_PF_SEEN = set()


def prefetch_osm(bbox):
    """Größeres Gebiet im Hintergrund auf die Festplatte holen (nur Download, kein Aufbau). Tippt jemand in der
    Karte auf „+“, liegen die Daten dann schon bereit."""
    if bbox in _PF_SEEN:
        return
    _PF_SEEN.add(bbox)
    if len(_PF_SEEN) > 50:
        _PF_SEEN.clear()

    def work():
        try:
            time.sleep(12)  # erst die Route des Nutzers fertig werden lassen (Toilettenabfrage u. a. nutzt dieselben Server)
            fetch_osm(bbox, time.time() + 40)
        except Exception:
            _PF_SEEN.discard(bbox)
    threading.Thread(target=work, daemon=True).start()


_WG_CACHE = {}


def scan_water_geo(bbox):
    """Nur die Gewässer (Seen, Flüsse) aus den Vektorkacheln als GeoJSON. Deutlich schneller als der volle Grün-Scan,
    denn es werden nur die Ebenen water und waterway gelesen und nichts in Raster umgerechnet."""
    hit = _WG_CACHE.get(bbox)
    if hit is not None:
        return hit
    w, s, e, n = bbox
    tpl = _ofm_template()
    x0, y0 = _tile_of(n, w, VT_ZOOM)
    x1, y1 = _tile_of(s, e, VT_ZOOM)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    if len(tiles) > 64:
        raise ValueError("Gebiet zu groß")
    with ThreadPoolExecutor(max_workers=8) as ex:
        datas = list(ex.map(lambda t: _fetch_tile(tpl, VT_ZOOM, t[0], t[1]), tiles))
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    ky = 111_320.0
    kx = ky * math.cos(math.radians(lat0))
    wgeo = []
    for (tx, ty), data in zip(tiles, datas):
        if not data:
            continue
        for lname, feats in mvt_decode(data, {"water", "waterway"}).items():
            for extent, gtype, parts, props in feats:
                if len(wgeo) >= 600:
                    break
                if gtype == 3 and lname == "water" and props.get("class") != "ocean":
                    for ring in parts:
                        if len(ring) < 4:
                            continue
                        lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, ring)
                        X, Y = (lon - lon0) * kx, (lat - lat0) * ky
                        if 0.5 * abs(np.dot(X[:-1], Y[1:]) - np.dot(X[1:], Y[:-1])) >= 400:
                            step = max(1, len(lon) // 140)
                            ring_ll = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon[::step], lat[::step])]
                            ring_ll.append(ring_ll[0])
                            wgeo.append({"type": "Feature", "properties": {},
                                         "geometry": {"type": "Polygon", "coordinates": [ring_ll]}})
                elif gtype == 2 and lname == "waterway":
                    for part in parts:
                        if len(part) < 2:
                            continue
                        lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, part)
                        wgeo.append({"type": "Feature", "properties": {}, "geometry": {
                            "type": "LineString", "coordinates": [[round(float(a), 5), round(float(b), 5)]
                                                                  for a, b in zip(lon[::2], lat[::2])]}})
    out = {"type": "FeatureCollection", "features": wgeo}
    if len(_WG_CACHE) > 8:
        _WG_CACHE.clear()
    _WG_CACHE[bbox] = out
    return out


def prefetch_water(bbox):
    """Wassernähe aktiv: Kacheldaten schon während des Wegenetz-Ladens holen (landen im Cache von scan_area)."""
    try:
        if not _water():
            return None
    except Exception:
        return None

    def work():
        try:
            scan_water_geo(tuple(bbox))
        except Exception:
            pass
    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t


def ensure_water(area, log=None):
    """Zweite, unabhängige Gewässerquelle: Seen und Flüsse aus den Karten-Vektorkacheln. Ergänzt die OSM-Daten,
    falls dort ein großer See (z. B. als Multipolygon) fehlt oder unvollständig ankam. Nur bei aktiver Wassernähe."""
    if not _water() or area.get("_wt") or not area.get("bbox"):
        return
    area["_wt"] = True
    try:
        if log:
            log("💧 Gewässer prüfen und ergänzen")
        feats = scan_water_geo(tuple(area["bbox"])).get("features", [])
        if not feats:
            return
        gs = gpd.GeoDataFrame(geometry=[shapely.geometry.shape(f["geometry"]) for f in feats], crs=4326).to_crs(area["crs"])
        gs = gs[gs.geometry.notna() & ~gs.geometry.is_empty].reset_index(drop=True)
        if not len(gs):
            return
        tree = STRtree(list(gs.geometry))
        DW = area["DW"]
        for r, f in enumerate((np.arange(SAMPLES) + 0.5) / SAMPLES):
            pts = shapely.line_interpolate_point(area["lines"], f, normalized=True)
            idx, dist = tree.query_nearest(pts, max_distance=MAX_BUF, return_distance=True, all_matches=False)
            DW[r, idx[0]] = np.minimum(DW[r, idx[0]], dist)
        area.pop("_w", None)
        base = water_geo(area) or {"type": "FeatureCollection", "features": []}
        base["features"] = (base["features"] + feats)[:1200]
        area["_wgeo"] = base
    except Exception:
        pass


_AREA_REG = []   # geladene Gebiete: (bbox, area), die letzten drei
_AREA_PEND = []  # Gebiete, die gerade im Hintergrund geladen werden: (bbox, Event)
_AREA_LOCK = threading.Lock()


def _bbox_covers(old, new, shrink=0.15):
    """Deckt das alte Gebiet das neue ab (Rand um 15 % verkleinert, damit kleine Verschiebungen des Startpunkts
    nicht zählen)? Und ist es nicht viel größer (sonst wird das Rechnen unnötig langsam)?"""
    w, s_, e, n = new
    dx, dy = (e - w) * shrink / 2, (n - s_) * shrink / 2
    ok = old[0] <= w + dx and old[1] <= s_ + dy and old[2] >= e - dx and old[3] >= n - dy
    big = (old[2] - old[0]) * (old[3] - old[1]) <= 6.0 * max((e - w) * (n - s_), 1e-9)
    return ok and big


def _area_lookup(bbox):
    with _AREA_LOCK:
        for bb, ar in reversed(_AREA_REG):
            if _bbox_covers(bb, bbox):
                return ar
    return None


def load_area_any(bbox, own=None, load_bbox=None):
    """Wie load_area, nutzt aber ein schon geladenes (oder gerade ladendes) Gebiet, das den Bedarf (bbox) abdeckt. Wer in
    der Nähe schon einmal gerechnet hat, wartet so nicht erneut auf die Kartenserver. Muss nachgeladen werden, wird
    load_bbox (etwas größer, in Stufen) geladen, damit Änderungen wie die Extra-Strecke kein neues Gebiet brauchen."""
    ar = _area_lookup(bbox)
    if ar is not None:
        return ar
    with _AREA_LOCK:
        pend = [ev for bb, ev in _AREA_PEND if ev is not own and _bbox_covers(bb, bbox)]
    for ev in pend[:1]:
        ev.wait(40)  # läuft schon im Hintergrund: abwarten statt doppelt zu laden
        ar = _area_lookup(bbox)
        if ar is not None:
            return ar
    ar = load_area(load_bbox or bbox)
    with _AREA_LOCK:
        _AREA_REG.append((tuple(ar.get("bbox") or load_bbox or bbox), ar))
        del _AREA_REG[:-3]
    return ar


_PREWARMED = set()


def _prewarm_bbox(bb, key, need=None):
    """Gebiet bb im Hintergrund laden (samt Gewässerdaten), falls es nicht schon da ist oder gerade geladen wird."""
    if key in _PREWARMED:
        return False
    _PREWARMED.add(key)
    if len(_PREWARMED) > 100:
        _PREWARMED.clear()
    need = need or bb
    if _area_lookup(need) is not None:
        return False
    ev = threading.Event()
    with _AREA_LOCK:
        if any(_bbox_covers(pb, need) for pb, _e in _AREA_PEND):
            return False
        _AREA_PEND.append((bb, ev))

    def work():
        try:
            ar_ = load_area_any(need, own=ev, load_bbox=bb)
            try:
                scan_water_geo(tuple(ar_["bbox"]))  # Gewässerdaten schon bereit, falls „Wassernähe“ gewählt wird
            except Exception:
                pass
        except Exception:
            _PREWARMED.discard(key)
        finally:
            with _AREA_LOCK:
                try:
                    _AREA_PEND.remove((bb, ev))
                except ValueError:
                    pass
            ev.set()
    threading.Thread(target=work, daemon=True).start()
    return True


def prewarm(lat, lon):
    """Sobald der Standort bekannt ist, im Hintergrund Wegenetz (für ein Ründchen von ca. 3 km und Routen in der Nähe)
    und Grünflächen holen. Die erste Berechnung findet die Daten dann schon vor."""
    key = (round(lat, 2), round(lon, 2))
    if key in _PREWARMED:
        return
    bb = loop_bbox((lat, lon), (lat, lon), LOAD_TIERS[0] * 1000)
    _prewarm_bbox(bb, key)
    try:  # Ründchen über 3,5 km brauchen das nächstgrößere Gebiet: Download schon vorab (erst nach kurzer Pause)
        prefetch_osm(loop_bbox((lat, lon), (lat, lon), LOAD_TIERS[1] * 1000))
    except Exception:
        pass

    def work_green():
        try:
            glat, glon = green_center(lat, lon)
            for v in GREEN_QUERIES.values():
                _green_part(glat, glon, GREEN_RADIUS + 900, v)
        except Exception:
            pass
    threading.Thread(target=work_green, daemon=True).start()


def prewarm_tour(start_v, stops_v, end_v, extra_km):
    """Ründchen mit Zwischenziel: sobald Start und Ziele gewählt sind (noch vor dem Klick auf „finden“), das passende
    Gebiet im Hintergrund laden. Wer danach noch Länge, Grünstufe oder Optionen einstellt, muss nicht mehr warten."""
    try:
        pts, _n, _s = tour_points(start_v, stops_v, end_v)
        need, load, span = tour_bboxes(pts, extra_km * 1000.0)
        if span <= MAX_SPAN_M:
            _prewarm_bbox(load, ("tour",) + tuple(round(x, 3) for x in load), need)
    except Exception:
        pass


def prewarm_ab(start_v, dest_v):
    """Route A -> B: sobald Start und Ziel gewählt sind (noch vor dem Klick), den Korridor im Hintergrund laden.
    Es ist genau das Gebiet, das make_ab_view braucht; der Klick wartet dann nur noch auf den Rest."""
    try:
        p1, p2 = tuple(_unpack(start_v)[:2]), tuple(_unpack(dest_v)[:2])
        bb, span = corridor_bbox(p1, p2)
        if 30 < span <= MAX_SPAN_M:
            _prewarm_bbox(tuple(bb), ("ab",) + tuple(round(x, 3) for x in bb))
    except Exception:
        pass


def green_center(lat, lon):
    """Suchmitte für Grünflächen auf ein 1-km-Raster gelegt: Wer in der Nähe schon gesucht hat, findet die Daten im Speicher."""
    return round(round(lat * 100) / 100, 2), round(round(lon * 100) / 100, 2)


def water_geo(area):
    if "_wgeo" in area:
        return area["_wgeo"]
    geo = None
    if area.get("water_json"):
        try:
            geo = json.loads(area["water_json"])
        except Exception:
            geo = None
    area["_wgeo"] = geo
    return geo


# ------------------------------------------------------- Routing
def _csr_struct(area):
    """Einmal je Gebiet: Aufbau des Kantenfeldes für die schnelle Wegsuche (CSR-Struktur, Knoten-Index)."""
    c = area.get("_csr")
    if c is None:
        eu, ev = np.asarray(area["eu"]), np.asarray(area["ev"])
        nl = area["node_list"]
        n = len(nl)
        perm = np.lexsort((ev, eu))
        indptr = np.zeros(n + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(np.bincount(eu, minlength=n))
        c = (n, perm, indptr, ev[perm].astype(np.int32), {nd: i for i, nd in enumerate(nl)})
        area["_csr"] = c
    return c


def fast_path(area, cost, s, t, limit=np.inf):
    """Günstigster Weg s -> t bei Kantenkosten cost (Feld je Kantenindex; np.inf = gesperrt).
    Ersetzt nx.shortest_path mit Python-Gewichtsfunktion (rund 20-mal schneller). Gibt Knoten-IDs zurück."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n, perm, indptr, indices, nidx = _csr_struct(area)
    if s not in nidx or t not in nidx:
        raise nx.NodeNotFound(f"{s} / {t}")
    if s == t:
        return [s]
    data = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)[perm]
    dist, pred = dijkstra(csr_matrix((data, indices, indptr), shape=(n, n)), directed=True,
                          indices=nidx[s], return_predecessors=True, limit=limit)
    si, ti = nidx[s], nidx[t]
    if not np.isfinite(dist[ti]):
        raise nx.NetworkXNoPath(f"Kein Weg {s} -> {t}")
    seq = [ti]
    while seq[-1] != si:
        seq.append(int(pred[seq[-1]]))
    nl = area["node_list"]
    return [nl[i] for i in reversed(seq)]


def mark_used_both(area, usedm, path):
    """Wie mark_used, zusätzlich die Gegenrichtung jeder Kante (bereits gelaufen heißt in beide Richtungen)."""
    H = area["H"]
    mark_used(area, usedm, path)
    for a_, b_ in zip(path[:-1], path[1:]):
        if H.has_edge(b_, a_):
            usedm[H[b_][a_]["i"]] = True


def solve(area, o, t, lam, buf):
    g = np.clip(1 - area["D"] / buf, 0, 1).mean(axis=0)
    L, H = area["lengths"], area["H"]
    weights = {"short": L, "green": L * (1 + lam * (1 - g))}
    out = {}
    for name, w in weights.items():
        path = fast_path(area, w, o, t)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum())
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        out[name] = {"path": path, "dist": dist, "green": share}
    return out


# ------------------------------------------------------- Doppelwege
# Ein Weg gilt als doppelt gelaufen, wenn er nah (DUP_REACH) und parallel zu einem früheren Wegstück verläuft,
# auch auf der anderen Straßenseite oder auf der anderen Seite eines Baumstreifens. Eine Parallelstraße
# (mindestens ein Häuserblock entfernt) zählt nicht.
DUP_REACH = 35.0       # m: Abstand, bis zu dem „gleiche Straße“ angenommen wird
DUP_MIN_ARC = 80.0     # m Wegstrecke zwischen beiden Besuchen (sonst nur eine Kurve)
DUP_ANGLE = 35.0       # Grad: Parallelität
DUP_STEP = 10.0
TWIN_REACH = 32.0
TWIN_LATERAL = 3.5     # m seitlicher Versatz: Wegstücke nebeneinander (nicht hintereinander)


def dup_analysis(X, Y):
    """Polylinie (m) -> (Anteil doppelt, Maske je Abtastpunkt, s, px, py)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if len(X) < 2:
        return 0.0, np.zeros(0, bool), np.zeros(0), np.zeros(0), np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    total = float(cum[-1])
    if total < 2 * DUP_MIN_ARC:
        return 0.0, np.zeros(0, bool), np.zeros(0), np.zeros(0), np.zeros(0)
    s = np.arange(DUP_STEP / 2, total, DUP_STEP)
    px, py = np.interp(s, cum, X), np.interp(s, cum, Y)
    s2a, s2b = np.clip(s - 8, 0, total), np.clip(s + 8, 0, total)
    dx = np.interp(s2b, cum, X) - np.interp(s2a, cum, X)
    dy = np.interp(s2b, cum, Y) - np.interp(s2a, cum, Y)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    from scipy.spatial import cKDTree
    pairs = cKDTree(np.c_[px, py]).query_pairs(DUP_REACH, output_type="ndarray")
    dup = np.zeros(len(s), dtype=bool)
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        i, j = np.minimum(i, j), np.maximum(i, j)
        d = np.abs(ang[i] - ang[j])
        d = np.minimum(d, 180.0 - d)
        ok = ((s[j] - s[i]) > DUP_MIN_ARC) & (d < DUP_ANGLE)
        ok &= ~((s[i] < 25) & (s[j] > total - 25))  # Start und Ziel am selben Punkt zählen nicht
        dup[j[ok]] = True
    return float(dup.mean()), dup, s, px, py


def dup_from_coords(coords):
    """[[lat, lon], ...] -> (Anteil doppelt, Liste von Teilstrecken [[lat, lon], ...] für die Karte)."""
    try:
        a = np.asarray(coords, float)
        if len(a) < 2:
            return 0.0, []
        lat0, lon0 = float(a[:, 0].mean()), float(a[:, 1].mean())
        ky, kx = 110540.0, 111320.0 * math.cos(math.radians(lat0))
        X, Y = (a[:, 1] - lon0) * kx, (a[:, 0] - lat0) * ky
        share, mask, s, px, py = dup_analysis(X, Y)
        if not mask.any():
            return share, []
        idx = np.nonzero(mask)[0]
        runs, start, prev = [], idx[0], idx[0]
        for v in idx[1:]:
            if v - prev > 3:
                runs.append((start, prev)); start = v
            prev = v
        runs.append((start, prev))
        segs = []
        for r0, r1 in runs:
            if r1 - r0 < 2:
                continue
            seg = [[round(float(lat0 + py[k] / ky), 5), round(float(lon0 + px[k] / kx), 5)] for k in range(r0, r1 + 1)]
            segs.append(seg)
            if len(segs) >= 60:
                break
        return share, segs
    except Exception:
        return 0.0, []


def area_twins(area):
    """Je Kante die Nachbarkanten, die parallel daneben liegen (andere Straßenseite, Gehweg, Baumstreifen).
    Wird beim Routing wie ein bereits gelaufenes Wegstück bestraft."""
    tw = area.get("_twins")
    if tw is not None:
        return tw
    n = len(area["lengths"])
    ptr, idx = np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
        fr = (0.2, 0.5, 0.8)
        pts = [shapely.line_interpolate_point(area["lines"], f, normalized=True) for f in fr]
        xy = np.concatenate([np.c_[shapely.get_x(p), shapely.get_y(p)] for p in pts])
        owner = np.tile(np.arange(n), len(fr))
        pairs = cKDTree(xy).query_pairs(TWIN_REACH, output_type="ndarray")
        a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
        pa, pb = xy[pairs[:, 0]], xy[pairs[:, 1]]
        eu, ev, P = area["eu"], area["ev"], area["xy"]
        vec = P[ev] - P[eu]
        ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 180.0
        d = np.abs(ang[a] - ang[b])
        d = np.minimum(d, 180.0 - d)
        th = np.radians(ang[a])
        lat = np.abs(-(pb[:, 0] - pa[:, 0]) * np.sin(th) + (pb[:, 1] - pa[:, 1]) * np.cos(th))
        shared = (eu[a] == eu[b]) | (eu[a] == ev[b]) | (ev[a] == eu[b]) | (ev[a] == ev[b])
        ok = (a != b) & (d < DUP_ANGLE) & (lat >= TWIN_LATERAL) & ~shared
        a, b = a[ok], b[ok]
        both = np.unique(np.concatenate([a * n + b, b * n + a]))
        src, dst = both // n, both % n
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
        idx = dst
    except Exception:
        pass
    area["_twins"] = (ptr, idx)
    return area["_twins"]


def mark_used(area, usedm, path):
    """Kanten eines Teilwegs samt paralleler Nachbarkanten als „schon gelaufen“ markieren."""
    ptr, tw = area_twins(area)
    H = area["H"]
    for a_, b_ in zip(path[:-1], path[1:]):
        i = H[a_][b_]["i"]
        usedm[i] = True
        if ptr[i + 1] > ptr[i]:
            usedm[tw[ptr[i]:ptr[i + 1]]] = True


def path_dup_share(area, path):
    pos = area["pos"]
    try:
        xs = [pos[n_][0] for n_ in path]
        ys = [pos[n_][1] for n_ in path]
        return dup_analysis(xs, ys)[0]
    except Exception:
        return 0.0


# ------------------------------------------------------- Ründchen
LOOP_BEARINGS = 8      # Richtungen, in die Runden probiert werden
LOOP_TOLERANCE = 0.07  # akzeptierte Längenabweichung beim Anpassen (±7 %)
LOOP_REUSE_PENALTY = 4.0  # Strafe für bereits gelaufene Wegstücke (vermeidet Hin-und-zurück)
LOOP_MAX_KM = 10.0


LOAD_TIERS = (3.5, 5.0, 6.5, 8.0, LOOP_MAX_KM)  # Gebietsgrößen (km Rundenlänge), die geladen werden


def load_tier(km):
    """Gebiet für eine Rundenlänge: in Stufen mit etwas Reserve, damit „+“ in der Karte selten neu laden muss."""
    for t in LOAD_TIERS:
        if t >= km:
            return t
    return LOOP_MAX_KM


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
    half = (0.38 if target <= 6000 else 0.31) * target + 400 + dist_m(p1, p2) / 2  # lange Runden: Gebiet knapper halten
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


def build_loop_path(area, g, ng, lam, o, t, theta, rho, mu=None, wq=0.0):
    """Runde A -> 3 Wegpunkte auf einem Kreis -> B. Jede Teilstrecke wird grün-optimal geroutet;
    schon genutzte Wegstücke werden teurer, damit möglichst keine Doppelstrecken entstehen."""
    H, L = area["H"], area["lengths"]
    nz = area.get("noise")
    nz = np.zeros(len(L)) if nz is None else nz
    wd = area.get("wide")
    wd = np.zeros(len(L)) if wd is None else wd
    ap = area_accp(area)
    mu = (QUIET if lam > 0 else 0.0) if mu is None else mu
    ax, ay = area["pos"][o]
    cx, cy = ax + rho * math.cos(theta), ay + rho * math.sin(theta)
    wps = []
    for alpha in (90, 180, 270):
        ang = theta + math.pi + math.radians(alpha)
        wps.append(pick_waypoint(area, ng, cx + rho * math.cos(ang), cy + rho * math.sin(ang), 0.3 * rho))
    return route_seq(area, g, lam, [o] + wps + [t], mu, wq)


def route_seq(area, g, lam, seq, mu, wq):
    """Teilstrecken über die Punkte in seq, jede grün-optimal; Gelaufenes wird teurer (keine Doppelwege)."""
    L = area["lengths"]
    nz = area.get("noise")
    nz = np.zeros(len(L)) if nz is None else nz
    wd = area.get("wide")
    wd = np.zeros(len(L)) if wd is None else wd
    ap = area_accp(area)
    full = [seq[0]]
    usedm = np.zeros(len(L), dtype=bool)
    base = L * (1 + lam * (1 - g) + mu * nz + wq * wd + ap)
    for s, e in zip(seq[:-1], seq[1:]):
        if s == e:
            continue
        p = fast_path(area, np.where(usedm, base * LOOP_REUSE_PENALTY, base), s, e)
        full += p[1:]
        mark_used_both(area, usedm, p)
    return full


PARK_W = 0.25          # Bonus in der Rundenwahl für Strecke mitten im Park (nicht nur am Rand entlang)
PARK_MIN_LEN = 120.0   # m Wege im Park, ab denen ein Park gezielt angesteuert wird


def park_inside(area):
    """Kanten, die (fast) ganz im Park liegen."""
    ins = area.get("_pins")
    if ins is None:
        ins = (area["D"] <= 0.5).mean(axis=0) >= 0.7
        area["_pins"] = ins
    return ins


def park_clusters(area):
    """Zusammenhängende Parkwege (Knotenindizes + Länge je Park)."""
    c = area.get("_pcl")
    if c is not None:
        return c
    c = []
    ins = park_inside(area)
    if int(ins.sum()) >= 2:
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            n = len(area["node_list"])
            eu, ev = area["eu"][ins], area["ev"][ins]
            _, lab = connected_components(coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(n, n)), directed=False)
            tot = np.bincount(lab[eu], weights=area["lengths"][ins], minlength=int(lab.max()) + 1)
            nodes_in = np.unique(np.concatenate([eu, ev]))
            for comp in np.unique(lab[nodes_in]):
                if tot[comp] / 2 >= PARK_MIN_LEN:
                    c.append({"nodes": nodes_in[lab[nodes_in] == comp], "len": float(tot[comp] / 2)})
        except Exception:
            c = []
    area["_pcl"] = c
    return c


def park_pairs(area, o, target, limit=2):
    """Je nahem Park: Eingang (nächster Parkknoten) und fernster Punkt im Park. Die Runde führt hinein und quer hindurch."""
    nl, xy = area["node_list"], area["xy"]
    ox, oy = area["pos"][o]
    found = []
    for cl in park_clusters(area):
        nd = cl["nodes"]
        d_o = np.hypot(xy[nd, 0] - ox, xy[nd, 1] - oy)
        if d_o.min() > 0.4 * target + 100:
            continue
        entry = nd[int(d_o.argmin())]
        d_e = np.hypot(xy[nd, 0] - xy[entry, 0], xy[nd, 1] - xy[entry, 1])
        d_e = np.where(d_e <= 0.5 * target, d_e, 0)
        far = nd[int(d_e.argmax())]
        if d_e[int(d_e.argmax())] < 60:
            continue
        found.append((float(d_o.min()), nl[entry], nl[far]))
    found.sort(key=lambda z: z[0])
    return [(e, f) for _, e, f in found[:limit]]


def path_inside_share(area, path):
    H, L, ins = area["H"], area["lengths"], park_inside(area)
    idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
    tot = float(L[idx].sum()) if idx else 0.0
    return float(L[idx][ins[idx]].sum() / tot) if tot > 0 else 0.0


def edge_set(path):
    return frozenset((min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:]))


def undouble(area, path, keep=(), target=None, extra_cap=60.0):
    """Hin-und-zurück-Stücke (Stichstraße: ein Stück hinein und auf demselben Weg wieder heraus) beseitigen.
    Zwei Möglichkeiten je Stück: (A) kürzen, die Stichstraße entfällt, oder (B) verlängern, ein kurzer Bogen um
    den Block statt hin und zurück. Bei vorgegebener Länge (target) gewinnt, was näher dran liegt. Knoten in
    keep (Zwischenziele) bleiben immer erhalten."""
    H, L = area["H"], area["lengths"]
    keep = set(keep)

    def elen(a, b):
        return float(L[H[a][b]["i"]])

    def plen(q):
        return sum(elen(a, b) for a, b in zip(q[:-1], q[1:]))

    p = list(path)
    for _ in range(16):
        hit = None
        for i in range(1, len(p) - 1):
            if p[i - 1] != p[i + 1]:
                continue
            m = 1
            while i - m - 1 >= 0 and i + m + 1 < len(p) and p[i - m - 1] == p[i + m + 1]:
                m += 1
            stem = p[i - m:i + 1]  # a, ..., t
            if set(stem[1:]) & keep:
                continue
            hit = (i, m, stem)
            break
        if hit is None:
            break
        i, m, stem = hit
        a = stem[0]
        spur = plen(stem)
        total0 = plen(p)
        dA = total0 - 2 * spur
        banned = set(zip(stem[:-1], stem[1:])) | set(zip(stem[1:], stem[:-1]))
        used = edge_set(p)
        blocked = L.astype(np.float64).copy()
        for (u_, v_) in banned | used | {(y, x_) for (x_, y) in used}:
            if H.has_edge(u_, v_):
                blocked[H[u_][v_]["i"]] = np.inf
        best = None
        try:
            for x in list(H.successors(a)):
                ex = (min(a, x), max(a, x))
                if (a, x) in banned or ex in used:
                    continue

                cst = blocked.copy()
                if H.has_edge(x, a):
                    cst[H[x][a]["i"]] = np.inf
                try:
                    back = fast_path(area, cst, x, a, limit=2 * spur + extra_cap)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                cyc = [a] + back
                if len(set(cyc[:-1])) != len(cyc) - 1:
                    continue
                cl = plen(cyc)
                if cl <= 2 * spur + extra_cap and (best is None or cl < best[0]):
                    best = (cl, cyc)
        except Exception:
            best = None
        use_b = False
        if best is not None and target:
            dB = total0 - 2 * spur + best[0]
            use_b = abs(dB - target) < abs(dA - target) - 20.0
        if use_b:
            p = p[:i - m] + best[1] + p[i + m + 1:]
        else:
            p = p[:i - m + 1] + p[i + m + 1:]
    return p


def marks_for(path, nodes):
    """Positionen der Zwischenziele (nodes[1:-1]) in path, in Reihenfolge."""
    marks, pos = [], 0
    for n in nodes[1:-1]:
        try:
            k = path.index(n, pos)
        except ValueError:
            k = min(pos, len(path) - 1)
        marks.append(k)
        pos = k
    return marks


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
    seen, rep_edges = set(), 0.0
    for (a, b), li in zip(pairs, l):
        k = (min(a, b), max(a, b))
        if k in seen:
            rep_edges += li
        else:
            seen.add(k)
    rep = max(rep_edges / dist, path_dup_share(area, path))  # auch die andere Straßenseite zählt als Doppelweg
    nz = area.get("noise")
    noise = float((l * nz[idx]).sum() / dist) if nz is not None else 0.0
    return dist, float((l * g[idx]).sum() / dist), float(rep), noise


def find_loops(area, o, t, target, lam, buf, progress=None, quiet=None, bearings=None):
    """Probiert mehrere Richtungen, passt den Kreisradius an die Wunschlänge an und
    liefert bis zu 3 deutlich verschiedene Varianten (beste zuerst)."""
    g = edge_green(area, buf)
    ng = node_green(area, g)
    cands = []
    nb = bearings or LOOP_BEARINGS
    for k in range(nb):
        theta = 2 * math.pi * k / nb
        rho = target / (2 * math.pi * 1.35)
        for _ in range(3):
            try:
                path = build_loop_path(area, g, ng, lam, o, t, theta, rho, *(quiet or (None, 0.0)))
                path = undouble(area, path, target=target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                break
            st_ = loop_stats(area, g, path)
            if not st_:
                break
            dist, green, rep, noise = st_
            cands.append({"path": path, "dist": dist, "green": green, "rep": rep, "noise": noise})
            ratio = dist / target
            if abs(ratio - 1) <= LOOP_TOLERANCE:
                break
            rho /= ratio
        if progress:
            progress((k + 1) / nb)
    # Zusätzlich: Runden, die gezielt in einen nahen Park hinein und quer hindurch führen (statt nur am Rand entlang)
    if lam > 0:
        mu_, wq_ = quiet or (None, 0.0)
        mu_ = (QUIET if lam > 0 else 0.0) if mu_ is None else mu_
        try:
            pairs = park_pairs(area, o, target)
        except Exception:
            pairs = []
        for entry, far in pairs:
            ox, oy = area["pos"][o]
            fx, fy = area["pos"][far]
            ax_, ay_ = fx - ox, fy - oy
            nrm = math.hypot(ax_, ay_) or 1.0
            variants_ = [None] + [(sd, 0.2 * target) for sd in (1, -1)]
            for var in variants_:
                rho, sd = (None, 0) if var is None else (var[1], var[0])
                for _ in range(1 if var is None else 3):
                    try:
                        seq = [o, entry, far]
                        if rho:
                            wx = (ox + fx) / 2 - sd * ay_ / nrm * rho
                            wy = (oy + fy) / 2 + sd * ax_ / nrm * rho
                            seq.append(pick_waypoint(area, ng, wx, wy, max(0.3 * rho, 60.0)))
                        seq.append(t)
                        path = undouble(area, route_seq(area, g, lam, seq, mu_, wq_), target=target)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        break
                    st_ = loop_stats(area, g, path)
                    if not st_:
                        break
                    dist, green, rep, noise = st_
                    cands.append({"path": path, "dist": dist, "green": green, "rep": rep, "noise": noise})
                    ratio = dist / target
                    if rho is None or abs(ratio - 1) <= LOOP_TOLERANCE:
                        break
                    rho = min(max(rho / ratio, 0.04 * target), 0.45 * target)
    for c in cands:
        c["inside"] = path_inside_share(area, c["path"]) if lam > 0 else 0.0
        c["score"] = (c["green"] + PARK_W * c["inside"] - REP_W * c["rep"] - 0.4 * ((quiet[0] if quiet else QUIET) if lam > 0 else 0) * c["noise"]
                      - 2.0 * abs(c["dist"] / target - 1))
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


def extreme_loop(area, o, t, target, buf, existing=()):
    """Ründchen „Extrem grün“: bis zu 35 % länger als gewünscht, aber über 70 % Grün. Gesucht wird mit dem stärksten
    Grün-Gewicht eine Runde um etwa 1,22 × Wunschlänge; nur was ≤ 1,35 × Wunschlänge bleibt und über 70 % Grün hat, zählt."""
    lam = LEVELS["Grünster Umweg"]
    vs = find_loops(area, o, t, target * 1.22, lam, buf, quiet=LEVEL_QUIET["Grünster Umweg"], bearings=6)
    have = {tuple(v["path"]) for v in existing}
    ok = [v for v in vs if v["dist"] <= (1 + EXTREME_DETOUR) * target and v["green"] > EXTREME_GREEN_MIN
          and tuple(v["path"]) not in have]
    return max(ok, key=lambda v: (v["green"] - REP_W * v["rep"])) if ok else None


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


def build_nav(area, path, title, loop=False, waypoints=None):
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
    for pi, name in (waypoints or []):
        steps.append({"idx": node_idx[min(pi, len(node_idx) - 1)], "i": "📍",
                      "t": f"Zwischenziel erreicht: {name}"})
    steps.sort(key=lambda z: (z["idx"], z["i"] == "🏁"))
    return {
        "title": title, "total": round(cum[-1]),
        "coords": [[round(la, 6), round(lo, 6)] for la, lo in ll],
        "cum": [round(c, 1) for c in cum],
        "nidx": list(node_idx),
        "steps": [{"d": round(cum[s["idx"]]), "lat": round(ll[s["idx"]][0], 6),
                   "lon": round(ll[s["idx"]][1], 6), "t": s["t"], "i": s["i"]} for s in steps],
    }


MAP_TEMPLATE = r"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="only light"><meta name="supported-color-schemes" content="light"><meta name="darkreader-lock">
<style>:root{color-scheme:only light}</style>
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
#thanks{position:fixed;z-index:1100;left:16px;right:16px;top:46%;transform:translateY(-50%);max-width:400px;margin:0 auto;display:none;background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:20px 20px 16px;box-shadow:0 14px 40px rgba(60,50,30,.35);text-align:center}
#thanks.on{display:block;animation:thxin .9s cubic-bezier(.25,.1,.25,1) both}
#sheet,#banner{transition:opacity .7s ease}
body.fin #sheet,body.fin #banner,body.thx #side{opacity:0;pointer-events:none}
#thx{font-size:1.08rem;line-height:1.45;color:var(--text);margin-bottom:14px}
#thxb{border:0;border-radius:12px;background:var(--moss);color:#FBF7EE;padding:10px 22px;font-size:1rem}
#sum{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:0 0 12px}
#sum div{background:#fff;border:1px solid var(--line);border-radius:12px;padding:6px 10px;min-width:84px}
#sum small{display:block;font-size:.66rem;letter-spacing:.05em;text-transform:uppercase;opacity:.65}
#sum b{font-size:1.02rem}
#gpxb{display:block;margin:0 auto 10px;border:1px solid var(--line);border-radius:12px;background:#fff;color:var(--text);padding:8px 16px;font-size:.92rem}
body.dog #gpxb{border-color:#E3A96B}
@keyframes thxin{0%{opacity:0;transform:translateY(-46%)}100%{opacity:1;transform:translateY(-50%)}}
#banner.off{background:#B07A35}#banner.done{background:#3F5B3A}
#ico{font-size:36px;line-height:1;width:46px;text-align:center;flex:none}
#dist{font-size:1.5rem;font-weight:700;line-height:1.1}
#ins{font-size:1rem;margin-top:2px}
#side{position:fixed;z-index:1000;right:10px;top:calc(var(--top) + 8px);display:flex;flex-direction:column;gap:8px}
body.nav #side{top:calc(var(--top) + 92px)}
#side button.on{background:#DCEBF6;border-color:#8DBBD6}
#side button{width:44px;height:44px;border-radius:12px;border:1px solid var(--line);background:var(--paper);font-size:20px;box-shadow:0 3px 10px rgba(60,50,30,.25)}
#sheet{position:fixed;z-index:1000;left:10px;right:10px;bottom:10px;max-width:620px;margin:0 auto;background:rgba(251,248,241,.98);border:1px solid var(--line);border-radius:20px;padding:12px;box-shadow:0 8px 28px rgba(60,50,30,.3)}
#ttl{font-size:.85rem;opacity:.75;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#chips{display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;-webkit-overflow-scrolling:touch}
.chip{flex:none;border:1px solid var(--line);background:#fff;border-radius:14px;padding:8px 12px;text-align:left;min-width:108px;color:var(--text)}
.chip b{display:block;font-size:.92rem}.chip span{font-size:.78rem;opacity:.7}
#glg{display:flex;flex-wrap:wrap;gap:3px 12px;font-size:.76rem;margin:4px 0 2px}#glg:empty{display:none}
#glg i{display:inline-block;width:16px;height:5px;border-radius:3px;margin-right:5px;vertical-align:middle}
.gbar{display:flex;height:5px;border-radius:3px;overflow:hidden;margin-top:6px;background:#E5E0D3}
.chip{transition:background .25s,border-color .25s,transform .18s,box-shadow .25s}.chip:active{transform:scale(.95)}
.chip.on{background:#E7EDDB;border-color:var(--sage);box-shadow:0 3px 10px rgba(78,110,69,.25);transform:translateY(-1px)}
#stats>div{animation:statpop .55s cubic-bezier(.2,.9,.3,1.2) both}#stats>div:nth-child(2){animation-delay:.07s}#stats>div:nth-child(3){animation-delay:.14s}#stats>div:nth-child(4){animation-delay:.21s}
@keyframes statpop{from{opacity:0;transform:translateY(10px) scale(.94)}to{opacity:1;transform:none}}
button{transition:transform .15s,background .25s,box-shadow .25s}button:active{transform:scale(.96)}
#stats{display:flex;gap:8px;margin:2px 0 6px}
#stats div{flex:1;background:#F2EDE0;border-radius:12px;padding:6px 8px;min-width:0}
#stats span{display:block;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;opacity:.65;white-space:nowrap}
#stats b{font-size:1.05rem}
#hint{font-size:.8rem;opacity:.7;min-height:1em}
#hint.warn{opacity:1;color:#8A5A1E;font-weight:600}
#dogbox{max-height:24vh;overflow:auto;-webkit-overflow-scrolling:touch}
#dogbox:empty{display:none}
#dogbox .dsurf{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:.82rem;font-weight:600;margin:6px 0 2px}
#dogbox .dsurf i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
#dogbox .dnote,#dogbox .dok,#dogbox .dbad{font-size:.76rem;margin-top:3px;opacity:.82}
#dogbox .dok{color:#2E6B3A;opacity:1;font-weight:600}
#dogbox .dbad{color:#B3261E;opacity:1;font-weight:700}
#dogbox .dwind{margin-top:6px;padding:8px 10px;border-radius:12px;background:#E3F0F8;border:1px solid #9CC4DC;color:#1E4A66;font-size:.78rem;line-height:1.35;font-weight:600}
#dogbox .dpaw{margin-top:6px;padding:8px 10px;border-radius:12px;background:#FDE6DD;border:1px solid #E9A58E;color:#8A2A12;font-size:.78rem;line-height:1.35;font-weight:600}
body.dog{--moss:#B8571A;--sage:#E3A96B;--sand:#F3E2C8;--paper:#FFF8EE;--line:#E8CFAA}
body.dog #go{background:#C8661E}
body.dog #thxb{background:#C8661E}
body.dog .chip.on,body.dog .sq.on{background:#FBE3C4;border-color:#E3A96B;box-shadow:0 3px 10px rgba(184,87,26,.25)}
body.dog #prog>div{background:linear-gradient(90deg,#F0C48A,#B8571A)}
body.dog #stats div{background:#F8EAD3}
body.dog #sheet:before{content:"Hundemodus · Chay ist dabei";padding-left:26px;min-height:20px;line-height:20px;background:var(--chay) no-repeat left center/20px;display:block;font-size:.68rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#B8571A;margin:-2px 0 4px}
#navbox{display:none}
body.nav #navbox{display:block}body.nav #pre{display:none}
#rem{font-weight:600;font-size:.98rem}
#prog{height:7px;border-radius:5px;background:#DDD3BF;overflow:hidden;margin-top:6px}
#prog>div{height:100%;width:0;background:linear-gradient(90deg,#9DB38A,#4E6E45)}
#btns{display:flex;gap:8px;margin-top:8px}
#go{flex:1;height:48px;border:0;border-radius:14px;background:#5E7F52;color:#FBF7EE;font-size:1rem;font-weight:700}
.sq{width:48px;height:48px;border-radius:14px;border:1px solid var(--line);background:#fff;font-size:20px}
.sq.on{background:#E7EDDB;border-color:var(--sage)}
#smenu{display:none;position:absolute;left:0;right:0;bottom:calc(100% + 8px);background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:6px;box-shadow:0 8px 28px rgba(60,50,30,.3)}
#smenu.on{display:block}
#smenu button{display:block;width:100%;text-align:left;border:0;background:transparent;border-radius:12px;padding:10px 12px;font-size:.98rem;font-weight:600;color:var(--text)}
#smenu button:active{background:#E7EDDB}
#smenu small{display:block;font-weight:400;font-size:.76rem;opacity:.7;margin-top:2px}
#sbox{display:none;position:fixed;inset:0;z-index:1300;background:rgba(59,58,51,.45);align-items:center;justify-content:center;padding:16px}
#sbox.on{display:flex}
#sbox>div{background:var(--paper);border-radius:18px;padding:14px;max-width:420px;width:100%;box-shadow:0 14px 40px rgba(60,50,30,.35)}
#sbox textarea{display:block;width:100%;margin:8px 0;border:1px solid var(--line);border-radius:10px;padding:8px;font:inherit;font-size:.9rem}
#sbclose{border:0;border-radius:12px;background:var(--moss);color:#FBF7EE;padding:8px 18px;font-size:.95rem}
#vbtn,#fbtn{display:none}body.nav #vbtn,body.nav #fbtn{display:block}
#map .cone{background:none;border:none;pointer-events:none}.coneIn{width:120px;height:120px;transition:transform .35s ease-out;transform-origin:60px 60px;filter:drop-shadow(0 0 3px rgba(255,255,255,.7))}.cone svg{overflow:visible}
#styletip{position:fixed;z-index:1200;left:50%;top:calc(var(--top) + 14px);transform:translateX(-50%);max-width:min(92vw,420px);text-align:center;background:rgba(59,58,51,.88);color:#FBF7EE;padding:6px 12px;border-radius:10px;font-size:.85rem;opacity:0;transition:opacity .3s;pointer-events:none}
#clouds{position:fixed;inset:0;z-index:3000;pointer-events:none;background:#F4EFE6;animation:cbase 1.2s .25s cubic-bezier(.4,0,.6,1) forwards;overflow:hidden}
.cl{position:absolute;border-radius:50%;filter:blur(30px);background:radial-gradient(closest-side,rgba(255,255,255,.96),rgba(255,255,255,.55) 55%,rgba(255,255,255,0));will-change:transform,opacity;animation:cpart 1.9s .15s cubic-bezier(.45,0,.2,1) forwards}
@keyframes cbase{to{opacity:0}}
@keyframes cpart{0%{transform:translate(0,0) scale(1);opacity:1}60%{opacity:.9}100%{transform:translate(var(--dx),var(--dy)) scale(1.9);opacity:0}}
#sheet.arrive,#side.arrive,#banner.arrive{animation:crise .7s 1s cubic-bezier(.2,.8,.2,1) both}
@keyframes crise{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){#clouds{display:none}#map.arrive,.arrive{animation:none!important}}
#list{position:fixed;z-index:1100;top:calc(var(--top) + 8px);left:10px;right:10px;bottom:10px;max-width:620px;margin:0 auto;background:var(--paper);border:1px solid var(--line);border-radius:20px;overflow:auto;padding:12px 14px;display:none;box-shadow:0 8px 28px rgba(60,50,30,.3)}
.row{display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #E4DBC8}
.ri{font-size:22px;width:30px;text-align:center;flex:none}.rt{flex:1}.rd{opacity:.65;font-size:.85rem;white-space:nowrap}
</style></head><body>
<div id="map"></div>
<div id="clouds"></div>
<div id="thanks"><div id="thx"></div><div id="sum"></div><button id="gpxb">💾 Strecke als GPX sichern</button><button id="thxb">Bis bald im Grünen 🌿</button></div>
<div id="banner"><div id="ico">🧭</div><div><div id="dist"></div><div id="ins"></div></div></div>
<div id="side"><button id="layers" title="Kartenstil wechseln">🗺️</button><button id="sat" title="Satellit / Luftbild">🛰️</button><button id="fit" title="Ganze Route zeigen">⤢</button><button id="wcbtn" title="Toiletten ein-/ausblenden" style="display:none">🚻</button><button id="dwbtn" title="Trinkwasserstellen zeigen">🚰</button><button id="home" title="Zurück zum Start">🏠</button></div>
<div id="sheet">
 <div id="smenu"><button id="m_gpx">💾 Als GPX speichern<small>für Uhr, Komoot, Organic Maps und andere Apps</small></button><button id="m_off">📴 Offline-Blatt speichern<small>Skizze und Anweisungen als Datei, ohne Netz nutzbar</small></button><button id="m_loc">📍 Standort teilen<small>Link mit deiner Position und der Restzeit</small></button></div>
 <div id="ttl"></div>
 <div id="pre"><div id="chips"></div><div id="stats"></div><div id="hint"></div><div id="glg"></div><div id="dogbox"></div></div>
 <div id="navbox"><div id="rem"></div><div id="prog"><div></div></div></div>
 <div id="btns"><button id="go">▶ Navigation starten</button><button class="sq on" id="vbtn" title="Sprachansagen">🔊</button><button class="sq on" id="fbtn" title="Karte folgt mir">🎯</button><button class="sq" id="lbtn" title="Alle Anweisungen">📋</button><button class="sq" id="gpx" title="Route als GPX-Datei speichern (für Uhr, Komoot, Wander-Apps)">💾</button><button class="sq" id="demo" title="Demo ohne Gehen">🧪</button></div>
</div>
<div id="list"></div>
<div id="sbox"><div><b>Text zum Teilen</b><textarea id="sbta" readonly rows="5"></textarea><button id="sbclose">Schließen</button></div></div>
<div id="styletip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const WALK = D.speed || 80, TOP = D.topPad || 0, ky = 110540;
const GO = D.mode === 'ab' ? '▶ Route starten' : '▶ Ründchen starten';
document.documentElement.style.setProperty('--top', TOP + 'px');
if (D.dog) { document.body.classList.add('dog'); if (D.chay) document.documentElement.style.setProperty('--chay', 'url(' + D.chay + ')'); }
const RC = D.dog ? '#B8571A' : '#3F6B3A', RDONE = D.dog ? '#EBCBA3' : '#B9C9A8', RDOT = D.dog ? '#B8571A' : '#4E6E45';

// Übergang: Wolkendecke, die aufbricht und zur Seite zieht, dahinter zoomt die Karte heran
(function () {
  let seen = false; const key = 'gw_cloud_' + JSON.stringify([D.p1, D.p2, D.mode]);
  try { seen = sessionStorage.getItem(key) === '1'; sessionStorage.setItem(key, '1'); } catch (e) {}
  const c = $('clouds');
  if (seen) { c.style.display = 'none'; return; }
  window.INTRO = true; setTimeout(() => { window.INTRO = false; }, 3600);
  const W = innerWidth, H = innerHeight, N = 14;
  for (let i = 0; i < N; i++) {
    const d = document.createElement('div'); d.className = 'cl';
    const w = (0.5 + Math.random() * 0.5) * Math.max(W, H) * 0.7, h = w * (0.55 + Math.random() * 0.25);
    const cx = (i % 4) / 3 * W, cy = Math.floor(i / 4) / 3 * H;  // 4 x 4 Raster (14 Wolken), leicht verschoben
    d.style.cssText = 'width:' + w + 'px;height:' + h + 'px;left:' + (cx - w / 2 + (Math.random() - .5) * 80) + 'px;top:' + (cy - h / 2 + (Math.random() - .5) * 80) +
      'px;--dx:' + ((cx - W / 2) * 1.1 + (Math.random() - .5) * 120) + 'px;--dy:' + ((cy - H / 2) * 1.1 + (Math.random() - .5) * 120) + 'px;animation-delay:' + (0.05 + Math.random() * 0.5) + 's;animation-duration:' + (1.6 + Math.random() * 0.9) + 's';
    c.appendChild(d);
  }
  ['sheet', 'side', 'banner'].forEach(id => $(id).classList.add('arrive'));
  setTimeout(() => { c.style.display = 'none'; ['sheet', 'side', 'banner'].forEach(id => $(id).classList.remove('arrive')); }, 3000);
})();
const map = L.map('map', {zoomControl: false, zoomAnimation: false, fadeAnimation: false, markerZoomAnimation: false}).setView(D.p1, 15);

// Kartenhintergründe. Standard ohne Schlüssel und ohne Referer-Pflicht (in eingebetteten Fenstern
// fehlt der Referer oft; OpenStreetMap blockiert dann). Lädt eine Quelle nicht, wird gewechselt.
const TILES = [
  {n: 'Karte', u: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', o: {maxZoom: 19, maxNativeZoom: 18, attribution: 'Kartenbild © Esri, OpenStreetMap-Mitwirkende u. a.'}},
  {n: 'Straßen', u: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', o: {maxZoom: 19, maxNativeZoom: 18, attribution: 'Kartenbild © Esri u. a.'}},
  {n: 'Gelände', u: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', o: {subdomains: 'abc', maxZoom: 19, maxNativeZoom: 17, attribution: '© OpenStreetMap-Mitwirkende, SRTM | © OpenTopoMap (CC-BY-SA)'}},
  {n: 'OpenStreetMap', u: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', o: {maxZoom: 19, referrerPolicy: 'strict-origin-when-cross-origin', attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>-Mitwirkende'}},
  // Satellit / Luftbild (sat: true) -> nur über den 🛰️-Knopf, nie als automatischer Ersatz
  {n: 'Satellit (Esri, aktuell)', sat: true, u: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', o: {maxZoom: 20, maxNativeZoom: 18, attribution: 'Bild © Esri, Maxar, Earthstar Geographics'}},
  {n: 'Luftbild NRW (10 cm)', sat: true, wms: true, u: 'https://www.wms.nrw.de/geobasis/wms_nw_dop', o: {layers: 'nw_dop_rgb', format: 'image/jpeg', version: '1.3.0', maxZoom: 21, attribution: 'Luftbild © Geobasis NRW (dl-de/zero-2-0)'}},
  {n: 'Sentinel-2 (2025, 10 m)', sat: true, u: 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg', o: {maxZoom: 19, maxNativeZoom: 13, attribution: 'Sentinel-2 cloudless 2025 © EOX IT Services GmbH (enthält modifizierte Copernicus-Sentinel-Daten)'}}
];
const BASE = TILES.map((t, i) => i).filter(i => !TILES[i].sat), SAT = TILES.map((t, i) => i).filter(i => TILES[i].sat);
let tIdx = 0, tLayer = null, tOk = 0, tErr = 0, tTimer = null, tTried = new Set(), tipTimer = null;
function tip(t, ms) {
  const el = $('styletip'); el.textContent = t; el.style.opacity = 1;
  clearTimeout(tipTimer); tipTimer = setTimeout(() => { el.style.opacity = 0; }, ms || 1600);
}
function nextWorking() {
  // automatischer Wechsel: nächste noch nicht probierte Quelle (Satellit nur als letzte)
  for (let k = 1; k <= TILES.length; k++) {
    const j = (tIdx + k) % TILES.length;
    if (!tTried.has(j) && !TILES[j].sat) return j;
  }
  return -1;
}
function setTiles(i, auto) {
  if (tLayer) map.removeLayer(tLayer);
  clearTimeout(tTimer);
  tIdx = i; tOk = 0; tErr = 0; tTried.add(i);
  const layer = (TILES[i].wms ? L.tileLayer.wms(TILES[i].u, TILES[i].o) : L.tileLayer(TILES[i].u, TILES[i].o)).addTo(map);
  tLayer = layer;
  layer.on('tileload', () => { tOk++; });
  const fail = () => {
    if (tLayer !== layer || tOk > 0) return;
    const j = TILES[i].sat ? BASE[0] : nextWorking();
    if (j >= 0) { setTiles(j, true); tip((TILES[i].sat ? TILES[i].n + ' nicht erreichbar, ' : '') + 'Kartenstil: ' + TILES[j].n); }
  };
  layer.on('tileerror', () => { tErr++; if (tErr >= 4 && tOk === 0) fail(); });
  tTimer = setTimeout(fail, TILES[i].sat ? 12000 : 7000);  // nichts geladen (auch ohne Fehlermeldung) -> wechseln
}
setTiles(0);
(D.attrib || []).forEach(a => map.attributionControl.addAttribution(a));

if (D.water) {
  L.geoJSON(D.water, {style: {color: '#7FA8B8', weight: 1.4, fillColor: '#B9D4DE', fillOpacity: .55}, interactive: false}).addTo(map);
}
if (D.parks) {
  L.geoJSON(D.parks, {style: {color: '#7C9F69', weight: 1, fillColor: '#B5CDA0', fillOpacity: .5},
    onEachFeature: (f, l) => { if (f.properties && f.properties.name) l.bindTooltip(f.properties.name, {sticky: true}); }}).addTo(map);
}
if (D.dogz) {
  L.geoJSON(D.dogz, {style: {color: '#C0392B', weight: 1.6, dashArray: '5 4', fillColor: '#E8836F', fillOpacity: .28},
    onEachFeature: (f, l) => l.bindTooltip('🚫 Hunde: ' + ((f.properties && f.properties.k) || 'Verbot'), {sticky: true})}).addTo(map);
}
const pin = e => L.divIcon({html: '<div style="font-size:30px;line-height:30px;filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))">' + e + '</div>',
  className: '', iconSize: [32, 32], iconAnchor: [16, 28]});
const chayIcon = () => L.divIcon({html: '<img src="' + D.chay + '" width="34" height="34" style="display:block;filter:drop-shadow(0 2px 3px rgba(0,0,0,.4))" alt="">',
  className: '', iconSize: [34, 34], iconAnchor: [-4, 32]});  // etwas rechts neben dem Standort: Chay läuft mit
const chayMk = D.dog && D.chay ? L.marker(D.p1, {icon: chayIcon(), interactive: false, keyboard: false, zIndexOffset: 700}) : null;
const startPin = L.marker(D.p1, {icon: pin('🚶')}).addTo(map);  // nur vor dem Start sichtbar (in der Navigation wäre er doppelt zum Standortpunkt)
if (chayMk) chayMk.addTo(map);
if (D.p1[0] !== D.p2[0] || D.p1[1] !== D.p2[1]) L.marker(D.p2, {icon: pin('🏁')}).addTo(map);
(D.stops || []).forEach(s => L.marker([s[0], s[1]], {icon: pin('📍')}).addTo(map).bindTooltip(s[2], {direction: 'top'}));
const userDot = L.circleMarker([0, 0], {radius: 9, color: '#fff', weight: 3, fillColor: '#2F6F8F', fillOpacity: 1});
// Richtungskegel: zeigt, wohin man blickt/läuft; grün = richtig, gelb = leicht daneben, rot = falsche Richtung
const userCone = L.marker([0, 0], {interactive: false, keyboard: false, zIndexOffset: 800, icon: L.divIcon({className: 'cone', iconSize: [120, 120], iconAnchor: [60, 60],
  html: '<div class="coneIn"><svg viewBox="-60 -60 120 120" width="120" height="120"><defs><radialGradient id="cg" cx="0" cy="0" r="58" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="currentColor" stop-opacity=".85"/><stop offset="1" stop-color="currentColor" stop-opacity="0"/></radialGradient></defs><path d="M0 0 L-27 -52 A58 58 0 0 1 27 -52 Z" fill="url(#cg)"/></svg></div>'})});
let hdg = null, hdgGps = 0, prevFix = null, compassOn = false;
const angDiff = (a, b) => { let d = (a - b) % 360; if (d > 180) d -= 360; if (d < -180) d += 360; return d; };
function bearingLL(a, b) {
  const dy = (b[0] - a[0]) * 110540, dx = (b[1] - a[1]) * 111320 * Math.cos(a[0] * Math.PI / 180);
  return (Math.atan2(dx, dy) * 180 / Math.PI + 360) % 360;
}
function updateCone(h) {
  if (h === null || h === undefined || isNaN(h) || !lastPos) return;
  hdg = hdg === null ? h : (hdg + angDiff(h, hdg) * 0.45 + 360) % 360;  // glätten
  let col = '#2F6F8F';
  if (navOn && coords.length > 1 && !finished) {
    const a = pointAt(Math.max(0, along - 3)), b = pointAt(Math.min(total, along + 28));
    const d = Math.abs(angDiff(hdg, bearingLL(a, b)));
    col = d <= 35 ? '#3E8E41' : (d <= 80 ? '#E0A526' : '#D2452D');
  }
  userCone.setLatLng(lastPos); if (!map.hasLayer(userCone)) userCone.addTo(map);
  const el = userCone.getElement();
  if (el) { el.style.color = col; const inn = el.firstElementChild; if (inn) inn.style.transform = 'rotate(' + hdg.toFixed(0) + 'deg)'; }
}
function onOrient(e) {
  if (Date.now() - hdgGps < 4000) return;  // beim Gehen gilt die Bewegungsrichtung, im Stehen der Kompass
  let h = null;
  if (typeof e.webkitCompassHeading === 'number') h = e.webkitCompassHeading;
  else if (e.absolute && e.alpha !== null && e.alpha !== undefined) h = (360 - e.alpha) % 360;
  if (h === null) return;
  h = (h + ((screen.orientation && screen.orientation.angle) || 0) + 360) % 360;
  updateCone(h);
}
function startCompass() {
  if (compassOn) return; compassOn = true;
  const go = () => {
    window.addEventListener('deviceorientationabsolute', onOrient, true);
    window.addEventListener('deviceorientation', onOrient, true);
  };
  try {
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function')
      DeviceOrientationEvent.requestPermission().then(r => { if (r === 'granted') go(); }).catch(() => {});
    else go();
  } catch (e) {}
}
function stopCompass() {
  compassOn = false; hdg = null; prevFix = null;
  window.removeEventListener('deviceorientationabsolute', onOrient, true);
  window.removeEventListener('deviceorientation', onOrient, true);
}
const userAcc = L.circle([0, 0], {radius: 10, color: '#2F6F8F', weight: 1, fillColor: '#2F6F8F', fillOpacity: .12});

let cur = -1, layers = [], coords = [], cum = [], steps = [], total = 1, P = [], lat0 = 0, lon0 = 0, kx = 1, done = null;
let navOn = false, along = 0, lastIdx = 0, offCount = 0, finished = false, follow = true, lastPos = null,
    said = {}, voice = true, watchId = null, sim = null, hadFix = false, wake = null, navT0 = 0, demoRun = false, dwPts = [], dwLayer = null, dwOn = false, dwBusy = false;

const toXY = (la, lo) => [(lo - lon0) * kx, (la - lat0) * ky];
function fmt(m) {
  if (m >= 1000) return (m / 1000).toFixed(1).replace('.', ',') + ' km';
  const r = m > 100 ? 10 : 5; return Math.max(0, Math.round(m / r) * r) + ' m';
}
function eta(rem) { return new Date(Date.now() + rem / WALK * 60000).toLocaleTimeString('de-DE', {hour: '2-digit', minute: '2-digit'}); }
function banner(cls, ico, dist, ins) { $('banner').className = cls; $('ico').textContent = ico; $('dist').textContent = dist; $('ins').textContent = ins; }
let VOICE = null;
function pickVoice() {
  // beste deutsche Stimme: neuronale/hochwertige zuerst, blecherne (compact/espeak) zuletzt
  try {
    const vs = (speechSynthesis.getVoices() || []).filter(v => /^de/i.test(v.lang));
    if (!vs.length) return;
    const score = v => (/neural|natural|online/i.test(v.name) ? 60 : 0) + (/premium|enhanced|erweitert|verbessert/i.test(v.name) ? 45 : 0) +
      (/google/i.test(v.name) ? 25 : 0) + (/anna|petra|katja|marlene|vicki|amala|conrad|florian|seraphina/i.test(v.name) ? 15 : 0) +
      (/de[-_]DE/i.test(v.lang) ? 5 : 0) - (/compact|eloquence|espeak|robot/i.test(v.name) ? 50 : 0);
    vs.sort((a, b) => score(b) - score(a));
    VOICE = vs[0];
  } catch (e) {}
}
if ('speechSynthesis' in window) { pickVoice(); try { speechSynthesis.addEventListener('voiceschanged', pickVoice); } catch (e) {} }
function say(t) {
  if (!voice || !('speechSynthesis' in window)) return;
  try {
    const clean = String(t).replace(/[\u{1F000}-\u{1FFFF}\u2600-\u27BF]/gu, '').replace(/\s+/g, ' ').replace(/ m\b/g, ' Meter').trim();
    const u = new SpeechSynthesisUtterance(clean);
    u.lang = 'de-DE'; if (VOICE) u.voice = VOICE;
    u.rate = 0.96; u.pitch = 1.0; u.volume = 1;
    speechSynthesis.cancel(); speechSynthesis.speak(u);
  } catch (e) {}
}
function fitRoute() {
  if (!coords.length) return;
  map.fitBounds(L.latLngBounds(coords), {paddingTopLeft: [24, TOP + 24], paddingBottomRight: [24, Math.max(D.dog ? 340 : 250, ($('sheet') ? $('sheet').offsetHeight : 0) + 24)], animate: false});
}
function buildList() {
  let h = '<div style="font-weight:700;margin-bottom:6px">Wegehinweise</div>';
  steps.forEach((s, i) => { h += '<div class="row"><span class="ri">' + esc(s.i) + '</span><span class="rt">' + esc(s.t) +
    '</span><span class="rd">' + (i === 0 ? 'Start' : 'bei ' + fmt(s.d)) + '</span></div>'; });
  $('list').innerHTML = h + '<p><button class="sq" id="closelist" style="width:auto;padding:0 18px">Schließen</button></p>';
  $('closelist').addEventListener('click', () => { $('list').style.display = 'none'; });
}
let INTRO_DONE = false;
// Route „zeichnet“ sich von Start bis Ziel (SVG-Strich mit normierter Pfadlänge)
function drawIn(line, delay) {
  try {
    const el = line.getElement(); if (!el || !el.setAttribute) return;
    el.setAttribute('pathLength', '1000');
    el.style.transition = 'none'; el.style.strokeDasharray = '1000'; el.style.strokeDashoffset = '1000';
    void el.getBoundingClientRect();
    setTimeout(() => {
      el.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(.45,.05,.25,1)'; el.style.strokeDashoffset = '0';
      setTimeout(() => { el.style.strokeDasharray = ''; el.style.strokeDashoffset = ''; el.style.transition = ''; }, 1700);
    }, delay);
  } catch (e) {}
}
const GCOL = ['#2E7D32', '#9CC25A', '#8C9188'], GNAME = ['viel Grün', 'etwas Grün', 'Straße'];
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
  if (o.gr && o.gr.length) {  // nach Grünanteil eingefärbt: dunkelgrün viel Grün, hellgrün etwas Grün, grau Straße
    o.gr.forEach(r => add(L.polyline(coords.slice(r[1], r[2] + 1), {color: GCOL[r[0]], weight: 7, opacity: 1, lineCap: 'round', lineJoin: 'round'})));
  } else {
    const mainLine = add(L.polyline(coords, {color: RC, weight: 7, opacity: 1}));
    drawIn(mainLine, (window.INTRO && !INTRO_DONE) ? 1900 : 0);
  }
  done = add(L.polyline([], {color: RDONE, weight: 7, opacity: 1}));
  (o.dups || []).forEach(sg => add(L.polyline(sg, {color: '#E08A1E', weight: 5, opacity: .95, dashArray: '1 9', lineCap: 'round'})));
  steps.forEach((s, k) => { if (k > 0 && k < steps.length - 1)
    add(L.circleMarker([s.lat, s.lon], {radius: 5, color: '#fff', weight: 2, fillColor: RDOT, fillOpacity: 1})); });
  (o.hot || []).forEach(sg => add(L.polyline(sg, {color: '#E2341D', weight: 5, opacity: .95, dashArray: '1 8', lineCap: 'round'})));
  (o.forbid || []).forEach(sg => add(L.polyline(sg, {color: '#C0392B', weight: 9, opacity: .8})));
  document.querySelectorAll('.chip').forEach((c, k) => c.classList.toggle('on', k === i));
  $('stats').innerHTML = o.stats.map(s => '<div><span>' + esc(s[0]) + '</span><b>' + esc(s[1]) + '</b></div>').join('');
  $('hint').textContent = (o.hint || '') + (showShort ? ' · gestrichelt: kürzeste Route' : '');
  $('hint').className = o.warn ? 'warn' : '';
  $('glg').innerHTML = o.gb ? o.gb.map((v, k) => v > 0 ? '<span><i style="background:' + GCOL[k] + '"></i>' + GNAME[k] + ' ' + v + ' %</span>' : '').join('') : '';
  const dg = o.dog; let dh = '';
  if (dg) {
    dh += '<div class="dsurf">' + (dg.surf || []).map(s => '<span><i style="background:' + s[2] + '"></i>' + esc(s[0]) + ' ' + s[1] + ' %</span>').join('') + '</div>';
    if (dg.est >= 10) dh += '<div class="dnote">Belag zu ' + dg.est + ' % aus der Wegart geschätzt (kein Eintrag in den Kartendaten).</div>';
    dh += dg.nocheck ? '<div class="dnote">Verbotszonen konnten gerade nicht geprüft werden.</div>' : dg.forb > 0 ? '<div class="dbad">⚠️ ' + dg.forb + ' m führen durch eine Hunde-Verbotszone, die sich nicht umgehen ließ. Bitte Schilder beachten.</div>'
                      : '<div class="dok">✅ Keine bekannten Hunde-Verbotszonen auf dieser Route</div>';
    if (dg.leash > 0) dh += '<div class="dnote">🦮 Leinenpflicht auf ' + dg.leash + ' m laut Kartendaten.</div>';
    if (dg.paw) dh += '<div class="dpaw">' + esc(dg.paw) + '</div>';
    if (dg.wind) dh += '<div class="dwind">' + esc(dg.wind) + '</div>';
    dh += '<div class="dnote">Nicht jedes Verbot steht in den Kartendaten. Schilder, Leinenpflicht und Brut- und Setzzeit gehen vor.</div>';
  }
  if (!dg && o.wind) dh += '<div class="dwind">' + esc(o.wind) + '</div>';
  $('dogbox').innerHTML = dh;
  $('rem').textContent = 'Gesamt ' + fmt(total) + ' · ca. ' + Math.max(1, Math.round(total / WALK)) + ' min';
  buildList();
  if (fit) fitRoute();
}

function snap(xy, fromAlong, windowed) {
  // Fortschritt nur im Fenster um die bisherige Strecke (−40 m … +160 m): Kreuzungen und Wegstücke, die später noch
  // einmal an derselben Stelle vorbeiführen (Ründchen, Rückweg), dürfen nicht zu einem Sprung ans Ende führen.
  let best = null;
  for (let i = 0; i < P.length - 1; i++) {
    if (windowed && (cum[i + 1] < fromAlong - 40 || cum[i] > fromAlong + 160)) continue;
    const a = P[i], b = P[i + 1], dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    let t = l2 > 0 ? ((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(xy[0] - (a[0] + t * dx), xy[1] - (a[1] + t * dy));
    if (!best || d < best.off) best = {i: i, off: d, along: cum[i] + t * Math.sqrt(l2)};
  }
  return best || {i: 0, off: 0, along: 0};
}
const THX_WALK = [
  'Geschafft! Dein Körper sagt Danke: Grün senkt den Puls und hebt die Laune.',
  'Schön gewesen. Ein Weg im Grünen ist Balsam für Kopf und Beine.',
  'Frische Luft getankt, Akku geladen. Mehr Grün, mehr Gelassenheit.',
  'Deine Beine sind zufrieden, dein Kopf auch. Das war grün gedacht.',
  'Ziel erreicht. Lunge, Herz und Laune klatschen leise im Hintergrund.',
  'Danke fürs Grünwandern! Jeder Schritt zwischen Bäumen tut dir gut.',
  'Gut gemacht. Der Weg war grün, jetzt darfst du auf dem Sofa angeben.',
  'Schon ein paar Minuten im Grünen entspannen messbar. Du hast mehr geschafft.'];
const THX_DOG = [
  'Geschafft! Dein Hund sagt Wuff, das heißt Danke. Wasser und ein Leckerli verdient er sich jetzt.',
  'Schöne Runde! Pfoten kurz checken, dann darf er auf dem Sofa schnarchen.',
  'Ziel erreicht. Schwanzwedeln im Dauerbetrieb: Das war ein Spaziergang nach seinem Geschmack.',
  'Gut gelaufen! Frische Luft, viele Schnuppermomente und ein zufriedener Hund.',
  'Angekommen. Nasenarbeit erledigt, Laune bestens. Danke fürs Gassigehen!'];
const THX_GREEN = [
  'Angekommen! Jetzt viel Spaß im Grün. Schuhe aus, wenn der Rasen es erlaubt.',
  'Da bist du. Hier darf getrödelt, getobt und gestaunt werden.',
  'Willkommen im Grünen! Für Bänke, Enten und gute Laune ist ab jetzt Zeit.',
  'Ziel erreicht. Ab hier zählt nur noch Spaß. Viel Freude im Grün!',
  'Grün erreicht. Tief durchatmen, losziehen, Spaß haben.',
  'Geschafft, jetzt beginnt der schöne Teil. Viel Spaß da draußen!'];
const THX_DOG_PARK = [
  'Angekommen im Grünen! Jetzt darf geschnuppert werden. Leinenregeln im Park bitte im Blick behalten.',
  'Park erreicht. Nase hoch, Schwanz hoch: Hier darf er sich Zeit lassen. Schilder und Leinenpflicht beachten.',
  'Da seid ihr. Ein paar Schnüffelminuten im Grünen, dann geht es entspannt weiter.',
  'Ziel im Park erreicht. Dein Hund hat jetzt Auslauf verdient. Bitte nur dort ableinen, wo es erlaubt ist.',
  'Willkommen im Grünen! Zeit zum Schnuppern, Spielen und Stöckchen suchen.'];
const THX_DOG_FREE = [
  'Hundefreilauf erreicht! Wenn hier laut Schild Freilauf gilt: Leine ab und los.',
  'Angekommen auf der Hundewiese. Toben, schnüffeln, Hundefreunde treffen: Jetzt ist seine Zeit.',
  'Freilauffläche erreicht. Bitte kurz die Hinweise vor Ort prüfen, dann darf er flitzen.'];
const THX_DOG_WATER = [
  'Am Wasser angekommen! Eine Pfotenabkühlung tut gut. Bei Algen oder Schaum aber lieber nicht trinken lassen.',
  'Wasser erreicht. Kurz planschen oder trinken lassen, sofern das Wasser sauber aussieht.'];
const THX_DOG_ARRIVE = [
  'Angekommen! Dein Hund hat sich eine Pause und einen Schluck Wasser verdient.',
  'Ziel erreicht. Gut gelaufen, gut geschnüffelt. Danke fürs Gassigehen!',
  'Geschafft! Schwanzwedeln im Dauerbetrieb, Pause ist jetzt dran.'];
const THX_WATER = [
  'Am Wasser angekommen. Durchatmen, schauen, Ruhe genießen.',
  'Ziel erreicht. Wasser, Weite und frische Luft: Besser lässt sich ein Weg kaum beenden.'];
function thxList() {
  const o = D.options[Math.max(cur, 0)] || {}, ctx = o.endctx || '';
  if (D.dog) {
    if (ctx === 'dogrun') return THX_DOG_FREE;
    if (ctx === 'park' || D.kind === 'green') return THX_DOG_PARK;
    if (ctx === 'water') return THX_DOG_WATER;
    return D.mode === 'ab' ? THX_DOG_ARRIVE : THX_DOG;  // Ründchen: zurück zu Hause, Pfoten checken, Sofa
  }
  if (D.kind === 'green') return THX_GREEN;
  if (D.mode === 'ab' && ctx === 'park') return THX_GREEN;
  if (D.mode === 'ab' && ctx === 'water') return THX_WATER;
  return THX_WALK;
}
function doneFlag(on) {  // merkt sich im Adressfeld der App: Strecke beendet (Python liest es beim Zurückgehen)
  try {
    const u = new parent.URL(parent.location.href);
    if (on) u.searchParams.set('gwdone', '1'); else u.searchParams.delete('gwdone');
    parent.history.replaceState(null, '', u.toString());
  } catch (e) {}
}
function showThanks() {
  document.body.classList.add('fin', 'thx'); doneFlag(true);
  const list = thxList(), lk = 'gw_thx_' + (D.dog ? 'd' : '') + list.length + (D.kind || 'walk');
  let i = Math.floor(Math.random() * list.length), last = -1;
  try { last = +(localStorage.getItem(lk) || -1); } catch (e) {}
  if (i === last) i = (i + 1) % list.length;
  try { localStorage.setItem(lk, String(i)); } catch (e) {}
  $('thx').textContent = list[i];
  $('sum').innerHTML = summaryHtml();
  if (D.dog && D.chay && !$('thxc')) { const im = document.createElement('img'); im.id = 'thxc'; im.src = D.chay; im.width = 64; im.height = 64; im.alt = D.chayName || ''; im.style.cssText = 'display:block;margin:-46px auto 8px;filter:drop-shadow(0 3px 5px rgba(0,0,0,.3))'; $('thanks').insertBefore(im, $('thx')); }
  $('thxb').textContent = D.dog ? 'Bis zum nächsten Gassi 🐾' : 'Bis zum nächsten Spaziergang 🌿';
  $('thanks').classList.add('on');
}
function statOf(o, label) {
  const r = (o.stats || []).find(x => x[0] === label);
  return r ? r[1] : null;
}
function summaryHtml() {
  const o = D.options[Math.max(cur, 0)] || {};
  const cells = [];
  const len = statOf(o, 'Länge'), plan = statOf(o, 'Dauer'), gr = statOf(o, 'Grünanteil'), lit = statOf(o, 'Beleuchtet');
  if (len) cells.push(['📏 Strecke', len]);
  const mins = navT0 && !demoRun ? Math.round((Date.now() - navT0) / 60000) : 0;
  if (mins >= 1) cells.push(['⏱️ Unterwegs', mins + ' min' + (plan ? ' (geplant ' + plan + ')' : '')]);
  else if (plan) cells.push(['⏱️ Dauer', plan + ' (geplant)']);
  if (gr && gr !== '–') cells.push(['🌿 Grünanteil', gr]);
  if (lit) cells.push(['💡 Beleuchtet', lit]);
  if (o.label) cells.push(['🧭 Variante', o.label]);
  return cells.map(c => '<div><small>' + esc(c[0]) + '</small><b>' + esc(c[1]) + '</b></div>').join('');
}
function gpxText(o) {
  const x = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const wpt = (p, n) => '<wpt lat="' + p[0].toFixed(6) + '" lon="' + p[1].toFixed(6) + '"><name>' + x(n) + '</name></wpt>';
  const loop = D.p1[0] === D.p2[0] && D.p1[1] === D.p2[1];
  const name = loop ? 'Ründchen ab ' + D.t1 : D.t1 + ' → ' + D.t2;
  const desc = (o.label || '') + ' · ' + (o.stats || []).map(r => r[0] + ' ' + r[1]).join(' · ');
  let w = wpt(D.p1, (loop ? 'Start/Ziel: ' : 'Start: ') + D.t1);
  (D.stops || []).forEach(s => { w += wpt([s[0], s[1]], 'Zwischenziel: ' + s[2]); });
  if (!loop) w += wpt(D.p2, 'Ziel: ' + D.t2);
  (dwOn ? dwNear() : []).forEach(p => { w += wpt([p.la, p.lo], 'Trinkwasser' + (p.name ? ': ' + p.name : '')); });
  const pts = (o.coords || []).map(c => '<trkpt lat="' + c[0].toFixed(6) + '" lon="' + c[1].toFixed(6) + '"></trkpt>').join('');
  return '<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="GreenWalker" xmlns="http://www.topografix.com/GPX/1/1">'
    + '<metadata><name>' + x(name) + '</name><desc>' + x(desc) + '</desc></metadata>' + w
    + '<trk><name>' + x(name + ' (' + (o.label || '') + ')') + '</name><trkseg>' + pts + '</trkseg></trk></gpx>';
}
function saveFile(text, mime, fn) {
  const blob = new Blob([text], {type: mime});
  return (async () => {
    try {  // Am Handy: Teilen-Menü (In Dateien sichern, an Uhr oder App senden)
      if (navigator.canShare && /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
        const f = new File([blob], fn, {type: mime});
        if (navigator.canShare({files: [f]})) { await navigator.share({files: [f], title: 'GreenWalker'}); return; }
      }
    } catch (e) { if (e && e.name === 'AbortError') return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = fn; a.style.display = 'none'; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
    tip('Gespeichert: ' + fn);
  })();
}
function baseName(o) {
  return 'GreenWalker_' + ((D.t1 || '') + '_' + (D.t2 || '') + '_' + (o.label || '')).replace(/[^A-Za-z0-9äöüÄÖÜß]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60);
}
function saveGpx() {
  const o = D.options[Math.max(cur, 0)];
  if (!o || !o.coords || o.coords.length < 2) { tip('Keine Route zum Speichern'); return; }
  return saveFile(gpxText(o), 'application/gpx+xml', baseName(o) + '.gpx');
}
function saveOffline() {
  const o = D.options[Math.max(cur, 0)];
  if (!o || !o.coords || o.coords.length < 2) { tip('Keine Route zum Speichern'); return; }
  return saveFile(offlineHtml(o), 'text/html', baseName(o) + '_offline.html');
}
// Offline-Blatt: eine einzige HTML-Datei ohne Netz: Skizze (Route, Parks, Wasser), Anweisungen, optional GPS-Punkt
function offlineHtml(o) {
  const x = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const W = 760, H = 560, M = 34, cs = o.coords;
  let la0 = 0; cs.forEach(c => { la0 += c[0]; }); la0 /= cs.length;
  const k = Math.cos(la0 * Math.PI / 180);
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  cs.forEach(c => { const px = c[1] * k, py = -c[0]; x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py); });
  const sc = Math.min((W - 2 * M) / Math.max(x1 - x0, 1e-9), (H - 2 * M) / Math.max(y1 - y0, 1e-9));
  const ox = (W - (x1 - x0) * sc) / 2, oy = (H - (y1 - y0) * sc) / 2;
  const P = (la, lo) => [ox + (lo * k - x0) * sc, oy + (-la - y0) * sc];
  const f1 = n => n.toFixed(1);
  const ring = r => {
    let out = [], last = null;
    r.forEach(c => { const p = P(c[1], c[0]); if (!last || Math.abs(p[0] - last[0]) + Math.abs(p[1] - last[1]) > 1.2) { out.push(p); last = p; } });
    if (out.length < 3) return '';
    if (out.every(p => p[0] < -20) || out.every(p => p[0] > W + 20) || out.every(p => p[1] < -20) || out.every(p => p[1] > H + 20)) return '';
    return 'M' + out.map(p => f1(p[0]) + ' ' + f1(p[1])).join('L') + 'Z';
  };
  const fillOf = gj => {
    let d = '';
    ((gj && gj.features) || []).forEach(ft => {
      const g = ft.geometry || {};
      if (g.type === 'Polygon') g.coordinates.forEach(r => { d += ring(r); });
      else if (g.type === 'MultiPolygon') g.coordinates.forEach(pg => pg.forEach(r => { d += ring(r); }));
    });
    return d;
  };
  const linesOf = gj => {
    let d = '';
    ((gj && gj.features) || []).forEach(ft => {
      const g = ft.geometry || {};
      const ls = g.type === 'LineString' ? [g.coordinates] : (g.type === 'MultiLineString' ? g.coordinates : []);
      ls.forEach(l => { d += 'M' + l.map(c => { const p = P(c[1], c[0]); return f1(p[0]) + ' ' + f1(p[1]); }).join('L'); });
    });
    return d;
  };
  const pts = cs.map(c => P(c[0], c[1]));
  const seg = (a, b) => 'M' + pts.slice(a, b + 1).map(p => f1(p[0]) + ' ' + f1(p[1])).join('L');
  const runs = o.gr && o.gr.length ? o.gr : [[-1, 0, cs.length - 1]];
  const casing = '<path d="' + seg(0, cs.length - 1) + '" fill="none" stroke="#fff" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>';
  const line = runs.map(r => '<path d="' + seg(r[1], r[2]) + '" fill="none" stroke="' + (r[0] < 0 ? '#4E6E45' : GCOL[r[0]]) + '" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>').join('');
  const st = o.steps || [];
  let marks = '', rows = '';
  st.forEach((s, i) => {
    const last = i === st.length - 1, first = i === 0, p = P(s.lat, s.lon);
    if (!first && !last) marks += '<g><circle cx="' + f1(p[0]) + '" cy="' + f1(p[1]) + '" r="12" fill="#fff" stroke="#3B3A33" stroke-width="2"/><text x="' + f1(p[0]) + '" y="' + f1(p[1] + 5) + '" text-anchor="middle" font-size="14" font-weight="700" fill="#3B3A33">' + i + '</text></g>';
    rows += '<li id="s' + i + '" data-d="' + (s.d || 0) + '"><span class="n">' + (first ? 'S' : (last ? 'Z' : i)) + '</span><span class="ic">' + x(s.i || '') + '</span><span class="t">' + x(s.t) + '</span><span class="d">' + (s.d ? fmt(s.d) : '') + '</span></li>';
  });
  const a = pts[0], b = pts[pts.length - 1];
  const dws = dwOn ? dwNear() : [];
  const dwSvg = dws.map(p => { const q = P(p.la, p.lo); return '<g><circle cx="' + f1(q[0]) + '" cy="' + f1(q[1]) + '" r="8" fill="#2F8FCB" stroke="#fff" stroke-width="2.5"/><circle cx="' + f1(q[0]) + '" cy="' + f1(q[1]) + '" r="2.6" fill="#fff"/></g>'; }).join('');
  const stops = (D.stops || []).map(s => { const p = P(s[0], s[1]); return '<g><circle cx="' + f1(p[0]) + '" cy="' + f1(p[1]) + '" r="9" fill="#B07A35" stroke="#fff" stroke-width="2.5"/><text x="' + f1(p[0] + 13) + '" y="' + f1(p[1] + 6) + '" font-size="18" fill="#3B3A33" stroke="#fff" stroke-width="4" paint-order="stroke">' + x(s[2]) + '</text></g>'; }).join('');
  const loop = Math.hypot(a[0] - b[0], a[1] - b[1]) < 6;
  const pxm = sc / 111320;
  let sb = 100; [100, 200, 250, 500, 1000, 2000].forEach(v => { if (v * pxm <= 150) sb = v; });
  const stats = (o.stats || []).map(r => '<div><small>' + x(r[0]) + '</small><b>' + x(r[1]) + '</b></div>').join('');
  const title = loop ? 'Ründchen ab ' + D.t1 : D.t1 + ' → ' + D.t2;
  const data = JSON.stringify({la0: la0, k: k, x0: x0, y0: y0, sc: sc, ox: ox, oy: oy, W: W, H: H, c: cs.map(c => [+c[0].toFixed(5), +c[1].toFixed(5)]), cum: o.cum});
  const css = 'body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#F4EFE6;color:#3B3A33}main{max-width:820px;margin:0 auto;padding:14px}h1{font-size:1.25rem;margin:4px 0}.sub{opacity:.7;font-size:.85rem;margin-bottom:10px}'
    + '.st{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.st div{background:#fff;border:1px solid #D9CFBC;border-radius:12px;padding:6px 10px}.st small{display:block;font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;opacity:.65}'
    + 'svg{width:100%;height:auto;background:#EFE9DB;border:1px solid #D9CFBC;border-radius:14px}ol{list-style:none;padding:0;margin:12px 0}li{display:flex;gap:10px;align-items:center;padding:9px 6px;border-bottom:1px solid #E4DBC8}'
    + 'li.cur{background:#E7EDDB;border-radius:10px;font-weight:700}.n{width:24px;height:24px;border-radius:50%;background:#fff;border:1.6px solid #3B3A33;text-align:center;font-size:.72rem;line-height:21px;flex:none}.ic{width:26px;text-align:center;font-size:1.2rem;flex:none}.t{flex:1}.d{opacity:.65;font-size:.85rem;white-space:nowrap}'
    + '.leg{display:flex;gap:12px;flex-wrap:wrap;font-size:.8rem;margin:6px 0}.leg i{display:inline-block;width:18px;height:5px;border-radius:3px;margin-right:5px;vertical-align:middle}.foot{font-size:.72rem;opacity:.65;margin-top:14px}#gps{font-size:.85rem;margin:6px 0;opacity:.85}@media print{#gps{display:none}}';
  const leg = (o.gr && o.gr.length ? '<div class="leg">' + [0, 1, 2].map(i => '<span><i style="background:' + GCOL[i] + '"></i>' + GNAME[i] + '</span>').join('')
    + (dws.length ? '<span><i style="background:#2F8FCB;height:10px;width:10px;border-radius:50%"></i>Trinkwasser</span>' : '') + '</div>'
    : (dws.length ? '<div class="leg"><span><i style="background:#2F8FCB;height:10px;width:10px;border-radius:50%"></i>Trinkwasser</span></div>' : ''))
    + (dws.length ? '<div class="sub">🚰 ' + dws.length + ' Trinkwasserstelle' + (dws.length > 1 ? 'n' : '') + ' bis ' + DW_NEAR + ' m neben der Route, nächste ca. ' + Math.round(dws[0].d / 10) * 10 + ' m (Stand ' + new Date().toLocaleDateString('de-DE') + ', oft nur im Sommer in Betrieb)</div>' : '');
  const js = '(function(){var D=' + data + ',dot=document.getElementById("dot"),g=document.getElementById("gps"),li=document.querySelectorAll("li");'
    + 'function hav(a,b,c,d){var R=6371000,p=Math.PI/180,u=(c-a)*p,v=(d-b)*p,h=Math.sin(u/2)*Math.sin(u/2)+Math.cos(a*p)*Math.cos(c*p)*Math.sin(v/2)*Math.sin(v/2);return 2*R*Math.asin(Math.sqrt(h))}'
    + 'function upd(p){var la=p.coords.latitude,lo=p.coords.longitude,px=D.ox+(lo*D.k-D.x0)*D.sc,py=D.oy+(-la-D.y0)*D.sc,bi=0,bd=1e12;'
    + 'D.c.forEach(function(c,i){var d=hav(la,lo,c[0],c[1]);if(d<bd){bd=d;bi=i}});'
    + 'if(px>-20&&px<D.W+20&&py>-20&&py<D.H+20){dot.setAttribute("cx",px);dot.setAttribute("cy",py);dot.style.display="block"}else dot.style.display="none";'
    + 'var al=D.cum[bi]||0,nx=-1;li.forEach(function(e,i){e.className="";if(nx<0&&+e.getAttribute("data-d")>al-8)nx=i});'
    + 'if(nx>=0){li[nx].className="cur"}'
    + 'g.textContent=bd>150?"Du bist ca. "+Math.round(bd)+" m von der Route entfernt.":"Auf der Route · noch ca. "+Math.max(0,Math.round(D.cum[D.cum.length-1]-al))+" m";}'
    + 'if(navigator.geolocation){g.textContent="GPS wird gesucht …";navigator.geolocation.watchPosition(upd,function(){g.textContent="GPS nicht verfügbar. Die Skizze und die Liste funktionieren trotzdem."},{enableHighAccuracy:true,maximumAge:2000,timeout:20000})}else g.textContent=""})();';
  return '<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + x(title) + ' · GreenWalker offline</title><style>' + css + '</style></head><body><main>'
    + '<h1>' + x(title) + '</h1><div class="sub">GreenWalker · ' + x(o.label || '') + ' · Offline-Blatt, funktioniert ohne Netz</div><div class="st">' + stats + '</div>'
    + '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Skizze der Route">'
    + '<path d="' + fillOf(D.parks) + '" fill="#CFE0B8" fill-rule="evenodd" stroke="#A9C58C" stroke-width="1"/>'
    + '<path d="' + fillOf(D.water) + '" fill="#BCD9EA" fill-rule="evenodd" stroke="#8DBBD6" stroke-width="1"/>'
    + '<path d="' + linesOf(D.water) + '" fill="none" stroke="#8DBBD6" stroke-width="2.5"/>'
    + casing + line + marks + stops + dwSvg
    + '<circle cx="' + f1(a[0]) + '" cy="' + f1(a[1]) + '" r="9" fill="#4E6E45" stroke="#fff" stroke-width="3"/>'
    + (loop ? '' : '<circle cx="' + f1(b[0]) + '" cy="' + f1(b[1]) + '" r="9" fill="#B8571A" stroke="#fff" stroke-width="3"/>')
    + '<circle id="dot" cx="0" cy="0" r="8" fill="#1E6FE0" stroke="#fff" stroke-width="3" style="display:none"/>'
    + '<g transform="translate(' + (W - 30) + ',34)"><path d="M0 -18 L8 8 L0 3 L-8 8Z" fill="#3B3A33"/><text y="26" text-anchor="middle" font-size="16" font-weight="700" fill="#3B3A33">N</text></g>'
    + '<g transform="translate(' + M + ',' + (H - 20) + ')"><path d="M0 0H' + f1(sb * pxm) + '" stroke="#3B3A33" stroke-width="3"/><text x="0" y="-9" font-size="16" fill="#3B3A33" stroke="#EFE9DB" stroke-width="4" paint-order="stroke">' + (sb >= 1000 ? (sb / 1000) + ' km' : sb + ' m') + '</text></g>'
    + '</svg>' + leg + '<div id="gps"></div><ol>' + rows + '</ol>'
    + '<div class="foot">Kartendaten © OpenStreetMap-Mitwirkende (ODbL). Skizze ohne Kartenhintergrund. Die Route folgt dem, was in OpenStreetMap erfasst ist: Bitte vor Ort auf Hinweise achten.</div>'
    + '<scr' + 'ipt>' + js + '</scr' + 'ipt></main></body></html>';
}
// Standort teilen: Link mit aktueller Position und Restzeit (einmalig, kein Live-Tracking)
function locText(p) {
  const url = 'https://www.openstreetmap.org/?mlat=' + p[0].toFixed(5) + '&mlon=' + p[1].toFixed(5) + '#map=17/' + p[0].toFixed(5) + '/' + p[1].toFixed(5);
  let t = (D.dog ? (D.chayName || 'Der Hund') + ' und ich sind' : 'Ich bin') + ' gerade hier unterwegs: ' + url;
  if (navOn && !finished && total > 0) {
    const rem = Math.max(0, total - along), mins = Math.max(1, Math.round(rem / (D.speed || 80)));
    t += '\nNoch ' + fmt(rem) + ' bis ' + (D.t2 || 'zum Ziel') + ', ca. ' + mins + ' min.';
  } else if (finished) t += '\nIch bin angekommen.';
  return t + '\n– GreenWalker';
}
function showShareBox(t) {
  $('sbta').value = t; $('sbox').classList.add('on');
  try { $('sbta').focus(); $('sbta').select(); } catch (e) {}
}
async function shareLoc() {
  let p = lastPos;
  if (!p && navigator.geolocation) {
    tip('Standort wird ermittelt …');
    p = await new Promise(res => navigator.geolocation.getCurrentPosition(q => res([q.coords.latitude, q.coords.longitude]), () => res(null), {enableHighAccuracy: true, timeout: 8000, maximumAge: 30000}));
  }
  if (!p) { tip('Standort nicht verfügbar. Bitte Freigabe erlauben.'); return; }
  const t = locText(p);
  try { if (navigator.share) { await navigator.share({title: 'Mein Standort', text: t}); return; } } catch (e) { if (e && e.name === 'AbortError') return; }
  try { await navigator.clipboard.writeText(t); tip('Text mit Standort kopiert'); return; } catch (e) {}
  showShareBox(t);
}
function backToApp() {  // „←“-Knopf der App auslösen (Zurück zur Startseite)
  try {
    const b = parent.document.querySelector('.st-key-gw_back button') || parent.document.querySelector('[class*="st-key-gw_back"] button');
    if (b) { b.click(); return true; }
  } catch (e) {}
  return false;
}
let homeArmed = false;
$('home').addEventListener('click', () => {
  if (navOn && !finished && !homeArmed) { homeArmed = true; tip('Nochmal tippen: Navigation beenden, zurück zum Start'); setTimeout(() => { homeArmed = false; }, 3200); return; }
  homeArmed = false;
  if (!backToApp()) tip('Bitte oben links ← tippen');
});
function toggleMenu(on) { $('smenu').classList.toggle('on', on === undefined ? !$('smenu').classList.contains('on') : on); }
$('gpx').addEventListener('click', e => { e.stopPropagation(); toggleMenu(); });
document.addEventListener('click', e => { if (!e.target.closest || !e.target.closest('#smenu')) toggleMenu(false); });
$('m_gpx').addEventListener('click', () => { toggleMenu(false); saveGpx(); });
$('m_off').addEventListener('click', () => { toggleMenu(false); saveOffline(); });
$('m_loc').addEventListener('click', () => { toggleMenu(false); shareLoc(); });
$('sbclose').addEventListener('click', () => { $('sbox').classList.remove('on'); });
$('gpxb').addEventListener('click', saveGpx);
$('thxb').addEventListener('click', () => {
  $('thanks').classList.remove('on');
  // Zurück zur Startseite: den „←“-Knopf der App auslösen (gwdone ist gesetzt, es geht ohne Vorauswahl zurück)
  try {
    const b = parent.document.querySelector('.st-key-gw_back button') || parent.document.querySelector('[class*="st-key-gw_back"] button');
    if (b) { b.click(); return; }
  } catch (e) {}
  document.body.classList.remove('thx');
});
function finish() {
  finished = true; stopWatch();
  setTimeout(showThanks, 1800);
  banner('done', '🏁', 'Geschafft!', steps[steps.length - 1].t); say(steps[steps.length - 1].t);
  $('rem').textContent = 'Ziel erreicht'; $('prog').firstElementChild.style.width = '100%';
}
function onFix(lat, lon, acc, gpsHeading, speed) {
  if (finished || !navOn) return;
  lastPos = [lat, lon];
  const xy = toXY(lat, lon);
  let s = snap(xy, along, true);
  if (s.off > 60) { const g = snap(xy, 0, false); if (g.off < s.off * 0.6) s = g; }
  lastIdx = s.i; along = s.along;
  userDot.setLatLng(lastPos); if (!map.hasLayer(userDot)) userDot.addTo(map);
  if (chayMk) { chayMk.setLatLng(lastPos); if (!map.hasLayer(chayMk)) chayMk.addTo(map); }  // Chay läuft mit
  // Richtung: GPS-Kurs (nur in Bewegung), sonst aus den letzten Positionen, sonst Kompass (onOrient)
  let h = null;
  if (typeof gpsHeading === 'number' && !isNaN(gpsHeading) && (speed || 0) > 0.7) h = gpsHeading;
  else if (prevFix && Math.hypot((lat - prevFix[0]) * 110540, (lon - prevFix[1]) * 111320 * Math.cos(lat * Math.PI / 180)) >= 5) h = bearingLL(prevFix, [lat, lon]);
  if (h !== null) { prevFix = [lat, lon]; hdgGps = Date.now(); updateCone(h); }
  else if (!prevFix) prevFix = [lat, lon];
  else updateCone(hdg);
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
  const endD = Math.hypot(xy[0] - P[P.length - 1][0], xy[1] - P[P.length - 1][1]);
  if (!far && rem < 20 && along > total * 0.85 && endD < Math.max(30, (acc || 0) * 1.2)) { finish(); return; }
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
  stopCompass();
  if (watchId !== null) { navigator.geolocation.clearWatch(watchId); watchId = null; }
  if (sim) { clearInterval(sim); sim = null; $('demo').classList.remove('on'); }
  try { if (wake) wake.release(); } catch (e) {} wake = null;
}
function enterNav() {
  navOn = true; finished = false; said = {}; hadFix = false; lastIdx = 0; along = 0; offCount = 0; navT0 = Date.now(); demoRun = false;
  document.body.classList.add('nav'); $('go').textContent = '⏹ Beenden';
  if (map.hasLayer(startPin)) map.removeLayer(startPin);
  if (chayMk && map.hasLayer(chayMk)) map.removeLayer(chayMk);
  try { if (navigator.wakeLock) navigator.wakeLock.request('screen').then(l => { wake = l; }).catch(() => {}); } catch (e) {}
  say('Navigation gestartet.');
  banner('', '📡', '…', 'Standort wird gesucht');
}
function exitNav() {
  stopWatch(); navOn = false; finished = false; $('thanks').classList.remove('on');
  document.body.classList.remove('fin', 'thx'); doneFlag(false);
  document.body.classList.remove('nav'); $('go').textContent = GO;
  if (!map.hasLayer(startPin)) startPin.addTo(map);
  if (chayMk) { chayMk.setLatLng(D.p1); if (!map.hasLayer(chayMk)) chayMk.addTo(map); }
  [userDot, userAcc, userCone].forEach(l => { if (map.hasLayer(l)) map.removeLayer(l); });
  if (done) done.setLatLngs([]);
  showOption(cur, true);
}
function startGps() {
  enterNav(); startCompass();
  if (!navigator.geolocation) { banner('off', '📡', 'Kein GPS', 'Dieses Gerät unterstützt keine Standortabfrage.'); return; }
  watchId = navigator.geolocation.watchPosition(
    p => onFix(p.coords.latitude, p.coords.longitude, p.coords.accuracy, p.coords.heading, p.coords.speed),
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
  enterNav(); demoRun = true; $('demo').classList.add('on'); follow = true; $('fbtn').classList.add('on');
  let s = 0;
  sim = setInterval(() => {
    s = Math.min(total, s + 10); const p = pointAt(s); onFix(p[0], p[1], 5, null, 0);
    if (s >= total || finished) { clearInterval(sim); sim = null; $('demo').classList.remove('on'); }
  }, 500);
}

$('chips').innerHTML = D.options.map((o, i) => '<button class="chip" data-i="' + i + '"><b>' + esc(o.label) + '</b><span>' + esc(o.sub) + '</span>' + (o.gb ? '<div class="gbar">' + o.gb.map((v, k) => '<i style="width:' + v + '%;background:' + GCOL[k] + '"></i>').join('') + '</div>' : '') + '</button>').join('');
$('chips').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b && !navOn) showOption(+b.dataset.i, true); });
$('ttl').textContent = (D.p1[0] === D.p2[0] && D.p1[1] === D.p2[1]) ? '🔄 Ründchen ab ' + D.t1 : '🚶 ' + D.t1 + '  →  🏁 ' + D.t2;
$('go').textContent = GO;
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
$('layers').addEventListener('click', () => {
  tTried = new Set();
  const k = BASE.indexOf(tIdx);
  setTiles(k < 0 ? BASE[0] : BASE[(k + 1) % BASE.length]); tip('Kartenstil: ' + TILES[tIdx].n);
});
$('sat').addEventListener('click', () => {
  tTried = new Set();
  const k = SAT.indexOf(tIdx);
  const j = k < 0 ? SAT[0] : (k + 1 < SAT.length ? SAT[k + 1] : BASE[0]);  // Satellit -> Luftbild -> Sentinel -> zurück zur Karte
  setTiles(j); tip(TILES[j].sat ? '🛰️ ' + TILES[j].n : 'Kartenstil: ' + TILES[j].n);
});
$('fit').addEventListener('click', fitRoute);
map.on('dragstart', () => { follow = false; $('fbtn').classList.remove('on'); });

function fix() { map.invalidateSize({animate: false}); }
window.addEventListener('resize', fix);
if (window.visualViewport) window.visualViewport.addEventListener('resize', fix);
if (window.ResizeObserver) new ResizeObserver(fix).observe($('map'));
showOption(D.default || 0, false);
if (D.san && ((D.san.pub || []).length || (D.san.cust || []).length)) {
  const wcLayer = L.layerGroup().addTo(map);
  const wcIcon = c => L.divIcon({className: '', iconSize: [30, 30], iconAnchor: [15, 15],
    html: '<div style="width:30px;height:30px;border-radius:50%;background:' + c + ';border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;font-size:16px">🚻</div>'});
  const addWc = (arr, color, label) => (arr || []).forEach(t => {
    L.marker([t[0], t[1]], {icon: wcIcon(color), zIndexOffset: 300}).addTo(wcLayer)
      .bindPopup('<b>' + esc(t[2]) + '</b><br><i>' + label + '</i>' + (t[3] ? '<br>' + esc(t[3]) : ''));
  });
  addWc(D.san.pub, '#2F6F8F', 'Öffentliche Toilette');
  addWc(D.san.cust, '#E0A526', 'Kundentoilette / Gaststätte');
  const wcb = $('wcbtn'); wcb.style.display = ''; let wcOn = true;
  wcb.addEventListener('click', () => { wcOn = !wcOn; wcOn ? wcLayer.addTo(map) : map.removeLayer(wcLayer); wcb.style.opacity = wcOn ? 1 : .45; });
}
// Trinkwasser-Ebene: Trinkbrunnen aus OpenStreetMap, beim Antippen direkt vom Gerät bei Overpass geholt (24 h gemerkt)
const OVP = ['https://overpass-api.de/api/interpreter', 'https://z.overpass-api.de/api/interpreter',
             'https://overpass.kumi.systems/api/interpreter', 'https://overpass.private.coffee/api/interpreter'];
const DW_NEAR = 300;
function dwBox() {
  let s = 90, w = 180, n = -90, e = -180;
  D.options.forEach(o => (o.coords || []).forEach(c => { s = Math.min(s, c[0]); n = Math.max(n, c[0]); w = Math.min(w, c[1]); e = Math.max(e, c[1]); }));
  const pad = 0.004;  // ca. 400 m
  return [s - pad, w - pad, n + pad, e + pad];
}
function segDistM(la, lo, a, b, kxl) {
  const ky = 110540, px = lo * kxl, py = la * ky, ax = a[1] * kxl, ay = a[0] * ky, bx = b[1] * kxl, by = b[0] * ky;
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
function distToRoute(la, lo, cs) {
  const kxl = 111320 * Math.cos(la * Math.PI / 180);
  let best = 1e9;
  for (let i = 0; i < cs.length - 1; i++) {
    if (Math.abs(cs[i][0] - la) > 0.01 && Math.abs(cs[i + 1][0] - la) > 0.01) continue;  // weit weg: überspringen
    best = Math.min(best, segDistM(la, lo, cs[i], cs[i + 1], kxl));
  }
  return best;
}
async function fetchDw(bb) {
  const key = 'gw_dw_' + bb.map(v => v.toFixed(3)).join('_');
  try {
    const c = JSON.parse(localStorage.getItem(key) || 'null');
    if (c && Date.now() - c.t < 86400000) return c.p;
  } catch (e) {}
  const b = '(' + bb.map(v => v.toFixed(5)).join(',') + ')';
  const q = '[out:json][timeout:20];(nwr["amenity"="drinking_water"]' + b + ';nwr["man_made"="water_tap"]["drinking_water"="yes"]' + b + ';);out center 300;';
  let lastErr = null;
  for (const u of OVP) {
    try {
      const ctl = new AbortController(), tm = setTimeout(() => ctl.abort(), 12000);
      const r = await fetch(u, {method: 'POST', body: 'data=' + encodeURIComponent(q), headers: {'Content-Type': 'application/x-www-form-urlencoded'}, signal: ctl.signal});
      clearTimeout(tm);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const js = await r.json();
      const pts = (js.elements || []).map(el => {
        const t = el.tags || {}, la = el.lat !== undefined ? el.lat : (el.center && el.center.lat), lo = el.lon !== undefined ? el.lon : (el.center && el.center.lon);
        if (la === undefined || lo === undefined || t.drinking_water === 'no' || t.access === 'private') return null;
        return {la: la, lo: lo, name: t.name || '', seasonal: !!t.seasonal && t.seasonal !== 'no', dog: t.dog === 'yes' || t['drinking_water:dog'] === 'yes'};
      }).filter(Boolean);
      try { localStorage.setItem(key, JSON.stringify({t: Date.now(), p: pts})); } catch (e) {}
      return pts;
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error('Overpass nicht erreichbar');
}
function dwNear() {
  const o = D.options[Math.max(cur, 0)] || {};
  return dwPts.map(p => ({...p, d: distToRoute(p.la, p.lo, o.coords || [])})).filter(p => p.d <= DW_NEAR).sort((a, b) => a.d - b.d);
}
function dwRender() {
  if (!dwLayer) dwLayer = L.layerGroup();
  dwLayer.clearLayers();
  const near = dwNear();
  near.forEach(p => {
    L.marker([p.la, p.lo], {zIndexOffset: 250, icon: L.divIcon({className: '', iconSize: [30, 30], iconAnchor: [15, 15],
      html: '<div style="width:30px;height:30px;border-radius:50%;background:#2F8FCB;border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;font-size:16px">🚰</div>'})})
      .addTo(dwLayer).bindPopup('<b>' + esc(p.name || 'Trinkwasser') + '</b><br>ca. ' + Math.round(p.d / 10) * 10 + ' m von der Route'
        + (p.dog ? '<br>🐕 für Hunde ausgewiesen' : '') + (p.seasonal ? '<br><i>saisonal: im Winter oft abgestellt</i>' : ''));
  });
  if (!map.hasLayer(dwLayer)) dwLayer.addTo(map);
  return near;
}
async function toggleDw() {
  const b = $('dwbtn');
  if (dwOn) { dwOn = false; if (dwLayer) map.removeLayer(dwLayer); b.style.opacity = 1; b.classList.remove('on'); return; }
  if (dwBusy) return;
  dwBusy = true; tip('🚰 Trinkwasserstellen werden gesucht …');
  try {
    if (!dwPts.length) dwPts = await fetchDw(dwBox());
    dwOn = true; b.classList.add('on');
    const near = dwRender();
    const mo = new Date().getMonth(), winter = mo >= 10 || mo <= 2;
    tip(near.length ? '🚰 ' + near.length + ' Trinkwasserstelle' + (near.length > 1 ? 'n' : '') + ' bis ' + DW_NEAR + ' m neben der Route, nächste ca. ' + Math.round(near[0].d / 10) * 10 + ' m'
      + (winter ? '. Im Winter sind viele abgestellt.' : '')
      : '🚰 Keine Trinkwasserstelle bis ' + DW_NEAR + ' m neben der Route in OpenStreetMap gefunden', 4200);
  } catch (e) { tip('🚰 Trinkwasser konnte nicht geladen werden. Bitte später nochmal.', 3000); }
  dwBusy = false;
}
$('dwbtn').addEventListener('click', toggleDw);
const _so0 = showOption; showOption = function(i, f) { _so0(i, f); if (dwOn) dwRender(); };
if (window.INTRO) {
  // Anflug: erst weit herausgezoomt, dann sanfter Flug auf die Route, während die Wolken aufreißen
  setTimeout(() => {
    fix(); fitRoute();
    const c = map.getCenter(), z = map.getZoom();
    map.setView(c, Math.max(z - 3, 3), {animate: false});
    setTimeout(() => { map.flyTo(c, z, {duration: 2.4, easeLinearity: .2}); INTRO_DONE = true; }, 450);
    setTimeout(() => { fix(); }, 3200);
  }, 120);
} else {
  setTimeout(() => { fix(); if (!navOn) fitRoute(); }, 150);
  setTimeout(() => { fix(); if (!navOn) fitRoute(); }, 900);
}
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
    g = edge_green(area, buf)
    L, H = area["lengths"], area["H"]
    nz = area.get("noise")
    nz = np.zeros(len(L)) if nz is None else nz
    wd = area.get("wide")
    wd = np.zeros(len(L)) if wd is None else wd
    ap = area_accp(area)

    def run(w):
        path = fast_path(area, w, o, t)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum()) if idx else 0.0
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        noise = float((L[idx] * nz[idx]).sum() / dist) if dist > 0 else 0.0
        narrow = float((L[idx] * (wd[idx] <= NARROW)).sum() / dist) if dist > 0 else 0.0
        return {"path": path, "dist": dist, "green": share, "noise": noise, "narrow": narrow}

    def cost(lam, mu, wq):
        return L * (1 + lam * (1 - g) + mu * nz + wq * wd + ap)

    short = run(L * (1 + ap))
    out = []
    for name, lam in LEVELS.items():
        mu, wq = level_quiet(name)
        if lam >= 30:
            # „Grünster Umweg“: darf deutlich länger sein, aber nicht ausufern und muss wirklich grüner sein
            best = max((r_ for _, r_ in out), key=lambda r_: r_["green"])
            r = None
            for f in (1.0, 0.5, 0.25):
                cand = run(cost(lam * f, mu, wq))
                if cand["dist"] <= 3.0 * short["dist"] + 600:
                    r = cand
                    break
            if r is None or r["path"] == best["path"] or r["green"] < best["green"] + 0.01:
                continue
            out.append((name, r))
            continue
        out.append((name, short if lam == 0 else run(cost(lam, mu, wq))))
    return short, out


def solve_extreme(area, o, t, buf, short):
    """Route A -> B „Extrem grün“: grünste Strecke mit höchstens 35 % Umweg. Nur mit mehr als 70 % Grün, sonst None."""
    g = edge_green(area, buf)
    L, H = area["lengths"], area["H"]
    nz = area.get("noise")
    nz = np.zeros(len(L)) if nz is None else nz
    wd = area.get("wide")
    wd = np.zeros(len(L)) if wd is None else wd
    ap = area_accp(area)

    def run(lam, mu, wq):
        path = fast_path(area, L * (1 + lam * (1 - g) + mu * nz + wq * wd + ap), o, t)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum()) if idx else 0.0
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        noise = float((L[idx] * nz[idx]).sum() / dist) if dist > 0 else 0.0
        narrow = float((L[idx] * (wd[idx] <= NARROW)).sum() / dist) if dist > 0 else 0.0
        return {"path": path, "dist": dist, "green": share, "noise": noise, "narrow": narrow}

    best = greenest_within(run, (1 + EXTREME_DETOUR) * short["dist"])
    return best if best and best["green"] > EXTREME_GREEN_MIN else None


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


def dogz_geo(area):
    """Hunde-Verbotszonen als GeoJSON für die Karte (nur im Hundemodus)."""
    if not _dog() or not area or not area.get("dogz_json"):
        return None
    g = area.get("_dogz")
    if g is None:
        try:
            g = json.loads(area["dogz_json"])
        except Exception:
            g = {}
        area["_dogz"] = g
    return g or None


def map_html(payload):
    if _dog():
        payload["dog"] = True
        payload["chay"] = chay_icon_uri(72)
        payload["chayName"] = DOG_NAME
        if payload.get("dogz") is None:
            payload.pop("dogz", None)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return MAP_TEMPLATE.replace("__DATA__", data)


def dog_forbidden_tags(t):
    """OSM-Tags einer Fläche: ist sie für Hunde (meist) verboten?"""
    return (t.get("landuse") == "cemetery" or t.get("amenity") == "grave_yard" or t.get("leisure") in ("playground", "zoo")
            or t.get("tourism") == "zoo" or str(t.get("dog", "")).lower() in ("no", "prohibited"))


@st.cache_data(ttl=900, show_spinner=False)
def dog_zone_rings(s, w, n, e):
    """Hunde-Verbotszonen in einem Rechteck als [(Name, Ring [(lon, lat), ...])] (Overpass, mit Geometrie)."""
    b = f"({s},{w},{n},{e})"
    q = ("[out:json][timeout:20];(" + "".join(f'way["{k}"="{v}"]{b};relation["{k}"="{v}"]{b};' for k, v in DOG_ZONE_TAGS) + ");out geom;")
    data = _overpass_race(q, 22.0, _ranked_mirrors()[:4])
    rings = []
    for el in data.get("elements", []):
        t = el.get("tags", {})
        name = next((DOG_ZONE_LABEL[v] for k, v in DOG_ZONE_TAGS if t.get(k) == v and v in DOG_ZONE_LABEL), "Hundeverbot")
        geoms = []
        if el.get("type") == "way" and el.get("geometry"):
            geoms.append(el["geometry"])
        for m in el.get("members", []) or []:
            if m.get("role") == "outer" and m.get("geometry"):
                geoms.append(m["geometry"])
        for g in geoms:
            ring = [(p["lon"], p["lat"]) for p in g]
            if len(ring) >= 4:
                rings.append((name, ring))
    return rings


_DZ_CACHE = {}
_DZ_EX = ThreadPoolExecutor(max_workers=3)


def dog_zone_rings_async(s, w, n, e):
    """Verbotszonen im Rechteck als Future. Ein schon laufender oder fertiger Abruf, der das Rechteck abdeckt, wird
    mitbenutzt: So kann er starten, während die Route noch berechnet wird."""
    now = time.time()
    for k, (t, f) in list(_DZ_CACHE.items()):
        if now - t < 1800 and k[0] <= s and k[1] <= w and k[2] >= n and k[3] >= e and not (f.done() and f.exception()):
            return f
    f = _DZ_EX.submit(dog_zone_rings, s, w, n, e)
    if len(_DZ_CACHE) > 20:
        _DZ_CACHE.clear()
    _DZ_CACHE[(s, w, n, e)] = (now, f)
    return f


def dog_zones_prefetch(p1, p2, pad=0.02):
    """Schon beim Start der Berechnung die Verbotszonen zwischen Start und Ziel holen (nur Hundemodus)."""
    try:
        if _dog():
            s, n = min(p1[0], p2[0]) - pad, max(p1[0], p2[0]) + pad
            w, e = min(p1[1], p2[1]) - pad, max(p1[1], p2[1]) + pad
            dog_zone_rings_async(math.floor(s * 100) / 100, math.floor(w * 100) / 100,
                                 math.ceil(n * 100) / 100, math.ceil(e * 100) / 100)
    except Exception:
        pass


def dog_osrm_check(coords):
    """Läuft eine Route (Liste [lat, lon]) durch eine bekannte Hunde-Verbotszone? -> (Meter, Linien, Namen)."""
    if not coords or len(coords) < 2:
        return 0.0, [], []
    la = [c[0] for c in coords]
    lo = [c[1] for c in coords]
    pad = 0.0015
    s, w = math.floor((min(la) - pad) * 100) / 100, math.floor((min(lo) - pad) * 100) / 100
    n, e = math.ceil((max(la) + pad) * 100) / 100, math.ceil((max(lo) + pad) * 100) / 100
    rings = dog_zone_rings_async(round(s, 2), round(w, 2), round(n, 2), round(e, 2)).result(timeout=16)
    if not rings:
        return 0.0, [], []
    boxes = [(min(p[0] for p in r), min(p[1] for p in r), max(p[0] for p in r), max(p[1] for p in r)) for _, r in rings]
    flag, names = [], set()
    for lat, lon in coords:
        hit = None
        for (nm, r), (x0, y0, x1, y1) in zip(rings, boxes):
            if x0 <= lon <= x1 and y0 <= lat <= y1 and _in_poly(lon, lat, r):
                hit = nm
                break
        flag.append(hit)
        if hit:
            names.add(hit)
    runs, meters, i = [], 0.0, 0
    ky = 110540.0
    while i < len(coords):
        if flag[i]:
            j = i
            while j + 1 < len(coords) and flag[j + 1]:
                j += 1
            seg = [list(c) for c in coords[max(0, i - 1):min(len(coords), j + 2)]]
            if len(seg) >= 2:
                runs.append(seg)
                kx = 111320.0 * math.cos(math.radians(seg[0][0]))
                meters += sum(math.hypot((q[0] - p[0]) * ky, (q[1] - p[1]) * kx) for p, q in zip(seg[:-1], seg[1:]))
            i = j + 1
        else:
            i += 1
    return meters, runs[:60], sorted(names)


def end_context(area, path):
    """Was liegt am Ende der Route? "park" (im bzw. am Park), "water" (am Wasser) oder "" – aus den Kartendaten, für die
    Abschlussnachricht. Endet die Route dort, wo sie begann (Ründchen, Zuhause), zählt nur „mitten im Park“."""
    try:
        if area is None or not path or len(path) < 2:
            return ""
        i = area["H"][path[-2]][path[-1]]["i"]
        pos = area["pos"]
        home = math.dist(pos[path[0]], pos[path[-1]]) < 80
        if park_inside(area)[i]:
            return "park"
        if home:
            return ""
        if float((area["D"][:, i] <= 30).mean()) >= 0.5:
            return "park"
        w = edge_water(area)
        if w is not None and float(w[i]) > WATER_NEAR:
            return "water"
    except Exception:
        pass
    return ""


GREEN_CLASS_HI, GREEN_CLASS_MID = 0.6, 0.25  # Grünscore einer Kante: viel Grün / etwas Grün / Straße


def green_runs(area, path, nidx, buf=150):
    """Route nach Grünanteil einfärben: Abschnitte [Klasse, Index von, Index bis] (Klasse 0 viel Grün, 1 etwas, 2 Straße)
    als Indizes in die Routenlinie (spart Daten) und der Längenanteil je Klasse in Prozent."""
    if not nidx or len(nidx) != len(path):
        return [], []
    H, L = area["H"], area["lengths"]
    idx = np.array([H[a][b]["i"] for a, b in zip(path[:-1], path[1:])], dtype=int)
    if not len(idx):
        return [], []
    g = edge_green(area, buf)[idx]
    cls = np.where(g >= GREEN_CLASS_HI, 0, np.where(g >= GREEN_CLASS_MID, 1, 2))
    ln = L[idx]
    runs = []  # [Klasse, i0, i1, Länge]
    for k in range(len(idx)):
        c, i0, i1 = int(cls[k]), int(nidx[k]), int(nidx[k + 1])
        if runs and runs[-1][0] == c:
            runs[-1][2] = i1
            runs[-1][3] += float(ln[k])
        else:
            runs.append([c, i0, i1, float(ln[k])])
    merged = []  # ganz kurze Abschnitte (unter 40 m) in den Vorgänger aufnehmen, sonst wird die Linie unruhig
    for r in runs:
        if merged and r[3] < 40.0:
            merged[-1][2] = r[2]
            merged[-1][3] += r[3]
        elif merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
            merged[-1][3] += r[3]
        else:
            merged.append(r)
    tot = float(ln.sum()) or 1.0
    share = [round(100 * float(ln[cls == c].sum()) / tot) for c in range(3)]
    return [[c, i0, i1] for c, i0, i1, _ in merged], share


def lit_share(area, path):
    """Anteil der Strecke (nach Länge) mit lit=yes bzw. lit=no; der Rest ist in OpenStreetMap nicht erfasst."""
    lit = area.get("lit")
    if lit is None or not path or len(path) < 2:
        return None
    idx = [area["H"][a][b]["i"] for a, b in zip(path[:-1], path[1:])]
    L = area["lengths"][idx]
    tot = float(L.sum())
    if tot <= 0:
        return None
    c = np.asarray(lit)[idx]
    return float(L[c == 1].sum() / tot), float(L[c == 2].sum() / tot)


def _option(nav, label, sub, hint, stats, area=None, path=None, endctx=None):
    _resolve_weather()
    if _lit() and area is not None and path:
        try:
            ls = lit_share(area, path)
            if ls:
                yes, no = ls
                hint += f" · 💡 {yes:.0%} beleuchtet erfasst" + (f", {no:.0%} unbeleuchtet" if no >= 0.05 else "")
                stats = list(stats) + [["Beleuchtet", f"{yes:.0%}"]]
        except Exception:
            pass
    nidx = nav.get("nidx")
    nav = {k: v for k, v in nav.items() if k != "nidx"}
    gr, gb = [], []
    if area is not None and path and nidx:
        try:
            gr, gb = green_runs(area, path, nidx)
        except Exception:
            gr, gb = [], []
    if endctx is None:
        endctx = end_context(area, path) if (area is not None and path) else ""
    extra = {}
    if _dog() and area is not None and path:
        try:
            di = dog_info(area, path, st.session_state.get("_temp"))
            if di:
                info, segs_ = di
                extra = {"dog": info, "hot": segs_.get("hot", []), "forbid": segs_.get("forbid", [])}
        except Exception:
            extra = {}
    elif _dog() and area is None:
        # Routen ohne eigenes Wegenetz (Kartenserver-Routing): Verbotszonen nachträglich prüfen, Belag ist unbekannt
        try:
            m, runs, names = dog_osrm_check(nav.get("coords") or [])
        except Exception:
            m, runs, names = None, [], []
        info = {"surf": [], "est": 0, "forb": round(m or 0), "leash": 0, "temp": None, "paw": "",
                "nocheck": m is None, "wind": st.session_state.get("_windnote", "") or ""}
        t_ = st.session_state.get("_temp")
        if t_ is not None and t_ >= HOT_C:
            info["paw"] = (f"🐾 Pfotenwarnung bei {t_:.0f} °C: Der Belag dieser Route ist nicht bekannt. In der Stadt sind Wege meist "
                           "Asphalt oder Pflaster und können die Pfoten verbrennen. Handrücken-Test: keine 5 Sekunden Hand auf dem Boden "
                           "halten können heißt zu heiß. Lieber früh morgens oder abends gehen.")
        extra = {"dog": info, "forbid": runs}
    elif not _dog() and st.session_state.get("_windnote"):
        extra = {"wind": st.session_state.get("_windnote")}
    share, segs = dup_from_coords(nav.get("coords") or [])
    if re.search(r"\d+% ohne Doppelweg", hint):
        hint = re.sub(r"\d+% ohne Doppelweg", f"{1 - share:.0%} ohne Doppelweg", hint)
    elif share >= 0.05:
        hint += f" · {share:.0%} Doppelweg"
    if segs and share >= 0.02:
        hint += " · orange markiert = doppelt gelaufen"
    if gr:
        extra = {**extra, "gr": gr, "gb": gb}
    return {**nav, "label": label, "sub": sub, "hint": hint, "stats": stats, "dups": segs, "endctx": endctx, **extra}


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
        prefetch_water(bbox)
        area = load_area_any(bbox)
        bbox = tuple(area.get("bbox") or bbox)
        log(f"✅ {fmt_m(area['n_edges'])} Wegstücke, {area['n_parks']} Grünflächen ({time.time() - t0:.0f} s{_tm_txt(area)})")
    ensure_water(area, log)
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
            nav, name, f"{fmt_km(r['dist'])} · {r['green']:.0%} grün",
            f"{LEVEL_HINT[name]} · Umweg {detour:+.0%}"
            + (f" · {r['narrow']:.0%} kleine Wege" if LEVELS[name] >= 12 else "")
            + (water_note(area, r['path']) if _water() else ""),
            [["Länge", fmt_km(r["dist"])], ["Dauer", f"{mins} min"],
             ["Grünanteil", f"{r['green']:.0%}"], ["Hauptstraße", f"{r['noise']:.0%}"]],
            area=area, path=r["path"]))
    if not options:
        raise ValueError("Start und Ziel liegen zu nah beieinander.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == "Ausgewogen"), 0)
    try:  # Zusatz: „Extrem grün“ (bis 35 % Umweg, über 70 % Grün)
        ext = solve_extreme(area, o, t, buf, short)
        if ext and ext["path"] not in [r_["path"] for _, r_ in levels]:
            nav = build_nav(area, ext["path"], f"{t1} → {t2}")
            if nav:
                detour = ext["dist"] / short["dist"] - 1 if short["dist"] else 0.0
                mins = max(1, round(ext["dist"] / M_PER_MIN))
                options.append(_option(
                    nav, EXTREME_NAME, f"{fmt_km(ext['dist'])} · {ext['green']:.0%} grün",
                    extreme_hint(f" · Umweg {detour:+.0%}") + (f" · {ext['narrow']:.0%} kleine Wege")
                    + (water_note(area, ext['path']) if _water() else ""),
                    [["Länge", fmt_km(ext["dist"])], ["Dauer", f"{mins} min"],
                     ["Grünanteil", f"{ext['green']:.0%}"], ["Hauptstraße", f"{ext['noise']:.0%}"]],
                    area=area, path=ext["path"]))
        elif not ext and short["dist"] > 300:
            options[default]["hint"] += extreme_note(False)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=parks_geo(area), water=water_geo(area), dogz=dogz_geo(area),
                   options=options, default=default, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": dest_v, "area": area, "bbox": bbox, "buf": buf, "water": _water(), "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


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
        # Gleich etwas mehr Fläche laden, als die Wunschlänge braucht: Wer in der Karte „+“ tippt, soll nicht jedes Mal
        # neu auf die Kartenserver warten müssen.
        load_km = load_tier(km)
        log("🗺️ Wegenetz und Grünflächen laden")
        t0 = time.time()
        _lb = loop_bbox(p1, p2, load_km * 1000)
        prefetch_water(_lb)
        area = load_area_any(_lb)
        bbox_km = load_km
        log(f"✅ {fmt_m(area['n_edges'])} Wegstücke, {area['n_parks']} Grünflächen ({time.time() - t0:.0f} s{_tm_txt(area)})")
    if bbox_km < LOOP_MAX_KM:  # nächstgrößeres Gebiet schon im Hintergrund holen
        prefetch_osm(loop_bbox(p1, p2, next(t for t in LOAD_TIERS if t > bbox_km) * 1000))
    ensure_water(area, log)
    o, t = nearest_node(area, *p1), nearest_node(area, *p2)
    log("🔄 Runden in verschiedene Richtungen probieren")
    variants = find_loops(area, o, t, target, LEVELS[level], buf, progress=progress, quiet=level_quiet(level))
    t1, t2 = lab1.split(" · ")[0], lab2.split(" · ")[0]
    options = []
    log("🧭 Wegehinweise und Varianten aufbauen")
    for i, v in enumerate(variants):
        nav = build_nav(area, v["path"], f"Ründchen {i + 1}", loop=True)
        if not nav:
            continue
        mins = max(1, round(v["dist"] / M_PER_MIN))
        diff = v["dist"] / target - 1
        options.append(_option(
            nav, f"Variante {i + 1}", f"{fmt_km(v['dist'])} · {v['green']:.0%} grün",
            f"Wunschlänge {fmt_km(target)} ({diff:+.0%}) · {1 - v['rep']:.0%} ohne Doppelweg"
            + (water_note(area, v['path']) if _water() else ""),
            [["Länge", fmt_km(v["dist"])], ["Dauer", f"{mins} min"],
             ["Grünanteil", f"{v['green']:.0%}"], ["Hauptstraße", f"{v.get('noise', 0):.0%}"]],
            area=area, path=v["path"]))
    if not options:
        raise ValueError("Hier konnte ich keine passende Runde finden. Versuche eine andere Länge "
                         "oder einen anderen Startpunkt.")
    try:  # Zusatz: „Extrem grün“ (bis 35 % länger als gewünscht, über 70 % Grün)
        log("🌿 Extrem grüne Variante suchen")
        ext = extreme_loop(area, o, t, target, buf, variants)
        log("🧭 Extrem grüne Variante aufbauen")
        if ext:
            nav = build_nav(area, ext["path"], "Ründchen extrem grün", loop=True)
            if nav:
                mins = max(1, round(ext["dist"] / M_PER_MIN))
                options.append(_option(
                    nav, EXTREME_NAME, f"{fmt_km(ext['dist'])} · {ext['green']:.0%} grün",
                    f"Wunschlänge {fmt_km(target)} ({ext['dist'] / target - 1:+.0%}) · "
                    + extreme_hint(f" · {1 - ext['rep']:.0%} ohne Doppelweg")
                    + (water_note(area, ext['path']) if _water() else ""),
                    [["Länge", fmt_km(ext["dist"])], ["Dauer", f"{mins} min"],
                     ["Grünanteil", f"{ext['green']:.0%}"], ["Hauptstraße", f"{ext.get('noise', 0):.0%}"]],
                    area=area, path=ext["path"]))
        else:
            options[0]["hint"] += extreme_note(False)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    payload = dict(mode="loop", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=parks_geo(area), water=water_geo(area), dogz=dogz_geo(area),
                   options=options, default=0, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "loop", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "end": end_v, "km": km, "km_ok": km, "level": level, "buf": buf,
            "area": area, "bbox_km": bbox_km, "water": _water(), "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


# ------------------------------------------------------- Auffangnetz ohne Overpass
OSRM_FOOT = "https://routing.openstreetmap.de/routed-foot/route/v1/foot/"
STEP_DE = {
    "left": ("↰", "Links abbiegen"), "right": ("↱", "Rechts abbiegen"),
    "slight left": ("↖", "Halb links halten"), "slight right": ("↗", "Halb rechts halten"),
    "sharp left": ("↰", "Scharf links abbiegen"), "sharp right": ("↱", "Scharf rechts abbiegen"),
    "uturn": ("↩", "Wenden"),
}


def _hav(p, q):
    la1, lo1, la2, lo2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12_742_000 * math.asin(math.sqrt(h))


def osrm_route(p1, p2):
    return osrm_request([p1, p2])[0]


def osrm_nav(route, title):
    """OSRM-Antwort -> dasselbe Format wie build_nav (Linie, Entfernungen, Wegehinweise auf Deutsch)."""
    coords = [[round(y, 6), round(x, 6)] for x, y in route["geometry"]["coordinates"]]
    if len(coords) < 2:
        return None
    cum = [0.0]
    for a, b in zip(coords[:-1], coords[1:]):
        cum.append(cum[-1] + _hav(a, b))
    total = cum[-1]
    if total < 5:
        return None
    legs = route.get("legs") or []
    raw = []
    for li, leg in enumerate(legs):
        for s in leg.get("steps", []):
            typ = (s.get("maneuver") or {}).get("type", "")
            if typ == "depart" and li > 0:
                continue  # Zwischenpunkt: kein neues "Los geht's"
            if typ == "arrive" and li < len(legs) - 1:
                continue  # Zwischenpunkt: noch nicht am Ziel
            raw.append(s)
    sum_raw = sum(float(s.get("distance", 0)) for leg in legs for s in leg.get("steps", [])) or 1.0
    scale = total / sum_raw
    steps, acc = [], 0.0
    for s in raw:
        start_d = acc
        acc += float(s.get("distance", 0))
        m = s.get("maneuver", {})
        typ, mod = m.get("type", ""), m.get("modifier", "")
        name = (s.get("name") or "").strip()
        loc = m.get("location") or [coords[0][1], coords[0][0]]
        on = f" auf {name}" if name else ""
        if typ == "depart":
            b = m.get("bearing_after")
            where = f" Richtung {COMPASS[int(round(b / 45)) % 8]}" if b is not None else ""
            icon, text = "🚶", f"Los geht’s{where}{on}"
        elif typ == "arrive":
            icon, text, start_d = "🏁", "Du hast dein Ziel erreicht", sum_raw
        elif typ in ("roundabout", "rotary", "roundabout turn"):
            ex = m.get("exit")
            icon = "🔄"
            text = (f"Im Kreisverkehr die {ex}. Ausfahrt nehmen" if ex else "In den Kreisverkehr einfahren") + on
        elif mod in ("straight", ""):
            if not name:
                continue
            icon, text = "⬆", f"Weiter geradeaus{on}"
        else:
            icon, text = STEP_DE.get(mod, ("⬆", "Weiter"))
            if typ == "fork" and mod in ("slight left", "slight right"):
                text = "Links halten" if mod == "slight left" else "Rechts halten"
            text += on
        steps.append({"d": round(start_d * scale), "lat": round(loc[1], 6), "lon": round(loc[0], 6),
                      "t": text, "i": icon})
    if not steps or steps[0]["d"] != 0:
        steps.insert(0, {"d": 0, "lat": coords[0][0], "lon": coords[0][1], "t": "Los geht’s", "i": "🚶"})
    if steps[-1]["i"] != "🏁":
        steps.append({"d": round(total), "lat": coords[-1][0], "lon": coords[-1][1],
                      "t": "Du hast dein Ziel erreicht", "i": "🏁"})
    return {"title": title, "total": round(total), "coords": coords,
            "cum": [round(c, 1) for c in cum], "steps": steps}


def osrm_view(start_v, dest_v):
    """Einfache Fußroute (ohne Grün-Daten), wenn die Overpass-Server nicht erreichbar sind."""
    lat1, lon1, lab1 = _unpack(start_v)
    lat2, lon2, lab2 = _unpack(dest_v)
    p1, p2 = (lat1, lon1), (lat2, lon2)
    if dist_m(p1, p2) > 15000:
        raise ValueError("Start und Ziel sind mehr als 15 km auseinander. Bitte näher beieinander wählen.")
    t1, t2 = lab1.split(" · ")[0], lab2.split(" · ")[0]
    nav = osrm_nav(osrm_route(p1, p2), f"{t1} → {t2}")
    if not nav:
        raise ValueError("Start und Ziel liegen zu nah beieinander.")
    mins = max(1, round(nav["total"] / M_PER_MIN))
    opt = _option(nav, "Schnellste", f"{fmt_km(nav['total'])} · ohne Grün",
                  "⚠️ Grün-Server überlastet: einfache Fußroute. Mit 🌿 oben erneut versuchen.",
                  [["Länge", fmt_km(nav["total"])], ["Dauer", f"{mins} min"],
                   ["Grünanteil", "–"], ["Umweg", "–"]])
    opt["warn"] = True
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=None,
                   options=[opt], default=0, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": dest_v, "area": None, "bbox": None, "buf": 150, "fallback": True,
            "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


def make_ab_view_safe(start_v, dest_v, buf=150, area=None, bbox=None, log=lambda *_: None):
    """Grüne Route; wenn die Kartenserver überlastet sind, automatisch die einfache Fußroute."""
    dog_weather(start_v)
    try:
        return make_ab_view(start_v, dest_v, buf, area, bbox, log)
    except ValueError:
        raise
    except Exception as e:
        log("⚠️ OSM-Server überlastet: ich nutze den Grün-Scan")
        try:
            return scan_ab_view(start_v, dest_v, buf, log)
        except ValueError:
            raise
        except Exception:
            pass
        log("⚠️ Grün-Scan nicht möglich: einfache Fußroute")
        try:
            return osrm_view(start_v, dest_v)
        except ValueError:
            raise
        except Exception:
            raise e


# ------------------------------------------------------- Grün-Scan (Ersatz, wenn Overpass nicht antwortet)
# Grünflächen und Hauptstraßen kommen aus freien Vektorkacheln (OpenFreeMap, OpenMapTiles-Schema),
# Wegvarianten vom FOSSGIS-Routenserver. Die Karte wird auf ein feines Raster gezeichnet; daraus
# entstehen zwei Abstandskarten ("wie weit bis zum nächsten Grün", "wie nah an einer Hauptstraße").
# Jede Wegvariante wird alle paar Meter abgetastet und bewertet; die beste je Grünstufe gewinnt.
OFM_TILEJSON = "https://tiles.openfreemap.org/planet"
VT_ZOOM = 14
INSIDE_FACTOR = 0.65      # Wert für „nur am Parkrand entlang“ gegenüber „im Park“ (1.0)
GREEN_FALLOFF = 1.5       # >1: Grün zählt nur nahe am Park viel, nicht schon 100 m davon entfernt
REP_W = 1.4               # Gewicht für Doppelwege (Strecke zweimal gehen) bei der Auswahl von Runden
WATER_SCAN_NEAR = 70.0    # m: so nah am Gewässer gilt ein Wegstück im Scan als „am Wasser“
SCAN_STEP = 10.0          # Abtastabstand entlang einer Route (m)
NOISE_DIST = 28.0         # bis zu diesem Abstand zählt eine Hauptstraße als laut (m)
QUIET = 1.0               # Gewicht "Hauptstraßen meiden" (bei "Schnellste" 0)
PARK_MIN_M2 = 3000        # kleinere Grünflächen werden nicht gezielt angesteuert
ROAD_W = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 0.8, "tertiary": 0.4}
OSM_ROAD_W = {"trunk": 1.0, "trunk_link": 1.0, "primary": 1.0, "primary_link": 1.0,
              "secondary": 0.8, "secondary_link": 0.8, "tertiary": 0.4, "tertiary_link": 0.4}
GREEN_LC = {"wood", "grass", "wetland"}
GREEN_SUB = {"park", "garden", "recreation_ground", "village_green", "allotments", "meadow", "grassland",
             "scrub", "shrubbery", "forest", "wood", "heath", "orchard", "golf_course", "flowerbed",
             "wet_meadow", "plant_nursery"}
GREEN_LU = {"cemetery"}
GREEN_DE = {"park": "Park", "garden": "Garten", "forest": "Wald", "wood": "Wald", "grass": "Wiese",
            "meadow": "Wiese", "grassland": "Wiese", "cemetery": "Friedhof", "allotments": "Kleingärten",
            "recreation_ground": "Grünanlage", "village_green": "Grünanlage", "scrub": "Gebüsch",
            "heath": "Heide", "wetland": "Feuchtgebiet", "orchard": "Obstwiese", "golf_course": "Golfplatz",
            "nature_reserve": "Naturschutzgebiet", "protected_area": "Schutzgebiet"}


# ---------- Mapbox-Vektorkacheln lesen (kleiner eigener Decoder, keine Zusatzbibliothek)
def _pb_varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _pb_fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _pb_varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _pb_varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _pb_varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError("unbekannter Protobuf-Typ")
        yield f, wt, v


def _pb_packed(v, wt):
    if wt == 0:
        return [v]
    out, i = [], 0
    while i < len(v):
        x, i = _pb_varint(v, i)
        out.append(x)
    return out


def _pb_value(b):
    for f, wt, v in _pb_fields(b):
        if f == 1:
            return bytes(v).decode("utf-8", "replace")
        if f == 2:
            return struct.unpack("<f", v)[0]
        if f == 3:
            return struct.unpack("<d", v)[0]
        if f == 4:
            return v - (1 << 64) if v >= (1 << 63) else v
        if f == 5:
            return v
        if f == 6:
            return (v >> 1) ^ -(v & 1)
        if f == 7:
            return bool(v)
    return None


def _mvt_geom(c):
    """Geometrie-Befehle (MoveTo/LineTo/ClosePath) -> Liste von Punktfolgen in Kachelkoordinaten."""
    parts, cur, x, y, i = [], None, 0, 0, 0
    while i < len(c):
        cmd, cnt = c[i] & 7, c[i] >> 3
        i += 1
        if cmd in (1, 2):
            for _ in range(cnt):
                if i + 1 >= len(c):
                    return parts
                dx, dy = c[i], c[i + 1]
                i += 2
                x += (dx >> 1) ^ -(dx & 1)
                y += (dy >> 1) ^ -(dy & 1)
                if cmd == 1:
                    cur = [(x, y)]
                    parts.append(cur)
                elif cur is not None:
                    cur.append((x, y))
        elif cmd == 7:
            if cur:
                cur.append(cur[0])
        else:
            break
    return parts


def mvt_decode(data, want):
    """Liest die gewünschten Ebenen: {name: [(extent, typ, teile, eigenschaften), ...]}."""
    out = {}
    for f, wt, v in _pb_fields(data):
        if f != 3 or wt != 2:
            continue
        name, keys, vals, feats, extent = None, [], [], [], 4096
        for f2, wt2, v2 in _pb_fields(v):
            if f2 == 1:
                name = bytes(v2).decode("utf-8", "replace")
            elif f2 == 2:
                feats.append(v2)
            elif f2 == 3:
                keys.append(bytes(v2).decode("utf-8", "replace"))
            elif f2 == 4:
                vals.append(v2)
            elif f2 == 5:
                extent = v2
        if name not in want:
            continue
        vals = [_pb_value(x) for x in vals]
        lst = out.setdefault(name, [])
        for fb in feats:
            tags, gtype, geom = [], 0, []
            for f3, wt3, v3 in _pb_fields(fb):
                if f3 == 2:
                    tags += _pb_packed(v3, wt3)
                elif f3 == 3:
                    gtype = v3
                elif f3 == 4:
                    geom += _pb_packed(v3, wt3)
            props = {}
            for k in range(0, len(tags) - 1, 2):
                if tags[k] < len(keys) and tags[k + 1] < len(vals):
                    props[keys[tags[k]]] = vals[tags[k + 1]]
            lst.append((extent, gtype, _mvt_geom(geom), props))
    return out


def _tile_of(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _tile_lonlat(tx, ty, z, extent, pts):
    a = np.asarray(pts, dtype=float)
    n = 2.0 ** z
    lon = (tx + a[:, 0] / extent) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (ty + a[:, 1] / extent) / n))))
    return lon, lat


@st.cache_data(ttl=86400, show_spinner=False)
def _ofm_template():
    r = requests.get(OFM_TILEJSON, headers=UA, timeout=8)
    r.raise_for_status()
    return r.json()["tiles"][0]


def _fetch_tile(tpl, z, x, y):
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    path = os.path.join(OSM_CACHE_DIR, "vt_" + hashlib.md5(f"{tpl}{z}/{x}/{y}".encode()).hexdigest() + ".pbf")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 14 * 86400:
        with open(path, "rb") as f:
            return f.read()
    url = tpl.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    r = requests.get(url, headers=UA, timeout=10)
    if r.status_code in (204, 404):
        data = b""
    else:
        r.raise_for_status()
        data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    with open(path + f".{os.getpid()}", "wb") as f:
        f.write(data)
    os.replace(path + f".{os.getpid()}", path)
    return data


def _is_green(layer, cls, sub):
    if layer == "park":
        return True
    if layer == "landcover":
        return cls in GREEN_LC or sub in GREEN_SUB
    if layer == "landuse":
        return cls in GREEN_LU
    return False


# ---------- Raster
def _fill_ring(mask, cx, cy):
    """Füllt einen Ring (in Rasterkoordinaten) per Scanline (gerade-ungerade-Regel)."""
    ny, nx_ = mask.shape
    r0, r1 = max(0, int(math.floor(cy.min()))), min(ny - 1, int(math.ceil(cy.max())))
    if r0 > r1 or cx.max() < 0 or cx.min() > nx_:
        return
    x0, y0, x1, y1 = cx[:-1], cy[:-1], cx[1:], cy[1:]
    for b0 in range(r0, r1 + 1, 256):
        rows = np.arange(b0, min(r1, b0 + 255) + 1) + 0.5
        R = rows[:, None]
        cond = ((y0 <= R) & (y1 > R)) | ((y1 <= R) & (y0 > R))
        if not cond.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x0 + (R - y0) * (x1 - x0) / (y1 - y0)
        for k in np.nonzero(cond.any(axis=1))[0]:
            xs = np.sort(xi[k][cond[k]])
            for a, b in zip(xs[0::2], xs[1::2]):
                c0, c1 = max(0, int(math.ceil(a - 0.5))), min(nx_ - 1, int(math.floor(b - 0.5)))
                if c1 >= c0:
                    mask[b0 + k, c0:c1 + 1] = True


def _draw_line(mask, cx, cy):
    ny, nx_ = mask.shape
    seg = np.hypot(np.diff(cx), np.diff(cy))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return
    s = np.linspace(0, cum[-1], int(cum[-1] * 2) + 2)
    px, py = np.interp(s, cum, cx), np.interp(s, cum, cy)
    c, r = np.floor(px).astype(int), np.floor(py).astype(int)
    ok = (c >= 0) & (c < nx_) & (r >= 0) & (r < ny)
    mask[r[ok], c[ok]] = True


def _edt(mask, cell):
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return (ndimage.distance_transform_edt(~mask) * cell).astype(np.float32)


@st.cache_resource(show_spinner=False, max_entries=2, ttl=3600)
def scan_area(bbox, water=False, dog=False):
    """Lädt die Vektorkacheln für das Gebiet und baut die Abstandskarten."""
    if ndimage is None:
        raise RuntimeError("scipy fehlt (requirements.txt aktualisieren)")
    w, s, e, n = bbox
    tpl = _ofm_template()
    x0, y0 = _tile_of(n, w, VT_ZOOM)
    x1, y1 = _tile_of(s, e, VT_ZOOM)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    if len(tiles) > 64:
        raise ValueError("Das Gebiet ist für den Grün-Scan zu groß. Bitte eine kürzere Strecke wählen.")
    with ThreadPoolExecutor(max_workers=6) as ex:
        datas = list(ex.map(lambda t: _fetch_tile(tpl, VT_ZOOM, t[0], t[1]), tiles))

    lat0, lon0 = (s + n) / 2, (w + e) / 2
    ky = 111_320.0
    kx = ky * math.cos(math.radians(lat0))
    xmin, ymin = (w - lon0) * kx, (s - lat0) * ky
    width, height = (e - w) * kx, (n - s) * ky
    cell = max(5.0, math.sqrt(width * height / 2.5e6))
    ncol, nrow = int(math.ceil(width / cell)) + 1, int(math.ceil(height / cell)) + 1
    green = np.zeros((nrow, ncol), dtype=bool)
    wmask = np.zeros((nrow, ncol), dtype=bool)
    wgeo = []
    roads = {wt: np.zeros((nrow, ncol), dtype=bool) for wt in sorted(set(ROAD_W.values()))}
    geo, n_green = [], 0

    for (tx, ty), data in zip(tiles, datas):
        if not data:
            continue
        layers = mvt_decode(data, {"landcover", "landuse", "park", "transportation", "water", "waterway"})
        for lname, feats in layers.items():
            for extent, gtype, parts, props in feats:
                cls, sub = props.get("class"), props.get("subclass")
                if gtype == 3 and _is_green(lname, cls, sub) and not (dog and lname == "landuse" and cls in ("cemetery", "playground")):
                    for ring in parts:
                        if len(ring) < 4:
                            continue
                        lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, ring)
                        X, Y = (lon - lon0) * kx, (lat - lat0) * ky
                        area_m2 = 0.5 * abs(np.dot(X[:-1], Y[1:]) - np.dot(X[1:], Y[:-1]))
                        if area_m2 < 50:
                            continue
                        _fill_ring(green, (X - xmin) / cell, (Y - ymin) / cell)
                        n_green += 1
                        if area_m2 >= 2500 and len(geo) < 900:
                            step = max(1, len(lon) // 60)
                            ring_ll = [[round(float(a), 5), round(float(b), 5)]
                                       for a, b in zip(lon[::step], lat[::step])]
                            ring_ll.append(ring_ll[0])
                            name = props.get("name") or GREEN_DE.get(sub) or GREEN_DE.get(cls) or "Grünfläche"
                            geo.append({"type": "Feature", "properties": {"name": str(name)},
                                        "geometry": {"type": "Polygon", "coordinates": [ring_ll]}})
                elif gtype == 3 and lname == "water" and cls != "ocean":
                    for ring in parts:
                        if len(ring) < 4:
                            continue
                        lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, ring)
                        X, Y = (lon - lon0) * kx, (lat - lat0) * ky
                        _fill_ring(wmask, (X - xmin) / cell, (Y - ymin) / cell)
                        if 0.5 * abs(np.dot(X[:-1], Y[1:]) - np.dot(X[1:], Y[:-1])) >= 400 and len(wgeo) < 600:
                            step = max(1, len(lon) // 140)
                            ring_ll = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon[::step], lat[::step])]
                            ring_ll.append(ring_ll[0])
                            wgeo.append({"type": "Feature", "properties": {},
                                         "geometry": {"type": "Polygon", "coordinates": [ring_ll]}})
                elif gtype == 2 and lname == "waterway":
                    for part in parts:
                        if len(part) < 2:
                            continue
                        lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, part)
                        X, Y = (lon - lon0) * kx, (lat - lat0) * ky
                        _draw_line(wmask, (X - xmin) / cell, (Y - ymin) / cell)
                        if len(wgeo) < 600:
                            wgeo.append({"type": "Feature", "properties": {}, "geometry": {
                                "type": "LineString", "coordinates": [[round(float(a), 5), round(float(b), 5)]
                                                                      for a, b in zip(lon[::2], lat[::2])]}})
                elif gtype == 2 and lname == "transportation" and cls in ROAD_W:
                    for part in parts:
                        if len(part) < 2:
                            continue
                        lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, part)
                        X, Y = (lon - lon0) * kx, (lat - lat0) * ky
                        _draw_line(roads[ROAD_W[cls]], (X - xmin) / cell, (Y - ymin) / cell)

    if water:
        green |= wmask  # Wassernähe bevorzugt: Wasser zählt wie Grün (Abstand, Zielpunkte, Bewertung)
    gdist = _edt(green, cell)
    depth = (ndimage.distance_transform_edt(green) * cell).astype(np.float32) if green.any() \
        else np.zeros(green.shape, dtype=np.float32)
    noise = np.zeros(green.shape, dtype=np.float32)
    for wt, m in roads.items():
        noise = np.maximum(noise, np.where(_edt(m, cell) < NOISE_DIST, wt, 0.0).astype(np.float32))
    lab, nlab = ndimage.label(green)
    sizes = np.bincount(lab.ravel(), minlength=nlab + 1).astype(np.float64) * cell * cell
    sizes[0] = 0.0
    return {"lat0": lat0, "lon0": lon0, "kx": kx, "ky": ky, "xmin": xmin, "ymin": ymin, "cell": cell,
            "gdist": gdist, "depth": depth, "noise": noise, "lab": lab, "sizes": sizes,
            "wdist": _edt(wmask, cell) if wmask.any() else None,
            "geo": {"type": "FeatureCollection", "features": geo} if geo else None,
            "wgeo": {"type": "FeatureCollection", "features": wgeo} if wgeo else None,
            "n_green": n_green, "green_share": float(green.mean())}


def _to_xy(sa, lat, lon):
    return (np.asarray(lon, float) - sa["lon0"]) * sa["kx"], (np.asarray(lat, float) - sa["lat0"]) * sa["ky"]


def _to_ll(sa, x, y):
    return sa["lat0"] + y / sa["ky"], sa["lon0"] + x / sa["kx"]


def _cells(sa, X, Y):
    c = np.floor((X - sa["xmin"]) / sa["cell"]).astype(int)
    r = np.floor((Y - sa["ymin"]) / sa["cell"]).astype(int)
    ny, nx_ = sa["gdist"].shape
    ok = (c >= 0) & (c < nx_) & (r >= 0) & (r < ny)
    return np.clip(r, 0, ny - 1), np.clip(c, 0, nx_ - 1), ok


def scan_score(sa, coords, buf):
    """Bewertet eine Route (Liste [lat, lon]): Grünanteil, Hauptstraßen-Anteil, Doppelwege."""
    a = np.asarray(coords, dtype=float)
    X, Y = _to_xy(sa, a[:, 0], a[:, 1])
    seg = np.hypot(np.diff(X), np.diff(Y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return {"dist": 0.0, "green": 0.0, "noise": 0.0, "rep": 0.0, "water": None}
    s = np.arange(SCAN_STEP / 2, total, SCAN_STEP)
    px, py = np.interp(s, cum, X), np.interp(s, cum, Y)
    r, c, ok = _cells(sa, px, py)
    gd = np.where(ok, sa["gdist"][r, c], np.inf)
    # mitten im Park zählt voll, am Parkrand entlang etwas weniger: so führt die Route durch den Park statt außen herum
    g = np.where(gd <= 0.0, 1.0, INSIDE_FACTOR * np.clip(1.0 - gd / buf, 0.0, 1.0) ** GREEN_FALLOFF)
    wdm = sa.get("wdist")
    water = None if wdm is None else float(np.mean(np.where(ok, wdm[r, c], np.inf) <= WATER_SCAN_NEAR))
    nz = np.where(ok, sa["noise"][r, c], 0.0)
    rep = dup_analysis(X, Y)[0]
    return {"dist": total, "green": float(g.mean()), "noise": float(nz.mean()), "rep": float(rep), "water": water}


def _green_targets(sa, A, B, k=4, slack=1.0):
    """Grünflächen nahe der Luftlinie A-B. Je Fläche ein Eingang (Seite von A) und ein Ausgang
    (Seite von B) im Inneren, damit die Route durch den Park führt statt nur am Rand vorbei."""
    D = math.hypot(B[0] - A[0], B[1] - A[1])
    big = np.nonzero(sa["sizes"] >= PARK_MIN_M2)[0]
    if not len(big):
        return []
    deep = (sa["depth"] >= min(12.0, 2 * sa["cell"])) & np.isin(sa["lab"], big)
    r, c = np.nonzero(deep)
    if not len(r):
        return []
    X = sa["xmin"] + (c + 0.5) * sa["cell"]
    Y = sa["ymin"] + (r + 0.5) * sa["cell"]
    dA, dB = np.hypot(X - A[0], Y - A[1]), np.hypot(X - B[0], Y - B[1])
    keep = dA + dB - D <= (0.6 * D + 300) * slack
    if not keep.any():
        return []
    X, Y, dA, dB = X[keep], Y[keep], dA[keep], dB[keep]
    labs = sa["lab"][r[keep], c[keep]]
    oa = np.lexsort((dA, labs))
    ul, fa = np.unique(labs[oa], return_index=True)
    ob = np.lexsort((dB, labs))
    _, fb = np.unique(labs[ob], return_index=True)
    ia, ib = oa[fa], ob[fb]
    inner = np.hypot(X[ia] - X[ib], Y[ia] - Y[ib])
    detour = dA[ia] + inner + dB[ib] - D
    ok = detour <= (0.7 * D + 300) * slack
    score = np.sqrt(sa["sizes"][ul]) / (1.0 + detour / (0.25 * D + 150)) \
        * (1.0 + 0.5 * np.minimum(inner / (0.3 * D + 50), 1.0))
    score[~ok] = -1
    out = []
    for j in np.argsort(-score):
        if score[j] < 0:
            break
        e, x = (float(X[ia[j]]), float(Y[ia[j]])), (float(X[ib[j]]), float(Y[ib[j]]))
        via = [e] if math.hypot(e[0] - x[0], e[1] - x[1]) < 60 else [e, x]
        mid = via[len(via) // 2]
        if all(math.hypot(mid[0] - q[-1][0], mid[1] - q[-1][1]) > 200 for q in out):
            out.append(via)
        if len(out) == k:
            break
    return out


def _snap_green(sa, x, y, radius):
    """Verschiebt einen Wegpunkt ins Innere einer nahen, möglichst großen Grünfläche."""
    cell = sa["cell"]
    ny, nx_ = sa["lab"].shape
    c0, c1 = int((x - radius - sa["xmin"]) / cell), int((x + radius - sa["xmin"]) / cell) + 1
    r0, r1 = int((y - radius - sa["ymin"]) / cell), int((y + radius - sa["ymin"]) / cell) + 1
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, nx_), min(r1, ny)
    if c0 >= c1 or r0 >= r1:
        return x, y
    lab = sa["lab"][r0:r1, c0:c1]
    ok = (sa["depth"][r0:r1, c0:c1] >= min(12.0, 2 * cell)) & (sa["sizes"][lab] >= PARK_MIN_M2)
    rr, cc = np.nonzero(ok)
    if not len(rr):
        return x, y
    X = sa["xmin"] + (cc + c0 + 0.5) * cell
    Y = sa["ymin"] + (rr + r0 + 0.5) * cell
    d = np.hypot(X - x, Y - y)
    inside = d <= radius
    if not inside.any():
        return x, y
    sc = np.sqrt(sa["sizes"][lab[rr, cc]]) / (1.0 + 3.0 * d / radius)
    sc[~inside] = -1
    i = int(np.argmax(sc))
    return float(X[i]), float(Y[i])


# ---------- Routenserver (FOSSGIS OSRM, höchstens 1 Anfrage pro Sekunde)
_OSRM_LOCK = threading.Lock()
_OSRM_LAST = [0.0]


def osrm_request(points, alternatives=False):
    with _OSRM_LOCK:
        wait = _OSRM_LAST[0] + 1.05 - time.time()
        if wait > 0:
            time.sleep(wait)
        _OSRM_LAST[0] = time.time()
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    params = {"overview": "full", "geometries": "geojson", "steps": "true"}
    if alternatives and len(points) == 2:
        params["alternatives"] = "3"
    if len(points) > 2:
        params["continue_straight"] = "true"
    js, err = None, None
    for attempt in range(3):  # der freie Routenserver ist gelegentlich kurz überlastet (429/50x/Zeitüberschreitung)
        try:
            r = requests.get(OSRM_FOOT + coords, headers=UA, timeout=12, params=params)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Routenserver antwortet mit {r.status_code}")
            r.raise_for_status()
            js = r.json()
            break
        except ValueError:
            raise
        except Exception as e:
            err = e
            time.sleep(1.5 + attempt)
    if js is None:
        raise RuntimeError(f"Der Routenserver antwortet gerade nicht ({type(err).__name__}).")
    if js.get("code") != "Ok" or not js.get("routes"):
        raise ValueError("Für diese Punkte habe ich keinen Fußweg gefunden.")
    return js["routes"]


def quick_route(pts, timeout=6):
    """Nur für die Sofort-Vorschau: schnellste Fußroute (vereinfachte Linie, ohne Wegehinweise), ein Versuch.
    Rückgabe {"ll": [[lat, lon], ...], "m": Meter} oder None. Nie ein Fehler: die Vorschau ist nur ein Bonus."""
    try:
        if len(pts) < 2 or all(dist_m(pts[0], q) < 30 for q in pts[1:]):
            return None
        if dist_m(pts[0], pts[-1]) > 15000 and len(pts) == 2:
            return None
        with _OSRM_LOCK:
            wait = _OSRM_LAST[0] + 1.05 - time.time()
            if wait > 0:
                time.sleep(wait)
            _OSRM_LAST[0] = time.time()
        coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in pts)
        r = requests.get(OSRM_FOOT + coords, headers=UA, timeout=timeout,
                         params={"overview": "simplified", "geometries": "geojson", "steps": "false"})
        r.raise_for_status()
        js = r.json()
        rt = js["routes"][0]
        ll = [[y, x] for x, y in rt["geometry"]["coordinates"]]
        if len(ll) < 2:
            return None
        return {"ll": ll, "m": float(rt["distance"])}
    except Exception:
        return None


def preview_svg(prev, w=300, h=104, pad=10):
    """Kleine Linienskizze der Vorschau-Route (ohne Kartenkacheln, daher sofort und ohne weiteres Netz)."""
    ll = prev["ll"]
    lat0 = sum(p[0] for p in ll) / len(ll)
    kx = math.cos(math.radians(lat0))
    xs = [p[1] * kx for p in ll]
    ys = [-p[0] for p in ll]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sc = min((w - 2 * pad) / max(x1 - x0, 1e-9), (h - 2 * pad) / max(y1 - y0, 1e-9))
    ox = (w - (x1 - x0) * sc) / 2
    oy = (h - (y1 - y0) * sc) / 2
    pts = [(ox + (x - x0) * sc, oy + (y - y0) * sc) for x, y in zip(xs, ys)]
    d = " ".join(f"{'M' if i == 0 else 'L'}{a:.1f} {b:.1f}" for i, (a, b) in enumerate(pts))
    (sx, sy), (ex, ey) = pts[0], pts[-1]
    loop = math.hypot(sx - ex, sy - ey) < 6
    end = ("" if loop else f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5.5" fill="#B8571A" stroke="#fff" stroke-width="2"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Skizze der schnellsten Route">'
            f'<path d="{d}" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>'
            f'<path class="pv" d="{d}" fill="none" stroke="#7C857A" stroke-width="3.2" stroke-linecap="round" '
            f'stroke-linejoin="round" stroke-dasharray="1 7"/>{end}'
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#4E6E45" stroke="#fff" stroke-width="2"/></svg>')


def _route_ll(route):
    return [[y, x] for x, y in route["geometry"]["coordinates"]]


def _dedupe(routes):
    out, seen = [], set()
    for rt in routes:
        ll = _route_ll(rt)
        if len(ll) < 2:
            continue
        key = (round(float(rt.get("distance", 0)) / 5), tuple(round(v, 4) for v in ll[len(ll) // 2]))
        if key not in seen:
            seen.add(key)
            out.append(rt)
    return out


def scan_ab_candidates(sa, p1, p2, log=lambda *_: None, wide=True):
    A, B = _to_xy(sa, p1[0], p1[1]), _to_xy(sa, p2[0], p2[1])
    A, B = (float(A[0]), float(A[1])), (float(B[0]), float(B[1]))
    D = math.hypot(B[0] - A[0], B[1] - A[1])
    routes = list(osrm_request([p1, p2], alternatives=True))
    vias = [[_to_ll(sa, x, y) for x, y in via] for via in _green_targets(sa, A, B, k=4)]
    if D > 150:
        mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        nx_, ny_ = -(B[1] - A[1]) / D, (B[0] - A[0]) / D
        for sgn in (1, -1):
            x, y = _snap_green(sa, mx + sgn * 0.3 * D * nx_, my + sgn * 0.3 * D * ny_, 0.2 * D + 80)
            vias.append([_to_ll(sa, x, y)])
    # große Umwege für „Grünster Umweg“: weiter entfernte Parks und breitere Ausschläge zur Seite
    for via in (_green_targets(sa, A, B, k=3, slack=3.0) if wide else []):
        vias.append([_to_ll(sa, x, y) for x, y in via])
    if D > 150 and wide:
        for sgn in (1, -1):
            x, y = _snap_green(sa, mx + sgn * 0.7 * D * nx_, my + sgn * 0.7 * D * ny_, 0.35 * D + 120)
            vias.append([_to_ll(sa, x, y)])
    log(f"🌳 {len(vias)} grüne Umwege werden geprüft")
    for via in vias:
        try:
            routes += osrm_request([p1] + [(float(v[0]), float(v[1])) for v in via] + [p2])
        except Exception:
            continue
    return _dedupe(routes)


def scan_loop_candidates(sa, p1, p2, target, buf, progress=None):
    A = _to_xy(sa, p1[0], p1[1])
    ax, ay = float(A[0]), float(A[1])

    def build(theta, rho):
        cx, cy = ax + rho * math.cos(theta), ay + rho * math.sin(theta)
        pts = [p1]
        for alpha in (90, 180, 270):
            ang = theta + math.pi + math.radians(alpha)
            x, y = _snap_green(sa, cx + rho * math.cos(ang), cy + rho * math.sin(ang), 0.4 * rho)
            la, lo = _to_ll(sa, x, y)
            pts.append((float(la), float(lo)))
        pts.append(p2)
        rt = osrm_request(pts)[0]
        if len(_route_ll(rt)) < 2:
            raise ValueError("leere Route")
        sc = scan_score(sa, _route_ll(rt), buf)
        return {"route": rt, **sc}

    cands, first, errs = [], [], []
    bearings = [2 * math.pi * k / 6 for k in range(6)]
    rho0 = target / (2 * math.pi * 1.3)
    total_steps = len(bearings) + 3
    for k, th in enumerate(bearings):
        try:
            c = build(th, rho0)
            c["theta"], c["rho"] = th, rho0
            cands.append(c)
            first.append(c)
        except Exception as e:
            errs.append(e)
            if len(errs) >= 3 and not cands:  # Routenserver offenbar down: nicht weiter warten
                break
        if progress:
            progress((k + 1) / total_steps)
    if not cands and errs:
        raise ValueError(f"{errs[-1]} Bitte in ein bis zwei Minuten noch einmal versuchen.")
    # die drei vielversprechendsten Richtungen auf die Wunschlänge nachjustieren
    first.sort(key=lambda c: -(c["green"] - REP_W * c["rep"] - abs(c["dist"] / target - 1)))
    for j, c in enumerate(first[:3]):
        ratio = c["dist"] / target if target else 1
        if abs(ratio - 1) > 0.07 and ratio > 0:
            try:
                c2 = build(c["theta"], c["rho"] / ratio)
                cands.append(c2)
            except Exception:
                pass
        if progress:
            progress((len(bearings) + j + 1) / total_steps)
    return cands


def _grid_keys(coords):
    a = np.asarray(coords, dtype=float)
    return set(zip(np.round(a[:, 0] * 4000).astype(int), np.round(a[:, 1] * 2500).astype(int)))


def _scan_option(nav, label, sub, hint, sc, extra=None):
    mins = max(1, round(sc["dist"] / M_PER_MIN))
    stats = [["Länge", fmt_km(sc["dist"])], ["Dauer", f"{mins} min"],
             ["Grünanteil", f"{sc['green']:.0%}"], ["Hauptstraße", f"{sc['noise']:.0%}"]]
    return _option(nav, label, sub, hint, stats)


def scan_water_note(sc):
    """Hinweisende: Wasseranteil (bei aktiver Wassernähe), sonst nichts. Auch 0 % wird gezeigt."""
    if not _water():
        return ""
    w = sc.get("water")
    if w is None:
        return " · kein Gewässer im Gebiet gefunden"
    return f" · {w:.0%} am Wasser"
SCAN_ATTRIB = ['Grün-Scan: <a href="https://openfreemap.org">OpenFreeMap</a> © OpenMapTiles, OSM',
               'Routing: <a href="https://routing.openstreetmap.de">FOSSGIS</a>']


def scan_ab_view(start_v, dest_v, buf=150, log=lambda *_: None):
    """Grüne Route A -> B ohne Overpass: Varianten vom Routenserver, bewertet per Grün-Scan."""
    lat1, lon1, lab1 = _unpack(start_v)
    lat2, lon2, lab2 = _unpack(dest_v)
    p1, p2 = (lat1, lon1), (lat2, lon2)
    bbox, span = corridor_bbox(p1, p2)
    if span > MAX_SPAN_M:
        raise ValueError(f"Start und Ziel sind {span / 1000:.1f} km auseinander. "
                         f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
    log("🛰️ Grün-Scan: Grünflächen und Hauptstraßen aus Kartenkacheln laden")
    t0 = time.time()
    sa = scan_area(bbox, _water(), _dog())
    log(f"✅ {sa['n_green']} Grünflächen erkannt, {sa['green_share']:.0%} des Gebiets grün "
        f"({time.time() - t0:.0f} s)")
    log("🧭 Wegvarianten vergleichen (direkt, durch Parks, seitliche Umwege)")
    routes = scan_ab_candidates(sa, p1, p2, log)
    scored = [(rt, scan_score(sa, _route_ll(rt), buf)) for rt in routes]
    scored = [(rt, sc) for rt, sc in scored if sc["dist"] > 0]
    if not scored:
        raise ValueError("Für diese Punkte habe ich keinen Fußweg gefunden.")
    short = min(sc["dist"] for _, sc in scored)
    t1, t2 = lab1.split(" · ")[0], lab2.split(" · ")[0]
    options, used = [], set()
    for name, lam in LEVELS.items():
        mu = level_quiet(name)[0]
        pool = [i for i in range(len(scored)) if lam < 30 or scored[i][1]["dist"] <= 3.0 * short + 600]
        j = min(pool, key=lambda i: scored[i][1]["dist"] * (
            1 + lam * (1 - scored[i][1]["green"]) + mu * scored[i][1]["noise"]))
        if j in used or (lam >= 30 and scored[j][1]["green"] < max(scored[u][1]["green"] for u in used) + 0.01):
            continue
        used.add(j)
        rt, sc = scored[j]
        nav = osrm_nav(rt, f"{t1} → {t2}")
        if not nav:
            continue
        detour = sc["dist"] / short - 1 if short else 0.0
        options.append(_scan_option(nav, name, f"{fmt_km(sc['dist'])} · {sc['green']:.0%} grün",
                                    f"{LEVEL_HINT[name]} · Umweg {detour:+.0%}{scan_water_note(sc)}", sc))
    if not options:
        raise ValueError("Start und Ziel liegen zu nah beieinander.")
    order = list(LEVELS)
    default = min(range(len(options)), key=lambda i: abs(order.index(options[i]["label"]) - 2))
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=sa["geo"], water=sa.get("wgeo"),
                   options=options, default=default, speed=M_PER_MIN, topPad=TOP_PAD, attrib=SCAN_ATTRIB)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": dest_v, "area": None, "bbox": None, "buf": buf, "fallback": "scan", "water": _water(), "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


def scan_loop_view(start_v, end_v, km, level, buf=150, log=lambda *_: None, progress=None):
    """Grünes Ründchen ohne Overpass."""
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
    log("🛰️ Grün-Scan: Grünflächen und Hauptstraßen aus Kartenkacheln laden")
    t0 = time.time()
    sa = scan_area(loop_bbox(p1, p2, target), _water(), _dog())
    log(f"✅ {sa['n_green']} Grünflächen erkannt, {sa['green_share']:.0%} des Gebiets grün "
        f"({time.time() - t0:.0f} s)")
    log("🔄 Runden in verschiedene Richtungen probieren (ca. 10 s)")
    cands = scan_loop_candidates(sa, p1, p2, target, buf, progress)
    if not cands:
        raise ValueError("Für diesen Startpunkt habe ich keine Runde gefunden.")
    lam = LEVELS[level]
    mu = level_quiet(level)[0]
    w_green = 0.15 + lam / 12.0
    for c in cands:
        c["score"] = (w_green * c["green"] - REP_W * c["rep"] - 0.25 * mu * c["noise"]
                      - 2.0 * abs(c["dist"] / target - 1))
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:
        if abs(c["dist"] / target - 1) > 0.25:
            continue
        ks = _grid_keys(_route_ll(c["route"]))
        if all(len(ks & p["ks"]) / max(len(ks | p["ks"]), 1) < 0.6 for p in picked):
            c["ks"] = ks
            picked.append(c)
        if len(picked) == 3:
            break
    if not picked:
        picked = cands[:1]
    t1, t2 = lab1.split(" · ")[0], lab2.split(" · ")[0]
    options = []
    for i, c in enumerate(picked):
        nav = osrm_nav(c["route"], f"Ründchen {i + 1}")
        if not nav:
            continue
        diff = c["dist"] / target - 1
        options.append(_scan_option(
            nav, f"Variante {i + 1}", f"{fmt_km(c['dist'])} · {c['green']:.0%} grün",
            f"Wunschlänge {fmt_km(target)} ({diff:+.0%}) · {1 - c['rep']:.0%} ohne Doppelweg{scan_water_note(c)}", c))
    if not options:
        raise ValueError("Für diesen Startpunkt habe ich keine Runde gefunden.")
    payload = dict(mode="loop", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=sa["geo"], water=sa.get("wgeo"),
                   options=options, default=0, speed=M_PER_MIN, topPad=TOP_PAD, attrib=SCAN_ATTRIB)
    return {"mode": "loop", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "end": end_v, "km": km, "km_ok": km, "level": level, "buf": buf,
            "area": None, "bbox_km": None, "fallback": "scan", "water": _water(), "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


def make_loop_view_safe(start_v, end_v, km, level, buf=150, area=None, bbox_km=None,
                        log=lambda *_: None, progress=None, weather=True):
    """Ründchen mit Overpass-Daten; wenn die Server überlastet sind, per Grün-Scan."""
    if weather:  # beim Neuberechnen in der Karte ist das Wetter schon bekannt
        dog_weather(start_v)
    try:
        return make_loop_view(start_v, end_v, km, level, buf, area, bbox_km, log, progress)
    except ValueError:
        raise
    except Exception as e:
        log("⚠️ OSM-Server überlastet: ich nutze den Grün-Scan")
        try:
            return scan_loop_view(start_v, end_v, km, level, buf, log, progress)
        except ValueError:
            raise
        except Exception:
            raise e


# ------------------------------------------------------- Ründchen mit Zwischenziel
TOUR_LEVELS = ["Ausgewogen", "Sehr grün", "Maximal grün"]
TOUR_MAX_STOPS = 3      # auf der Startseite
TOUR_MAX_MAP_STOPS = 5  # in der Kartenansicht


def _short(lab):
    """Name ohne Symbol am Anfang und ohne Adresszusatz."""
    return re.sub(r"^[^\w]+", "", lab.split(" · ")[0]).strip() or lab


def tour_points(start_v, stops_v, end_v):
    """Start, Zwischenziele (Reihenfolge nach kürzestem Weg), Ende. Rückgabe: Punkte, Namen, Zwischenziel-Namen."""
    import itertools
    lat1, lon1, lab1 = _unpack(start_v)
    p1 = (lat1, lon1)
    if end_v:
        lat2, lon2, lab2 = _unpack(end_v)
        pe = (lat2, lon2)
    else:
        pe, lab2 = p1, lab1
    stops = [(lat, lon, lab) for lat, lon, lab in (_unpack(v) for v in stops_v)]
    if len(stops) > 1:
        def length(perm):
            seq = [p1] + [(a, b) for a, b, _ in perm] + [pe]
            return sum(dist_m(x, y) for x, y in zip(seq[:-1], seq[1:]))
        stops = list(min(itertools.permutations(stops), key=length))
    pts = [p1] + [(a, b) for a, b, _ in stops] + [pe]
    return pts, [_short(lab1)] + [_short(lab) for _, _, lab in stops] + [_short(lab2)], stops


def points_bbox(pts, extra_m=0.0):
    ky = 111_320.0
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    kx = ky * math.cos(math.radians(sum(lats) / len(lats)))
    span = max((max(lons) - min(lons)) * kx, (max(lats) - min(lats)) * ky)
    pad = max(600.0, 0.3 * span) + 0.4 * extra_m
    return (round(min(lons) - pad / kx, 4), round(min(lats) - pad / ky, 4),
            round(max(lons) + pad / kx, 4), round(max(lats) + pad / ky, 4)), span


def tour_bboxes(pts, extra_m=0.0):
    """(benötigtes Gebiet, zu ladendes Gebiet, Abstand der Punkte). Geladen wird für die nächsthöhere Stufe der
    Extra-Strecke (2/4/7/10 km), damit ein Verschieben des Reglers kein neues Gebiet braucht."""
    need, span = points_bbox(pts, extra_m)
    tier = 0.0 if extra_m <= 0 else next((t for t in (2000.0, 4000.0, 7000.0, 10000.0) if extra_m <= t), extra_m)
    load, _ = points_bbox(pts, tier)
    return need, load, span


def tour_path(area, g, lam, mu, wq, nodes):
    """Teilstrecken Start -> Zwischenziele -> Ende, jede grün-optimal; schon gelaufene Wegstücke werden
    teurer, damit der Rückweg möglichst ein anderer ist. marks: Index der Zwischenziele im Pfad."""
    H, L = area["H"], area["lengths"]
    nz = area.get("noise")
    nz = np.zeros(len(L)) if nz is None else nz
    wd = area.get("wide")
    wd = np.zeros(len(L)) if wd is None else wd
    ap = area_accp(area)
    full, marks = [nodes[0]], []
    usedm = np.zeros(len(L), dtype=bool)
    base = L * (1 + lam * (1 - g) + mu * nz + wq * wd + ap)
    for s_, e in zip(nodes[:-1], nodes[1:]):
        if s_ != e:
            p = fast_path(area, np.where(usedm, base * LOOP_REUSE_PENALTY, base), s_, e)
            full += p[1:]
            mark_used_both(area, usedm, p)
        marks.append(len(full) - 1)
    try:
        cleaned = undouble(area, full, keep=set(nodes))
        if len(cleaned) != len(full):
            return cleaned, marks_for(cleaned, nodes)
    except Exception:
        pass
    return full, marks[:-1]


def tour_extra_variants(area, g, lam, mu, wq, nodes, extra_m):
    """Fügt eine Extra-Schleife von etwa extra_m hinzu: ein grüner Umweg-Punkt wird vor dem ersten bzw. nach
    dem letzten Zwischenziel eingefügt, sein Abstand wird an die Wunschlänge angepasst."""
    ng = node_green(area, g)
    base, _ = tour_path(area, g, lam, mu, wq, nodes)
    st0 = loop_stats(area, g, base)
    base_len = st0[0] if st0 else 0.0
    cands = []
    for pos in sorted({0, len(nodes) - 2}):
        ax, ay = area["pos"][nodes[pos]]
        for k in range(6):
            theta = 2 * math.pi * k / 6
            rho = extra_m / 2.6
            for _ in range(3):
                w_node = pick_waypoint(area, ng, ax + rho * math.cos(theta), ay + rho * math.sin(theta), 0.3 * rho)
                seq = nodes[:pos + 1] + [w_node] + nodes[pos + 1:]
                try:
                    path, marks = tour_path(area, g, lam, mu, wq, seq)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    break
                st_ = loop_stats(area, g, path)
                if not st_:
                    break
                dist, green, rep, noise = st_
                got = dist - base_len
                cands.append({"path": path, "marks": marks[:pos] + marks[pos + 1:], "pos": pos, "dist": dist,
                              "green": green, "rep": rep, "noise": noise, "got": got})
                if abs(got / extra_m - 1) <= 0.12:
                    break
                rho *= min(max(extra_m / max(got, 0.2 * extra_m), 0.5), 2.0)
    for c in cands:
        c["score"] = c["green"] - REP_W * c["rep"] - 0.4 * mu * c["noise"] - 2.0 * abs(c["got"] / extra_m - 1)
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:
        if abs(c["got"] / extra_m - 1) > 0.35:
            continue
        es = edge_set(c["path"])
        if all(len(es & p["es"]) / len(es | p["es"]) < 0.6 for p in picked):
            c["es"] = es
            picked.append(c)
        if len(picked) == 3:
            break
    return picked


def tour_extreme(area, g, nodes, extra_m, existing=()):
    """Ründchen mit Zwischenziel „Extrem grün“: bis 35 % länger als der direkte Weg über die Zwischenziele (mit
    Extra-Strecke: als Wunschweg + Extra), aber über 70 % Grün. Ergebnis {"path", "marks", "dist", "green", ...} oder None."""
    p0, _ = tour_path(area, g, 0.0, 0.0, 0.0, nodes)
    st0 = loop_stats(area, g, p0)
    if not st0:
        return None
    budget = (1 + EXTREME_DETOUR) * (st0[0] + extra_m)
    have = {tuple(x) for x in existing}
    if extra_m > 0:
        q = LEVEL_QUIET["Grünster Umweg"]
        vs = tour_extra_variants(area, g, LEVELS["Grünster Umweg"], q[0], q[1], nodes, extra_m * 1.1)
        ok = [v for v in vs if v["dist"] <= budget and v["green"] > EXTREME_GREEN_MIN and tuple(v["path"]) not in have]
        return max(ok, key=lambda v: v["green"] - REP_W * v["rep"]) if ok else None

    def run(lam, mu, wq):
        path, marks = tour_path(area, g, lam, mu, wq, nodes)
        st_ = loop_stats(area, g, path)
        if not st_:
            return {"path": path, "marks": marks, "dist": 1e12, "green": 0.0, "rep": 1.0, "noise": 0.0}
        return {"path": path, "marks": marks, "dist": st_[0], "green": st_[1], "rep": st_[2], "noise": st_[3]}

    best = greenest_within(run, budget)
    if best and best["green"] > EXTREME_GREEN_MIN and tuple(best["path"]) not in have:
        return best
    return None


def _tour_payload(mode_extra, p1, pe, t1, t2, stops, options, parks, default, attrib=None, water=None, dogz=None):
    payload = dict(mode="tour", dogz=dogz, p1=list(p1), p2=list(pe), t1=t1, t2=t2, parks=parks, options=options,
                   default=default, speed=M_PER_MIN, topPad=TOP_PAD,
                   stops=[[a, b, _short(lab)] for a, b, lab in stops])
    if attrib:
        payload["attrib"] = attrib
    if water:
        payload["water"] = water
    return payload


def make_tour_view(start_v, stops_v, end_v, level, buf=150, log=lambda *_: None, extra_km=0):
    pts, names, stops = tour_points(start_v, stops_v, end_v)
    extra_m = extra_km * 1000.0
    bbox, load_bb, span = tour_bboxes(pts, extra_m)
    if span > MAX_SPAN_M:
        raise ValueError(f"Die Punkte liegen {span / 1000:.1f} km auseinander. "
                         f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
    log("🗺️ Wegenetz und Grünflächen laden")
    t0 = time.time()
    have = _area_lookup(bbox)
    prefetch_water(tuple(have["bbox"]) if have is not None and have.get("bbox") else load_bb)
    area = load_area_any(bbox, load_bbox=load_bb)
    log(f"✅ {fmt_m(area['n_edges'])} Wegstücke, {area['n_parks']} Grünflächen ({time.time() - t0:.0f} s{_tm_txt(area)})")
    ensure_water(area, log)
    nodes = [nearest_node(area, *p) for p in pts]
    if len(set(nodes)) < 2:
        raise ValueError("Start und Zwischenziele liegen zu nah beieinander.")
    g = edge_green(area, buf)
    log("🧭 Runde mit Zwischenzielen berechnen")
    loop = pts[0] == pts[-1]
    title = " → ".join(names)
    options, seen = [], set()
    runs = []  # (Name, Pfad, Marken, Hinweis)
    if extra_m > 0:
        log(f"➕ Extra-Strecke von etwa {extra_km} km einbauen")
        mu, wq = level_quiet(level)
        try:
            vs = tour_extra_variants(area, g, LEVELS[level], mu, wq, nodes, extra_m)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise ValueError("Zwischen den Punkten gibt es keinen Fußweg im geladenen Gebiet.")
        for i, v in enumerate(vs):
            runs.append((f"Variante {i + 1}", v["path"], v["marks"],
                         f"Extra {fmt_km(v['got'])} (Wunsch +{extra_km} km) · "))
    else:
        for name in TOUR_LEVELS:
            mu, wq = level_quiet(name)
            try:
                path, marks = tour_path(area, g, LEVELS[name], mu, wq, nodes)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raise ValueError("Zwischen den Punkten gibt es keinen Fußweg im geladenen Gebiet.")
            runs.append((name, path, marks, f"{LEVEL_HINT[name]} · "))
    ext_ok = None
    try:  # Zusatz: „Extrem grün“ (bis 35 % Umweg, über 70 % Grün)
        log("🌿 Extrem grüne Variante suchen")
        ext = tour_extreme(area, g, nodes, extra_m, [r_[1] for r_ in runs])
        ext_ok = bool(ext)
        if ext:
            runs.append((EXTREME_NAME, ext["path"], ext["marks"], extreme_hint(" · ")))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    for name, path, marks, pre in runs:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        st_ = loop_stats(area, g, path)
        nav = build_nav(area, path, title, loop=loop, waypoints=list(zip(marks, names[1:-1])))
        if not st_ or not nav:
            continue
        dist, green, rep, noise = st_
        idx = [area["H"][a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        Ls, wd = area["lengths"][idx], area["wide"][idx]
        narrow = float((Ls * (wd <= NARROW)).sum() / max(Ls.sum(), 1))
        mins = max(1, round(dist / M_PER_MIN))
        options.append(_option(
            nav, name, f"{fmt_km(dist)} · {green:.0%} grün",
            f"{pre}{1 - rep:.0%} ohne Doppelweg" + (f" · {narrow:.0%} kleine Wege" if LEVELS.get(level if (extra_m > 0 and name != EXTREME_NAME) else name,
                                                                                                LEVELS["Grünster Umweg"]) >= 12 else "")
            + (water_note(area, path) if _water() else ""),
            [["Länge", fmt_km(dist)], ["Dauer", f"{mins} min"], ["Grünanteil", f"{green:.0%}"],
             ["Hauptstraße", f"{noise:.0%}"]], area=area, path=path))
    if not options:
        raise ValueError("Ich konnte keine Runde mit diesen Zwischenzielen finden.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == level), 0)
    if ext_ok is False and options:
        options[default]["hint"] += extreme_note(False)
    payload = _tour_payload(None, pts[0], pts[-1], names[0], names[-1], stops, options, parks_geo(area), default,
                            water=water_geo(area), dogz=dogz_geo(area))
    return {"mode": "tour", "html": map_html(payload), "p1": pts[0], "p2": pts[-1], "t1": names[0],
            "t2": names[-1], "stops": names[1:-1], "start": start_v, "buf": buf, "stops_v": list(stops_v),
            "end_v": end_v, "level": level, "extra_km": extra_km, "water": _water(), "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


def _merge_routes(routes):
    coords, legs, marks, dist = [], [], [], 0.0
    for rt in routes:
        c = rt["geometry"]["coordinates"]
        coords += c if not coords else c[1:]
        legs += rt.get("legs") or [{"steps": []}]
        marks.append(len(coords) - 1)
        dist += float(rt.get("distance", 0))
    return {"distance": dist, "geometry": {"type": "LineString", "coordinates": coords}, "legs": legs}, marks


def _add_waypoint_steps(nav, marks, names):
    for m, name in zip(marks, names):
        la, lo = nav["coords"][min(m, len(nav["coords"]) - 1)]
        nav["steps"].append({"d": round(nav["cum"][min(m, len(nav["cum"]) - 1)]), "lat": la, "lon": lo,
                             "t": f"Zwischenziel erreicht: {name}", "i": "📍"})
    nav["steps"].sort(key=lambda z: (z["d"], z["i"] == "🏁"))
    return nav


def _scan_tour_extra(sa, pts, legs, chosen, lam, mu, buf, extra_m, log):
    """Extra-Schleife im Ersatzmodus: über einen grünen Umweg-Punkt in einer Teilstrecke fahren."""
    base_rt, base_marks = _merge_routes([c[0] for c in chosen])
    base_len = scan_score(sa, _route_ll(base_rt), buf)["dist"]
    cands = []

    def attempt(pos, theta, rho):
        a, b = pts[pos], pts[pos + 1]
        ax, ay = _to_xy(sa, a[0], a[1])
        x, y = _snap_green(sa, float(ax) + rho * math.cos(theta), float(ay) + rho * math.sin(theta), 0.4 * rho)
        wla, wlo = _to_ll(sa, x, y)
        rt = osrm_request([a, (float(wla), float(wlo)), b])[0]
        ks = _grid_keys(_route_ll(rt))
        ch = list(chosen)
        ch[pos] = (rt, None, ks)
        merged, marks = _merge_routes([c[0] for c in ch])
        sc = scan_score(sa, _route_ll(merged), buf)
        if sc["dist"] <= 0:
            return None
        c = {"merged": merged, "marks": marks, "sc": sc, "got": sc["dist"] - base_len, "pos": pos,
             "theta": theta, "rho": rho}
        c["score"] = (sc["green"] - REP_W * sc["rep"] - 0.25 * mu * sc["noise"] - 2.0 * abs(c["got"] / extra_m - 1))
        return c

    for pos in sorted({0, len(chosen) - 1}):
        for k in range(4):
            try:
                c = attempt(pos, 2 * math.pi * k / 4 + 0.4, extra_m / 2.6)
            except Exception:
                continue
            if c:
                cands.append(c)
    cands.sort(key=lambda c: -c["score"])
    for c in list(cands[:3]):  # einmal auf die Wunschlänge nachjustieren
        if abs(c["got"] / extra_m - 1) > 0.12 and c["got"] > 0:
            try:
                c2 = attempt(c["pos"], c["theta"], c["rho"] * min(max(extra_m / c["got"], 0.5), 2.0))
            except Exception:
                c2 = None
            if c2:
                cands.append(c2)
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:
        if abs(c["got"] / extra_m - 1) > 0.4:
            continue
        ks = _grid_keys(_route_ll(c["merged"]))
        if all(len(ks & p["ks"]) / max(len(ks | p["ks"]), 1) < 0.6 for p in picked):
            c["ks"] = ks
            picked.append(c)
        if len(picked) == 3:
            break
    return picked


def scan_tour_view(start_v, stops_v, end_v, level, buf=150, log=lambda *_: None, extra_km=0):
    """Ründchen mit Zwischenzielen ohne Overpass: je Teilstrecke Varianten vom Routenserver, per Grün-Scan bewertet."""
    pts, names, stops = tour_points(start_v, stops_v, end_v)
    extra_m = extra_km * 1000.0
    bbox, span = points_bbox(pts, extra_m)
    if span > MAX_SPAN_M:
        raise ValueError(f"Die Punkte liegen {span / 1000:.1f} km auseinander. "
                         f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
    log("🛰️ Grün-Scan: Grünflächen und Hauptstraßen aus Kartenkacheln laden")
    sa = scan_area(bbox, _water(), _dog())
    legs = []
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        log(f"🧭 Teilstrecke {k + 1} von {len(pts) - 1} prüfen")
        routes = scan_ab_candidates(sa, a, b, lambda *_: None, wide=False)
        sc = [(rt, scan_score(sa, _route_ll(rt), buf)) for rt in routes]
        sc = [(rt, x) for rt, x in sc if x["dist"] > 0]
        if not sc:
            raise ValueError("Für diese Punkte habe ich keinen Fußweg gefunden.")
        legs.append([(rt, x, _grid_keys(_route_ll(rt))) for rt, x in sc])
    loop = pts[0] == pts[-1]
    title = " → ".join(names)
    options, seen = [], set()
    if extra_m > 0:
        lam, mu = LEVELS[level], level_quiet(level)[0]
        chosen, used = [], set()
        for cands in legs:
            def cost(c):
                rt, x, ks = c
                ov = len(ks & used) / max(len(ks), 1) if used else 0.0
                return x["dist"] * (1 + lam * (1 - x["green"]) + mu * x["noise"]) * (1 + 1.5 * ov)
            best = min(cands, key=cost)
            chosen.append(best)
            used |= best[2]
        log(f"➕ Extra-Strecke von etwa {extra_km} km einbauen")
        for i, c in enumerate(_scan_tour_extra(sa, pts, legs, chosen, lam, mu, buf, extra_m, log)):
            nav = osrm_nav(c["merged"], title)
            if not nav:
                continue
            nav = _add_waypoint_steps(nav, c["marks"][:-1], names[1:-1])
            options.append(_scan_option(
                nav, f"Variante {i + 1}", f"{fmt_km(c['sc']['dist'])} · {c['sc']['green']:.0%} grün",
                f"Extra {fmt_km(c['got'])} (Wunsch +{extra_km} km) · {1 - c['sc']['rep']:.0%} ohne Doppelweg{scan_water_note(c['sc'])}",
                c["sc"]))
        runs_levels = []
    else:
        runs_levels = TOUR_LEVELS
    for name in runs_levels:
        lam, mu = LEVELS[name], level_quiet(name)[0]
        chosen, used = [], set()
        for cands in legs:
            def cost(c):
                rt, x, ks = c
                ov = len(ks & used) / max(len(ks), 1) if used else 0.0
                return x["dist"] * (1 + lam * (1 - x["green"]) + mu * x["noise"]) * (1 + 1.5 * ov)
            best = min(cands, key=cost)
            chosen.append(best)
            used |= best[2]
        key = tuple(id(c[0]) for c in chosen)
        if key in seen:
            continue
        seen.add(key)
        merged, marks = _merge_routes([c[0] for c in chosen])
        sc = scan_score(sa, _route_ll(merged), buf)
        nav = osrm_nav(merged, title)
        if not nav or sc["dist"] <= 0:
            continue
        nav = _add_waypoint_steps(nav, marks[:-1], names[1:-1])
        options.append(_scan_option(nav, name, f"{fmt_km(sc['dist'])} · {sc['green']:.0%} grün",
                                    f"{LEVEL_HINT[name]} · {1 - sc['rep']:.0%} ohne Doppelweg{scan_water_note(sc)}", sc))
    if not options:
        raise ValueError("Ich konnte keine Runde mit diesen Zwischenzielen finden.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == level), 0)
    payload = _tour_payload(None, pts[0], pts[-1], names[0], names[-1], stops, options, sa["geo"], default,
                            attrib=SCAN_ATTRIB, water=sa.get("wgeo"))
    return {"mode": "tour", "html": map_html(payload), "p1": pts[0], "p2": pts[-1], "t1": names[0],
            "t2": names[-1], "stops": names[1:-1], "start": start_v, "buf": buf, "fallback": "scan",
            "stops_v": list(stops_v), "end_v": end_v, "level": level, "extra_km": extra_km, "water": _water(), "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


def make_tour_view_safe(start_v, stops_v, end_v, level, buf=150, log=lambda *_: None, extra_km=0):
    dog_weather(start_v)
    try:
        return make_tour_view(start_v, stops_v, end_v, level, buf, log, extra_km)
    except ValueError:
        raise
    except Exception as e:
        log("⚠️ OSM-Server überlastet: ich nutze den Grün-Scan")
        try:
            return scan_tour_view(start_v, stops_v, end_v, level, buf, log, extra_km)
        except ValueError:
            raise
        except Exception:
            raise e


# ------------------------------------------------------- „Ins Grün“: passende Grünfläche in der Nähe finden
GREEN_RADIUS = 10000
GREEN_PREFS = ["Wassernähe", "Sanitäroption", "Sportangebote", "Spielplätze aller Art", "Sehenswürdigkeiten"]
GREEN_PREFS_PREFER = GREEN_PREFS + ["Barrierefreiheit"]
# Hundemodus: Spielplätze sind für Hunde tabu, dafür gibt es Hundefreilaufflächen
GREEN_PREFS_DOG = [("Hundefreilaufflächen" if p == "Spielplätze aller Art" else p) for p in GREEN_PREFS]
GREEN_PREFS_PREFER_DOG = GREEN_PREFS_DOG + ["Barrierefreiheit"]
GREEN_CAT = {"Wassernähe": "water", "Sanitäroption": "san", "Sportangebote": "sport",
             "Spielplätze aller Art": "play", "Sehenswürdigkeiten": "sight", "Barrierefreiheit": "access",
             "Hundefreilaufflächen": "dogrun"}
GREEN_ICON = {"water": "💧", "san": "🚻", "sport": "⚽", "play": "🛝", "sight": "🏛️", "access": "♿", "dogrun": "🐕"}
GREEN_SIZES = {"Klein (bis 2 ha)": (0.2, 2.0), "Mittel (2–15 ha)": (2.0, 15.0),
               "Groß (15–100 ha)": (15.0, 100.0), "Sehr groß (ab 100 ha)": (100.0, 1e9)}
SPORT_SPORTS = ("basketball|table_tennis|skateboard|soccer|volleyball|tennis|climbing|bmx|handball|"
                "beachvolleyball|boules|streetball|fitness|multi")


def _box(lat, lon, r):
    d = r / 110540.0
    dx = d / max(math.cos(math.radians(lat)), 0.3)
    return f"({lat - d:.4f},{lon - dx:.4f},{lat + d:.4f},{lon + dx:.4f})"


# Mehrere kleine Abfragen statt einer großen: Overpass lehnt große Stadtabfragen oft mit 500/504 ab.
GREEN_QUERIES = {
    "park": 'wr["leisure"~"^(park|nature_reserve|common|dog_park)$"]',
    "wald": 'wr["landuse"~"^(forest|recreation_ground|village_green)$"];wr["natural"~"^(wood|heath)$"]',
}
FEAT_QUERIES = {
    "san": 'nwr["amenity"="toilets"]',
    "play": 'nwr["leisure"="playground"]',
    "dogrun": ('nwr["leisure"="dog_park"];nwr["dog"="off_leash"];'
               'nwr["leisure"~"^(park|garden|pitch|common|nature_reserve)$"]["name"~"Hundewiese|Hundeauslauf|Hundefreilauf|Freilauffläche|Hundespielwiese",i]'),
    "sport": 'nwr["leisure"~"^(pitch|fitness_station|sports_centre|skate_park|dog_park|track)$"];nwr["amenity"="boat_rental"]',
    "water": 'wr["natural"="water"];wr["landuse"~"^(reservoir|basin)$"];wr["waterway"~"^(river|canal|riverbank)$"]',
    "access": 'nwr["amenity"="toilets"]["wheelchair"~"^(yes|designated)$"];wr["highway"~"^(footway|path|pedestrian|cycleway)$"]["wheelchair"~"^(yes|designated)$"];nwr["leisure"~"^(park|playground|garden)$"]["wheelchair"~"^(yes|designated)$"]',
    "sight": 'nwr["tourism"~"^(attraction|viewpoint|zoo)$"];nwr["historic"~"^(monument|castle|ruins|fort|archaeological_site|tower)$"]',
}


@st.cache_data(ttl=90, show_spinner=False)
def _ranked_mirrors():
    """Server mit freien Plätzen zuerst (wie beim Routen-Abruf), nicht erreichbare weglassen."""
    try:
        return rank_mirrors()
    except Exception:
        return list(OVERPASS_MIRRORS)


@st.cache_data(ttl=21600, show_spinner=False)
def _green_part(lat, lon, r, sel):
    """Eine kleine Abfrage (Bereichsfilter statt Umkreis = schneller). Alle Server werden der Reihe nach probiert."""
    box = _box(lat, lon, r)
    body = ";".join(f"{x}{box}" for x in sel.split(";"))
    q = f"[out:json][timeout:30];({body};);out tags bb 6000;"  # bb: Rechteck je Objekt (nicht mit „center“ mischen!)
    last = None
    ms = _ranked_mirrors()
    for chunk, to in ((ms[:4], 26.0), (ms[2:6], 20.0)):
        if not chunk:
            continue
        try:
            res = _overpass_race(q, to, chunk)
            _GP_OK.add((lat, lon, r, sel))
            return res
        except Exception as e:
            last = e
    raise last or RuntimeError("Overpass nicht erreichbar")


_GP_OK = set()  # (lat, lon, r, Auswahl) der Abfragen, die schon im Zwischenspeicher liegen


def _green_ready(lat, lon, r, cats):
    """Liegen alle Abfragen für diesen Umkreis schon im Zwischenspeicher (z. B. durch das Vorwärmen)?"""
    sels = list(GREEN_QUERIES.values()) + [FEAT_QUERIES[c] for c in cats]
    return all((lat, lon, r, v) in _GP_OK for v in sels)


def _green_fetch(lat, lon, r, cats):
    """Grünflächen (Pflicht) und die gewählten Ausstattungs-Kategorien (nur diese, spart Last), alles gleichzeitig.
    Fällt eine Ausstattungs-Abfrage aus, wird sie übersprungen; Rückgabe (greens, feats, ausgefallene Kategorien)."""
    jobs = {("g", k): v for k, v in GREEN_QUERIES.items()}
    jobs.update({("f", c): FEAT_QUERIES[c] for c in cats})
    greens, feats, failed = [], [], []
    ex = ThreadPoolExecutor(max_workers=len(jobs))
    futs = {k: ex.submit(_green_part, lat, lon, r, v) for k, v in jobs.items()}
    ex.shutdown(wait=False)  # nicht fertige Abrufe laufen im Hintergrund weiter und landen im Zwischenspeicher
    t_end = time.time() + 28.0  # Zeitbudget: langsame Zusatzangaben werden übersprungen statt minutenlang abgewartet
    for (kind, name), f in sorted(futs.items(), key=lambda kv: kv[0][0] != "g"):
        try:
            left = max(0.5, t_end - time.time())
            els = f.result(timeout=left).get("elements", [])
        except Exception as e:
            if kind == "g":
                failed.append(("g", e))
            else:
                failed.append((name, e))
            continue
        (greens if kind == "g" else feats).extend(els)
    if not greens:
        errs = [e for k, e in failed if k == "g"]
        raise errs[0] if errs else ValueError("Im Umkreis von 10 km habe ich keine Grünfläche gefunden.")
    # Ausgefallene Ausstattungs-Abfragen (oft nur kurz überlastete Server) noch einmal versuchen, mit weniger
    # gleichzeitigen Anfragen: Ohne sie wüsste die Suche nicht, welche Fläche Wasser, Toiletten usw. hat.
    redo = [k for k, _ in failed if k != "g"]
    if redo:
        time.sleep(1.5)
        ex2 = ThreadPoolExecutor(max_workers=2)
        futs2 = {c: ex2.submit(_green_part, lat, lon, r, FEAT_QUERIES[c]) for c in redo}
        ex2.shutdown(wait=False)
        t_end2 = time.time() + 22.0
        for c, f in futs2.items():
            try:
                feats.extend(f.result(timeout=max(0.5, t_end2 - time.time())).get("elements", []))
                failed = [(k, e) for k, e in failed if k != c]
            except Exception:
                pass
    return greens, feats, [k for k, _ in failed if k != "g"]


def _bounds(e):
    b = e.get("bounds")
    if b:
        return float(b["minlat"]), float(b["minlon"]), float(b["maxlat"]), float(b["maxlon"])
    c = e.get("center") or e
    if "lat" in c:
        return float(c["lat"]), float(c["lon"]), float(c["lat"]), float(c["lon"])
    return None


def feature_cats(t):
    """Wie feature_cat, aber mit Barrierefreiheit zusätzlich: Liste der Kategorien."""
    out = []
    c = feature_cat(t)
    if c:
        out.append(c)
    if t.get("wheelchair") in ("yes", "designated") and (t.get("amenity") == "toilets" or t.get("highway")
                                                          or t.get("leisure") in ("park", "playground", "garden")):
        out.append("access")
    return out


def feature_cat(t):
    """OSM-Tags → Kategorie (water/san/sport/play/sight) oder None."""
    if t.get("amenity") == "toilets":
        return None if t.get("access") in ("private", "no", "customers") else "san"
    if t.get("leisure") == "playground":
        return "play"
    if _dog() and (t.get("leisure") == "dog_park" or t.get("dog") == "off_leash"
                   or re.search(r"hundewiese|hundeauslauf|hundefreilauf|freilauffläche|hundespielwiese", str(t.get("name", "")), re.I)):
        return "dogrun"
    if t.get("leisure") in ("pitch", "fitness_station", "sports_centre", "skate_park", "dog_park", "track") \
            or t.get("amenity") == "boat_rental" or (t.get("leisure") and t.get("sport")):
        return "sport"
    if t.get("natural") == "water" or t.get("waterway") in ("river", "canal"):
        return "water"
    if t.get("tourism") in ("attraction", "viewpoint", "zoo") or t.get("historic"):
        return "sight"
    return None


def _area_ha(b):
    la0, lo0, la1, lo1 = b
    ky, kx = 110540.0, 111320.0 * math.cos(math.radians((la0 + la1) / 2))
    return (la1 - la0) * ky * (lo1 - lo0) * kx * 0.6 / 10000.0  # Rechteck → grob auf die echte Fläche


def rank_greens(greens, feats, prefer, avoid, size_range, pos, min_ha=0.2, min_d=100.0):
    """Grünflächen nach Passung sortieren. greens/feats sind Overpass-Elemente (mit bounds/center)."""
    lat0, lon0 = pos
    ky, kx = 110540.0, 111320.0 * math.cos(math.radians(lat0))
    items = []
    for e in feats:
        b = _bounds(e)
        if b:
            for cat in feature_cats(e.get("tags", {})):
                items.append((cat, b))
    lo_ha, hi_ha = size_range
    out = []
    for e in greens:
        if _dog() and dog_forbidden_tags(e.get("tags", {})):
            continue  # Hundemodus: Friedhöfe, Spielplätze, Zoos und dog=no kommen nicht als Ziel infrage
        b = _bounds(e)
        if not b:
            continue
        ha = _area_ha(b)
        if ha < min_ha:
            continue
        la0, lo0, la1, lo1 = b
        cla, clo = (la0 + la1) / 2, (lo0 + lo1) / 2
        tla, tlo = min(max(lat0, la0), la1), min(max(lon0, lo0), lo1)
        d = math.hypot((tla - lat0) * ky, (tlo - lon0) * kx)
        target = (tla, tlo)
        if ha <= 6:  # kleine Fläche: Mitte (Rechteck); genauer Umriss wird danach für die besten Flächen geholt
            target = (cla, clo)
        else:  # große Fläche: vom Rand ein Stück zur Mitte hin
            vx, vy = (cla - tla) * ky, (clo - tlo) * kx
            nv = math.hypot(vx, vy)
            if nv > 150:
                target = (tla + vx / nv * 70 / ky, tlo + vy / nv * 70 / kx)
        if d < 200:  # Standort liegt im Rechteck der Fläche: Mitte ansteuern
            target = (cla, clo)
            d = math.hypot((cla - lat0) * ky, (clo - lon0) * kx)
        if d < min_d or d > GREEN_RADIUS * 1.15:
            continue
        m = 60.0 / ky
        mx = m * ky / kx
        counts, pts = {}, {}
        for cat, (fa0, fo0, fa1, fo1) in items:
            pad = 100.0 / ky if cat == "water" else m
            padx = pad * ky / kx
            if fa1 >= la0 - pad and fa0 <= la1 + pad and fo1 >= lo0 - padx and fo0 <= lo1 + padx:
                counts[cat] = counts.get(cat, 0) + 1
                pts.setdefault(cat, []).append(((fa0 + fa1) / 2, (fo0 + fo1) / 2))
        # Ziel: bevorzugte Ausstattung (z. B. Teich, Spielplatz), die dem Standort am nächsten liegt – dort soll es hingehen
        goal = None
        for c in prefer:
            for (fla, flo) in pts.get(c, []):
                dd = math.hypot((fla - lat0) * ky, (flo - lon0) * kx)
                if goal is None or dd < goal[0]:
                    goal = (dd, (fla, flo), c)
        if goal:
            target = goal[1]
        # Größe: 1 im Wunschbereich, nach außen fallend (je Zehnerpotenz Abweichung -1)
        if lo_ha <= ha <= hi_ha:
            fit = 1.0
        else:
            ref = lo_ha if ha < lo_ha else hi_ha
            fit = max(0.0, 1.0 - abs(math.log10(ha / ref)))
        score = 2.0 * fit - 1.2 * d / GREEN_RADIUS
        for c in prefer:
            if counts.get(c):  # Barrierefreiheit ist immer das höchste Gut
                score += 8.0 if c == "access" else 1.5 + 0.3 * min(counts[c] - 1, 3)
        for c in avoid:
            if counts.get(c):
                score -= 2.5
        t = e.get("tags", {})
        kind = (t.get("leisure") or t.get("landuse") or t.get("natural") or "")
        name = t.get("name") or {"park": "Park", "forest": "Wald", "wood": "Wald", "nature_reserve": "Naturschutzgebiet",
                                  "common": "Grünfläche", "recreation_ground": "Erholungsfläche", "heath": "Heide",
                                  "village_green": "Grünanlage", "dog_park": "Hundepark"}.get(kind, "Grünfläche")
        out.append({"name": name, "kind": kind, "ha": ha, "d": d, "target": target, "counts": counts,
                    "score": score, "fit": fit, "goal": goal[2] if goal else None,
                    "id": (e.get("type"), e.get("id")), "inside": goal is not None})
    out.sort(key=lambda g: -g["score"])
    return out


def green_reasons(g):
    bits = []
    for c, n in g["counts"].items():
        bits.append(f"{GREEN_ICON.get(c, '')} {CAT_NAME.get(c, c)}" + (f" ×{n}" if n > 1 and c != "water" else ""))
    return bits


def fmt_ha(ha):
    return f"{ha:.0f} ha" if ha >= 10 else f"{ha:.1f} ha".replace(".", ",")


def green_view(start_v, g, prefer, avoid):
    """Schnellster Fußweg zur gewählten Grünfläche, in der Kartenansicht mit Begründung."""
    dog_weather(start_v)
    lat1, lon1, lab1 = _unpack(start_v)
    p1, p2 = (lat1, lon1), g["target"]
    dog_zones_prefetch(p1, p2)  # Verbotszonen holen, während der Routenserver rechnet
    t1 = lab1.split(" · ")[0]
    t2 = f"{g['name']} ({fmt_ha(g['ha'])})"
    nav = osrm_nav(osrm_route(p1, p2), f"{t1} → {t2}")
    if not nav:
        raise ValueError("Die Grünfläche liegt zu nah.")
    mins = max(1, round(nav["total"] / M_PER_MIN))
    why = green_reasons(g)
    if g.get("ok") is not None:
        hint = "Schnellster Weg zu deinem Grün"
        if g["ok"]:
            hint += " · ✓ " + " · ".join(g["ok"])
        if g["dev"]:
            hint += " · Abweichung: " + " · ".join(g["dev"])
    else:
        hint = "Schnellster Weg zur am besten passenden Grünfläche" + (" · " + " · ".join(why) if why else "")
    if g.get("failed"):
        hint += " · ℹ️ nicht abrufbar: " + " ".join(GREEN_ICON[c] for c in g["failed"])
    if g.get("note"):
        hint += " · ⚠️ " + g["note"]
    opt = _option(nav, "Schnellste", f"{fmt_km(nav['total'])} · {mins} min", hint,
                  [["Länge", fmt_km(nav["total"])], ["Dauer", f"{mins} min"], ["Fläche", fmt_ha(g["ha"])],
                   ["Passt zu", " ".join(GREEN_ICON[c] for c in g["counts"]) or "–"]],
                  endctx="dogrun" if "dogrun" in g.get("counts", {}) else "park")
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=None, kind="green",
                   options=[opt], default=0, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": _pack(p2[0], p2[1], t2), "area": None, "bbox": None, "buf": 150,
            "dog": _dog(), "temp": st.session_state.get("_temp"), "wnote": st.session_state.get("_windnote", "")}


def _poly_centroid(pts):
    """Schwerpunkt eines Polygons (Liste (lon, lat)); None bei entartetem Umriss."""
    a2 = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cr = x0 * y1 - x1 * y0
        a2 += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    if abs(a2) < 1e-18:
        return None
    return cx / (3 * a2), cy / (3 * a2)


def _in_poly(x, y, pts):
    inside = False
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def inside_point_of(pts, pos, ha):
    """pts: Umriss als [(lon, lat)]. Kleine Fläche: ein Punkt in der Mitte (liegt er nicht darin, der dichteste
    Randpunkt zur Mitte hin verschoben). Große Fläche: ca. 70 m hinter dem Rand, an der Stelle nächst dem Standort."""
    if len(pts) > 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    cen = _poly_centroid(pts) or (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    ky = 110540.0
    kx = 111320.0 * math.cos(math.radians(pos[0]))
    if not _in_poly(cen[0], cen[1], pts):  # unregelmäßige Form: einen Punkt innen suchen
        best = None
        for p in pts:
            q = ((p[0] + cen[0]) / 2, (p[1] + cen[1]) / 2)
            if _in_poly(q[0], q[1], pts):
                best = q
                break
        cen = best or pts[0]
    if ha <= 6:
        return cen[1], cen[0]
    near = min(pts, key=lambda p: math.hypot((p[1] - pos[0]) * ky, (p[0] - pos[1]) * kx))
    dx, dy = (cen[0] - near[0]) * kx, (cen[1] - near[1]) * ky
    n = math.hypot(dx, dy)
    if n < 90:
        return cen[1], cen[0]
    cand = (near[0] + dx / n * 70 / kx, near[1] + dy / n * 70 / ky)
    return (cand[1], cand[0]) if _in_poly(cand[0], cand[1], pts) else (cen[1], cen[0])


def _assemble_rings(segs):
    """Außenränder einer Relation bestehen oft aus mehreren Wegstücken: zu geschlossenen Ringen zusammensetzen."""
    segs = [list(sg) for sg in segs if len(sg) > 1]
    rings = [sg for sg in segs if len(sg) > 3 and sg[0] == sg[-1]]
    open_ = [sg for sg in segs if sg[0] != sg[-1]]
    while open_:
        cur = open_.pop(0)
        grew = True
        while grew and cur[0] != cur[-1]:
            grew = False
            for i, o in enumerate(open_):
                if o[0] == cur[-1]:
                    cur += o[1:]
                elif o[-1] == cur[-1]:
                    cur += o[::-1][1:]
                elif o[-1] == cur[0]:
                    cur = o[:-1] + cur
                elif o[0] == cur[0]:
                    cur = o[::-1][:-1] + cur
                else:
                    continue
                open_.pop(i)
                grew = True
                break
        if cur[0] == cur[-1] and len(cur) > 3:
            rings.append(cur)
    return rings


def green_inside_point(g, pos):
    """Punkt IM Grün statt am Rand/an der Straße: nutzt den echten Umriss der Fläche (ein kleiner Abruf je Fläche)."""
    typ, oid = g.get("id", (None, None))
    if not typ or not oid:
        return None
    js = _overpass_race(f"[out:json][timeout:25];{typ}({oid});out geom;", 22.0, _ranked_mirrors()[:3])
    el = (js.get("elements") or [None])[0]
    if not el:
        return None
    if typ == "way":
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry") or []]
    else:  # Relation: größter geschlossener Außenring; sonst alle Randpunkte
        segs = []
        for m in el.get("members", []):
            if m.get("role") in ("outer", "") and len(m.get("geometry") or []) > 1:
                segs.append([(p["lon"], p["lat"]) for p in m["geometry"]])
        rings = _assemble_rings(segs)
        if not rings:
            return None
        pts = max(rings, key=lambda r: (max(x for x, _ in r) - min(x for x, _ in r)) * (max(y for _, y in r) - min(y for _, y in r)))
    if len(pts) < 4:
        return None
    return inside_point_of(pts, pos, g["ha"])


CAT_NAME = {"water": "Wasser", "san": "Toilette", "sport": "Sportangebot", "play": "Spielplatz",
            "sight": "Sehenswürdigkeit", "access": "Barrierefreiheit", "dogrun": "Hundefreilauf"}
SIZE_WORD = {"Klein (bis 2 ha)": "klein", "Mittel (2–15 ha)": "mittel", "Groß (15–100 ha)": "groß",
             "Sehr groß (ab 100 ha)": "sehr groß"}


def size_word(ha):
    for name, (lo, hi) in GREEN_SIZES.items():
        if lo <= ha < hi:
            return SIZE_WORD[name]
    return "klein" if ha < 2 else "sehr groß"


def green_eval(g, prefer, avoid, size_range, size_name):
    """Was passt, was weicht ab? Ergebnis: ok/dev (Texte), Abweichungs-Schlüssel und Gewicht
    (Barrierefreiheit wiegt am schwersten)."""
    ok, dev, keys, cost = [], [], [], 0.0
    want = SIZE_WORD.get(size_name, "")
    if g["fit"] >= 0.99:
        ok.append(f"Größe passt ({fmt_ha(g['ha'])}, {want})")
    else:
        dev.append(f"Größe: {fmt_ha(g['ha'])} ({size_word(g['ha'])}) statt {want}")
        keys.append("size")
        cost += 0.5 if g["fit"] > 0.5 else 1.0
    failed = set(g.get("failed", []))
    for c in prefer:
        if c in failed:
            continue
        if g["counts"].get(c):
            ok.append(f"{GREEN_ICON[c]} {CAT_NAME[c]} vorhanden")
        else:
            dev.append(f"{GREEN_ICON[c]} {CAT_NAME[c]} nicht gefunden")
            keys.append("p" + c)
            cost += 100.0 if c == "access" else 1.0
    for c in avoid:
        if c in failed:
            continue
        if g["counts"].get(c):
            dev.append(f"{GREEN_ICON[c]} {CAT_NAME[c]} vorhanden (du wolltest das lieber vermeiden)")
            keys.append("a" + c)
            cost += 1.0
        else:
            ok.append(f"kein {GREEN_ICON[c]} {CAT_NAME[c]}")
    return {"ok": ok, "dev": dev, "keys": frozenset(keys), "cost": cost}


def propose_options(ranked, prefer, avoid, size_range, size_name, limit=3):
    """Wünsche gegeneinander abwägen: perfekte Treffer zuerst; sonst die besten Kompromisse mit je anderer
    Abweichung (z. B. „groß, mit Wasser, dafür Spielplatz“ oder „mittel, Wasser, ohne Spielplatz“)."""
    best = {}
    for g in ranked[:60]:
        ev = green_eval(g, prefer, avoid, size_range, size_name)
        g.update(ok=ev["ok"], dev=ev["dev"], cost=ev["cost"], keys=ev["keys"])
        cur = best.get(ev["keys"])
        if cur is None or g["score"] > cur["score"]:
            best[ev["keys"]] = g
    groups = sorted(best.values(), key=lambda g: (g["cost"], -g["score"]))
    if not groups:
        return []
    top = groups[0]["cost"]
    return [g for g in groups if g["cost"] <= top + 2.0 or g["cost"] < 100][:limit] if top < 100 else groups[:limit]


class NoGreen(Exception):
    """Keine Grünfläche gefunden – kein Programmfehler, sondern ein Hinweis an die Person."""


def _nominatim_greens(pos):
    """Notlösung, wenn der Kartenabruf nichts liefert: Parks und Wälder per Ortssuche (Nominatim)."""
    out = []
    for word in ("Park", "Wald", "Grünanlage"):
        try:
            for la, lo, name, *_ in _nominatim_poi(word, pos[0], pos[1], half_deg=0.09):
                d = 100.0 / 110540.0
                dx = d / max(math.cos(math.radians(la)), 0.3)
                out.append({"tags": {"name": name, "leisure": "park"},
                            "bounds": {"minlat": la - d, "minlon": lo - dx, "maxlat": la + d, "maxlon": lo + dx}})
        except Exception:
            pass
    return out


GREEN_NEAR_RADIUS = 4500


def _green_good(ranked, use_p):
    """Reicht das Ergebnis, um nicht weiter zu suchen? Mehrere Kandidaten, und (falls Ausstattung gewünscht)
    mindestens einer der ersten davon hat sie."""
    if len(ranked) < 3:
        return False
    return (not use_p) or any(all(c in g["counts"] for c in use_p) for g in ranked[:3])


def _green_progressive(clat, clon, cats, prefer, avoid, size_range, pos, log):
    """Erst der nahe Umkreis (schnell, kleine Abfragen), der große nur bei Bedarf. Liegt der große schon im
    Zwischenspeicher (Vorwärmen), wird gleich er genommen. Rückgabe: greens, feats, ausgefallene, Rangliste."""
    big_r = GREEN_RADIUS + 900
    if _green_ready(clat, clon, big_r, cats):
        g_, f_, fl_ = _green_fetch(clat, clon, big_r, cats)
        return g_, f_, fl_, None
    ex = ThreadPoolExecutor(max_workers=2)
    f_small = ex.submit(_green_fetch, clat, clon, GREEN_NEAR_RADIUS, cats)
    f_big = None
    t0 = time.time()
    small = None
    try:
        small = f_small.result(timeout=4.0)  # meist schnell da
    except Exception:
        f_big = ex.submit(_green_fetch, clat, clon, big_r, cats)  # langsam oder gescheitert: den großen schon starten
        try:
            small = f_small.result(timeout=14.0)
        except Exception:
            small = None
    ranked = None
    if small is not None:
        g_, f_, fl_ = small
        up = [c for c in prefer if c not in fl_]
        ua = [c for c in avoid if c not in fl_]
        ranked = rank_greens(g_, f_, up, ua, size_range, pos)
        if _green_good(ranked, up) and not any(c in fl_ for c in prefer):  # fehlende Angaben: nicht raten, weiter suchen
            ex.shutdown(wait=False)
            if f_big is None:
                log("🌳 Grünflächen in der Nähe gefunden")
            return g_, f_, fl_, ranked
    log("🌳 Zu wenig Passendes in der Nähe, ich suche im Umkreis von 10 km")
    if f_big is None:
        f_big = ex.submit(_green_fetch, clat, clon, big_r, cats)
    ex.shutdown(wait=False)
    try:
        return (*f_big.result(timeout=max(8.0, 30.0 - (time.time() - t0))), None)
    except Exception:
        if small is not None:
            return small[0], small[1], small[2], ranked  # lieber das Nahe als gar nichts
        raise


def find_green(pos, prefer, avoid, size_range, log=lambda *_: None):
    """Beste Grünfläche samt Hinweisen. Fehlende Daten führen nie zum Abbruch: es wird gelockert, ausgewichen
    und der Hinweis später an der Route angezeigt."""
    log("🌳 Grünflächen in der Nähe suchen")
    cats = list(dict.fromkeys(list(prefer) + list(avoid)))
    greens, feats, failed = [], [], []
    clat, clon = green_center(pos[0], pos[1])
    ranked = None
    try:
        greens, feats, failed, ranked = _green_progressive(clat, clon, cats, prefer, avoid, size_range, pos, log)
    except Exception:
        log("⚠️ Kartenserver antworten nicht, ich nutze die Ortssuche")
    if failed:
        log("⚠️ Nicht alle Angaben abrufbar: " + ", ".join(GREEN_ICON[c] for c in failed))
    log("🔎 Ausstattung vergleichen")
    use_p = [c for c in prefer if c not in failed]
    use_a = [c for c in avoid if c not in failed]
    note = ""
    if ranked is None:
        ranked = rank_greens(greens, feats, use_p, use_a, size_range, pos)
    if not ranked and greens:
        log("ℹ️ Nichts Passendes in Wunschgröße, ich lockere die Suche")
        ranked = rank_greens(greens, feats, use_p, use_a, size_range, pos, min_ha=0.05, min_d=30.0)
    if not ranked:
        log("🌿 Weitere Suche über die Ortssuche")
        extra = _nominatim_greens(pos)
        if extra:
            ranked = rank_greens(extra, [], [], [], size_range, pos, min_ha=0.05, min_d=30.0)
            note = "Genauere Daten waren nicht abrufbar: Fläche und Ausstattung sind nicht geprüft."
            use_p, use_a = [], []
    if not ranked:
        raise NoGreen("Im Umkreis von 10 km habe ich keine Grünfläche gefunden. Das kann an fehlenden "
                      "OpenStreetMap-Daten oder überlasteten Servern liegen. Bitte in ein paar Minuten noch einmal "
                      "versuchen oder den Start an einen anderen Ort legen." + f" (Treffer im Kartenabruf: {len(greens)})")
    todo = [g for g in ranked[:3] if not g.get("inside")]  # Ziel mitten in die Fläche legen (nicht an den Rand/die Straße)
    if todo:
        ex_ = ThreadPoolExecutor(max_workers=len(todo))
        futs_ = [(g, ex_.submit(green_inside_point, g, pos)) for g in todo]
        ex_.shutdown(wait=False)
        t_end = time.time() + 12.0  # gleichzeitig und mit Zeitgrenze statt nacheinander (große Flächen haben viel Geometrie)
        for g, f in futs_:
            try:
                g["target"] = f.result(timeout=max(0.3, t_end - time.time())) or g["target"]
            except Exception:
                pass
    for g in ranked:
        g["counts"] = {c: n for c, n in g["counts"].items() if c in cats}
        g["missing"] = [c for c in prefer if c not in g["counts"]]
        g["failed"] = [c for c in failed if c in cats]
        g["note"] = note
    return ranked


# ------------------------------------------------------- Sanitäroptionen (Toiletten am Weg)
# Datenquelle: OpenStreetMap (amenity=toilets, Zugang/Gebühr/Rollstuhl/Öffnungszeiten). Die Daten sind
# freiwillig gepflegt und können unvollständig sein; Kundentoiletten in Gaststätten sind nicht garantiert.
SAN_GASTRO = "restaurant|cafe|pub|bar|fast_food|biergarten"
SAN_REACHES = [100, 150, 250, 400]
SAN_NOTE = "Benutzung der Sanitäranlagen nicht garantiert. Nutzung nur für Gäste, Kundentoilette."


def payload_from_html(html):
    """Die Kartendaten (JSON) aus dem fertigen Karten-HTML zurückholen."""
    i = html.index("const D = ") + len("const D = ")
    j = html.index(";\nconst $ = id =>", i)
    return json.loads(html[i:j].replace("<\\/", "</"))


def _thin(coords, n=70):
    if len(coords) <= n:
        return [list(c) for c in coords]
    step = len(coords) / float(n)
    out = [list(coords[int(k * step)]) for k in range(n)]
    out.append(list(coords[-1]))
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _san_overpass(q):
    return _overpass_race(q, 22.0, _ranked_mirrors()[:4])


def _san_bbox(option_coords, reach):
    allp = [c for coords in option_coords for c in coords]
    la0, la1 = min(c[0] for c in allp), max(c[0] for c in allp)
    lo0, lo1 = min(c[1] for c in allp), max(c[1] for c in allp)
    pad = reach / 110540.0
    padx = pad / max(math.cos(math.radians((la0 + la1) / 2)), 0.3)
    return la0 - pad, lo0 - padx, la1 + pad, lo1 + padx  # s, w, n, e


_SAN_CACHE = {}


def fetch_sanitary(option_coords, reach, region=None):
    """Toiletten und Gaststätten im Rechteck um alle Routen. Zwei kleine Abfragen parallel (Toiletten /
    Gaststätten); die genaue Entfernung zum Weg wird danach hier berechnet. Klappt nur eine, wird sie genutzt.
    Mit region (w, s, e, n; das geladene Wegenetz) wird einmal für das ganze Gebiet gefragt und gemerkt: Jede weitere
    Route dort (z. B. beim Ändern der Länge in der Karte) braucht keine neue Abfrage mehr."""
    s_, w_, n_, e_ = _san_bbox(option_coords, reach)
    if region:
        try:
            rw, rs, re_, rn = region
            if (rs <= s_ and rw <= w_ and rn >= n_ and re_ >= e_
                    and (rn - rs) * 110540.0 <= 6500 and (re_ - rw) * 70000.0 <= 6500):
                s_, w_, n_, e_ = rs, rw, rn, re_
        except Exception:
            pass
    bb = f"({s_:.5f},{w_:.5f},{n_:.5f},{e_:.5f})"
    hit = _SAN_CACHE.get(bb)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    qs = (f'[out:json][timeout:20];nwr["amenity"="toilets"]{bb};out center tags 4000;',
          f'[out:json][timeout:20];nwr["amenity"~"^({SAN_GASTRO})$"]["name"]{bb};out center tags 6000;')
    res, errs = [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_san_overpass, q) for q in qs]
        for f in futs:
            try:
                res.append(f.result().get("elements", []))
            except Exception as e:
                errs.append(e)
    if not res:
        raise errs[0]
    out = [el for part in res for el in part]
    if not errs:
        if len(_SAN_CACHE) > 12:
            _SAN_CACHE.clear()
        _SAN_CACHE[bb] = (time.time(), out)
    return out


_TILE_GASTRO = {"restaurant", "cafe", "bar", "pub", "biergarten", "bakery", "ice_cream", "food_court"}


def tile_pois(option_coords, reach):
    """Ersatz ohne Overpass: Cafés, Restaurants, Kneipen (und Toiletten, falls enthalten) aus den freien
    Vektorkacheln von OpenFreeMap (POI-Ebene). Liefert Einträge im Format von classify_sanitary."""
    s_, w_, n_, e_ = _san_bbox(option_coords, reach)
    tpl = _ofm_template()
    # nur die Kacheln, die die Route(n) samt Randstreifen berühren (nicht das ganze Rechteck)
    dlat = reach / 110540.0
    dlon = dlat / max(math.cos(math.radians((s_ + n_) / 2)), 0.3)
    tset = set()
    for coords in option_coords:
        a = np.asarray(coords, float)
        for la, lo in a[::max(1, len(a) // 400)]:
            for dy in (-dlat, 0.0, dlat):
                for dx in (-dlon, 0.0, dlon):
                    tset.add(_tile_of(float(la) + dy, float(lo) + dx, VT_ZOOM))
    tiles = sorted(tset)
    if len(tiles) > 60:
        raise ValueError("Gebiet zu groß")
    with ThreadPoolExecutor(max_workers=6) as ex:
        datas = list(ex.map(lambda t: _fetch_tile(tpl, VT_ZOOM, t[0], t[1]), tiles))
    out = []
    for (tx, ty), data in zip(tiles, datas):
        if not data:
            continue
        for extent, gtype, parts, props in mvt_decode(data, {"poi"}).get("poi", []):
            if gtype != 1 or not parts:
                continue
            cls, sub = str(props.get("class") or ""), str(props.get("subclass") or "")
            name = props.get("name") or props.get("name:latin") or ""
            toilet = bool({cls, sub} & {"toilet", "toilets"})
            if not toilet and (not name or not ({cls, sub} & _TILE_GASTRO)):
                continue
            for part in parts:
                lon, lat = _tile_lonlat(tx, ty, VT_ZOOM, extent, part[:1])
                la, lo = float(lat[0]), float(lon[0])
                if s_ <= la <= n_ and w_ <= lo <= e_:
                    out.append({"kind": "pub" if toilet else "gastro", "lat": la, "lon": lo,
                                "name": str(name) or "Öffentliche Toilette", "info": "öffentlich" if toilet else "",
                                "kv": sub or cls})
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _sweep_point(lat, lon, word):
    return _nominatim_poi(word, lat, lon, half_deg=0.005)


def sweep_gastro(coords, step_m=450.0, max_pts=12):
    """Notlösung, wenn der Kartenabruf nichts liefert: die Route ablaufen und an Punkten entlang des Wegs
    nach Restaurants und Cafés im Umfeld fragen (Nominatim)."""
    a = np.asarray(coords, float)
    lat0, lon0, kx, ky = _route_frame(coords)
    X, Y = (a[:, 1] - lon0) * kx, (a[:, 0] - lat0) * ky
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    total = float(cum[-1])
    n = max(2, min(max_pts, int(total // step_m) + 1))
    pts = []
    for sdist in np.linspace(0, total, n):
        k = int(np.searchsorted(cum, sdist))
        k = min(k, len(a) - 1)
        pts.append((round(float(a[k, 0]), 4), round(float(a[k, 1]), 4)))
    jobs = [(la, lo, w) for la, lo in pts for w in ("Restaurant", "Café")]
    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(_sweep_point, *j) for j in jobs]
        for f in futs:
            try:
                for la, lo, name, *_ in f.result():
                    if name:
                        out.append({"kind": "gastro", "lat": la, "lon": lo, "name": name, "info": "", "kv": "restaurant"})
            except Exception:
                pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _sweep_photon_point(lat, lon, word, tag):
    r = requests.get("https://photon.komoot.io/api/", headers=UA, timeout=6,
                     params={"q": word, "lat": lat, "lon": lon, "limit": 25, "lang": "de", "osm_tag": tag,
                             "location_bias_scale": 0.1})
    r.raise_for_status()
    out = []
    for f in r.json().get("features", []):
        pr = f.get("properties", {})
        co = (f.get("geometry") or {}).get("coordinates") or []
        if pr.get("name") and len(co) == 2:
            out.append((float(co[1]), float(co[0]), pr["name"]))
    return out


def sweep_photon(coords, step_m=450.0, max_pts=14):
    """Zusätzliche Quelle (Photon): an Punkten entlang der Route nach Restaurants, Cafés und Kneipen im Umfeld fragen."""
    a = np.asarray(coords, float)
    lat0, lon0, kx, ky = _route_frame(coords)
    X, Y = (a[:, 1] - lon0) * kx, (a[:, 0] - lat0) * ky
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    total = float(cum[-1])
    n = max(2, min(max_pts, int(total // step_m) + 1))
    pts = []
    for sdist in np.linspace(0, total, n):
        k = min(int(np.searchsorted(cum, sdist)), len(a) - 1)
        pts.append((round(float(a[k, 0]), 3), round(float(a[k, 1]), 3)))
    jobs = [(la, lo, w, t) for la, lo in pts
            for w, t in (("Restaurant", "amenity:restaurant"), ("Café", "amenity:cafe"), ("Kneipe", "amenity:pub"))]
    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_sweep_photon_point, *j) for j in jobs]
        for f in futs:
            try:
                for la, lo, name in f.result():
                    k = (_norm_name(name), round(la, 4), round(lo, 4))
                    if k not in seen:
                        seen.add(k)
                        out.append({"kind": "gastro", "lat": la, "lon": lo, "name": name, "info": "", "kv": "restaurant"})
            except Exception:
                pass
    return out


def _san_info(t, kind):
    bits = []
    if kind == "pub":
        bits.append("öffentlich")
    fee = t.get("fee")
    if fee in ("yes", "no"):
        bits.append("gebührenpflichtig" if fee == "yes" else "kostenlos")
        if fee == "yes" and t.get("charge"):
            bits[-1] += f" ({t['charge']})"
    if t.get("wheelchair") in ("yes", "designated"):
        bits.append("rollstuhlgerecht")
    if t.get("opening_hours"):
        bits.append("Zeiten: " + t["opening_hours"][:60])
    if t.get("operator") and kind == "pub":
        bits.append(t["operator"][:40])
    return " · ".join(bits)


def classify_sanitary(elements):
    """-> Liste {kind: pub|cust|gastro, lat, lon, name, info}. Privat/gesperrte Toiletten fallen weg."""
    out = []
    for e in elements:
        c = e.get("center") or e
        if "lat" not in c:
            continue
        t = e.get("tags", {})
        if t.get("amenity") == "toilets":
            acc = t.get("access", "")
            if acc in ("private", "no"):
                continue
            kind = "cust" if acc in ("customers", "permit") else "pub"
            name = t.get("name") or ("Öffentliche Toilette" if kind == "pub" else "Kundentoilette")
        else:
            kind, name = "gastro", t.get("name", "")
            if not name:
                continue
        out.append({"kind": kind, "lat": float(c["lat"]), "lon": float(c["lon"]), "name": name,
                    "info": _san_info(t, kind), "kv": t.get("amenity", "")})
    return out


def _route_frame(coords):
    a = np.asarray(coords, float)
    lat0 = float(a[:, 0].mean())
    ky, kx = 110540.0, 111320.0 * math.cos(math.radians(lat0))
    return lat0, float(a[:, 1].mean()), kx, ky


def _near_route(coords, pts, reach):
    """Je Punkt (lat, lon): (Abstand zur Route in m, Position entlang der Route in m)."""
    from scipy.spatial import cKDTree
    a = np.asarray(coords, float)
    lat0, lon0, kx, ky = _route_frame(coords)
    X, Y = (a[:, 1] - lon0) * kx, (a[:, 0] - lat0) * ky
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    total = float(cum[-1])
    if total <= 0 or not pts:
        return [(1e9, 0.0)] * len(pts)
    s = np.arange(0.0, total + 5.0, 5.0)
    px, py = np.interp(s, cum, X), np.interp(s, cum, Y)
    d, idx = cKDTree(np.c_[px, py]).query(
        np.c_[[(p[1] - lon0) * kx for p in pts], [(p[0] - lat0) * ky for p in pts]])
    return [(float(d[k]), float(s[idx[k]])) for k in range(len(pts))]


SAN_SHORT_M = 3000  # Strecken bis hierher brauchen nur eine Toilette (etwa in der Mitte)
SAN_GASTRO_REACH = 250  # so weit abseits der Route zählt ein Café/Restaurant als Toilettenmöglichkeit (m)
SAN_WIDE = 500  # so weit abseits der Route werden Ausweichziele (Toilette/Gaststätte) angeboten
SAN_MAX_STOPS = 7


def san_zones(total, stop_s):
    """Wo sollte eine Toilette liegen? Nach einem und nach zwei Dritteln jedes Abschnitts (Start → Zwischenziel →
    … → Ende). Sehr kurze Abschnitte brauchen keine."""
    bounds = [0.0] + sorted(stop_s) + [total]
    multi = len(bounds) > 2
    zones = []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        seg = hi - lo
        if seg < 500:
            continue
        if seg <= SAN_SHORT_M:  # bis 3 km reicht eine Toilette etwa in der Mitte
            label = f"Abschnitt {i + 1} · Mitte" if multi else "Mitte der Route"
            zones.append({"label": label, "center": lo + seg / 2.0, "tol": min(700.0, max(300.0, 0.25 * seg))})
            continue
        for k, frac in ((1, "⅓"), (2, "⅔")):
            label = (f"Abschnitt {i + 1} · {frac}" if multi else f"{frac} der Route")
            zones.append({"label": label, "center": lo + seg * k / 3.0, "tol": min(700.0, max(250.0, 0.17 * seg))})
    return zones


def add_sanitary(view, reach=150, log=lambda *_: None):
    """Sanitäroptionen einarbeiten: Es soll etwa nach einem und nach zwei Dritteln (je Abschnitt) eine Toilette
    liegen. Fehlt eine, werden Ausweichziele angeboten und die Route lässt sich darüber neu berechnen."""
    payload = payload_from_html(view["html"])
    opts = payload["options"]
    default = int(payload.get("default") or 0)
    log("🚻 Toiletten entlang der Route suchen")
    coords_all = [o["coords"] for o in opts]
    failed_op, items, items_t = False, [], []

    def _op():
        return classify_sanitary(fetch_sanitary(coords_all, SAN_WIDE + 60, (view.get("area") or {}).get("bbox")))

    def _tiles():
        return tile_pois(coords_all, SAN_WIDE + 60)

    # Zwei voneinander unabhängige Quellen gleichzeitig: Overpass (Toiletten + Gaststätten) und die POI-Ebene der
    # Vektorkacheln (Gaststätten). Fällt eine aus oder ist sie lückenhaft, trägt die andere.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_op, f_t = ex.submit(_op), ex.submit(_tiles)
        try:
            items = f_op.result()
        except Exception:
            failed_op = True
        try:
            items_t = f_t.result()
        except Exception:
            items_t = []
    known = {(_norm_name(it["name"]), round(it["lat"], 3), round(it["lon"], 3)) for it in items if it["kind"] == "gastro"}
    for it in items_t:
        if it["kind"] == "gastro" and (_norm_name(it["name"]), round(it["lat"], 3), round(it["lon"], 3)) in known:
            continue
        items.append(it)
    failed = failed_op and not items_t

    def measure(items_):
        pts_ = [(it["lat"], it["lon"]) for it in items_]
        res = []
        for o in opts:
            info = _near_route(o["coords"], pts_, SAN_WIDE)
            res.append([{**it, "d": d, "s": s_} for it, (d, s_) in zip(items_, info) if d <= SAN_WIDE])
        return res

    per_wide = measure(items)

    def n_gastro():
        return sum(1 for m in per_wide[default] if m["kind"] in ("cust", "gastro", "pub") and m["d"] <= SAN_GASTRO_REACH)

    a_ = np.asarray(opts[default]["coords"], float)
    _lat0, _, _kx, _ky = _route_frame(opts[default]["coords"])
    route_m = float(np.hypot(np.diff(a_[:, 1]) * _kx, np.diff(a_[:, 0]) * _ky).sum())
    # Auf einer Stadtstrecke liegen viele Lokale. Sind es auffällig wenige, ist eine Quelle lückenhaft: weitere befragen.
    if n_gastro() < max(5, route_m / 1000 * 4):
        log("🍽️ Weitere Quelle: Cafés und Restaurants entlang der Route")
        try:
            extra = sweep_photon(opts[default]["coords"])
            if extra:
                failed = False
                items = items + extra
                per_wide = measure(items)
        except Exception:
            pass
    if n_gastro() < max(3, route_m / 1000 * 2):
        log("🍽️ Noch eine Quelle: ich laufe die Route nach Cafés und Restaurants ab")
        try:
            extra = sweep_gastro(opts[default]["coords"])
            if extra:
                failed = False
                items = items + extra
                per_wide = measure(items)
        except Exception:
            pass
    all_stops = payload.get("stops", [])
    stops = [(st_[0], st_[1]) for st_ in all_stops if not str(st_[2]).startswith("🚻")]
    wc_stops = [(st_[0], st_[1]) for st_ in all_stops if str(st_[2]).startswith("🚻")]  # schon eingeplant

    def zones_for(i):
        coords = opts[i]["coords"]
        a_ = np.asarray(coords, float)
        lat0, _, kx, ky = _route_frame(coords)
        total = float(np.hypot(np.diff(a_[:, 1]) * kx, np.diff(a_[:, 0]) * ky).sum())
        stop_s = [s_ for _, s_ in _near_route(coords, stops, 1e9)] if stops else []
        zs = san_zones(total, stop_s)
        wc_s = [s_ for _, s_ in _near_route(coords, wc_stops, 1e9)] if wc_stops else []
        greach = max(reach, SAN_GASTRO_REACH)
        for z in zs:
            z["pub"] = (any(m["kind"] == "pub" and m["d"] <= reach and abs(m["s"] - z["center"]) <= z["tol"]
                            for m in per_wide[i])
                        or any(abs(w - z["center"]) <= z["tol"] for w in wc_s))
            # Cafés, Restaurants und Kundentoiletten gelten ebenfalls als Möglichkeit (nur für Gäste)
            z["gas"] = any(m["kind"] in ("cust", "gastro") and m["d"] <= greach and abs(m["s"] - z["center"]) <= z["tol"]
                           for m in per_wide[i])
            z["ok"] = z["pub"] or z["gas"]
        return zs

    all_zones = [zones_for(i) for i in range(len(opts))]
    for o, zs in zip(opts, all_zones):
        ok = sum(1 for z in zs if z["ok"])
        n_pz = sum(1 for z in zs if z["pub"])
        if failed:
            note = " · 🚻 Toilettendaten nicht abrufbar"
        elif not zs:
            note = " · 🚻 kurze Strecke, keine Toilette nötig"
        elif ok == len(zs) and n_pz == len(zs):
            note = f" · 🚻 Toilette auf der Route vorhanden ({ok}/{len(zs)})"
        elif ok == len(zs):
            note = (f" · 🚻 Toilette auf der Route: {n_pz} öffentlich, {ok - n_pz} über Cafés/Restaurants "
                    f"(nur für Gäste)")
        else:
            note = f" · 🚻 Toiletten an {ok} von {len(zs)} Stellen"
        o["hint"] = (o.get("hint") or "") + note
    zones = all_zones[default]
    wide_def = per_wide[default]
    for z in zones:
        z["cands"] = []
        if z["ok"]:
            continue
        win = max(z["tol"] * 2, 400.0)
        seen_k = set()
        for kind in ("pub", "gastro"):
            pool = [m for m in wide_def if m["kind"] == kind and abs(m["s"] - z["center"]) <= win]
            pool.sort(key=lambda m: m["d"] + 0.5 * abs(m["s"] - z["center"]))
            for m in pool[:4]:
                k = (kind, _norm_name(m["name"]) if kind == "gastro" else (round(m["lat"], 4), round(m["lon"], 4)))
                if k in seen_k:
                    continue
                seen_k.add(k)
                z["cands"].append({"name": m["name"], "lat": m["lat"], "lon": m["lon"], "d": m["d"],
                                   "s": m["s"], "kind": kind, "info": m["info"]})
    n_public = sum(1 for m in wide_def if m["kind"] == "pub" and m["d"] <= reach)
    n_cust = sum(1 for m in wide_def if m["kind"] == "cust" and m["d"] <= reach)
    gastro_marks = {(round(c["lat"], 5), round(c["lon"], 5)): c for z in zones for c in z["cands"] if c["kind"] == "gastro"}
    seen, pubs, custs = set(), [], []
    for mine in per_wide:
        for m in mine:
            if m["d"] > reach:
                continue
            key = (round(m["lat"], 5), round(m["lon"], 5), m["kind"])
            if key in seen:
                continue
            seen.add(key)
            if m["kind"] == "pub":
                pubs.append([round(m["lat"], 5), round(m["lon"], 5), m["name"], m["info"]])
            elif m["kind"] == "cust":
                custs.append([round(m["lat"], 5), round(m["lon"], 5), m["name"],
                              (m["info"] + " · " if m["info"] else "") + SAN_NOTE])
    for (la, lo), c in gastro_marks.items():
        custs.append([round(la, 5), round(lo, 5), c["name"], SAN_NOTE])
    # Für Abschnitte ohne öffentliche Toilette die nächsten Cafés/Restaurants auf der Karte zeigen
    greach = max(reach, SAN_GASTRO_REACH)
    have = {(c[0], c[1]) for c in custs}
    for z in zones:
        if z.get("pub"):
            continue
        near = sorted((m for m in wide_def if m["kind"] in ("cust", "gastro") and m["d"] <= greach
                       and abs(m["s"] - z["center"]) <= max(z["tol"] * 2, 300.0)),
                      key=lambda m: m["d"] + 0.5 * abs(m["s"] - z["center"]))
        for m in near[:4]:
            k = (round(m["lat"], 5), round(m["lon"], 5))
            if k in have:
                continue
            have.add(k)
            custs.append([k[0], k[1], m["name"], (m["info"] + " · " if m["info"] else "") + SAN_NOTE])
    payload["san"] = {"pub": pubs[:80], "cust": custs[:40]}
    view["html"] = map_html(payload)
    view["san"] = {"n_public": n_public, "n_cust": n_cust, "zones": zones, "reach": reach, "failed": failed,
                   "missing": sum(1 for z in zones if not z["ok"])}
    return view["san"]


# ------------------------------------------------------- Logo / App-Ikon (eingebettet, keine Zusatzdateien nötig)
# Blatt mit Zielpunkt: Kleinform fürs Ikon (Browser-Tab 64 px, Home-Bildschirm 180 px)
ICON_FAV_B64 = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAALg0lEQVR4nNRbe3AV1Rn/7WYhb0LIJMQAEYEESkTAGZGK1ldHfNDagUaxKloKI9gZR8A+BrQTWxnGUrHTsZ0iRXmWN/pHgYaHPA1xWkiCGggPKa+QIPJICCGQ7Pbb3XvvnrP37OveC+I3udk95/v2u+ec3/ke55y9CnzQSzPHPqxK2uOSJA2DqhVoQHdoWjokQFOpJElU1PQisfWrWQ7zpRBfJ/Makie+KuDbn3fVH9JHNy1000jleqr5TIO2buUf12316pvkxCgre0A5nZw/gZS9QcoLzNYZf/qXhYuhOrMcuYT4lqjG3PN8rszqlwTykYtAntVv8evp35uN6S0fbCvb1g4/A1BWViY3pBx8StPU31OxyC8yLLJ+kOfkRfpZeYiR18uq4PuNmcXKAwdVFa+vnrNutTWMJiXZB6D/yPynSOZDEuvOIyvxyDOdC/M5ZCXJhoyNz3U+pB+MfvD6jc6AnRnmjcTqD/Ej8pb+HEnGEwPvKfqituJQHdtfmS1MmvXMGHr4n3SbqoFFDsy0ZBHjyywfGv88j3y0HjjUi+SF+uEibzLSJE1aWzr1safBwRCiyTOfuUOVtUrSk+poY5E6Fvlv0eY92qPx/8KPtZL8iDVzNlTpBWMGlK4sTVIldRn8IA8BYizyECMPNyS9ZpZdHv7kJViDw9Sn0hcuKi0tTYoMQLcjSROpcqCTTWseZUNe8pAX8eEsL7J5oX6JkWeRB+ug7e3Vbld7tkw0ase//eNMRU07QpW51sgFiMM3Ls7z+iGQD/MlgTyr3/y+M6kdUpHcWU0bE+68H+QlOHt/obxTtJDEM0OyeXO+89EzR4Q8nNoDTj6vNUkbLRMAD1gjhcTZvKBeqF9zt3lH23byEfCQZ/kahit03xt+bTaIzSMxNi+K81HIO80UMfIMHwNkquzNtALf2TgfBHmrv/kKjUM+/503kc0j9HxibD56pqhavkL3yZw3hntuz480k9sz3txCQLLNLO/cnkNecsztkZGWgcL8QvTI64FeeT2xZ/8eVNXVIMrbi9pr6c9SnJC8meJ8XnZ39OvZF73ye1Gn6dP9VnTNzAJLp87Uk3y1J/IWhCZfiYw03OM8hwwCxHkWSfiL85npXTCobwlK+pifnKycSEcv1tfh3JebcTkzB3n97oKSnG7UZ2VkeSMvmOnKzWTzuV1z8fBdD+P+O3+A9NR02Ong9sU4tGNJpJyc0Q3ff/EdpGcXID0lzRt5QfuVmG2elUfsNq9f+/Xoix/dNwp3DrgTTqSpHfiqcg1X13bpHE7t24Li+5+HoighDPj2qLzNR810oQ+4UXF+cNFgjLp3FIoLi+BFrU1fo+Nqa3T9hUbjKjm0127zXHvCPuBG23y3Ljl44YlxGFo8BH4prWs+MnNvRfPXx7j67MISsx2hMudTBDZvb49yo21+5PCRKH3op+jcqTOC0rXew5HUdBYdbS1GWXeCPQc9FOFrNp/iZ+YqNyrOpyWnYfLoSbij6A7EQvsOf46/bS1Hr9xCvDnhl0jJyEFyZjdrcK5dZdrlbPP2mRH2HN42H0ec75HbE9N+NoULZ0Epu0s2mU428inxybol2mdcaL4ojPNce2z9QcQHXEebLy4sxpSxryKNwlQ8pGd782b83ZF/vvmCL5sP88P9la+nzQ8tHooZL06Pq/OXr1zGxsrNuHT5kqvcpZZmXzZv9xEyrtN6vk9BH7w8ZjLipaXlyzH3o3lYvmmlq9zZC2fN77etGqPaa+uXHJZJZJzP7ZqHqWTzsXh6OxX16mdcB972PUcZlU49jp0+4W7zEGeEyvWI8889+iwy0zKRCHqA0mL940anv2lA27U2V5sXZbx6f+VEx/n7htyHIcWDEQ+VV27ChJmT8J/aPb7kDx47FGkPfCIf5ssRm9fis3n92jWzK555ZCzipSMnj+B803nUHq31JV9TV+3b5jUb31oLJGA9P2zgMGSkZiAotV29gm17duCewfeQ6WQYafKgfoNwL5W9SG/T3gPVjnFehDyYmS6LkJQiWvwhr5NMCh8f8RhioVVb1uL9j+fjffL2OqWnppEpjQhlpu5Ue3Q/hcrWKGSF7ZeE7Y7P5sOC+sImOzMbsVBetzyjs8NKhiEo7aquENo8W7bPjOi1ACxvz9uQv/W8zr/79uEISifOnDQyvEfu/qHxCUrtHe2o3PeZp7eHyyqXzwPi2LcfHHCRs+Bfi/HqO9OMLC9W2ly5BU2Xmjy9vdNaR2fIsdo8K9+XNixTk1MRhMKZW0pyMmKhq7T6W7Fptau3lzx8gk5KrDaPcJHKhbcUIii98vTL5DQfdc3w3Gjdrg24SOh7xXkh8oy8zNt8MOTD174Ft8EP6bLb9+40bL9zp+SYO3/m3Bms2mTuD3rFeSHyDN9xLRBk396v9/903278ZcV7eHfpnxEPzfvoQ1y52uYrzrPtBRh5UR4QCHnG26b7XO52z85FJ6UTBseRKq/YuIpOgfb6jvNuV739SiLO57tQCuxG4XBXRLu/y2cuQaxUvnsTVpRbb7r5tXk2Wtn5shfyTiMXNUMcaOm/lxnhbuG6xYiHyis2Yu6af8Rk824zI7IWiOd8PrwhqVMz7dw0kpPq17OPUZYl80281OQUxEIdtNaft3Y+Nu7W84XgcV6EPJj+Rp8LmNK2kXTYYwvxr7Zbb6H+bm4ZjjecwK+fn0bZ4TCMfvBJDO0/BP1vLUZQOt90AbM+eBuHT37luJ5XPc4xzH45Z7RKIs7n25gTm5TOKQbq7XSUpVMylQf07o+g9OWRWsxe/C6ampvghLzmI8477naHuqmwyAc5nzfr6eyNGtdwrpEQNjv5+vjpJKsay9qgpD9XUVOJleTpTzaeiim3jyDv8xxD8RPnnZDXy73yCghx65VjfSkblOr+dxC7aiqw/b870NLaEpqIUkLifFjePpPDHVNisfnQFpKx9fXac9MgyzKq6qqo8ZeNDZEMGoRMOq/PSs+kXJ9fI9SfPY1ThO6JxpOoO1aHLw7Xkgm18T4o4DmFH5uPIA9GHqG1AOzIe9i8zhjQewCmPjvF6LxOO/buwu7PKw0hfqTB+xSGb94j6vshmolwt2m/Ns/5CJh5QJs1kmCUsSNsXfXH9NdVfjPuV1CSzJO1in2VZucB8OfzAj0Itm8fb5yP0s9fL+o7WQ0c8iKbZ0ZOf0dn+s9/S97dXMZW0Ybkeyv+aqLphAysstUmD++NxMR5D/0N+gxoECFjR0CXyM/JJy8/I2LX+2k/7k+L5xje2+9b5k7Ixrqe95y5cJWnAVClaj82n9OlG974xQzj9TSdjlByMmvBbHR0dAiRBKKRdCo7Ie9k0who85rmkNFCqyYX1vGx5DaioWGYOHqCcUSt07HTx/HW/Jm0K9PmOsJhBBJt8who887905bLy2avL6eqKq/37dvbzczuNIWxP8x/C61XrjgjwzxvtSkaWWjXP86LkDfENNRsXvBppZHBlIzo30hSY60RtroRjrN7DlShqaUJizcsRbN+FG3jR8VhBnnR2SKPLP98kDgf1h9B3hbn2efBZLySLE0+Wn38gGQOIaSnXxu1h1zZ0EiwjowcM71CrJjjPP9PqJ+TZ/gWJOCQj8hLgdpT88miXcYbWuFfjZEvVMcTtxUibwyxzX4H4rzo2kr4jwv12/rZ3KrZ66tJ5gWrDbxPkOCSgcEmr900cd6yebOskp7ntyzauS9qAHRaPWfdKhL5Cd1edkMeopG/CeO87UrI48mtC3dyr5tKENCYqY/pvmA53RZ/GzbPPQ9vm9c45IXtOUjAPrV1ya4a2EiGgPQfFX6T1VpCil+i1VQ924gbYfOJivNE9bR8nKweV0pEnddJOAPsNPqVkQ+S5BOk826atwWkvDv1K50daX5QYOscz7fYNuRDkDkib+Pb9LdQexqpS7STolZomrz+k4U7dsCD/g8AAP//uy5qcwAAAAZJREFUAwDfGrshkryDKgAAAABJRU5ErkJggg=="
ICON_TOUCH_B64 = "iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAIAAACyr5FlAAAQAElEQVR4nOxdB2AUxf6eC+khIaSQS0hIQnojJLQgVURaUBQQLBR9yuOJguWv8sRnRxHloaLyFBWfhSeiSHsiCDyK9FCSCCEN0nvvPfv/3e3u7OzdbXIN5pD9xMvc3Gz/5ptvfrsza91YU4xkyNAFayRDhgRkcsiQhEwOGZKQySFDEjI5ZEhCJocMScjkkCEJmRwyJCGTQ4YkZHLIkIRMDhmSkMkhQxIyOWRIQiaHDEnI5JAhCZkcMiRx05CjobkhqyiruLKoqr6qprGmur66oaURMQz8xLAl1En4UHBJ1X8KpFD/ReIiOM3+r1qCz+ZLEt8Zcq0Mn1LnK0Rb57ckfKgKuTi5uLu4ubm4ebh6DvLyiwgId3Z0RjcDFBb+JFhBRcGZtNOZhZmVdZVwWdTXQ8FddfWF4q6W6o+Cu1rqS8mVJC6mKkddBqnzFQpEpNl8oQyxfpzml+rGaYU6zVKmx6VEe6gY0H9AREDYuNixAd4ByIJhueSoqK34NWlvytUUoRYLtVMh1gxSAXBRhaj26/5fIVYTdmWkZhBLE5qhpVi6NUNqPdyKFGhE+PC5k+Z49HNHFglLJEd9U/3+c/uTMs50dnVx7QISaUY3r+caNZJhujVqMNYMpGD5pCqjpR9cGQWvB93dnK5gbeBWIKUZ3fz6e9MMdovdQlphZWU1adjtM8fMsMC2xuLI0dza9OWvX+aW5SLNiq9TM5BQFfky19Vn9KwZSLyAhqARqyT5qsoI9QtePveJvo59kSXBsshRXlu+6ZfPahpqsE4gdP18hpZm6Okz+MUM8hm9uhzP/p5Pz1vu4+mDLAYWRI6swsx/7/+qtaNNonb+KXyG1lKkgjnYOSybvTQ6KBpZBqyQZaCxpXHr4a3ADK5uqTO5uoU1gz+bXK1Vpflai5mh4FbIn3E2n80W2IM1Awl1F/FpoRXg6zrWAG7ViM/nSKvaQ2Gv+PJ4r0TrUah/QIjQP8Stubm1ecv+rdBpR5YBiyBHV3cXaEZtUy1fO7nzzrcsCjKiIFxFhIhfFVL1GLdN2poh1nZB84mWC4nbCHYFGi0XzlEQ3kKVxksxDCJbHHYPBcYg3MYpSqpKPtr2CZwQZAGwCHLsOP5zTmkOd/3IOo3YesbWQqwZotaE1BUW7LVUQ1szFLgM0qzH/La067qGZiB+KQ3NELyIoAeEDqnWI2gSoRlCy6LOycjP+vbX/yALAH1yXC3OPpV2CmElUGhqBtJuBXDtZwhmCB+4doo0A19vRGoGYngzo+DT2OGKar+gGcKeCDlI2CtNzSDXjMSagQjNUB82t7bD54+k52Yg2qBPDghpXB+fwVwfn6Ewr8/Q2Gd8WvYc/wXRBmVygGxcLcpGBvkMRk+foTDFZyBJn4E3Jsox1GdoaoaYGZBx6drl3JI8RBWUyXHowiFNH9Crz1AY6jOQaP36+QysBzp9Bp+vQ1ekfQajWzMYQufE2rbz6G5EFTTJAZ4cIqE6NQOZ02dwmzPYZyDz+gxFzz6DX41wFFdy0zu7OhE90CRHVmFWa3srMqfPuDniGVI+g1tKBVV+S2tLWs4VRA80yfFHzh+qP7dkPAPp8BnCUXBtDWJSsy8heqBJjuKqIn19BjLUZ9wE8Qwtn4H3mTsF8Ce7IBvRA80nweDW/C0ez9D2GYjgE/ypa6xD9EBNObq7u+ua6uR4hobPQEh0vNX1tV1d1ELp1MgBd1LUdxBu9XgG0vIZhCapOnRwSxJRAjVyNDTX3wrxDKWbl4OdPbesrngG3jj2GYz4nNTSa1loeg5z+wxutbTiGfZ2Dv7KQQM9vb3cld7uSqWHcqDnQCix9ut3kzNT9PcZgmypi3Z0dCBKoEYOfFEUQquvx3Og2GcQXsR8z4EiXtT1eg7Uwc4x0Ns/0CcwwCcg0CdA6a5U6YcWHO0cdPoMfv+7RT15vtXr5o8FKRAtUCOHQqHLZ9wM8Qw/pW9caFxcWFyw72CFovd22cHBSewzSM3Q4TMQWWcQPngKoKkcRI3U5TOIvomCcwbamoH5cb3Gm7BlrKz6RAZEDA0ZEhceP6C/p+axdHflJe2uKUrvaGt09vAPHHWvvYtQxsneUTg6rm/C8CsmvIhCfE6QYItpwRI8h0XHM5Tu3hPix48Zclt/l/46j6K5piR559qaQi7OXZF9riB5X+ysF7xCE9gcR3snQ30GsT80QXU4pGE+Qz3exPw+g+8xiTXD3tZ+7NAx42LHDh44uOeDyEnahZnBoqO1Kf3g556D462sbeGrk4MTMtBnkOcE0QNl5bDAeIadjd2k4bfPHJPo7KTHKCOmO//8Xu3sxqrC0oyTPlETIW1jbd1rPAOR7azYeyF6oEkOjZZYb58hLqO/zxAugo6l4I+Njd3kEZP0pYUadWU53Z1tOn+qL73KkoOs+734DEYUF9aI8dx40FUO7g/1eIZ1H5uJwybcNTbR1bk/MgR93XwQuU8EXJRBwpEa4DOwD1PgIXO0QNdzoN58xo2IZ0QGRjw07SHfAb7IcPSxdfAKGVWWeVo7382PHJukp8/Q0D8+TkoJ1D0HzXgG9EQWzVgQPdikEWZR056szE3pam8hMyMnP2bv4oGPU594hrZmIGFn6YCu56AZzxgdnfDo3Y/a2tgiE1BdV/3m5rWoW3m3r3N3XVFHa6PzgMDgsQ/gfiziaCntM3R7L3ZsMLqleyvqP/jjBsUzwGE8nLh43NCxyGTsOf5LfmkBJLb3sfv4Bd0jkVpamnvzGdzZIDQD36mmqR2UPceNj2cM6D9gxbzlfl5+yByob6xnE55uA6TK1DbU6hPP0PAZ5DlBlECVHLp9xnWMZ/h6+q5c9IKLkwsyExYlLkjPzWjvbF8w9X6pMk2tzfrEM7R8Bu4E3ZrKcWPjGYHeAc8vfAHudCDzoV/ffp+s3NBzGZVy6BPP0KUZt7DnUH3coHhGuH/4sw88Y2drh244autrGT3iGdqawR4jRVAfK8vWD/YkMNoeDQnKgfg03wYxvT4HyjXZg5SDnn3QnMxIzUpd+806PUekQbOCWzqtJ9ox74XjxeeEOxZ6oOs5dGgGMnc8o6+D89P3PwV3TJCZ0NLW/PoXb0Hi7OUkpbsyIXpkD4Vb21uLK0sYfeMZCvKccB1ceqCrHNqawZ0L8gwioqfHRwv0HW9iZWW1Yt6T7mady7GwrAinPV09ei5cWlnGdHcj3OvQ1AyFTs3Q0FFaoEmOG/B8xv133h/mH4bMipBBIXcMvx0SQ4Jjgnx7uaFfVFGsbzxD4njpdVYoNivcgTPXKZ4Bv4b6hUxNmIKuA5bd9zf4p0/J4ooiXid6j2cQvW7ROaEFesqhMMlnoN6ez7C1sVtyzxJkPuw4suvzXZvZkd/6g22D9IlniH0Gg2gzA9F++vy6xDPY9ENTHvBy80JmwuY9X/9yXPVQD2z3sbsf0XMp2JPkjBRNnyGhGZr3koj2kRboPX2u+rhez2cE+wbfrrYF5sKZS2fYRGVtpf5L5ZbkN7U0I/3iGRrHy65B51iHGwZqzQqDRB4NmSOegce13jPhbmRWzJs8Fz7hFu7C6Q/qv1RKRrJB8Qy2DnDHcisrh3BezPd8Bpvj7+0fHRSDTENKVkpLWxuOYdwxYlJ0UDRcZqUhTdXFjBQjfQZ/vIgqKJKDVAtzjje5Z8I9Jqrxxh8/PXTuMCSW3rtkSsJkNtNL+r6rTnR0dsA9OR3PZ+jhM/j0LRznQOaLZ+Acfy+/+LA4ZBqOJR9nE5euGj+xzm+nDnR1dekfz9CO8XCeDFEDZXKY4DMUGj6DfapqVHSC6SYuccx0+PRwdb9v8hxkLI6cO2qYz0A6jpf7lRIsY8SbCfEMjZz48HhkMhbOeGji8Al+Rj1yzCIrL+tacS5ChvgMRPgMIQKmuEWVw+zzZ/i4e/t4eCOjcPrSWTCh+KspzADsP3UQSWiG5r0kQjPwGUBiHaUFus9zmBrPEOqZOsdo2dh3cj9EPyHxyMxFM8clItNQVlV29PxRQ+MZIp9BKiI9WILn0N9nIJ0+gy+PIgLDkVH434WjbOIUH+wyBT8e/LmL4LG+8Qyx9yLqADVQvSsr4TP0jGcg8fwZ8HugdyAyChP4J9HnTpqNTINKNs4dNSaewYg0g/pdN2QZ41aMjGfw9Y9L+3j46D/GVQOJY2eEDgrxcPWQmmdBf/x44OduhjEiniG+/0ycE3qwBM9hTDwD+wy2dkJ6kGmjDUIGhSCTcT7twuFzR0z3GQyhiIgeqD99buT8GQqhneby/X38Ddp0Q3Pjll+3dHZ1PzjtfjeTBQOpnwj8bPvnGpqBdD2fweC7yupjxJqhfn5FU0cRPVAeZW90PEN7KRenfoZsHK377n0+AMo8OW8ZMhlf7/m2sq7a+HgGblnE3otiiJRqnMPYeAbRf1HwFZVx7WsYOWobathEPwMX1IkrOen7Tx5gd87YeAYf4yHvOiGaoNtbMTKegX0G7ulAjqOBo5VWzH/SX+kXGRh5l8mBjbyS/NVfvGM2n4GIdlNdElECzWdIEUISPkOIAegzHyibdnQwjBxBvoPXP7MOmYyquuo3N73V0tbSk89QewhpnyHpvTjVoQTLmBPMkHiGhmZg9jja9UKOuoa6Lfu+hwUWTn/I6E6vBppbm9d89W51fS2706bEMwifIZwHPC8lFdB7hpTVWMPjGQrCu+HIAXx29PbGq3VbPkjLSUPq7T4+ZykyGRDveu3TN8tqKswTzxA/G0umaYHmM6SammG4zyBb9I7O9p632NbBPTje1t6GTMbVwmtvbHq7oalB0AzT4hmM2HtRny0OUX2GlP3D9Px8BsJsQISfx3VL6AWgjg6Rcvx8eOcrn72+Zd9WnJM4ZgZ8jhky+uGZi5BpOP3HmZc+fkX90nlG8r4JEvfFiL6JRj8LCTqhebwU+7HIAubnMDieQfoM7uyri5DKkZqVqrIXCF2+lhYfPjQiQHVDbkL8OPiHTENTS9OG7z9OSrvA6QUyWzyD9Bl4KXpmVAULmk2QVwIDfAYi2uZ6lcJz6NNHOC4zvukoNeuPj7ZurKqtMs99E+ljQUhBaCo1WMSTYMhYn8EQLXp5dRlec9TgyHsnzsrIyxoWERcdFIVMRmZe1nd7/3P5app6/65PPENDM/AZoAfK91aMiGcQ9QyR9a9cPNxogSEDTHoAGM+t+7edv3JR+9lx88YzdGkG2URRgAXMCWZgPIP0GXxadd4rVF1Ks6GitvL4xRP/O3u4uKIEIWTG5zMYiXgG6TOwaiKqEXTL8RwG+QyRZvh4eP919mOeWq9BMQKNzY0nU88cu3AsPSeD4a+NceNNDI9nCJqhURNogfrc58b5DNy6M15unq8uedmlr2qCwPaO9gvpF12cnPs5u/Z3du35bgsEN4sriosqSkoqSnJL8vJL8surKxDi7vmZYbyJTp+h0MtnMIQbowiaI97E3sIYn9HfxfUfj77EMgOpx7LuOrYnpygH8VEQJ0cnBzsHJmwLFwAAEABJREFUe1s7B1sHuJZtHe3NLc1tHW3NrS2dnR2Gzgeq22eQtV+P5zN69hla3gtRBN2xssbEM3Dt7NfX+ZXH/uFBTLyUnpvOMoOr9oiBZqKxqZH7JtR+dgUI12BT588wRzxDwnshiqAWIVW9M7HH5zOQqBVHhBqryjjaO4BmKN2VeIWlVaXrv/sQidWYP79CveczcFqBmdTzWHgzPp/B7wm/Hp5DiPQZ/DmB9hFRAjXlUE0jbFQ8A77Z2dqtemQV+RKM8pqK1za9Wd9cz4hqOdYGBbEtHe1675phis/QL54h3h9BR13pkYOectg7ghsg6xAjfd+E1AzrPtYrFz83eGAgXhWELF/77I3ahlpuKRUUGu2F2EMg4+YDZRgDxpvodd9EgUTPuQkyxx2vg72jrY0NogSaT4KpxMPAeEafPlbPPPh0REAEXglw4rVNb1TXV2s7A/4aMOStGwWhFkgfzUC6fAZj/niGyGfw+waOG9EDTXL4efn15jNEmmGlUDx1/4q4sKF4DfWN9aAZELBiCPPWk89ASFQ7DZsP1ACfIdIMhDWmV58haAabDjTwkXrzgiY54kKG6h/PgPTS2UtHRA7Hi0NP5PXP3yytLjPcZ7A5lhXPwJqh4LcOHwkxIxE90CRHTFB0H6s++vgMQFxY3Pg44fU5wIw3vlhdVFFsuM9gjPQZqGefoUsz8FJaPoN8ZkXDZ+Bjt7a2HhZp6kQ0poAmOcCQhvuHM73FM9jaOXrIaLwgBDff2rxG9YokrloSOo90aMbNEc/Q8l4xwVHQY0f0QHmU/cT48UjTZzAamsGe2cNJh9lFgBlvb16bU5zLCBetR59xk8QzuDUTOjptzJ2IKijfso8LjQv2Dc4qzCI0Q0eLDum0nCv/3PK+0s3r9+QTbK8VGeYzhJbLMuMZ2GewexgRGJYQMwJRBfX3raDZE2ZJ+QxxnWaSLp/b8/svf8p4Bnm87NE9PGsBog365AjzD4seHEX6DEasGZgBHESawfw54hmkZkCZkdHDQTkQbdAnB+DhxEUujs4aXl31A5ej0V/4E8YzSAa7OPZdOvcvyAJgEeRwc3F7cu4yW2tbDZ9B9l+I+AdR+xnmzxHPwJphY23zj7+u9HA159uDjIZFkAMQ4hf82N2q6qLhM3ABw33GTRPPwEcHX59duDw8IBRZBiyFHACIfv511mO2qpexMVo+g/0gdB4xf5p4BlsGNOO5xU/dFjsKWQwUjTXFyJKQV5r//tYNNfU1/DUQtQv8qWZIV6gQajmjMQqGuAaIuE4KsRLofqJTYINR4016fp6eLAPLQjvyjyUrAwf6I0uCxZEDqW+0fvzTv7IKspHQCqgh6Aeu/Rr5QprjE0MsSv6vaykuW9SWEUuLyyh62rqwsFBGgST2WRHmH/zKX/9uroH/ZoQlkgPQ1d197OLvO4/uqm2o49uIbl4/RJohfg5UL81AQo/jeo43IZ8D1dwrLu3u6nbfnfdOvW0y3GNClgcLJQeL9s72Q0mHfztzsKq2iq+TYs1ARE3XUzMYpKsia4pDj5qh0FAp8a6JGj/CKuO0qohHf/eZ46Yljptma2OLLBUWTQ6M/LKC8+kXL6RfyCnO0+UztGunqKU3g88wfLyJTs0Y7BMwMmb4qOjhg30DkcXj5iAHifqm+ubWlpa25tY20TQbjI6yjOEZOn8mqnzPS+n6wd7OHm6uOjk4mmVmuhuJm48cMm4YaA+klmHBkMkhQxIyOWRIQiaHDEnI5JAhCZkcMiQhk0OGJGRyyJCETA4ZkpDJIUMSMjlkSEImhwxJyOSQIQmZHDIkIZNDhiRkcsiQhEwOGZKQySFDEuYhR3NzS35BfkFBYVl5eXd3d0x01NDYWNV8LBaPmvqaoooiNu3l7uXpKppdv7ymAr/Gxd8nwNmhL/lrXmleA/8GoCDfYAc7e9Tr5hpqL2amsOm40Nj+zq6NLY25xblsToBPQF/xJujCVHLU1tZ+9fU3+w8cFI0+Qkjp5fXA/HnTpk5Blg24/G9/8x6bHh2TsHzu4+SvWw/+ePrSGTb94JT5M9VvicP48IdPStXUsbKy2rzqM6QHgIhf7N7MplctfgHIAczAO7Bq0fPRQdHIYmDSWNnc3Nwnnnpm328HNJgBKC0re3/DR2+tfbfdHK9ivH4IHRSCR45k5Wdp/JqWcwWnr+RmkD81NDeU8qIS6htsycNPjIbx5Ghvb1+9Zm1lJfd+JBsbm7sSE+fPnZswciR+xdqxY7+v/+AjZMGAti9mcCSbrqyrUk12y6O4sqS+qR5/zRRT59LVSzgdZY43hVkgjG9Wftqxo6CwkE1Pmjhh2eN/c+7LtZdVVdVvrllz5Uo6pA8fPZqYOD0mSsfp6+rqyr561cXZ2dvbm8xvaGzIyc2DyxYZHt6nT0/jBEtKS8vLyh2dnEKCg1BvqKuvKy4q8fT0AJD5oOTnM5LZ9JXc9DFDbmPTl6+lkcWaWptyS3IDvAPYr+m5mfin2OAYnO5muqvqqoBnTDcT6BPgYHddpgMsqymvqC53tHcip/mWQkFpQVtne7Bv76dIA0aSo6Wldfv2HWw6MCDg2aefsiGm6HZ3d3v5xb8vfORR9s2Me3/dx5JjauJdbIFnnlre3tb25b+/bm1tg+v68YcfQCaI0Hfff5907gJWI7iQ8+6bO3P6dGjUya1nZGZ+/8O2lNQ/mpub2ZzQkJCHFy0cFi/M2om39erLL3l7KT/8+OMr6Rl84eAXV77gwzNyCHFpr+RlYHKk53PlfTy8QUWQqpVJF8hRwAmJo51D0MDBkGhoadx17L9HLxwDGrE/Ab/D/cMenr7AT+mHDAEw7O2v38WN2tPzl49Uz86bXXh1x5FdsGMtbdwLlAO8/edNmjM0NBYv+/qXqzPUIndbTMLkEXe8v3WD+v236MtVnxrKVCPJcSE5ubGJOwXz75trozV5u7u7+/vr3m1tUR2Dg6PmPpWVlv/w0/Yu4gXjebl5/7fyRdAMslhFReUnGz9NSjr/yksv4k0cPnJ07bp/ariczKysVS+/snjRggfnz9fYVm1t3QcffVxXW0cUzn5u5d83b9pkb28HX73dlf1d+kO3BdIZRNtx+Rp3bYaFxxUfV5HjSl76jNumQaKlraWwnFPN6MGRQILCiuJ3vnmPbJWQesgkSNEbX615c+mrSjcvpDe+3PNvzIyF0x5kmXEi9eTGnzdpHHhuSd67W9bPuf2eORPv1VgJlNu08wuWGcbBSM9x6fJlnI6L0z3LblhoaGzsEPgH1Vrjpx27dwEznPs6R0dFenp4Qu/3rffew8wYEh1976y7I8K5GdPOJiVt+uJLNg0+d/2HG9gTBC3O2DG3zbr7Lm8l99aVr7/57vcTJzS3tXMnMAPICh0onAkN387du/HXWL6PUFxR3NSiUiNsOMBpjozkpnzMyOOoAyzBFylKvew3e7/FzBgVPXLWuLvGD+XmWwYhgYuN9Mbek78ePn+UTU9LmDJ99FSk7lVt2rWZ3SjoKPAVVMGTfxHR9sM7T186q7GeCxkXwTJb97GBpmfggIHIcBipHI2NjWzC1tbWtZ8Lzgedz8rK1i4fExNNNg3QKi1f9njijOlsLOTk6dOgHOxPy59YNnPGdDa9dduP0E+GBHSIHlm8yNHRcefuPWCEVfttbf3Jh+8HBARAeslfHnlrzTunzqjOzs5du8eNGUNuurCo+KUXV44fq7pUBw4dWrf+AzY/M1MwDTFBUUcu/o7UdT09L31YeDw2HCF+wWAd7G3tWtvbICYBsQ1/pT/ZcxkaMgSWGh4xrJ9Tv6zC7HGxY+bczlVioC97mWFt7R3t+vRokjNTtvz2A5seETEMZINN7z21r6OzQ3XgfaxXL311kNcgSHd2dazf+lGyOnDy6+n9CdGiidLb2tvGDx23eMYCfQIwOmEkObo6udc8a7zvGSzqC6te0i6/Z8dPtrZ2+Ov48eNmJgoxg5MnT7MJMBmYGYD75szeum0bMAkIcTEldczohBMnTnJrGDeWZQZSd5QeuH8+S47LaVdAgUCT8EomThjPMgNw5x137Nv326U01YUvLBYGCccQtiOzIAvIkZ7PUQdMg5XCKjwgnL0G6XmZQI7sAq4CKN2VbPWdMnIy/NM4ajsb4ZDrmuo0ImwciFBhUWXJ9we2sfIA/nHFvCdwIPHclQtsYkTkcJYZSEUUm9kTZrE7BnYEuEvG0Fz6uiy951FTQpFGksOprxObAHJoXAx94KNUkl8Li7j2GzzK8ytf1LlIUWEhUKS8gnt5bFraFbIkNExsAs5sUVFxeJgwiadygKix9/LyYsnR0tyCM+GcgrOD9hupWvF8+LxWeI39KTJQ9W6XCP8w7hoUXJ066s5rfEwzKlB480tVbdXRlN9BVDrVVby4spRs77sZiaH5RP7Xe79jE3BFn3voGTyjC6gObrMgGAOWEy/SRRx4SWUp6Bz+aYCrp4lBaiPJ4TVgAE7n5uXjnqq7m9tDD9zPpqESJ6eoTig0ATZiRdXooHZ2cs60obEx9dIlnVusqq7u7BRUCswH/NNZsrJK5AoVVqITJHW+oGVhyZFXlg8XFfqKSF01Q9WnG379/oCqWE5JTmlVKavw6nzOrJxNO/fpjk2tZor4wZX++fCOhxMXsV+7uoUDr1T3k3UuVd1QQ37tY2XqbIBGkgPunuD0kSPHMDkghLBowUNs+t1169kEOMGeKQyBCjbhO3DghPHjdJYJDQ1xdHQA48KKRHx8XESY7jl+/Xx8kOGAy7zn+F6kfsEPNhzhAaFs9YUerJO9Y1Nrc0lVWQ4vG7AzLDngjskn2z9lGWNrYwOZoPxBvoGFZcVbD25DhoA1N5D47ewhMMKsbkEXFJo26N8ilZJFhkhELHzclMisMJIc0AHx8/Vlg2B79+27feL4aHGYq7y8/MQpzh8MjR3S89r8/fxSUlJVe2Ntg7mlE4P8BuXm5ULC18en55KGIiIgHLxep7p3feZyEpsZSbwvDAqcS78AdRpEgs2B8Abr9ZLSkrCWvPP4avzOysqaSmQIoJ8cHxa3+qs17NdPfv5s3ZNr2OCEt4eyqEJlknw8vOZPnotuCIxUHlCChfy1gar82hurDx0+jG+jZF/Nfn312xDgYr/OmD6157WNTuAm34QLz8ZVWVRVVT2wcPGrr7/5w48/Vakbi9EJnCH//cRJHAEDHD9xcuHDf3nn3XX//WUvbqQMAihEZGA4m07OTmUT0XxkHRDFp4lf+frA66KNtQ1mBlT046mnkCEYGhwTGRDO9l2R+o4xdiHx/MvLzqadb24V3NLpy2eXr39mw48bfztzoLPLmAPvAcaHzyeMG5ucnLx3336k9grQiOB2hAT0PoIG9xK4jY+LCw4KglA6pF965bXExGkR4eFXr177df/+6urq02fPpvyROm3qnezaduzaBbSrqalZtuLpxGlTfQYOvHDh4v4DBzo6OsqPHi0rLyf7QQYhKjAyNQMNCZAAAAKaSURBVFvleNrULLeztWNDn9yvvIq08XUAGw4/Ly4ACvrx1S/fjgiPg37HofNHCssKkeF44M77Uq9eKipXPUhwLPn4qMgRcWFDoSu0/8xBcKYQfXnxXy9PGj7B2907NTv1WPIJ4ERV3ZnK2sopo8z8fhaTbtmvePIJ6Lb8tH0HI2HFY4cMWbrkUT3WhF5c+fzzf18FVGhqbtr243byJ/Czzyxf3s9FNaEWeJrnn3327bXvQQytpKTki6/+TZaESNeKJ5YhYxEbEgM9SfwVOrGkVfL18nV2dMYdEAhahA3ignuQgJ4ndCYhfeDsQfjH5seFxF7MSkEGAlzwU3OXrfrsVVYJPt35+boVa937uf/t3iWf/PQpmNOK2oofDv5ELuLq7ProXYuRuWGSoYVz99gjj2z618ZpU+6E+2fkTxAYXfncc2vfXk2GN3oAWNGNGz4AYSD7QeD4ICKy8cMPJkwYjzMhKvrRB+vG3jaa7D9DfGz2vfds/GhDYGAAMhbgIp0dhTgBKLxGgSiilYkOjMLUAbf49Pwn40OFSDHk3DV2xpxJs5FRACJCRJxNNzQ3frn7K0gkRI18Y8krw8KGOtk74ZJgeqBr/c6y1RB9QeaG2SaMA/GoqKgoL6+EOu3rOxAqMTIWYCby8vL7Ojt5K71BNnooCbfoIPIB24JbrVZW9OdxV99zKWprb1W6e1/XNx+obvzWVrn16w+KAkRE1wfybIIyJCE/YCxDEjI5ZEhCJocMScjkkCEJmRwyJCGTQ4YkZHLIkIRMDhmSkMkhQxIyOWRIQiaHDEnI5JAhCZkcMiQhk0OGJGRyyJCETA4ZkpDJIUMSMjlkSEImhwxJyOSQIQmZHDIkIZNDhiRkcsiQhEwOGZKQySFDEv8PAAD//0bKhbQAAAAGSURBVAMA+Wvl1Ks5B+YAAAAASUVORK5CYII="
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="60 60 380 390" width="46" height="46">'
    '<g transform="translate(256 256) scale(1.08) translate(-256 -250)">'
    '<path d="M150 372C124 236 214 138 392 118C410 300 316 396 150 372Z" fill="#FBF7EE"/>'
    '<path d="M146 376L104 420" stroke="#FBF7EE" stroke-width="20" stroke-linecap="round"/>'
    '<path d="M176 346C238 330 238 276 284 250C328 226 336 190 356 160" fill="none" stroke="#4E6E45" '
    'stroke-width="17" stroke-linecap="round" stroke-dasharray="1 33"/>'
    '<circle cx="358" cy="156" r="21" fill="#B07A35"/></g></svg>'
)
ICON_JS = """<script>(function(){
var FAV='data:image/png;base64,__FAV__',TOUCH='data:image/png;base64,__TOUCH__',TITLE='GreenWalker';
function docs(){var r=[],w=window;for(var k=0;k<6;k++){try{r.push(w.document)}catch(e){break}
 if(w.parent===w)break;w=w.parent}return r}
function patch(fav,touch){docs().forEach(function(d){try{var h=d.head;
 d.querySelectorAll('link[rel*="icon"]').forEach(function(e){e.remove()});
 [['icon',fav,'64x64'],['shortcut icon',fav,'64x64'],['apple-touch-icon',touch,'180x180']].forEach(function(a){
  var l=d.createElement('link');l.rel=a[0];l.href=a[1];l.sizes=a[2];h.appendChild(l)});
 if(d.title!==TITLE)d.title=TITLE;
 try{d.documentElement.style.colorScheme='only light'}catch(e){}
 [['apple-mobile-web-app-title',TITLE],['application-name',TITLE],['apple-mobile-web-app-capable','yes'],['color-scheme','only light'],['supported-color-schemes','light'],['darkreader-lock','']].forEach(function(a){
  var m=d.querySelector('meta[name="'+a[0]+'"]');if(!m){m=d.createElement('meta');m.name=a[0];h.appendChild(m)}m.content=a[1]})
 }catch(e){}})}
var cur=[FAV,TOUCH];
patch(FAV,TOUCH);
/* Gespeichertes dunkles Streamlit-Design einmalig verwerfen (Streamlit merkt sich die Wahl im Browser) */
try{var PW=window.parent,LS=PW.localStorage,SS=PW.sessionStorage,hit=false;
 for(var i=LS.length-1;i>=0;i--){var kk=LS.key(i);
  if(kk&&kk.indexOf('stActiveTheme')===0){var vv=LS.getItem(kk)||'';
   if(vv.toLowerCase().indexOf('dark')>-1||vv.indexOf('"base":1')>-1){LS.removeItem(kk);hit=true}}}
 if(hit&&!SS.getItem('gwTheme')){SS.setItem('gwTheme','1');PW.location.reload()}}catch(e){}
/* Echte Dateien aus static/ bevorzugen (iOS liest sie zuverlässiger als eingebettete Daten) */
try{var P=window.parent,base=P.location.origin+P.location.pathname.replace(/[^/]*$/,'')+'app/static/';
 fetch(base+'apple-touch-icon.png',{method:'HEAD'}).then(function(r){
  if(r.ok&&/image/.test(r.headers.get('content-type')||'')){cur=[base+'favicon.png',base+'apple-touch-icon.png'];patch(cur[0],cur[1])}
 }).catch(function(){})}catch(e){}
/* Der Host (Streamlit-Rahmen) setzt Titel und Ikon gern nachträglich zurück: mehrfach erneut setzen */
[800,2500,6000,12000].forEach(function(t){setTimeout(function(){patch(cur[0],cur[1])},t)});
try{var T=window.parent.document.querySelector('title');
 if(T)new MutationObserver(function(){if(window.parent.document.title!==TITLE)window.parent.document.title=TITLE}).observe(T,{childList:true})}catch(e){}
})();</script>"""


SELECT_JS = """<script>(function(){
try{var P=window.parent,D=P.document;if(P.__gwSel)return;P.__gwSel=1;var swallow=0;
D.addEventListener('mousedown',function(e){
 var sel=e.target.closest&&e.target.closest('[data-baseweb="select"]');if(!sel)return;
 var svg=e.target.closest('svg');if(!svg||svg.closest('[data-baseweb="tag"]'))return;
 var all=[].slice.call(sel.querySelectorAll('svg')).filter(function(x){return!x.closest('[data-baseweb="tag"]')});
 if(svg!==all[all.length-1])return;
 var inp=sel.querySelector('input');
 if(inp&&inp.getAttribute('aria-expanded')==='true'){e.preventDefault();e.stopPropagation();swallow=Date.now();inp.blur();
  inp.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',keyCode:27,bubbles:true}))}
},true);
D.addEventListener('click',function(e){if(Date.now()-swallow<400&&e.target.closest&&e.target.closest('[data-baseweb="select"]')){e.preventDefault();e.stopPropagation()}},true);
}catch(e){}
})();</script>"""


# iPhone: Die Tastatur blendet über dem Suchfeld gern eigene Kontakt-Adressen ein („Arbeit …“, „Privat …“). Wer die
# antippt, schließt nur die Tastatur, und das Suchfeld leert sich. Darum werden Autofill und Autokorrektur in den
# Suchfeldern abgeschaltet (Suchfelder laufen in eigenen Rahmen; ein Wächter im Elternfenster erledigt das).
SB_JS = """<script>(function(){
try{var P=window.parent;if(P.__gwSb)return;P.__gwSb=1;
function guard(doc){try{if(doc.__gwG)return;doc.__gwG=1;var last=0,prev=0;
 function mark(e){var t=e.target;if(!t||t.tagName!=='INPUT')return;var v=t.value||'';
  if(v.length-prev>3||(e.inputType&&/Replacement|Autofill|Paste|Drop/i.test(e.inputType)))last=Date.now();prev=v.length}
 doc.addEventListener('input',mark,true);doc.addEventListener('change',mark,true);
 doc.addEventListener('focusout',function(e){if(Date.now()-last<1500&&e.target&&e.target.tagName==='INPUT'){e.stopImmediatePropagation();e.stopPropagation()}},true);
}catch(e){}}
function fix(doc){guard(doc);try{[].slice.call(doc.querySelectorAll('input')).forEach(function(i){
 if(i.__gw)return;i.__gw=1;
 i.setAttribute('autocomplete','off');i.setAttribute('autocorrect','off');i.setAttribute('autocapitalize','none');
 i.setAttribute('spellcheck','false');i.setAttribute('enterkeyhint','search');i.setAttribute('inputmode','search');
 i.setAttribute('name','gw'+Math.random().toString(36).slice(2,8));i.setAttribute('data-lpignore','true');i.setAttribute('data-form-type','other');
 i.setAttribute('aria-autocomplete','list');
})}catch(e){}}
function scan(){try{[].slice.call(P.document.querySelectorAll('iframe')).forEach(function(f){
 var u=(f.getAttribute('src')||'')+(f.title||'');
 if(u.indexOf('searchbox')<0)return;
 try{if(f.contentDocument)fix(f.contentDocument)}catch(e){}})}catch(e){}}
setInterval(scan,500);scan();
}catch(e){}
})();</script>"""


def page_icon():
    try:
        import io
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(ICON_FAV_B64)))
    except Exception:
        return "🌳"


def install_icons():
    """Setzt Tab-Ikon und Home-Bildschirm-Ikon (Apple Touch Icon) im Elternfenster."""
    try:
        st_html(ICON_JS.replace("__FAV__", ICON_FAV_B64).replace("__TOUCH__", ICON_TOUCH_B64) + SELECT_JS + SB_JS, height=0)
    except Exception:
        pass


# ------------------------------------------------------- UI
st.set_page_config(page_title="GreenWalker", page_icon=page_icon(), layout="centered",
                   initial_sidebar_state="collapsed")
install_icons()

# Hundemodus: eigener Look (warme Orange- und Cremetöne, Pfotenmuster), damit der Modus auf einen Blick erkennbar ist
DOG_CSS = """
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background-color:#FCEBD3!important;
 background-image:url("data:image/svg+xml,__PAW__")!important}
.stApp .hero{background:linear-gradient(135deg,#B8571A,#E58F3E)!important;box-shadow:0 6px 18px rgba(140,70,20,.35)}
.stApp .hero .ico{font-size:40px}
.stApp .hero .ico img{display:block;width:52px;height:52px;margin:-3px 0 -3px -3px}
.stApp [data-testid="stToggle"] input:checked+div,.stApp [data-testid="stToggle"] [aria-checked="true"]{background-color:#D2782A!important}
.stApp .stButton>button[kind="primary"]{background:#D2782A!important;border-color:#D2782A!important}
.stApp .stButton>button[kind="primary"]:hover{background:#B8571A!important;border-color:#B8571A!important;box-shadow:0 6px 18px rgba(184,87,26,.4)}
.stApp .stButton>button{border-color:#E8CFAA!important}
.stApp [data-testid="stButtonGroup"] button[aria-checked="true"],.stApp [data-testid="stButtonGroup"] button[kind*="Active"]{background:#FBE3C4!important;border-color:#E3A96B!important}
.st-key-card_route,.st-key-card_green,.st-key-card_loop,.st-key-card_tour{background:rgba(255,250,242,.98)!important;border-color:#EBCBA0!important;box-shadow:0 8px 28px rgba(140,70,20,.18)}
.pill{background:#FBE3C4;border-color:#E3A96B}
.fld .dot.s{background:#B8571A;box-shadow:0 0 0 3px rgba(184,87,26,.18)}
.stApp [data-baseweb="select"] > div,.stApp [data-baseweb="input"],.stApp [data-baseweb="base-input"]{background:#FFF8EE!important}
.st-key-dog_mode{border-color:#E3A96B!important;background:#FFF1DD!important}
</style>
""".replace("__PAW__", "%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140' viewBox='0 0 140 140'%3E%3Cg fill='%23B8571A' fill-opacity='.07'%3E%3Cg transform='translate(34 36) rotate(-18)'%3E%3Cellipse cx='0' cy='6' rx='9' ry='7.5'/%3E%3Cellipse cx='-11' cy='-7' rx='3.8' ry='5'/%3E%3Cellipse cx='-4' cy='-12' rx='3.8' ry='5'/%3E%3Cellipse cx='4' cy='-12' rx='3.8' ry='5'/%3E%3Cellipse cx='11' cy='-7' rx='3.8' ry='5'/%3E%3C/g%3E%3Cg transform='translate(100 104) rotate(14)'%3E%3Cellipse cx='0' cy='6' rx='9' ry='7.5'/%3E%3Cellipse cx='-11' cy='-7' rx='3.8' ry='5'/%3E%3Cellipse cx='-4' cy='-12' rx='3.8' ry='5'/%3E%3Cellipse cx='4' cy='-12' rx='3.8' ry='5'/%3E%3Cellipse cx='11' cy='-7' rx='3.8' ry='5'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")
DOG_BASE_CSS = """<style>
.st-key-dog_mode{background:rgba(251,248,241,.98);border:1px solid #D9CFBC;border-radius:16px;padding:8px 14px 4px;margin:0 0 10px;
 box-shadow:0 4px 14px rgba(60,50,30,.12)}
.chayfact{display:flex;gap:10px;align-items:center;font-size:.84rem;line-height:1.35;padding:8px 10px;margin:6px 0 4px;border-radius:12px;background:#FBEEDB;border:1px solid #EBCBA3;color:#5A3A1E}
.chayfact img{flex:none;border-radius:50%}
.dogchip{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;font-size:.86rem;font-weight:600;padding:8px 12px;margin:0 0 10px;
 border-radius:14px;background:#FFF1DD;border:1px solid #E3A96B;color:#5A3513}
.dogwarn{padding:10px 12px;border-radius:14px;background:#FDE6DD;border:1px solid #E9A58E;color:#8A2A12;font-size:.88rem;font-weight:600;margin:0 0 10px;line-height:1.4}
</style>"""

BASE_CSS = """
<style>
:root{--moss:#4E6E45;--sage:#9DB38A;--sand:#E9E1D2;--paper:#FBF8F1;--line:#D9CFBC;--text:#3B3A33}
html,body,.stApp{color-scheme:only light!important}
body,.stApp,.stApp *:not(svg):not(path):not(.hero):not(.hero *):not(.stButton>button[kind="primary"]):not(.stButton>button[kind="primary"] *){--text-color:#3B3A33}
.stApp [data-testid="stMarkdownContainer"] *,.stApp [data-testid="stWidgetLabel"] *,.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stToggle"] *,.stApp [data-testid="stCheckbox"] *,.stApp [data-baseweb="tab"] *,.stApp [data-testid="stExpander"] summary *{color:#3B3A33!important}
.stApp .hero,.stApp .hero *{color:#FBF7EE!important}
.stApp [data-testid="stToggle"] [role="switch"][aria-checked="false"]+div,.stApp [data-testid="stToggle"] label>div:first-child{background-color:#CFC6B2}
.stApp [data-testid="stToggle"] input:checked+div,.stApp [data-testid="stToggle"] [aria-checked="true"]{background-color:#5E7F52!important}
.stApp .stButton>button,.stApp [data-testid="stButtonGroup"] button{background:#fff!important;color:#3B3A33!important}
.stApp .stButton>button[kind="primary"]{background:#5E7F52!important;color:#FBF7EE!important}
.stApp .stButton>button[kind="primary"] *{color:#FBF7EE!important}
[data-testid="stHeader"]{background:transparent!important}
/* Immer hell: auch wenn Browser oder Streamlit ein dunkles Design anbieten, bleibt die App lesbar und bedienbar */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#F4EFE6!important;color:#3B3A33!important}
.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp small{color:#3B3A33}
[data-baseweb="select"] > div,[data-baseweb="input"],[data-baseweb="base-input"],textarea{background:#FBF8F1!important;color:#3B3A33!important}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"]{background:#FBF8F1!important;color:#3B3A33!important}
html,body,.stApp,.stApp input,.stApp textarea,.stApp button,.stApp label,.stApp p,.stApp li,.stApp div,
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp [data-baseweb]{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.block-container{padding-top:1.2rem;max-width:680px}
.hero{display:flex;gap:14px;align-items:center;border-radius:16px;padding:14px 16px;margin-bottom:14px;
 background:var(--moss);color:#FBF7EE;box-shadow:0 6px 18px rgba(60,50,30,.3)}
.hero .ico{font-size:36px;line-height:1;width:46px;text-align:center;flex:none}
.hero h1{margin:0;padding:0;font-size:1.5rem;font-weight:700;line-height:1.1;color:#FBF7EE;letter-spacing:0}
.hero p{margin:2px 0 0;font-size:1rem;color:#FBF7EE;opacity:.92}
.st-key-card_route,.st-key-card_green,.st-key-card_loop,.st-key-card_tour{background:rgba(251,248,241,.98);border:1px solid var(--line);border-radius:20px;
 padding:14px 14px 12px;box-shadow:0 8px 28px rgba(60,50,30,.18);margin-top:12px}
.st-key-card_route label p,.st-key-card_green label p,.st-key-card_loop label p,.st-key-card_tour label p{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
 opacity:.7;font-weight:600}
.hint{opacity:.7;font-size:.8rem;margin:-4px 0 8px}
.fld{display:flex;align-items:center;gap:8px;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;font-weight:700;
 color:var(--text);opacity:.75;margin:10px 0 4px}
.fld .dot{width:10px;height:10px;border-radius:50%;display:inline-block;box-shadow:0 0 0 3px rgba(78,110,69,.15)}
.fld .dot.s{background:var(--moss)}.fld .dot.z{background:#B07A35;box-shadow:0 0 0 3px rgba(176,122,53,.18)}
[data-testid="stToggle"] label p,[data-testid="stCheckbox"] label p{font-size:.92rem;font-weight:600}
[data-testid="stSlider"] [role="slider"]{box-shadow:0 2px 8px rgba(60,50,30,.3)}
.stCaption,[data-testid="stCaptionContainer"]{font-size:.8rem;opacity:.7}
.trip{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 10px}
.pill{border-radius:14px;padding:8px 12px;background:#E7EDDB;color:var(--text);border:1px solid var(--sage);
 font-size:.92rem;font-weight:600}
.stButton>button{border-radius:14px;font-weight:600;padding:.6rem 1rem;border:1px solid var(--line);background:#fff;color:var(--text)}
.stButton>button[kind="primary"]{background:#5E7F52;border-color:#5E7F52;color:#FBF7EE;height:3rem;font-size:1rem;font-weight:700}
.stButton>button[kind="primary"]:hover{background:var(--moss);border-color:var(--moss);color:#FBF7EE}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="base-input"]{border-radius:14px!important}
[data-testid="stButtonGroup"]{display:flex;width:100%;gap:8px;margin-bottom:4px}
[data-testid="stButtonGroup"] button{flex:1 1 0;min-height:46px;border-radius:14px;font-weight:700;white-space:normal;
 line-height:1.15;background:#fff;border:1px solid var(--line);color:var(--text)}
[data-testid="stButtonGroup"] button[aria-checked="true"],[data-testid="stButtonGroup"] button[kind*="Active"]{
 background:#E7EDDB;border-color:var(--sage)}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent;padding:0;overflow-x:auto}
.stTabs [data-baseweb="tab"]{flex:1 0 auto;white-space:nowrap;padding:0 14px;justify-content:center;height:46px;border-radius:14px;font-weight:700;
 color:var(--text);background:#fff;border:1px solid var(--line)}
.stTabs [aria-selected="true"]{background:#E7EDDB;border-color:var(--sage);color:var(--text);box-shadow:none}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
/* Bewegung: Karten gleiten ein, Bedienelemente reagieren weich */
@keyframes gwrise{from{opacity:0;transform:translateY(16px) scale(.985)}to{opacity:1;transform:none}}
.st-key-card_route,.st-key-card_green,.st-key-card_loop,.st-key-card_tour{animation:gwrise .55s cubic-bezier(.2,.8,.2,1) both}
.hero{animation:gwrise .6s cubic-bezier(.2,.8,.2,1) both}
.stButton button,[data-testid="stButtonGroup"] button{transition:transform .15s,background .25s,border-color .25s,box-shadow .25s}
.stButton button:active,[data-testid="stButtonGroup"] button:active{transform:scale(.97)}
.stButton button[kind="primary"]:hover{box-shadow:0 6px 18px rgba(78,110,69,.4);transform:translateY(-1px)}
[data-testid="stToggle"],[data-testid="stCheckbox"]{transition:opacity .2s}
@media (prefers-reduced-motion:reduce){.st-key-card_route,.st-key-card_green,.st-key-card_loop,.st-key-card_tour,.hero{animation:none}}
/* Handy: Auswahl als lesbare Tipp-Felder (Chips), die umbrechen statt zu quetschen */
.st-key-favsync{position:absolute!important;width:1px;height:1px;overflow:hidden;opacity:0;pointer-events:none;margin:0!important}
[class*="st-key-chips_"],[class*="st-key-chips_"] [data-testid="stVerticalBlock"]{flex-direction:row!important;flex-wrap:wrap!important;
 gap:.45rem!important;align-items:stretch!important}
[class*="st-key-chips_"] [data-testid="stElementContainer"],[class*="st-key-chips_"] .element-container,
[class*="st-key-chips_"] [data-testid="stButton"]{width:auto!important;flex:0 0 auto!important;min-width:0!important}
[class*="st-key-chips_"] button{white-space:nowrap!important;min-height:44px!important;padding:.3rem 1rem!important;
 border-radius:999px!important;font-size:.96rem!important;line-height:1.2!important;width:auto!important}
[class*="st-key-chips_"] button p{white-space:nowrap!important;margin:0!important}
[data-testid="stRadio"] label{padding:.5rem .2rem}
@media (max-width:640px){
 [data-testid="stRadio"] [role="radiogroup"]{gap:.15rem}
 [data-testid="stRadio"] label{min-height:44px;align-items:center}
}
</style>
"""


# ---- Chay: Black Tri Australian Shepherd (schwarz, weißer Blesse/Brust/Pfoten, kupferfarbene Abzeichen)
DOG_NAME = "Chay"
CH_BLACK, CH_WHITE, CH_TAN, CH_EYE = "#202026", "#F8F5EE", "#B5651D", "#C98A2B"


def _sp(d, dx, dy):
    """Absolute Koordinaten eines Pfads (M/L/Q/C) verschieben; relative Befehle bleiben."""
    out, cmd, nums = [], None, []
    for tok in re.findall(r"[MLQCZmlqczvVhH]|-?\d+\.?\d*", d):
        if tok.isalpha():
            cmd = tok
            out.append(tok)
        elif cmd in ("M", "L", "Q", "C"):
            nums.append(float(tok))
            if len(nums) == 2:
                out.append(f"{nums[0] + dx:g} {nums[1] + dy:g}")
                nums = []
        else:
            out.append(tok)
    return " ".join(out)


def chay_head(dx=0.0, dy=0.0, anim=False, glasses=""):
    """Chays Kopf (Vorderansicht) mit Mittelpunkt (50+dx, 52+dy), nach Fotos von Chayton: schwarzes, struppiges Fell,
    kleine, vorn geknickte Ohren, schmale weiße Blesse, kupferfarbene Wangen und Brauen, offenes Maul.
    anim: Klassen für die Ladeanimation (Ohren, Augen, Zunge)."""
    def E(cx, cy, rx, ry, fill, rot=0, cls="", extra=""):
        t = f' transform="rotate({rot} {cx + dx:g} {cy + dy:g})"' if rot else ""
        c = f' class="{cls}"' if cls else ""
        return f'<ellipse{c} cx="{cx + dx:g}" cy="{cy + dy:g}" rx="{rx:g}" ry="{ry:g}" fill="{fill}"{extra}{t}/>'

    def Pa(d, fill="none", cls="", stroke=""):
        c = f' class="{cls}"' if cls else ""
        return f'<path{c} d="{_sp(d, dx, dy)}" fill="{fill}"{stroke}/>'
    dark_line = ' stroke="#15151A" stroke-width="1.5" stroke-linecap="round"'
    ear_l = Pa("M31 24 Q14 18 7 38 Q9 46 20 44 Q31 42 38 32 Z", CH_BLACK, "ear l" if anim else "")
    ear_r = Pa("M69 24 Q86 18 93 38 Q91 46 80 44 Q69 42 62 32 Z", CH_BLACK, "ear r" if anim else "")
    px_, py_ = 37.5 + dx, 47 + dy
    eyes = ('<g class="eye">' if anim else "<g>") + (
        f'<circle cx="{px_:g}" cy="{py_:g}" r="4.8" fill="{CH_EYE}"/><circle cx="{62.5 + dx:g}" cy="{py_:g}" r="4.8" fill="{CH_EYE}"/>'
        f'<circle cx="{px_:g}" cy="{py_:g}" r="2.6" fill="#15110D"/><circle cx="{62.5 + dx:g}" cy="{py_:g}" r="2.6" fill="#15110D"/>'
        f'<circle cx="{px_ + 1.2:g}" cy="{py_ - 1.4:g}" r="1.1" fill="#fff"/><circle cx="{63.7 + dx:g}" cy="{py_ - 1.4:g}" r="1.1" fill="#fff"/></g>')
    tongue = Pa("M43.5 72 Q50 90 56.5 72 Z", "#F08A9A", "tongue" if anim else "")
    tufts = E(37, 25, 7, 6, CH_BLACK) + E(50, 22, 7, 6, CH_BLACK) + E(63, 25, 7, 6, CH_BLACK) + E(27, 36, 6, 8, CH_BLACK) + E(73, 36, 6, 8, CH_BLACK)
    return (ear_l + ear_r
            + E(50, 51, 29, 28, CH_BLACK) + tufts
            + E(25, 60, 8, 12, CH_BLACK, 12) + E(75, 60, 8, 12, CH_BLACK, -12)
            + Pa("M46.5 24 Q50 21 53.5 24 L55 48 Q56 58 59 68 L41 68 Q44 58 45 48 Z", CH_WHITE)
            + E(50, 67, 15, 12.5, CH_WHITE)
            + E(31, 62, 8, 10.5, CH_TAN, 12) + E(69, 62, 8, 10.5, CH_TAN, -12)
            + E(38.5, 67, 8, 9, CH_WHITE) + E(61.5, 67, 8, 9, CH_WHITE) + E(50, 72, 8, 6, CH_WHITE)
            + E(36, 37.5, 4.2, 3, CH_TAN, -14) + E(64, 37.5, 4.2, 3, CH_TAN, 14)
            + eyes + glasses
            + Pa("M40 71 Q50 69 60 71 Q58 82 50 83 Q42 82 40 71 Z", "#3A1A20")
            + tongue
            + Pa("M43 61.5 Q50 58 57 61.5 Q56 68 50 69 Q44 68 43 61.5 Z", "#15151A")
            + E(48, 61, 2.6, 1.0, "#fff", 0, "", ' fill-opacity=".45"'))


def _chay_chest():
    """Schultern, kupferfarbene Krause und helle Brust (für das Symbol, unter dem Kopf)."""
    return (f'<ellipse cx="50" cy="108" rx="40" ry="21" fill="{CH_BLACK}"/>'
            f'<ellipse cx="50" cy="91" rx="23" ry="12" fill="{CH_TAN}"/>'
            f'<ellipse cx="50" cy="102" rx="14" ry="15" fill="{CH_WHITE}"/>')


def chay_icon_svg(px=46, bg=True):
    """Kleines Chay-Symbol (Kopf, Brust). bg: heller Kreis dahinter."""
    if bg:
        body = ('<clipPath id="chc"><circle cx="50" cy="52" r="49"/></clipPath><g clip-path="url(#chc)">'
                '<circle cx="50" cy="52" r="49" fill="#FFF4E2"/><g transform="translate(7 6) scale(.86)">'
                + _chay_chest() + chay_head() + "</g></g>")
    else:
        body = _chay_chest() + chay_head()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 104" width="{px}" height="{px}" role="img" aria-label="{DOG_NAME}">{body}</svg>')


def chay_icon_uri(px=46, bg=True):
    return "data:image/svg+xml;base64," + base64.b64encode(chay_icon_svg(px, bg).encode()).decode()


# ---- Ladehund (Hundemodus): kleiner animierter Hund je Jahreszeit und Wetter
DOG_SCENE_CSS = """
#gwload .dogscene{position:relative;width:min(70vw,260px);filter:drop-shadow(0 8px 14px rgba(90,50,20,.18))}
#gwload .dogscene svg{display:block;width:100%;height:auto;overflow:visible}
#gwload .dogscene .tail{transform-origin:150px 126px;animation:gwd-wag .45s ease-in-out infinite alternate}
#gwload .dogscene .ear.l{transform-box:fill-box;transform-origin:85% 10%;animation:gwd-earl 1.6s ease-in-out infinite}
#gwload .dogscene .ear.r{transform-box:fill-box;transform-origin:15% 10%;animation:gwd-earr 1.6s ease-in-out infinite}
#gwload .dogscene .eye{transform-origin:120px 78px;animation:gwd-blink 3.4s infinite}
#gwload .dogscene .tongue{transform-origin:120px 102px;animation:gwd-tongue .7s ease-in-out infinite alternate}
#gwload .dogscene .sweat{animation:gwd-sweat 1.6s ease-in infinite}
#gwload .dogscene .fl{animation:gwd-fall 3.6s linear infinite}
#gwload .dogscene .lf{animation:gwd-leaf 4.2s ease-in-out infinite}
#gwload .dogscene .pop{animation:gwd-pop 1.3s ease-out infinite}
#gwload .dogscene .puff{transform-origin:center;transform-box:fill-box;animation:gwd-puff 1.1s ease-out infinite}
#gwload .dogscene .jet{animation:gwd-jet .5s linear infinite}
#gwload .dogscene .dp{animation:gwd-drop 1s ease-in infinite}
#gwload .dogscene .dirt{animation:gwd-dirt .7s ease-out infinite}
#gwload .dogscene .flw{animation:gwd-sway 2.6s ease-in-out infinite alternate}
#gwload .dogscene .bfly{animation:gwd-fly 7s ease-in-out infinite}
#gwload .dogscene .wing{transform-origin:0 0;animation:gwd-flap .25s ease-in-out infinite alternate}
#gwload .dogscene .ripple{transform-origin:center;transform-box:fill-box;animation:gwd-ripple 1.4s ease-out infinite}
#gwload .sc-winter .dog{animation:gwd-hop 1s ease-in-out infinite}
#gwload .sc-autumn .dog{animation:gwd-peek 1.7s ease-in-out infinite}
#gwload .sc-summer .dog{animation:gwd-jump 1.2s ease-in-out infinite}
#gwload .sc-spring .dog{animation:gwd-lean 1s ease-in-out infinite;transform-origin:120px 150px}
#gwload .sc-spring .paw.l{transform-origin:104px 140px;animation:gwd-dig .36s ease-in-out infinite alternate}
#gwload .sc-spring .paw.r{transform-origin:136px 140px;animation:gwd-dig .36s .18s ease-in-out infinite alternate}
#gwload .sc-summer .head{transform-origin:120px 100px;animation:gwd-look 1.2s ease-in-out infinite}
#gwload .sc-winter .head{transform-origin:120px 100px;animation:gwd-tilt 2s ease-in-out infinite}
#gwload .dogscene .wl{stroke-dasharray:26 18;animation:gwd-wind .7s linear infinite}
#gwload .dogscene.windy .ear.l{animation:gwd-earwl .22s ease-in-out infinite alternate}
#gwload .dogscene.windy .ear.r{animation:gwd-earwr .22s ease-in-out infinite alternate}
#gwload .dogscene.windy .tail{animation-duration:.3s}
@keyframes gwd-wind{to{stroke-dashoffset:-44}}
@keyframes gwd-earwl{from{transform:rotate(6deg)}to{transform:rotate(-14deg)}}
@keyframes gwd-earwr{from{transform:rotate(-6deg)}to{transform:rotate(14deg)}}
@keyframes gwd-wag{from{transform:rotate(-16deg)}to{transform:rotate(18deg)}}
@keyframes gwd-earl{0%,100%{transform:rotate(0)}50%{transform:rotate(-7deg)}}
@keyframes gwd-earr{0%,100%{transform:rotate(0)}50%{transform:rotate(7deg)}}
@keyframes gwd-blink{0%,92%,100%{transform:scaleY(1)}96%{transform:scaleY(.1)}}
@keyframes gwd-tongue{from{transform:scaleY(.8)}to{transform:scaleY(1.25)}}
@keyframes gwd-sweat{0%{opacity:0;transform:translateY(-6px)}25%{opacity:1}100%{opacity:0;transform:translateY(22px)}}
@keyframes gwd-fall{from{transform:translateY(-30px)}to{transform:translateY(190px)}}
@keyframes gwd-leaf{0%{transform:translate(0,-30px) rotate(0)}50%{transform:translate(14px,70px) rotate(150deg)}100%{transform:translate(-6px,170px) rotate(320deg)}}
@keyframes gwd-pop{0%{opacity:0;transform:translateY(8px) scale(.6)}25%{opacity:1}100%{opacity:0;transform:translateY(-34px) rotate(40deg) scale(1)}}
@keyframes gwd-puff{0%{opacity:.9;transform:scale(.4)}100%{opacity:0;transform:scale(2.2)}}
@keyframes gwd-jet{to{stroke-dashoffset:-14}}
@keyframes gwd-drop{0%{opacity:0;transform:translateY(-10px)}20%{opacity:1}100%{opacity:0;transform:translateY(34px)}}
@keyframes gwd-dirt{0%{opacity:1;transform:translate(0,0)}100%{opacity:0;transform:translate(var(--dx),-52px)}}
@keyframes gwd-sway{from{transform:rotate(-4deg)}to{transform:rotate(4deg)}}
@keyframes gwd-fly{0%{transform:translate(20px,50px)}25%{transform:translate(70px,20px)}50%{transform:translate(150px,44px)}75%{transform:translate(200px,16px)}100%{transform:translate(20px,50px)}}
@keyframes gwd-flap{from{transform:scaleX(1)}to{transform:scaleX(.25)}}
@keyframes gwd-ripple{0%{opacity:.7;transform:scale(.3)}100%{opacity:0;transform:scale(1.6)}}
@keyframes gwd-hop{0%,100%{transform:translateY(0)}45%{transform:translateY(-12px)}}
@keyframes gwd-peek{0%,100%{transform:translateY(14px)}50%{transform:translateY(-4px)}}
@keyframes gwd-jump{0%,100%{transform:translateY(0) rotate(0)}40%{transform:translateY(-16px) rotate(-3deg)}70%{transform:translateY(-4px) rotate(2deg)}}
@keyframes gwd-lean{0%,100%{transform:rotate(-4deg)}50%{transform:rotate(-1deg) translateY(2px)}}
@keyframes gwd-dig{from{transform:translateY(0) rotate(0)}to{transform:translateY(-9px) rotate(-18deg)}}
@keyframes gwd-look{0%,100%{transform:rotate(4deg)}50%{transform:rotate(-7deg)}}
@keyframes gwd-tilt{0%,100%{transform:rotate(-4deg)}50%{transform:rotate(5deg)}}
@media (prefers-reduced-motion:reduce){#gwload .dogscene *{animation:none!important}}
"""


def _dog_body(acc):
    """Der Hund (Vorderansicht). acc: Zubehör {'sun','rain','scarf','hot'}."""
    coat = ""
    hood = ""
    if "rain" in acc:
        coat = ('<path d="M88 118 Q120 100 152 118 L158 148 Q120 158 82 148Z" fill="#F2C230" stroke="#D9A81F" stroke-width="2"/>'
                '<circle cx="120" cy="124" r="2.6" fill="#B8861A"/><circle cx="120" cy="136" r="2.6" fill="#B8861A"/>')
        hood = ('<path d="M86 84 Q84 44 120 44 Q156 44 154 84 Q148 62 120 61 Q92 62 86 84Z" fill="#F2C230" stroke="#D9A81F" stroke-width="2"/>')
    scarf = ""
    if "scarf" in acc and "rain" not in acc:
        scarf = ('<path d="M92 106 Q120 120 148 106 L150 116 Q120 130 90 116Z" fill="#C8433A"/>'
                 '<rect x="138" y="112" width="10" height="24" rx="3.5" fill="#C8433A" transform="rotate(-8 140 112)"/>'
                 '<path d="M138 124 l9 -2 M139 130 l9 -2" stroke="#E9786F" stroke-width="2"/>')
    glasses = ""
    if "sun" in acc:
        glasses = ('<rect x="95" y="68" width="22" height="15" rx="6.5" fill="#1b1b20"/><rect x="123" y="68" width="22" height="15" rx="6.5" fill="#1b1b20"/>'
                   '<path d="M117 73 h6" stroke="#1b1b20" stroke-width="3"/><path d="M99 72 l6 -1 M127 72 l6 -1" stroke="#fff" stroke-opacity=".55" stroke-width="2" stroke-linecap="round"/>')
    sweat = '<path class="sweat" d="M158 64 q-6 10 0 13 q6 -3 0 -13z" fill="#7FD0F5"/>' if "hot" in acc else ""
    return (
        '<g class="dog">'
        '<ellipse class="shadow" cx="120" cy="153" rx="42" ry="6" fill="#000" fill-opacity=".13"/>'
        f'<path class="tail" d="M150 128 Q178 116 172 88" stroke="{CH_BLACK}" stroke-width="12" fill="none" stroke-linecap="round"/>'
        f'<ellipse cx="120" cy="126" rx="35" ry="28" fill="{CH_BLACK}"/><ellipse cx="120" cy="114" rx="24" ry="11" fill="{CH_TAN}"/>'
        f'<ellipse cx="120" cy="132" rx="16" ry="19" fill="{CH_WHITE}"/>'
        f'<ellipse cx="95" cy="134" rx="6" ry="9" fill="{CH_TAN}" fill-opacity=".8"/><ellipse cx="145" cy="134" rx="6" ry="9" fill="{CH_TAN}" fill-opacity=".8"/>'
        + coat + scarf +
        f'<ellipse cx="97" cy="151" rx="12" ry="6.5" fill="{CH_WHITE}"/><ellipse cx="143" cy="151" rx="12" ry="6.5" fill="{CH_WHITE}"/>'
        f'<ellipse class="paw l" cx="104" cy="148" rx="11" ry="7.5" fill="{CH_WHITE}" stroke="#D8D2C6" stroke-width="1.5"/>'
        f'<ellipse class="paw r" cx="136" cy="148" rx="11" ry="7.5" fill="{CH_WHITE}" stroke="#D8D2C6" stroke-width="1.5"/>'
        '<g class="head">' + chay_head(70, 30, True, glasses) + hood + sweat +
        '</g></g>')


def _flakes():
    xs = [14, 38, 62, 88, 110, 136, 160, 184, 208, 226, 50, 150]
    out = []
    for i, x in enumerate(xs):
        out.append(f'<circle class="fl" cx="{x}" cy="0" r="{2 + (i % 3) * .6:.1f}" fill="#fff" stroke="#BFD6E8" stroke-width=".6" '
                   f'style="animation-delay:-{(i * 0.37) % 3.6:.2f}s;animation-duration:{3.2 + (i % 4) * .5:.1f}s"/>')
    return "".join(out)


def _leaves():
    cols = ["#D9731F", "#B8381F", "#E8B534", "#9A5B22"]
    out = []
    for i, x in enumerate([22, 58, 96, 140, 178, 212]):
        c = cols[i % 4]
        out.append(f'<g transform="translate({x} 0)"><path class="lf" d="M0 0 q7 -9 14 0 q-7 9 -14 0z" fill="{c}" '
                   f'style="animation-delay:-{i * 0.7:.1f}s;animation-duration:{3.6 + (i % 3) * .6:.1f}s"/></g>')
    return "".join(out)


def _pile():
    cols = ["#D9731F", "#B8381F", "#E8B534", "#9A5B22", "#C9501F", "#EFC24A"]
    blobs = [(70, 152, 34, 14), (100, 156, 36, 15), (130, 156, 38, 15), (160, 152, 34, 14), (86, 144, 30, 12), (118, 146, 34, 13),
             (150, 144, 30, 12), (104, 138, 24, 9), (136, 139, 24, 9)]
    out = [f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{cols[i % 6]}"/>' for i, (x, y, rx, ry) in enumerate(blobs)]
    for i, x in enumerate([92, 120, 148, 108, 134]):
        out.append(f'<path class="pop" d="M{x} 140 q6 -8 12 0 q-6 8 -12 0z" fill="{cols[(i + 2) % 6]}" style="animation-delay:-{i * .27:.2f}s"/>')
    return "".join(out)


def dog_scene_html(scene="spring", acc=()):
    """Animierte Ladeszene: Hund im Schnee, im Laubhaufen, mit Wasserschlauch oder im Blumenbeet (je Jahreszeit)."""
    acc = set(acc)
    dog = _dog_body(acc)
    if scene == "winter":
        bg = ('<ellipse cx="120" cy="160" rx="114" ry="24" fill="#F6FAFD"/><ellipse cx="50" cy="152" rx="40" ry="12" fill="#fff"/>'
              '<ellipse cx="192" cy="153" rx="38" ry="11" fill="#fff"/><g>' + _flakes() + '</g>')
        fg = ('<ellipse cx="82" cy="156" rx="30" ry="9" fill="#fff"/><ellipse cx="160" cy="157" rx="30" ry="9" fill="#fff"/>'
              '<circle class="puff" cx="100" cy="150" r="7" fill="#fff" stroke="#CFE0EE"/><circle class="puff" cx="140" cy="150" r="6" fill="#fff" stroke="#CFE0EE" style="animation-delay:-.4s"/>'
              '<circle class="puff" cx="120" cy="154" r="5" fill="#fff" stroke="#CFE0EE" style="animation-delay:-.8s"/>'
              '<circle cx="120" cy="86" r="0" fill="#fff"/>')
    elif scene == "autumn":
        bg = '<ellipse cx="120" cy="160" rx="114" ry="22" fill="#E7C48B"/><g>' + _leaves() + '</g>'
        fg = _pile()
    elif scene == "summer":
        jets = "".join(
            f'<path class="jet" d="M206 96 Q{q} {qy} {ex} {ey}" stroke="#69C3F2" stroke-width="{w}" fill="none" stroke-dasharray="7 7" stroke-linecap="round" style="animation-delay:-{d}s"/>'
            for q, qy, ex, ey, w, d in ((178, 62, 148, 118, 3.5, 0), (182, 70, 160, 126, 3, .15), (172, 58, 134, 112, 2.6, .3)))
        drops = "".join(f'<circle class="dp" cx="{x}" cy="{y}" r="2.4" fill="#8AD3F7" style="animation-delay:-{d}s"/>'
                        for x, y, d in ((150, 112, 0), (158, 118, .3), (140, 108, .55), (164, 124, .8), (132, 104, .15)))
        bg = ('<ellipse cx="120" cy="160" rx="114" ry="22" fill="#A5D274"/>'
              '<path d="M150 118 a46 46 0 0 1 80 -8" stroke="#F4A3A3" stroke-width="3" fill="none" opacity=".4"/>'
              '<path d="M154 118 a42 42 0 0 1 72 -8" stroke="#F6D36B" stroke-width="3" fill="none" opacity=".4"/>'
              '<path d="M158 118 a38 38 0 0 1 64 -8" stroke="#8AD3A3" stroke-width="3" fill="none" opacity=".4"/>')
        fg = ('<ellipse cx="128" cy="156" rx="46" ry="8" fill="#9ED5F2" fill-opacity=".55"/><ellipse class="ripple" cx="128" cy="156" rx="30" ry="5" fill="none" stroke="#fff" stroke-width="2"/>'
              '<rect x="198" y="92" width="14" height="9" rx="3" fill="#F2B632" transform="rotate(-18 205 96)"/>'
              '<path d="M210 99 Q228 104 226 124 Q224 142 204 152" stroke="#2F8F5B" stroke-width="6" fill="none" stroke-linecap="round"/>'
              + jets + drops)
    else:  # spring
        flowers = ""
        for i, (x, col) in enumerate(((34, "#F26B8A"), (56, "#F6C945"), (186, "#B77BE0"), (210, "#F26B6B"), (20, "#FFFFFF"))):
            y0 = 150 - (i % 2) * 4
            flowers += (f'<g class="flw" style="transform-origin:{x}px {y0}px;animation-delay:-{i * .5}s">'
                        f'<path d="M{x} {y0} v-34" stroke="#4E8F3A" stroke-width="3" stroke-linecap="round"/>'
                        f'<path d="M{x} {y0 - 12} q-9 -4 -11 -12 q8 0 11 12z M{x} {y0 - 14} q9 -4 11 -12 q-8 0 -11 12z" fill="#5FA449"/>'
                        f'<path d="M{x - 8} {y0 - 36} q8 -16 16 0 q-2 11 -8 12 q-6 -1 -8 -12z" fill="{col}"/></g>')
        bg = ('<ellipse cx="120" cy="160" rx="114" ry="22" fill="#B9DD87"/><ellipse cx="120" cy="153" rx="84" ry="14" fill="#8B5E3C"/>'
              '<ellipse cx="96" cy="154" rx="14" ry="3" fill="#6F4A2E"/><ellipse cx="146" cy="152" rx="16" ry="3" fill="#6F4A2E"/>' + flowers +
              '<g class="bfly"><g class="wing"><ellipse cx="-5" cy="0" rx="6" ry="4" fill="#F6C945"/></g><g class="wing" style="transform:scaleX(-1)"><ellipse cx="-5" cy="0" rx="6" ry="4" fill="#F6C945"/></g></g>')
        fg = "".join(f'<circle class="dirt" cx="{x}" cy="150" r="{r}" fill="#7A5232" style="--dx:{dx}px;animation-delay:-{d}s"/>'
                     for x, r, dx, d in ((108, 3.2, -26, 0), (116, 2.6, -12, .15), (124, 3, 12, .3), (132, 3.4, 28, .45), (112, 2.2, -40, .55), (128, 2.2, 38, .1)))
    if "wind" in acc:
        bg += ('<g fill="none" stroke="#9CC4DC" stroke-width="3" stroke-linecap="round" opacity=".75">'
               '<path class="wl" d="M4 60 q40 -10 80 0 t70 0" style="animation-delay:-.1s"/>'
               '<path class="wl" d="M20 100 q36 -9 70 0 t60 0" style="animation-delay:-.35s"/>'
               '<path class="wl" d="M120 40 q30 -8 60 0 t52 0" style="animation-delay:-.5s"/></g>')
    return (f'<div class="dogscene sc-{scene}{" windy" if "wind" in acc else ""}"><svg viewBox="0 0 240 172" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            + bg + dog + fg + '</svg></div>')


def dog_scene_for(wx, temp, month, wind=None):
    """(Szene, Zubehör) aus Wetterlage (sun/rain/snow/cloud/None), Temperatur und Monat."""
    season = ("winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter")[month - 1]
    scene = season
    if wx == "snow" or (temp is not None and temp <= 0):
        scene = "winter"
    acc = []
    if wx == "rain":
        acc.append("rain")
    elif wx == "sun" and (temp is None or temp >= 10):
        acc.append("sun")
    if (scene == "winter" or (temp is not None and temp <= 4)) and "rain" not in acc:
        acc.append("scarf")
    if temp is not None and temp >= 30:
        acc.append("hot")
    if wind is not None and wind >= 30:
        acc.append("wind")
    return scene, acc


DOG_OVERLAY_CSS = """
#gwload{background:#FCEBD3}
#gwload .bar{background:rgba(184,87,26,.18)}
#gwload .bar:after{background:#C8661E}
#gwload .fact small{color:#B8571A}
#gwload .fact{border-color:#E8CFAA;background:rgba(255,248,238,.85)}
"""
DOG_FACTS = [
    "Hunde haben bis zu 300 Millionen Riechzellen, Menschen nur etwa 6 Millionen. Beim Schnüffeln liest dein Hund die Nachrichten der Nachbarschaft.",
    "Hunde kühlen sich vor allem durch Hecheln und über die Pfotenballen. Zum Schwitzen taugt ihre Haut kaum, deshalb sind heiße Tage so anstrengend.",
    "Asphalt kann an einem Sommertag deutlich heißer als 50 °C werden, auch wenn die Luft nur 30 °C hat. Der Handrücken-Test hilft: Fünf Sekunden müssen auszuhalten sein.",
    "Schnüffeln ist für Hunde Arbeit: Zehn Minuten Nasenspaziergang machen oft müder und zufriedener als reines Laufen.",
    "Zecken sind von März bis Oktober aktiv. Nach Spaziergängen durch hohes Gras lohnt sich ein kurzer Blick ins Fell.",
    "Schokolade, Weintrauben und Zwiebeln sind für Hunde giftig. Beim Picknick im Park also nichts davon abgeben.",
    "Auf Friedhöfen, Spielplätzen und in vielen Zoos sind Hunde verboten. In Grünanlagen gilt oft Leinenpflicht, besonders zur Brut- und Setzzeit.",
    "Hunde erkennen die Stimmung ihrer Menschen an Tonfall und Körpersprache. Ein entspannter Spaziergang tut beiden gut.",
]


# Ladeansicht beim Berechnen: Wolken ziehen, Logo atmet, Sprüche wechseln. Gleiche Farbe wie die Wolkendecke der
# Karte, damit der Übergang ohne Bruch in die Kartenansicht läuft.
OVERLAY_HTML = """
<style>
#gwload{position:fixed;inset:0;z-index:99999;background:#F4EFE6;display:flex;flex-direction:column;align-items:center;
 justify-content:center;gap:16px;animation:gwfade .35s ease both;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;padding:0 22px}
@keyframes gwfade{from{opacity:0}to{opacity:1}}
#gwload .cbg{position:absolute;inset:-24px;background-size:cover;background-position:center;filter:blur(4px) saturate(.9);
 transform:scale(1.04);animation:gwbgin 1.2s ease both}
#gwload .cmilk{position:absolute;inset:0;background:linear-gradient(rgba(244,239,230,.42),rgba(244,239,230,.58))}
@keyframes gwbgin{from{opacity:0}to{opacity:1}}
#gwload.hasbg .b{opacity:.3}
#gwload .credit{position:absolute;right:10px;bottom:6px;font-size:.62rem;color:#3B3A33;opacity:.55;z-index:2}
#gwload .b{position:absolute;border-radius:50%;filter:blur(34px);background:radial-gradient(closest-side,rgba(255,255,255,.95),rgba(255,255,255,.5) 55%,rgba(255,255,255,0));
 animation:gwdrift 9s ease-in-out infinite alternate}
#gwload .b:nth-child(1){width:70vmax;height:44vmax;left:-20vmax;top:-8vmax}
#gwload .b:nth-child(2){width:60vmax;height:38vmax;right:-18vmax;top:22vh;animation-duration:12s;animation-delay:-3s}
#gwload .b:nth-child(3){width:66vmax;height:40vmax;left:-12vmax;bottom:-12vmax;animation-duration:10s;animation-delay:-5s}
#gwload .b:nth-child(4){width:48vmax;height:30vmax;right:4vmax;bottom:4vh;animation-duration:14s;animation-delay:-8s}
@keyframes gwdrift{from{transform:translate(-4vw,0) scale(1)}to{transform:translate(6vw,3vh) scale(1.12)}}
#gwload .logo{position:relative;width:84px;height:84px;border-radius:24px;background:#4E6E45;display:flex;align-items:center;justify-content:center;
 box-shadow:0 10px 30px rgba(60,50,30,.3);animation:gwbreath 2.2s ease-in-out infinite}
#gwload .logo img{width:54px;height:54px}
#gwload .logo:before{content:"";position:absolute;inset:-10px;border-radius:32px;border:2px solid rgba(78,110,69,.35);animation:gwring 2.2s ease-out infinite}
@keyframes gwbreath{50%{transform:scale(1.07)}}
@keyframes gwring{from{opacity:.9;transform:scale(.92)}to{opacity:0;transform:scale(1.35)}}
#gwload .st{position:relative;text-align:center;color:#3B3A33;font-weight:650;font-size:1.02rem;max-width:min(88vw,380px);min-height:1.4em}
#gwload .bar{position:relative;width:min(60vw,220px);height:4px;border-radius:4px;background:rgba(78,110,69,.18);overflow:hidden}
#gwload .bar:after{content:"";position:absolute;inset:0;width:40%;border-radius:4px;background:#4E6E45;animation:gwslide 1.5s ease-in-out infinite}
@keyframes gwslide{from{transform:translateX(-110%)}to{transform:translateX(280%)}}
#gwload .fact{position:relative;width:min(88vw,380px);min-height:7.2em;margin-top:6px;padding:12px 16px 10px;border-radius:16px;
 background:rgba(251,248,241,.8);border:1px solid #D9CFBC;box-shadow:0 6px 20px rgba(60,50,30,.12);text-align:center}
#gwload .fact small{display:block;font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;font-weight:700;color:#4E6E45;margin-bottom:6px}
#gwload .fact p{position:absolute;left:16px;right:16px;top:34px;margin:0;opacity:0;color:#3B3A33;font-size:.95rem;line-height:1.38}
#gwload .prev{position:relative;width:min(88vw,380px);padding:8px 14px 8px;border-radius:16px;background:rgba(251,248,241,.8);
 border:1px solid #D9CFBC;box-shadow:0 6px 20px rgba(60,50,30,.12);text-align:center;animation:gwfade .5s ease both}
#gwload .prev b{display:block;font-size:.86rem;color:#3B3A33;margin-top:2px}
#gwload .prev small{display:block;font-size:.74rem;color:#6A685E;margin-top:1px}
#gwload .prev .pv{animation:gwmarch 1.6s linear infinite}
@keyframes gwmarch{to{stroke-dashoffset:-16}}
@media (prefers-reduced-motion:reduce){#gwload *{animation-duration:0s!important}#gwload .fact p:first-of-type{opacity:1}}
__FACTCSS__
</style>
<div id="gwload" class="__BGCLS__">__CITYBG__<i class="b"></i><i class="b"></i><i class="b"></i><i class="b"></i>
<div class="logo"><img src="data:image/svg+xml;base64,__LOGO__" alt=""></div>
<div class="st">__STAGE__</div><div class="bar"></div>__PREVIEW__
<div class="fact"><small>__LABEL__</small>__FACTS__</div></div>
"""

# ---- Natur- und Umweltfakten für die Ladeansicht (Ort aus dem Startpunkt, Jahreszeit, allgemein)
# (Stadt: Breite, Länge, Radius km, Fakten)
CITY_FACTS = [
    ("Köln", 50.94, 6.96, 16, [
        "Köln ist Papageien-Stadt: Halsbandsittiche leben hier seit rund 60 Jahren frei, gern in Platanen am Rhein und in Parks.",
        "Der Kölner Grüngürtel entstand in den 1920er-Jahren unter Oberbürgermeister Konrad Adenauer und ist bis heute die grüne Lunge der Stadt."]),
    ("Düsseldorf", 51.23, 6.78, 14, [
        "Auch in Düsseldorf leben wilde Halsbandsittiche, unter anderem in Parks wie dem Hofgarten.",
        "Der Düsseldorfer Hofgarten gehört zu den ältesten öffentlichen Parks Deutschlands."]),
    ("Bonn", 50.73, 7.10, 12, [
        "Im Frühjahr blühen in der Bonner Altstadt Zierkirschen so dicht, dass ganze Straßen rosa leuchten.",
        "In Bonn und am Rhein sind Halsbandsittiche zu Hause: grüne Papageien mitten im Rheinland."]),
    ("Essen", 51.46, 7.01, 12, [
        "Essen war 2017 Grüne Hauptstadt Europas: Aus Zechengelände wurden Parks und Wege, und die Stadt ist deutlich grüner, als ihr Ruf vermuten lässt."]),
    ("Berlin", 52.52, 13.40, 30, [
        "Berlin gilt als Hauptstadt der Nachtigallen: Im Mai singen sie nachts in Parks und Gebüschen.",
        "In Berlins Wäldern leben Wildschweine, besonders im Grunewald. Sie sind meist scheu, aber nachts durchaus in der Stadt unterwegs.",
        "Auf dem Tempelhofer Feld, dem alten Flughafen, brüten Feldlerchen mitten in der Großstadt."]),
    ("Hamburg", 53.55, 9.99, 22, [
        "Die Alster ist ein aufgestauter Fluss: Der Stau reicht bis ins 13. Jahrhundert zurück.",
        "An der Hamburger Elbe sind noch Ebbe und Flut zu spüren, das Tidewasser reicht flussaufwärts bis Geesthacht."]),
    ("München", 48.14, 11.58, 22, [
        "Die Isar wurde zwischen 2000 und 2011 in München wieder naturnah umgebaut. Heute baden die Münchner dort im Sommer.",
        "Der Englische Garten wurde 1789 angelegt und zählt zu den größten Stadtparks der Welt."]),
    ("Frankfurt", 50.11, 8.68, 18, [
        "Der Frankfurter Stadtwald gehört zu den größten Stadtwäldern Deutschlands.",
        "Wilde Halsbandsittiche gibt es auch im Rhein-Main-Gebiet, in Frankfurt, Mainz und Wiesbaden."]),
    ("Stuttgart", 48.78, 9.18, 16, [
        "Mitten in Stuttgart wachsen Reben: Weinberge und Streuobstwiesen ziehen sich bis in den Talkessel."]),
    ("Leipzig", 51.34, 12.37, 16, [
        "Der Leipziger Auwald ist einer der größten erhaltenen Auenwälder mitten in einer europäischen Großstadt."]),
    ("Hannover", 52.37, 9.74, 16, [
        "Die Eilenriede in Hannover zählt zu den größten Stadtwäldern Europas."]),
    ("Dresden", 51.05, 13.74, 18, [
        "Die Dresdner Heide ist ein großes Waldgebiet direkt am Stadtrand, und die Elbwiesen bleiben bewusst unbebaut."]),
    ("Bremen", 53.08, 8.80, 14, [
        "Der Bremer Bürgerpark wurde 1866 von Bürgern gegründet und wird bis heute von einem Verein getragen."]),
    ("Münster", 51.96, 7.63, 12, [
        "Die Promenade rund um die Münsteraner Altstadt liegt auf der früheren Stadtbefestigung. Heute ist sie eine Baumallee."]),
    ("Nürnberg", 49.45, 11.08, 18, [
        "Der Nürnberger Reichswald hieß im Mittelalter „Bienengarten des Reiches“. Hier wurde Honig im großen Stil gewonnen."]),
]
SEASON_FACTS = {
    "winter": ["Im Winter lassen sich Vögel leichter beobachten: Ohne Laub sieht man Nester, Misteln und Spechthöhlen.",
               "Rotkehlchen singen auch im Winter, denn sie verteidigen ihr Revier das ganze Jahr.",
               "Meisen schließen sich im Winter zu gemischten Trupps zusammen und suchen gemeinsam nach Futter.",
               "Raureif und Frost lassen Spinnennetze und Gräser sichtbar werden. Ein Winterspaziergang hat seine eigene Schönheit."],
    "spring": ["Im April und Mai singen die Vögel am lautesten: Der Morgenchor beginnt schon vor Sonnenaufgang.",
               "Bärlauch und Buschwindröschen färben im Frühjahr Auwälder und Parkränder weiß.",
               "Ab Ende April kehren die Mauersegler aus Afrika zurück und kreisen schreiend um die Häuser.",
               "Im Frühjahr treiben Bäume aus: Zuerst blühen oft Weiden und Haselsträucher, wichtige erste Nahrung für Insekten."],
    "summer": ["Linden blühen im Juni und Juli, und ihr Duft lockt Bienen und Hummeln an.",
               "An Hitzetagen sind Parks und Wälder oft mehrere Grad kühler als versiegelte Straßen.",
               "Im Juni und Juli leuchten in der Dämmerung an Wegrändern manchmal Glühwürmchen, die Leuchtkäfer.",
               "Auf Wiesen am Wegrand blühen Wildblumen wie Wiesen-Salbei und Margerite, ein Buffet für Schmetterlinge."],
    "autumn": ["Das Herbstlaub wird nicht „bunt“: Blattgrün wird abgebaut, darunter kommen Gelb und Orange zum Vorschein.",
               "Eichelhäher vergraben im Herbst Tausende Eicheln. Viele vergessen sie, und so entstehen neue Eichen.",
               "Im Herbst ziehen Kraniche in Keilformation über das Land. Ihr Rufen hört man oft, bevor man sie sieht.",
               "Pilze sind nur der Fruchtkörper: Ihr eigentlicher Organismus liegt als feines Geflecht im Boden."],
}
GENERAL_FACTS = [
    "Schon etwa 20 Minuten im Grünen können das Stresshormon Cortisol messbar senken.",
    "Große Laubbäume verdunsten an heißen Tagen viele Liter Wasser und kühlen so ihre Umgebung.",
    "Stadtamseln singen oft höher und lauter als Waldamseln, um den Verkehrslärm zu übertönen.",
    "Bäume tauschen über Pilzfäden im Boden Nährstoffe aus, ein Netz unter dem Waldboden.",
    "Der Buntspecht trommelt bis zu 20-mal pro Sekunde, und sein Schädel federt jeden Schlag ab.",
    "Fledermäuse jagen in der Dämmerung gern über Wegen und Gewässern, ein einzelnes Tier frisst viele Mücken pro Nacht.",
    "Auf Nebenstraßen und Gassen ist die Luft meist spürbar besser als an Hauptverkehrsstraßen.",
    "Ein Eichhörnchen findet viele seiner Verstecke wieder, und die vergessenen werden zu Bäumen.",
    "Gehen kann das kreative Denken fördern: Eine Stanford-Studie fand beim Spazieren deutlich mehr Ideen als im Sitzen.",
    "Moos kann ein Vielfaches seines Trockengewichts an Wasser speichern und wirkt wie ein kleiner Schwamm.",
    "Efeu schadet gesunden Bäumen meist nicht: Er klettert nur und zapft den Baum nicht an.",
    "Regenwürmer ziehen Laub in die Erde und lockern so den Boden, ein Gartenhelfer ohne Gehalt.",
    "Hummeln fliegen schon bei Temperaturen, bei denen Honigbienen im Stock bleiben.",
    "Platanen werfen ihre Rinde in großen Platten ab. So entsteht das typische Tarnmuster am Stamm.",
    "Krähen erkennen menschliche Gesichter und merken sich, wer ihnen freundlich oder feindlich begegnet ist.",
    "Der Mauersegler verbringt fast sein ganzes Leben in der Luft, er frisst, trinkt und schläft im Flug.",
    "Eine alte Eiche kann mehrere hundert Insektenarten beherbergen, ein ganzes Dorf in einem Baum.",
    "Brennnesseln sind wichtige Futterpflanzen für die Raupen von Tagpfauenauge und Kleinem Fuchs.",
    "Die Linde war in vielen Dörfern der Treffpunkt, an dem Feste gefeiert und Recht gesprochen wurde.",
    "Spechte hacken Höhlen in Bäume, die später Meisen, Stare und Fledermäuse als Wohnung nutzen.",
    "Totholz ist Lebensraum: Käfer, Pilze und Vögel leben von abgestorbenen Ästen und Stämmen.",
    "Eine einzelne Hecke kann Nistplatz, Windschutz und Futterquelle für Dutzende Vogelarten sein.",
]
WATER_FACTS = [
    "„Blaue Räume“: Wasser in Sichtweite macht Spaziergänge nach Studien erholsamer als reines Grün.",
    "Libellen gibt es seit über 300 Millionen Jahren. Ihre Larven leben im Wasser und zeigen saubere Gewässer an.",
    "Flüsse und Seen kühlen ihre Umgebung im Sommer spürbar. Am Ufer ist es oft ein paar Grad angenehmer.",
    "Graureiher stehen oft stundenlang regungslos im Flachwasser und warten auf Fische.",
    "Das Plätschern von Wasser überdeckt Verkehrslärm und wird von vielen Menschen als beruhigend empfunden.",
]


def region_facts(lat, lon):
    """Fakten zum Ort (nächstgelegene bekannte Stadt im Umkreis); leer, wenn keine passt."""
    best, bd = None, 1e9
    for name, la, lo, rad, facts in CITY_FACTS:
        d = dist_m((lat, lon), (la, lo)) / 1000
        if d <= rad and d < bd:
            best, bd = (name, facts), d
    return best


def pick_facts(lat=None, lon=None, water=False, month=None, seen=None):
    """Zufällige Auswahl aus allen Fakten (Ort, Jahreszeit, allgemein). Schon gezeigte Fakten kommen erst
    wieder, wenn alle durch sind; mindestens ein Ortsfakt ist unter den ersten drei, wenn es einen gibt."""
    month = month or time.localtime().tm_mon
    season = ("winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter")[month - 1]
    rnd = random.Random()
    local = []
    if lat is not None:
        reg = region_facts(lat, lon)
        if reg:
            local = list(reg[1])
    pool = list(dict.fromkeys(local + SEASON_FACTS[season] + GENERAL_FACTS + (WATER_FACTS if water else [])))
    seen = seen if seen is not None else set()
    fresh = [f for f in pool if f not in seen]
    if len(fresh) < 8:  # fast alles gezeigt: von vorn beginnen
        seen.clear()
        fresh = list(pool)
    rnd.shuffle(fresh)
    out = fresh[:8]
    loc_new = [f for f in local if f in out[:3]]
    if local and not loc_new:
        cand = [f for f in local if f in fresh] or local
        f = rnd.choice(cand)
        if f in out:
            out.remove(f)
        out.insert(rnd.randrange(0, 3), f)
        out = out[:8]
    seen.update(out)
    return out


# ---- Eigene Stadtsilhouetten (immer verfügbar, auch ohne Netz): liegen unter dem Foto und springen ein,
# wenn kein Foto geladen werden kann.
SKY_COL = ("#cfe0ec", "#e8eef0", "#f4efe6")


def _landmark(city):
    """SVG-Teile des Wahrzeichens (Grundlinie y=430, Mitte x=450)."""
    c = "#6f8798"
    if city == "Köln":  # Dom mit zwei Türmen
        return (f'<rect x="330" y="300" width="240" height="130" fill="{c}"/>'
                f'<polygon points="330,300 450,250 570,300" fill="{c}"/>'
                + "".join(f'<polygon points="{x},300 {x + 24},60 {x + 48},300" fill="{c}"/>'
                          f'<rect x="{x - 4}" y="240" width="56" height="190" fill="{c}"/>' for x in (340, 512))
                + "".join(f'<rect x="{x}" y="{y}" width="8" height="28" fill="#e8eef0" opacity=".5"/>' for x in (356, 528) for y in (290, 340)))
    if city in ("Düsseldorf", "Stuttgart"):  # Fernsehturm
        return (f'<polygon points="438,430 450,150 462,430" fill="{c}"/>'
                f'<ellipse cx="450" cy="190" rx="34" ry="14" fill="{c}"/><rect x="444" y="190" width="12" height="40" fill="{c}"/>'
                f'<rect x="448" y="40" width="4" height="120" fill="{c}"/>')
    if city == "Berlin":
        return (f'<polygon points="438,430 450,170 462,430" fill="{c}"/><circle cx="450" cy="170" r="30" fill="{c}"/>'
                f'<rect x="448" y="40" width="4" height="100" fill="{c}"/>')
    if city in ("Dresden", "Aachen"):  # Kuppelbau (Frauenkirche) bzw. Dom mit Kuppel
        return (f'<rect x="350" y="320" width="200" height="110" fill="{c}"/>'
                f'<path d="M365 320 Q450 150 535 320 Z" fill="{c}"/>'
                f'<rect x="440" y="120" width="20" height="60" fill="{c}"/><polygon points="438,120 450,60 462,120" fill="{c}"/>'
                + "".join(f'<rect x="{x}" y="250" width="22" height="70" fill="{c}"/>'
                          f'<polygon points="{x - 2},250 {x + 11},212 {x + 24},250" fill="{c}"/>' for x in (330, 548)))
    if city == "München":  # zwei Türme mit Hauben
        return (f'<rect x="340" y="290" width="220" height="140" fill="{c}"/>'
                + "".join(f'<rect x="{x}" y="170" width="50" height="130" fill="{c}"/>'
                          f'<path d="M{x - 4} 170 Q{x + 25} 90 {x + 54} 170 Z" fill="{c}"/>'
                          f'<rect x="{x + 22}" y="70" width="6" height="40" fill="{c}"/>' for x in (350, 500)))
    if city == "Frankfurt":  # Hochhäuser
        return "".join(f'<rect x="{x}" y="{y}" width="{w}" height="{430 - y}" fill="{c}"/>'
                       for x, y, w in ((330, 200, 50), (390, 110, 56), (456, 160, 50), (516, 230, 44), (300, 290, 34)))
    # Kirchturm mit Spitze (Rathaus/Kirche) als Allgemeinmotiv
    return (f'<rect x="410" y="210" width="80" height="220" fill="{c}"/><polygon points="404,210 450,90 496,210" fill="{c}"/>'
            f'<rect x="446" y="40" width="8" height="60" fill="{c}"/><rect x="300" y="320" width="110" height="110" fill="{c}"/>'
            f'<polygon points="296,320 355,270 414,320" fill="{c}"/><rect x="490" y="340" width="120" height="90" fill="{c}"/>')


def city_art(city):
    """Data-URI einer weichen Stadtsilhouette mit Wahrzeichen, Fluss und Bäumen."""
    rnd = random.Random(sum(ord(ch) for ch in (city or "x")))
    far = ""
    x = -20
    while x < 920:
        w, h = rnd.randint(34, 80), rnd.randint(50, 150)
        far += f'<rect x="{x}" y="{430 - h}" width="{w}" height="{h}" fill="#9db0bf" opacity=".55"/>'
        x += w + rnd.randint(0, 6)
    river = ""
    if city in ("Köln", "Düsseldorf", "Bonn", "Frankfurt", "Dresden", "Hamburg", "Bremen"):
        river = ('<rect x="0" y="430" width="900" height="70" fill="#a9c4d6"/>'
                 + "".join(f'<rect x="{rnd.randint(0, 800)}" y="{440 + 11 * k}" width="{rnd.randint(60, 140)}" height="3" '
                           f'fill="#e8f0f4" opacity=".7"/>' for k in range(6)))
    trees = ""
    for k in range(14):
        tx = rnd.randint(-20, 920)
        ty = 500 + rnd.randint(0, 70)
        r = rnd.randint(26, 52)
        trees += (f'<circle cx="{tx}" cy="{ty}" r="{r}" fill="#9DB38A" opacity=".85"/>'
                  f'<circle cx="{tx + r // 2}" cy="{ty - r // 3}" r="{int(r * .7)}" fill="#7f9c6e" opacity=".85"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 600" preserveAspectRatio="xMidYMid slice">'
           f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{SKY_COL[0]}"/>'
           f'<stop offset=".6" stop-color="{SKY_COL[1]}"/><stop offset="1" stop-color="{SKY_COL[2]}"/></linearGradient></defs>'
           f'<rect width="900" height="600" fill="url(#g)"/>{far}{_landmark(city)}'
           f'<rect x="0" y="430" width="900" height="170" fill="#c9d6bd"/>{river}{trees}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


_BG_CACHE = {}  # gerundete Koordinaten -> (Stadt, Bild-URL oder "")
_BG_LOCK = threading.Lock()


def _city_name(lat, lon):
    reg = region_facts(lat, lon)
    if reg:
        return reg[0]
    r = requests.get("https://photon.komoot.io/reverse", headers=UA, timeout=3,
                     params={"lat": lat, "lon": lon, "lang": "de", "limit": 1})
    pr = (r.json().get("features") or [{}])[0].get("properties", {})
    return pr.get("city") or pr.get("town") or pr.get("county") or pr.get("state") or ""


CITY_MOTIF = {  # Suchbegriffe für typische Motive (Wikimedia Commons)
    "Köln": "Köln Dom Skyline Rhein", "Düsseldorf": "Düsseldorf Rheinturm Skyline", "Aachen": "Aachener Dom",
    "Dresden": "Dresden Frauenkirche Altstadt", "Bonn": "Bonn Rhein Panorama", "Essen": "Essen Zollverein",
    "Berlin": "Berlin Skyline Fernsehturm", "Hamburg": "Hamburg Elbphilharmonie Skyline", "München": "München Frauenkirche Skyline",
    "Frankfurt": "Frankfurt Skyline Main", "Stuttgart": "Stuttgart Skyline", "Leipzig": "Leipzig Skyline Panorama",
    "Hannover": "Hannover Neues Rathaus", "Bremen": "Bremen Marktplatz Rathaus", "Münster": "Münster Prinzipalmarkt",
    "Nürnberg": "Nürnberg Kaiserburg Altstadt",
}


def _wiki_get(host, params):
    r = requests.get(f"https://{host}/w/api.php", headers=UA, timeout=3.5, params={"format": "json", **params})
    r.raise_for_status()
    return r.json().get("query", {}).get("pages") or {}


def _ok_img(u):
    return isinstance(u, str) and u.startswith("https://upload.wikimedia.org/") and not u.lower().endswith(".svg.png")


def _img_pageimage(city):
    for lang in ("de", "en"):
        for pg in _wiki_get(f"{lang}.wikipedia.org", {"action": "query", "titles": city, "redirects": 1,
                                                        "prop": "pageimages", "piprop": "thumbnail",
                                                        "pithumbsize": 900}).values():
            u = (pg.get("thumbnail") or {}).get("source", "")
            if _ok_img(u):
                return u
    return ""


def _img_commons(term):
    """Bildsuche in Wikimedia Commons (nur Fotos, Querformat), erstes passendes Ergebnis."""
    pages = _wiki_get("commons.wikimedia.org", {
        "action": "query", "generator": "search", "gsrsearch": f"{term} filetype:bitmap", "gsrnamespace": 6,
        "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|size", "iiurlwidth": 900})
    best = []
    for pg in pages.values():
        ii = (pg.get("imageinfo") or [{}])[0]
        u, w, h = ii.get("thumburl", ""), ii.get("width", 0), ii.get("height", 0)
        if _ok_img(u) and w >= 1200 and w > h * 1.15 and not str(pg.get("title", "")).lower().endswith((".gif", ".png")):
            best.append((pg.get("index", 99), u))
    return min(best)[1] if best else ""


def _city_image(city):
    """Freies Foto der Stadt: bei bekannten Städten zuerst ein Motiv-Treffer (Skyline/Wahrzeichen) aus Commons,
    sonst das Titelbild des Wikipedia-Artikels."""
    steps = []
    if city in CITY_MOTIF:
        steps.append(lambda: _img_commons(CITY_MOTIF[city]))
    steps += [lambda: _img_pageimage(city), lambda: _img_commons(f"{city} Skyline")]
    for fn in steps:
        try:
            u = fn()
            if u:
                return u
        except Exception:
            continue
    return ""


def city_background(lat, lon):
    """(Stadt, Bild-URL) zum Ort; leer, wenn nichts gefunden. Treffer werden gemerkt, Fehlschläge nur kurz
    (2 Minuten), damit ein einmaliger Netzfehler das Bild nicht dauerhaft verhindert."""
    key = (round(lat, 2), round(lon, 2))
    now = time.time()
    with _BG_LOCK:
        hit = _BG_CACHE.get(key)
        if hit and (hit[1] or now - hit[2] < 120):
            return hit[:2]
    try:
        city = _city_name(lat, lon)
        with _BG_LOCK:
            for v in _BG_CACHE.values():  # gleiche Stadt schon bekannt
                if v[0] == city and v[1]:
                    _BG_CACHE[key] = v
                    return v[:2]
        res = (city, _city_image(city) if city else "", now)
    except Exception:
        return ("", "")
    with _BG_LOCK:
        _BG_CACHE[key] = res
    return res[:2]


class Overlay:
    """Vollbild-Ladeansicht mit Arbeitsstand (aktueller Schritt) und wechselnden Natur-/Ortsfakten."""
    PER = 7.0  # Sekunden je Fakt

    def __init__(self, start=None, water=False):
        lat = lon = None
        try:
            lat, lon, _ = _unpack(start)
        except Exception:
            pass
        try:
            seen = st.session_state.setdefault("_facts_seen", set())
        except Exception:
            seen = None
        self.facts = pick_facts(lat, lon, water, seen=seen)
        self.dog = _dog()
        self.pos = (lat, lon) if lat is not None else None
        if self.dog:
            self.facts = random.sample(DOG_FACTS, 3) + self.facts[:5]
            if self.pos:
                try:
                    get_weather(lat, lon, wait=0.3)
                except Exception:
                    pass
        self.bg = ""
        self.city = ""
        self._bgt = None
        self._prev = None        # Sofort-Vorschau (schnellste Route), kommt aus einem Hintergrund-Thread
        self._prev_shown = False
        if lat is not None:
            def work():
                try:
                    self.city, self.bg = city_background(lat, lon)
                except Exception:
                    pass
            self._bgt = threading.Thread(target=work, daemon=True)
            self._bgt.start()
            self._bgt.join(0.3)  # kurz warten, sonst erscheint das Bild beim nächsten Schritt
        self.laps = [(time.time(), "Start")]
        try:
            st.session_state.pop("_laps_end", None)
        except Exception:
            pass
        self.stage = "Ich suche den grünsten Weg" + (" am Wasser entlang" if water else "") + " …"
        self.ph = st.empty()
        self.render()

    def render(self):
        try:
            n = len(self.facts)
            total = n * self.PER
            vis = (self.PER - 0.6) / total * 100
            fade = self.PER / total * 100
            css = (f"@keyframes gwfact{{0%{{opacity:0;transform:translateY(8px)}}{0.5 / total * 100:.2f}%,{vis:.2f}%{{opacity:1;transform:none}}"
                   f"{fade:.2f}%,100%{{opacity:0;transform:translateY(-8px)}}}}"
                   + "".join(f"#gwload .fact p:nth-of-type({k + 1}){{animation:gwfact {total:.1f}s {k * self.PER:.1f}s infinite}}" for k in range(n)))
            facts = "".join(f"<p>{escape(f)}</p>" for f in self.facts)
            logo = base64.b64encode(LOGO_SVG.encode()).decode()
            html = (OVERLAY_HTML.replace("__FACTCSS__", css + self._bg_css()).replace("__LOGO__", logo)
                    .replace("__STAGE__", escape(self.stage)).replace("__FACTS__", facts)
                    .replace("__LABEL__", "Wusstest du?").replace("__PREVIEW__", self._prev_html())
                    .replace("__CITYBG__", self._bg_html()).replace("__BGCLS__", "hasbg"))
            if self.dog:
                html = html.replace('<div class="logo"><img src="data:image/svg+xml;base64,' + logo + '" alt=""></div>', self._dog_scene())
                html = html.replace("</style>", DOG_SCENE_CSS + DOG_OVERLAY_CSS + "</style>", 1)
            self.ph.markdown(html, unsafe_allow_html=True)
        except Exception:
            pass

    def set_preview(self, pts, what="Schnellste Fußroute"):
        """Sofort-Vorschau: holt im Hintergrund die schnellste Route und zeigt sie als Skizze, sobald sie da ist.
        Die grüne Berechnung wartet nicht darauf."""
        self._prev_what = what

        def work():
            r = quick_route(pts)
            if r:
                self._prev = r
        try:
            threading.Thread(target=work, daemon=True).start()
        except Exception:
            pass

    def _prev_html(self):
        pv = self._prev
        if not pv:
            return ""
        try:
            m = pv["m"]
            mins = max(1, round(m / M_PER_MIN))
            return (f'<div class="prev">{preview_svg(pv)}<b>{escape(getattr(self, "_prev_what", "Schnellste Fußroute"))}: '
                    f'{fmt_km(m)} · {mins} min</b><small>Die grünere Route wird noch berechnet …</small></div>')
        except Exception:
            return ""

    def _poll_preview(self):
        """True, wenn die Vorschau seit dem letzten Zeichnen neu dazugekommen ist."""
        if self._prev and not self._prev_shown:
            self._prev_shown = True
            return True
        return False

    def _dog_scene(self):
        """Ladehund passend zu Jahreszeit und aktuellem Wetter (Sonnenbrille, Regencape, Schal, Schweißtropfen)."""
        w = None
        if self.pos:
            try:
                w = get_weather(self.pos[0], self.pos[1])
            except Exception:
                w = None
        temp = w["temp"] if w else None
        wx = wx_kind(w["code"], w["temp"], w["day"]) if w else None
        scene, acc = dog_scene_for(wx, temp, time.localtime().tm_mon, (w or {}).get("wind"))
        return dog_scene_html(scene, acc)

    def _photo(self):
        u = self.bg or ""
        if u.startswith("https://upload.wikimedia.org/") and not any(c in u for c in "\"'<> \\{};"):
            return u
        return ""

    def _bg_css(self):
        """Foto (falls vorhanden) über der eigenen Silhouette; die Silhouette ist immer da."""
        layers = []
        if self._photo():
            layers.append(f'url("{self._photo()}")')
        layers.append(f'url("{city_art(self.city)}")')
        return "#gwload .cbg{background-image:" + ",".join(layers) + "}"

    def _bg_html(self):
        credit = "<div class='credit'>Foto: Wikimedia Commons</div>" if self._photo() else ""
        return "<div class='cbg'></div><div class='cmilk'></div>" + credit

    def log(self, msg):
        """Schritt anzeigen (und weiter ins versteckte Statusfeld schreiben)."""
        self._lap(msg)
        try:
            st.write(msg)
        except Exception:
            pass
        txt = re.sub(r"\s*\([^)]*\)\s*$", "", str(msg)).strip()
        newp = self._poll_preview()
        if txt and txt != self.stage and len(txt) < 140:
            self.stage = txt
            self.render()
        elif newp:
            self.render()

    def step(self, msg):
        """Wie log, aber ohne Textzeile auf der Seite (für die Kartenansicht)."""
        self._lap(msg)
        txt = re.sub(r"\s*\([^)]*\)\s*$", "", str(msg)).strip()
        newp = self._poll_preview()
        if txt and txt != self.stage and len(txt) < 140:
            self.stage = txt
            self.render()
        elif newp:
            self.render()

    def _lap(self, msg):
        """Merkt, wie lange jeder Schritt gedauert hat (Anzeige unten auf der Startseite: „Zeiten der letzten Berechnung“)."""
        try:
            self.laps.append((time.time(), str(msg)))
            st.session_state["_laps"] = list(self.laps)
        except Exception:
            pass

    def empty(self):
        try:
            self.ph.empty()
        except Exception:
            pass


def loading_overlay(start=None, water=False):
    return Overlay(start, water)


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

PLACEHOLDER = "Wohin? Ort, Adresse, Café, Bier, Essen …"
SEARCH_TIP = ("Tipp: Tippe mindestens 3 Buchstaben, dann erscheinen Vorschläge. "
              "Tippfehler sind okay, es werden ähnliche Orte angezeigt. "
              "Du kannst auch Adressen einfügen oder Schlagworte wie „Bier“, „Essen“ oder „Bäcker“ tippen (dann kommen Kneipen, Restaurants bzw. Bäckereien in der Nähe). Der Start lässt sich überall in Deutschland setzen (am besten mit Stadt, z. B. „Torstraße Berlin“), "
              "das Ziel sollte höchstens 10 km vom Start entfernt sein und wird zuerst dort gesucht.")


SEARCH_STYLE = {
    "searchbox": {
        "control": {"borderRadius": 14, "minHeight": 50, "borderColor": "#D9CFBC", "backgroundColor": "#FFFFFF",
                    "boxShadow": "0 1px 2px rgba(60,50,30,.06)"},
        "menu": {"borderRadius": 14, "overflow": "hidden", "border": "1px solid #D9CFBC",
                 "boxShadow": "0 8px 24px rgba(60,50,30,.18)"},
        "option": {"padding": "14px 16px", "fontSize": "16px", "lineHeight": "1.3"},
    },
}


def _sb_gen():
    return st.session_state.get("_sbgen", 0)


def bump_sb_gen():
    """Suchfelder bekommen nach jedem Neuaufbau der Startseite einen frischen Schlüssel. Sonst bleibt im Hintergrund
    ein alter Suchzustand zurück, während das Feld im Browser neu startet. Das Feld reagiert dann nicht mehr."""
    old = _sb_gen()
    st.session_state["_sbgen"] = old + 1
    try:
        for k in [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(f"g{old}_") and k.endswith("_sb")]:
            del st.session_state[k]
    except Exception:
        pass


def searchbox(fn, **kw):
    """Eine einzige Suchleiste für alle Reiter und die Karte: gleiche Suche, gleiches Aussehen, gleiche Vorschläge
    (Orte, Adressen und Schlagworte wie „Bier“ oder „Essen“)."""
    try:
        return st_searchbox(fn, style_overrides=SEARCH_STYLE, **kw)
    except TypeError:  # ältere Version ohne style_overrides / default_searchterm
        for k in ("default", "default_searchterm"):
            kw.pop(k, None)
        try:
            return st_searchbox(fn, style_overrides=SEARCH_STYLE, **kw)
        except TypeError:
            return st_searchbox(fn, **kw)


def place_input(label, key):
    """Ortseingabe. Die zuletzt gewählte Adresse bleibt im Feld stehen, auch nach einem Ausflug in die Karte."""
    dot = "s" if label.startswith("Start") else "z"
    st.markdown(f'<div class="fld"><i class="dot {dot}"></i>{escape(label)}</div>', unsafe_allow_html=True)
    keep = f"_keep_{key}"
    kw = dict(key=f"g{_sb_gen()}_{key}", placeholder=PLACEHOLDER, clear_on_submit=False)
    saved = st.session_state.get(keep)
    if saved:
        try:
            kw.update(default=saved, default_searchterm=_unpack(saved)[2])
        except Exception:
            st.session_state.pop(keep, None)
    fn = search_start if label.startswith("Start") else search_places
    v = searchbox(fn, **kw)
    if v:
        st.session_state[keep] = v
    return v


@st.cache_data(ttl=3600, show_spinner=False)
def reverse_name(lat, lon):
    """Adresse zu Koordinaten (Photon), nur zur Anzeige."""
    try:
        r = requests.get("https://photon.komoot.io/reverse", headers=UA, timeout=2.0,
                         params={"lat": lat, "lon": lon, "lang": "de"})
        p = r.json()["features"][0]["properties"]
        street = " ".join(x for x in [p.get("street") or p.get("name"), p.get("housenumber")] if x)
        return ", ".join(x for x in [street, p.get("city") or p.get("district")] if x)
    except Exception:
        return ""


def start_input(label, key):
    """Startpunkt: per Suche oder per Standortfreigabe des Browsers (Schalter)."""
    use = st.toggle("📍 Von meinem Standort starten", key=f"{key}_useloc", value=True,
                    help="Der Browser fragt nach der Standortfreigabe. Der Standort wird nur für die Suche genutzt.")
    if not use:
        v = place_input(label, key)
        if v:
            st.session_state["bias"] = _unpack(v)[:2]
            try:
                prewarm(*_unpack(v)[:2])
            except Exception:
                pass
        return v

    skey = "loc_shared"  # ein Standort für alle Reiter: wird nur einmal abgefragt
    loc = st.session_state.get(skey)
    if loc is None:
        raw = get_geolocation(component_key="geo_shared")
        if isinstance(raw, dict) and raw.get("error"):
            st.warning("Der Standort ist nicht verfügbar. Bitte die Freigabe im Browser erlauben "
                       "(Schloss-Symbol in der Adressleiste) und den Schalter erneut setzen, "
                       "oder den Startpunkt über die Suche eingeben.")
            v = place_input(label, key)
            if v:
                st.session_state["bias"] = _unpack(v)[:2]
            return v
        c = raw.get("coords") if isinstance(raw, dict) else None
        if not c or c.get("latitude") is None:
            st.info("📡 Standort wird ermittelt … bitte im Browser die Freigabe erlauben.")
            return None
        loc = (float(c["latitude"]), float(c["longitude"]), c.get("accuracy"))
        st.session_state[skey] = loc
    lat, lon, acc = loc
    try:
        prewarm(lat, lon)
    except Exception:
        pass
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
    if any(x in msg for x in ("500", "502", "503", "504", "überlastet", "Timeout", "timed out", "capacity")):
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
        "- Ersatz bei überlasteten Servern (Grün-Scan): Vektorkacheln von OpenFreeMap (OpenMapTiles, "
        "OSM-Daten) + Routenserver von FOSSGIS (routing.openstreetmap.de)\n"
        "- Kartenhintergrund: Esri Topo (Standard), Esri Straßen, OpenTopoMap, OpenStreetMap\n"
        "- Satellit/Luftbild (🛰️): Esri World Imagery; Geobasis NRW Digitalorthophotos 10 cm (nur NRW, dl-de/zero); "
        "Sentinel-2 cloudless 2025 von EOX (10 m, nicht kommerziell, © EOX IT Services GmbH, Copernicus-Sentinel-Daten)\n"
        f"- Grünflächen-Tags: `{GREEN_TAGS}`\n"
        f"- Score je Abtastpunkt: 1,0 im Park, linear fallend auf 0 im Abstand von {buf} m\n"
        f"- Kantenscore: Mittel aus {SAMPLES} Abtastpunkten\n"
        f"- Kosten: Länge × (1 + λ × (1 − Grünscore) + {QUIET:g} × Hauptstraßen-Nähe), λ je Grünstufe: "
        + ", ".join(f"{k} {v:g}" for k, v in LEVELS.items())
        + f"\n- Hauptstraßen (Bundes-, Haupt-, Neben-, Kreisstraßen) zählen bis {NOISE_DIST:g} m Abstand als laut "
        "(bei „Schnellste“ ignoriert)\n"
        "- Trinkwasser (🚰): OSM-Tags `amenity=drinking_water` und `man_made=water_tap` mit `drinking_water=yes`, Overpass direkt vom Gerät, "
        "bis 300 m neben der Route; viele Brunnen sind im Winter abgestellt\n"
        "- Beleuchtete Wege (💡): OSM-Tag `lit`. Wege mit `lit=no` kosten +60 %, nicht erfasste Fuß- und Waldwege +30 %, "
        "nicht erfasste Straßen +8 % (nur wenn der Schalter aktiv ist)\n"
        f"- Extrem grün: grünste Strecke mit höchstens {EXTREME_DETOUR:.0%} Umweg (Ründchen: {EXTREME_DETOUR:.0%} länger als "
        f"gewünscht), nur angeboten bei über {EXTREME_GREEN_MIN:.0%} Grün\n"
        "- Hundemodus: Verbotszonen (Friedhof, Spielplatz, Zoo, dog=no) und Belag (surface) aus OpenStreetMap; "
        "Wetter für Pfotenwarnung (ab 30 °C) und Ladehund von Open-Meteo (open-meteo.com, nicht kommerziell)"
    )


# ---------- Wechsel zwischen Startseite und Kartenansicht
def open_map(view):
    try:
        if "gwdone" in st.query_params:
            del st.query_params["gwdone"]
    except Exception:
        pass
    st.session_state["gw_sid"] = st.session_state.get("gw_sid", 0) + 1
    st.session_state["_laps_end"] = time.time()
    view["sid"] = st.session_state["gw_sid"]
    st.session_state["gw"] = view
    st.session_state["bias"] = view["p1"]


def soft_back():
    """Nächster Aufbau der Startseite blendet weich ein (statt hart umzuschalten)."""
    st.session_state["_back"] = True


def close_map():
    """Zurück aus der Karte. Nach einer beendeten Strecke geht es direkt zur Startseite, nicht zur Vorauswahl."""
    st.session_state.pop("gw", None)
    soft_back()
    try:
        if st.query_params.get("gwdone"):
            st.session_state.pop("green_pending", None)
            st.session_state.pop("san_pending", None)
    except Exception:
        pass
    try:
        if "gwdone" in st.query_params:
            del st.query_params["gwdone"]
    except Exception:
        pass


# Eingaben der Startseite bleiben beim Wechsel in die Karte und zurück erhalten (Streamlit verwirft sonst
# den Zustand von Widgets, die in einem Durchlauf nicht gezeichnet werden).
_KEEP_KEYS = ("page_sel", "water_", "acc_", "lit_", "san_", "green_prefer", "green_avoid", "green_size", "level_", "loop_km",
              "loop_same", "tour_same", "tour_extra", "n_stops", "start_sb_useloc", "green_sb_useloc", "loop_a_sb_useloc",
              "tour_a_sb_useloc", "dog_mode")


_BUTTON_KEYS = {"san_plan", "san_show", "san_cancel"}


def snap_widgets():
    try:
        st.session_state["_saved_widgets"] = {k: v for k, v in st.session_state.items()
                                              if isinstance(k, str) and k.startswith(_KEEP_KEYS)
                                              and not k.endswith("_sb") and k != "green_pending"
                                              and not re.search(r"_c\d+$", k)
                                              and k not in _BUTTON_KEYS}
    except Exception:
        pass


def restore_widgets():
    try:
        for k, v in st.session_state.get("_saved_widgets", {}).items():
            if k not in st.session_state and k not in _BUTTON_KEYS:
                try:
                    st.session_state[k] = v
                except Exception:  # z. B. Button-Schlüssel: nicht setzbar, einfach überspringen
                    pass
    except Exception:
        pass


BACK_CSS = """<style>
@keyframes gwback{0%{opacity:1}35%{opacity:.85}100%{opacity:0;visibility:hidden}}
@keyframes gwbackin{0%{opacity:0;transform:translateY(8px)}100%{opacity:1;transform:none}}
.stApp::after{content:"";position:fixed;inset:0;background:#F3EEE5;z-index:99999;pointer-events:none;
 animation:gwback 1.25s cubic-bezier(.4,0,.2,1) forwards}
.block-container{animation:gwbackin 1.1s cubic-bezier(.25,.1,.25,1) both}
@media (prefers-reduced-motion:reduce){.stApp::after{display:none}.block-container{animation:none}}
</style>"""


def back_fx():
    if st.session_state.pop("_back", False):
        st.markdown(BACK_CSS, unsafe_allow_html=True)


def set_retry():
    gw = st.session_state.get("gw")
    if gw:
        gw["retry"] = True


def bump_km(delta):
    gw = st.session_state.get("gw")
    if gw:
        gw["km"] = round(min(max(gw["km"] + delta, 1.0), LOOP_MAX_KM), 1)
        gw["recalc"] = True


def _san_carry(old, new):
    """Sanitäroptionen bleiben beim Neuberechnen in der Karte aktiv."""
    new["access"] = old.get("access", False)
    new["lit"] = old.get("lit", False)
    if new["access"]:
        try:
            access_hint(new)
        except Exception:
            pass
    reach = old.get("san_reach")
    if not reach:
        return
    new["san_reach"] = reach
    try:
        add_sanitary(new, reach)
    except Exception:
        pass


@contextmanager
def map_busy(gw, preview_pts=None, preview_what="Schnellste Fußroute"):
    """Während in der Karte neu berechnet wird: Ladeansicht mit Arbeitsstand statt stiller Wartezeit."""
    ov = None
    try:
        ov = loading_overlay(gw.get("start"), bool(gw.get("water", False)))
        if preview_pts:
            ov.set_preview(preview_pts, preview_what)
    except Exception:
        pass
    try:
        yield (ov.step if ov else (lambda *_: None))
    finally:
        if ov:
            ov.empty()


VIEW_CACHE_MAX = 6


def view_snapshot(v):
    return {k: x for k, x in v.items() if k not in ("vcache", "recalc", "_carried")}


def render_map_view():
    """Vollbild-Karte mit Suchfeld (Ziel ändern) bzw. Längen-Regler (Ründchen) in der Karte."""
    gw = st.session_state["gw"]
    st.markdown(MAP_CSS, unsafe_allow_html=True)
    st.session_state["bias"] = gw["p1"]
    st.session_state["_water"] = bool(gw.get("water", False))
    st.session_state["_access"] = bool(gw.get("access", False))
    st.session_state["_lit"] = bool(gw.get("lit", False))
    st.session_state["_dog"] = bool(gw.get("dog", False))
    st.session_state["_temp"] = gw.get("temp")
    st.session_state["_windnote"] = gw.get("wnote", "")
    dest_new = add_new = None
    with st.container(key="gwtop"):
        if gw["mode"] == "ab":
            if gw.get("fallback"):
                c1, c2, c3 = st.columns([1, 5, 1])
                c3.button("🌿", key="gw_retry", use_container_width=True, on_click=set_retry,
                          help="Genauere OSM-Grünflächen erneut versuchen")
            else:
                c1, c2 = st.columns([1, 6])
            c1.button("←", key="gw_back", use_container_width=True, on_click=close_map, help="Zurück zur Suche")
            with c2:
                dest_new = searchbox(search_places, key=f"gw_dest_{gw['sid']}",
                                     placeholder=f"🏁 {gw['t2']} · Ziel ändern …", clear_on_submit=False)
        elif gw["mode"] == "tour":
            c1, c2 = st.columns([1, 6])
            c1.button("←", key="gw_back", use_container_width=True, on_click=close_map, help="Zurück zur Suche")
            n_ = len(gw.get("stops", []))
            with c2:
                if n_ < TOUR_MAX_MAP_STOPS:
                    add_new = searchbox(search_places, key=f"gw_add_{gw['sid']}", clear_on_submit=False,
                                        placeholder=f"🛍️ {n_} Zwischenziel{'e' if n_ != 1 else ''} · ➕ weiteres Ziel …")
                else:
                    st.markdown(f'<div class="looptitle">🛍️ Ründchen · <b>{n_} Zwischenziele</b> (Maximum)</div>',
                                unsafe_allow_html=True)
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
    if gw["mode"] == "ab" and gw.get("retry"):
        gw["retry"] = False
        st.toast("🌿 Versuche die grüne Route erneut …")
        try:
            new = make_ab_view_safe(gw["start"], gw["dest"], gw["buf"])
            if new.get("fallback"):
                st.toast("Die OSM-Server sind noch überlastet. Der Grün-Scan bleibt aktiv.")
                new = None
        except Exception as e:
            st.toast(f"⚠️ {e}"[:220])
    elif gw["mode"] == "ab" and dest_new and dest_new != gw["dest"] and dest_new != gw.get("failed"):
        try:
            try:
                _pp = [tuple(_unpack(gw["start"])[:2]), tuple(_unpack(dest_new)[:2])]
            except Exception:
                _pp = None
            with map_busy(gw, _pp) as step:
                if gw.get("fallback") == "scan":
                    new = scan_ab_view(gw["start"], dest_new, gw["buf"], step)
                elif gw.get("fallback"):
                    new = make_ab_view_safe(gw["start"], dest_new, gw["buf"], log=step)
                else:
                    new = make_ab_view_safe(gw["start"], dest_new, gw["buf"], gw["area"], gw["bbox"], step)
        except Exception as e:
            gw["failed"] = dest_new
            st.toast(f"⚠️ {e}"[:220])
    elif gw["mode"] == "tour" and add_new and add_new != gw.get("failed"):
        try:
            try:
                _pp = tour_points(gw["start"], gw["stops_v"] + [add_new], gw["end_v"])[0]
            except Exception:
                _pp = None
            with map_busy(gw, _pp, "Direkter Weg über die Zwischenziele") as step:
                new = make_tour_view_safe(gw["start"], gw["stops_v"] + [add_new], gw["end_v"], gw["level"], gw["buf"],
                                          log=step, extra_km=gw.get("extra_km", 0))
            new["sid"] = gw["sid"] + 1  # neues leeres Suchfeld
            _san_carry(gw, new)
            st.session_state["gw_sid"] = max(st.session_state.get("gw_sid", 0), new["sid"])
            st.session_state["gw"] = new
            st.rerun()
        except ValueError as e:
            gw["failed"] = add_new
            st.toast(f"⚠️ {e}"[:220])
        except Exception as e:
            gw["failed"] = add_new
            st.toast(f"⚠️ Neu berechnen nicht möglich: {e}"[:220])
    elif gw["mode"] == "loop" and gw.get("recalc"):
        gw["recalc"] = False
        vc = gw.setdefault("vcache", {})
        vc.setdefault(gw["km_ok"], {**view_snapshot(gw), "km": gw["km_ok"]})  # die gerade gezeigte Länge bleibt abrufbar
        hit = vc.get(gw["km"])
        if hit is not None:  # schon berechnet: sofort anzeigen
            new = dict(hit)
            new["vcache"] = vc
            new["sid"] = gw["sid"]
            st.session_state["gw"] = new
            st.rerun()
        try:
            with map_busy(gw) as step:
                if gw.get("fallback") == "scan":
                    new = scan_loop_view(gw["start"], gw["end"], gw["km"], gw["level"], gw["buf"], step)
                else:
                    new = make_loop_view_safe(gw["start"], gw["end"], gw["km"], gw["level"], gw["buf"],
                                              area=gw["area"], bbox_km=gw["bbox_km"], log=step, weather=False)
                new["sid"] = gw["sid"]
                step("🚻 Toiletten entlang der Route suchen") if gw.get("san_reach") else None
                _san_carry(gw, new)
                new["_carried"] = True
            vc[new["km"]] = view_snapshot(new)
            while len(vc) > VIEW_CACHE_MAX:
                vc.pop(next(iter(vc)))
            new["vcache"] = vc
        except Exception as e:
            new = None
            gw["km"] = gw["km_ok"]
            st.toast(f"⚠️ {e}"[:220])
    if new:
        new["sid"] = gw["sid"]
        if not new.pop("_carried", False):
            _san_carry(gw, new)
        st.session_state["gw"] = new
        st.rerun()

    with st.container(key="mapfull"):
        st_html(gw["html"], height=700, scrolling=False)


# ---------- Startseite
def access_toggle(prefix):
    """Schalter „Barrierefreiheit“ (wie Wassernähe): plant ohne Treppen, steile Stücke und schlechte Beläge."""
    return st.toggle("♿ Barrierefreiheit", key=f"acc_{prefix}", value=False,
                     help="Plant stufenfrei: Treppen, sehr steile Stücke und schlechte Beläge (Sand, grober Schotter, "
                          "Kopfsteinpflaster) werden gemieden. Es zählt, was in OpenStreetMap erfasst ist.")


def access_hint(view):
    """Hinweis zur Barrierefreiheit an jeder Routenvariante (im Ersatzmodus ehrlich: nicht prüfbar)."""
    if not view.get("access"):
        return view
    payload = payload_from_html(view["html"])
    note = (" · ♿ im Ersatzmodus nicht prüfbar, Treppen bitte selbst beachten" if view.get("fallback")
            else " · ♿ stufenfrei geplant: Treppen und schlechte Beläge gemieden")
    for o in payload["options"]:
        o["hint"] = (o.get("hint") or "") + note
    view["html"] = map_html(payload)
    return view


def lit_toggle(prefix):
    """Schalter „Beleuchtete Wege“ (für den Abend): bevorzugt Wege mit lit=yes, meidet unbeleuchtete."""
    return st.toggle("💡 Beleuchtete Wege", key=f"lit_{prefix}", value=False,
                     help="Für den Abend und die dunkle Jahreszeit: bevorzugt Wege, die in OpenStreetMap als beleuchtet "
                          "erfasst sind, und meidet Wege mit lit=no. Wege ohne Eintrag gelten als eher dunkel "
                          "(Fußwege stärker als Straßen). Es zählt, was erfasst ist.")


def lit_hint(view):
    """Im Ersatzmodus (ohne eigenes Wegenetz) ehrlich sagen, dass die Beleuchtung nicht geprüft werden kann."""
    if not view.get("lit") or not (view.get("fallback") or view.get("area") is None):
        return view
    payload = payload_from_html(view["html"])
    for o in payload["options"]:
        o["hint"] = (o.get("hint") or "") + " · 💡 im Ersatzmodus nicht prüfbar, bitte selbst auf Beleuchtung achten"
    view["html"] = map_html(payload)
    return view


def san_settings(prefix):
    """Schalter wie bei der Wassernähe. Rückgabe: Reichweite in m (fest 150) oder None."""
    on = st.toggle("🚻 Sanitäroptionen", key=f"san_{prefix}", value=False,
                   help="Zeigt öffentliche Toiletten entlang der Route (Daten: OpenStreetMap) und warnt vor der "
                        "Route, wenn keine in der Nähe ist. Alternativ lässt sich eine Gaststätte einplanen.")
    return SAN_REACHES[1] if on else None


def san_finish(view, reach, ov):
    """Toiletten einarbeiten. Gibt (view, wartet) zurück; fehlen Toiletten, wird die Route erst nach Rückfrage gezeigt."""
    if not reach:
        return view, False
    view["san_reach"] = reach
    try:
        res = add_sanitary(view, reach, log=ov.log)
    except Exception:
        view["san"] = {"n_public": 0, "n_cust": 0, "zones": [], "reach": reach, "failed": True, "missing": 0}
        return view, True
    return view, bool(res.get("failed") or res.get("missing"))


def san_gate(view, reach, ov):
    """Karte öffnen – oder bei fehlenden Toiletten zuerst die Warnung zeigen."""
    view["access"] = _access()
    view["lit"] = _lit()
    try:
        access_hint(view)
        lit_hint(view)
    except Exception:
        pass
    view, wait = san_finish(view, reach, ov)
    if wait:
        st.session_state["san_pending"] = view
    else:
        open_map(view)
    st.rerun()


def render_san_pending():
    back_fx()
    view = st.session_state["san_pending"]
    san = view.get("san", {})
    reach = san.get("reach", 150)
    st.markdown("### 🚻 Sanitäroptionen")
    zones = san.get("zones", [])
    missing = [z for z in zones if not z["ok"]]
    chosen = []
    if san.get("failed"):
        st.warning("Die Toilettendaten konnten gerade nicht geladen werden. Ich kann daher nicht sagen, ob auf der "
                   "Route Toiletten liegen.")
    else:
        names = ", ".join(z["label"] for z in missing)
        st.warning(f"An folgenden Stellen der Route habe ich im Umkreis von {reach} m keine öffentliche Toilette "
                   f"gefunden: {names}. Die Daten stammen aus OpenStreetMap und können unvollständig sein.")
        st.markdown("**Alternative:** Die Route so anpassen, dass sie dort an einer Toilette oder einer Gaststätte "
                    "mit Gästetoilette vorbeiführt (kleiner Umweg).")
        for i, z in enumerate(missing):
            cands = z.get("cands", [])
            if not cands:
                st.caption(f"{z['label']}: nichts in der Nähe gefunden.")
                continue
            labels = [("🚻 " if c["kind"] == "pub" else "🍽️ ") + f"{c['name']} · {_fmt_dist(c['d'])} vom Weg"
                      + (" · öffentlich" if c["kind"] == "pub" else " · private Sanitäranlagen") for c in cands]
            opts = [None] + list(range(len(cands)))
            pick = st.radio(z["label"], opts, index=1, key=f"san_pick_{i}",
                            format_func=lambda j, L=labels: "– nicht einplanen –" if j is None else L[j])
            if pick is not None:
                chosen.append(cands[pick])
        if any(c["kind"] != "pub" for c in chosen) or any(c.get("cands") and c["cands"][0]["kind"] != "pub" for c in missing):
            st.caption("🚻 Private Sanitäranlagen. " + SAN_NOTE)
    c1, c2, c3 = st.columns(3)
    if chosen and c1.button("Route anpassen", type="primary", use_container_width=True, key="san_plan"):
        try:
            with st.spinner("Route mit Toiletten-Stopps wird berechnet …"):
                new = replan_with_stops(view, chosen)
            open_map(new)
            st.rerun()
        except Exception as e:
            st.error(f"Einplanen nicht möglich: {e}")
    if c2.button("Route trotzdem anzeigen", use_container_width=True, key="san_show"):
        open_map(view)
        st.rerun()
    if c3.button("Abbrechen", use_container_width=True, key="san_cancel"):
        st.session_state.pop("san_pending", None)
        soft_back()
        st.rerun()


def replan_with_stops(view, cands):
    """Toiletten bzw. Gaststätten als Zwischenziele einplanen und neu rechnen (Typ je nach Modus)."""
    stops_new = [_pack(c["lat"], c["lon"], "🚻 " + c["name"] + ("" if c.get("kind") == "pub" else " (Gäste-WC)")) for c in cands]
    st.session_state["_water"] = bool(view.get("water"))
    st.session_state["_access"] = bool(view.get("access"))
    st.session_state["_lit"] = bool(view.get("lit"))
    st.session_state["_dog"] = bool(view.get("dog"))
    st.session_state["_temp"] = view.get("temp")
    st.session_state["_windnote"] = view.get("wnote", "")
    reach = view.get("san_reach")
    lvl = view.get("level") if view.get("level") in TOUR_LEVELS else "Sehr grün"
    if view["mode"] == "ab":
        new = make_tour_view_safe(view["start"], stops_new[:SAN_MAX_STOPS], view["dest"], "Sehr grün")
    elif view["mode"] == "loop":
        chain = [view["p1"]] + [(c["lat"], c["lon"]) for c in cands[:SAN_MAX_STOPS]] + [view["p1"]]
        length = sum(dist_m(chain[i], chain[i + 1]) for i in range(len(chain) - 1))
        extra = round(view["km"] - 1.4 * length / 1000.0)
        new = make_tour_view_safe(view["start"], stops_new[:SAN_MAX_STOPS], view.get("end"), lvl,
                                  extra_km=extra if extra >= 1 else 0)
    else:
        room = max(0, SAN_MAX_STOPS - len(view.get("stops_v", [])))
        new = make_tour_view_safe(view["start"], list(view["stops_v"]) + stops_new[:room], view.get("end_v"),
                                  view["level"], extra_km=view.get("extra_km", 0))
    new["san_reach"] = reach
    new["access"] = bool(view.get("access"))
    new["lit"] = bool(view.get("lit"))
    try:
        access_hint(new)
    except Exception:
        pass
    if reach:
        try:
            add_sanitary(new, reach)
        except Exception:
            pass
    return new


def _chip_click(key, opt, multi):
    cur = st.session_state.get(key)
    if multi:
        cur = list(cur or [])
        cur.remove(opt) if opt in cur else cur.append(opt)
        st.session_state[key] = cur
    else:
        st.session_state[key] = opt


def chips(label, options, key, multi=False, default=None, fmt=None, help=None):
    """Auswahl als Tipp-Felder, die auf dem Handy umbrechen (statt Aufklapp-Liste oder winziger Schalter).
    Gewählt = gefüllt. Rückgabe: Liste (multi) oder der gewählte Wert."""
    options = list(options)
    if key not in st.session_state:
        st.session_state[key] = [] if multi else (default if default in options else options[0])
    cur = st.session_state[key]
    if label:
        st.markdown(f'<div class="fld">{escape(label)}</div>', unsafe_allow_html=True)
    with st.container(key=f"chips_{key}"):
        for i, o in enumerate(options):
            on = (o in cur) if multi else (o == cur)
            st.button(fmt(o) if fmt else str(o), key=f"{key}_c{i}", type="primary" if on else "secondary",
                      on_click=_chip_click, args=(key, o, multi))
    if help:
        st.caption(help)
    return list(cur) if multi else cur


def home_green():
    with st.container(key="card_green"):
        st.markdown("Ich suche vom Standort aus die Grünfläche im Umkreis von 10 km, die am besten passt, "
                    "und zeige den schnellsten Weg dorthin.")
        start_v = start_input("Startpunkt", "green_sb")
        dogm = bool(st.session_state.get("dog_mode", False))
        opts_p, opts_a = (GREEN_PREFS_PREFER_DOG, GREEN_PREFS_DOG) if dogm else (GREEN_PREFS_PREFER, GREEN_PREFS)
        for k_, o_ in (("green_prefer", opts_p), ("green_avoid", opts_a)):  # beim Moduswechsel ungültige Auswahl entfernen
            try:
                if st.session_state.get(k_):
                    st.session_state[k_] = [v for v in st.session_state[k_] if v in o_]
            except Exception:
                pass
        prefer = chips("Bevorzugt", opts_p, "green_prefer", multi=True,
                                help=("Hundefreilaufflächen: Wiesen und Plätze, auf denen Hunde ohne Leine laufen dürfen (aus den Kartendaten). "
                                      "Sportangebote: Outdoor-Fitness, Skatepark, Tretboot u. a." if dogm else
                                      "Sportangebote: Fußball, Basketball, Tischtennis, Outdoor-Fitness, Skatepark, "
                                      "Hundepark, Tretboot u. a. (als Suchabfrage im Umkreis)."))
        avoid = chips("Vermeiden", opts_a, "green_avoid", multi=True)
        size = st.select_slider("Bevorzugte Grünflächengröße", options=list(GREEN_SIZES), value="Mittel (2–15 ha)",
                                key="green_size")
        go = st.button("Los, das Grün finden", type="primary", use_container_width=True, key="go_green")
    if not go:
        return
    if not start_v:
        st.warning("Ohne Standort kann ich nicht suchen. Bitte die Freigabe im Browser erlauben.")
        return
    both = set(prefer) & set(avoid)
    if both:
        st.warning("„" + "“, „".join(sorted(both)) + "“ steht bei Bevorzugt und Vermeiden. Bitte nur eins wählen.")
        return
    pos = _unpack(start_v)[:2]
    pc, ac = [GREEN_CAT[x] for x in prefer], [GREEN_CAT[x] for x in avoid]
    view = None
    st.session_state["_lit"] = False
    st.session_state["_dog"] = bool(st.session_state.get("dog_mode", False))
    st.session_state["_temp"] = None
    dog_prefetch(start_v)
    ov = loading_overlay(start_v, "Wassernähe" in prefer)
    with st.status("Grün wird gesucht …", expanded=True) as status:
        try:
            ranked = find_green(pos, pc, ac, GREEN_SIZES[size], log=ov.log)
            ov.log("⚖️ Wünsche abwägen")
            options = propose_options(ranked, pc, ac, GREEN_SIZES[size], size)
            if options and options[0]["cost"] > 0:
                # Wünsche passen nicht alle zusammen: erst zur Auswahl stellen
                st.session_state["green_pending"] = {"start_v": start_v, "options": options, "prefer": pc,
                                                     "avoid": ac, "size": size, "wishes": (prefer, avoid)}
                status.update(label="Vorschläge bereit", state="complete", expanded=False)
                st.rerun()
            ov.log("🚶 Schnellsten Weg berechnen")
            err, t_try = None, time.time()
            for g in (options[:1] + ranked)[:5]:
                try:
                    view = green_view(start_v, g, pc, ac)
                    break
                except Exception as e:
                    err = e
                    if not isinstance(e, ValueError) or time.time() - t_try > 45:
                        break  # Server nicht erreichbar: weitere Ziele probieren würde nur länger warten
            if not view:
                raise err or ValueError("Keine erreichbare Grünfläche gefunden.")
            status.update(label="Fertig", state="complete", expanded=False)
        except NoGreen as e:
            status.update(label="Nichts gefunden", state="complete", expanded=False)
            ov.empty()
            st.warning(str(e))
            return
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        open_map(view)
        st.rerun()
    ov.empty()


def render_green_pending():
    """Wünsche passen nicht alle zusammen: freundlich erklären und 2–3 Vorschläge zur Auswahl stellen."""
    back_fx()
    gp = st.session_state["green_pending"]
    prefer_n, avoid_n = gp["wishes"]
    st.markdown("### Das passt für dich")
    wishes = [f"Größe: {gp['size'].split(' (')[0].lower()}"] + [f"{GREEN_ICON[GREEN_CAT[x]]} {x}" for x in prefer_n] \
        + [f"kein {GREEN_ICON[GREEN_CAT[x]]} {x}" for x in avoid_n]
    st.markdown("**Deine Wünsche:** " + " · ".join(wishes))
    opts = gp["options"]
    st.caption("Alles zusammen gibt es hier leider nicht. Ich habe abgewogen und zeige dir die besten Möglichkeiten. "
               "Such dir die aus, die sich für dich richtig anfühlt.")
    if "access" in gp["prefer"]:
        if all("paccess" not in o["keys"] for o in opts):
            st.success("♿ Barrierefreiheit hat bei der Auswahl Vorrang. Alle Vorschläge erfüllen sie.")
        else:
            st.info("♿ Für Barrierefreiheit gibt es in OpenStreetMap oft keine Angaben. Ich konnte keine Fläche "
                    "sicher bestätigen. Die Vorschläge zeigen, was sonst am besten passt.")
    dmin = min(o["d"] for o in opts)
    for i, g in enumerate(opts):
        with st.container(border=True):
            tag = "Beste Gesamtpassung" if i == 0 else f"Alternative {i}"
            st.markdown(f"**{tag}: {g['name']}** · {fmt_ha(g['ha'])} · ca. {_fmt_dist(g['d'])} entfernt")
            if len(opts) > 1 and g["d"] <= dmin and any(o["d"] > dmin * 1.05 for o in opts):
                st.markdown("✓ Kürzeste Entfernung")
            for t in g["ok"]:
                st.markdown(f"✓ {t}")
            for t in g["dev"]:
                st.markdown(f"○ {t}")
            if st.button("Hierhin gehen", key=f"gp_pick_{i}", type="primary" if i == 0 else "secondary",
                         use_container_width=True):
                try:
                    with st.spinner("Weg wird berechnet …"):
                        view = green_view(gp["start_v"], g, gp["prefer"], gp["avoid"])
                    open_map(view)
                    st.rerun()
                except Exception as e:
                    show_error(e)
    if st.button("← Wünsche ändern", key="gp_back", use_container_width=True):
        st.session_state.pop("green_pending", None)
        soft_back()
        st.rerun()


# ---------- Favoriten und zuletzt genutzte Ziele (Schnellstart)
# Gespeichert im Browser (localStorage) des Geräts: bleibt nach dem Neuladen erhalten, ohne Konto und ohne Server.
FAV_MAX, REC_MAX = 8, 4
FAV_LS_KEY = "gw_favs_v1"
_FAV_JS_GET = "(function(){try{return localStorage.getItem('" + FAV_LS_KEY + "')||''}catch(e){return ''}})()"


def _fav_key(v):
    try:
        lat, lon, _ = _unpack(v)
        return (round(lat, 4), round(lon, 4))
    except Exception:
        return None


def _fav_state():
    return st.session_state.setdefault("_favs", {"fav": [], "rec": []})


def _fav_valid(v):
    return isinstance(v, str) and _fav_key(v) is not None


def _fav_merge(raw):
    """Gespeicherte Liste aus dem Browser mit dem Stand dieser Sitzung zusammenführen."""
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    stt = _fav_state()
    for name, mx in (("fav", FAV_MAX), ("rec", REC_MAX)):
        cur = list(stt[name])
        seen = {_fav_key(v) for v in cur}
        for v in (data.get(name) or []) if isinstance(data, dict) else []:
            if _fav_valid(v) and _fav_key(v) not in seen:
                cur.append(v)
                seen.add(_fav_key(v))
        stt[name] = cur[:mx]
    fk = {_fav_key(v) for v in stt["fav"]}
    stt["rec"] = [v for v in stt["rec"] if _fav_key(v) not in fk]


def _fav_add(v):
    stt = _fav_state()
    k = _fav_key(v)
    if k is None or any(_fav_key(x) == k for x in stt["fav"]):
        return
    if len(stt["fav"]) >= FAV_MAX:
        st.toast(f"Maximal {FAV_MAX} Favoriten. Bitte erst einen entfernen.")
        return
    stt["fav"].append(v)
    stt["rec"] = [x for x in stt["rec"] if _fav_key(x) != k]
    st.toast("⭐ Als Favorit gemerkt")


def _fav_del(i):
    stt = _fav_state()
    if 0 <= i < len(stt["fav"]):
        stt["fav"].pop(i)


def _rec_add(v):
    """Ziel einer gefundenen Route merken (neueste zuerst; Favoriten stehen schon oben)."""
    stt = _fav_state()
    k = _fav_key(v)
    if k is None or any(_fav_key(x) == k for x in stt["fav"]):
        return
    stt["rec"] = ([v] + [x for x in stt["rec"] if _fav_key(x) != k])[:REC_MAX]


def _fav_go(v):
    """Kachel getippt: Reiter „Route“ mit diesem Ziel öffnen und sofort losrechnen (sobald der Standort da ist)."""
    st.session_state["page_sel"] = "route"
    st.session_state["_keep_ziel_sb"] = v
    st.session_state["_quick_dest"] = v


def fav_sync():
    """Favoriten einmal aus dem Browser lesen und Änderungen zurückschreiben (unsichtbare Hilfskomponente)."""
    ss = st.session_state
    if streamlit_js_eval is None:
        ss["_fav_loaded"] = True
        return
    try:
        with st.container(key="favsync"):
            if not ss.get("_fav_loaded"):
                raw = streamlit_js_eval(js_expressions=_FAV_JS_GET, key="fav_read")
                if isinstance(raw, str):
                    _fav_merge(raw)
                    ss["_fav_loaded"] = True
                return
            payload = json.dumps(_fav_state(), ensure_ascii=False, separators=(",", ":"))
            js = ("(function(){try{localStorage.setItem('" + FAV_LS_KEY + "'," + json.dumps(payload, ensure_ascii=False)
                  + ");return 1}catch(e){return 0}})()")
            k = "fav_w" + hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
            try:
                streamlit_js_eval(js_expressions=js, want_output=False, key=k)
            except TypeError:
                streamlit_js_eval(js_expressions=js, key=k)
    except Exception:
        ss["_fav_loaded"] = True  # Speichern im Browser nicht möglich: Favoriten gelten für diese Sitzung


def home_favs():
    """Schnellstart-Kacheln: Favoriten (⭐) und zuletzt genutzte Ziele (🕘) auf einen Tipp."""
    stt = _fav_state()
    favs, recs = stt["fav"], stt["rec"]
    if not favs and not recs:
        return
    st.markdown('<div class="fld">⭐ Schnellstart · Route von hier</div>', unsafe_allow_html=True)
    with st.container(key="chips_favs"):
        for i, v in enumerate(favs):
            st.button("⭐ " + _short(_unpack(v)[2])[:24], key=f"favgo_c{i}", type="secondary",
                      on_click=_fav_go, args=(v,))
        for i, v in enumerate(recs):
            st.button("🕘 " + _short(_unpack(v)[2])[:24], key=f"recgo_c{i}", type="secondary",
                      on_click=_fav_go, args=(v,))
    if favs:
        with st.expander("Favoriten verwalten"):
            for i, v in enumerate(favs):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"⭐ {escape(_short(_unpack(v)[2])[:40])}")
                c2.button("✕", key=f"favdel_{i}", on_click=_fav_del, args=(i,), help="Favorit entfernen")


def home_route():
    with st.container(key="card_route"):
        start_v = start_input("Start", "start_sb")
        ziel_v = place_input("Ziel", "ziel_sb")
        st.caption(SEARCH_TIP)
        if ziel_v and _fav_key(ziel_v) not in {_fav_key(x) for x in _fav_state()["fav"]}:
            st.button("⭐ Ziel als Favorit merken", key="fav_add_btn", on_click=_fav_add, args=(ziel_v,))
        water_on = st.toggle("💧 Wassernähe bevorzugen", key="water_route", value=False,
                             help="Führt die Route bevorzugt an Seen, Teichen und Flüssen entlang – für zusätzliche Entspannung.")
        san_reach = san_settings("route")
        acc_on = access_toggle("route")
        lit_on = lit_toggle("route")
        go = st.button("Route finden", type="primary", use_container_width=True, key="go_ab")
    quick = st.session_state.get("_quick_dest")
    if quick:
        if ziel_v is None or ziel_v == quick:
            ziel_v = quick
            if start_v:  # Schnellstart: sofort losrechnen (wartet sonst, bis der Standort da ist)
                go = True
                st.session_state.pop("_quick_dest", None)
        else:  # Ziel wurde von Hand geändert
            st.session_state.pop("_quick_dest", None)
    if start_v and ziel_v:
        prewarm_ab(start_v, ziel_v)  # Korridor schon laden, während noch Optionen eingestellt werden
    if not go:
        return
    if not (start_v and ziel_v):
        st.warning("Bitte Start und Ziel jeweils aus den Vorschlägen auswählen.")
        return
    view = None
    st.session_state["_water"] = bool(water_on)
    st.session_state["_access"] = bool(acc_on)
    st.session_state["_lit"] = bool(lit_on)
    st.session_state["_dog"] = bool(st.session_state.get("dog_mode", False))
    st.session_state["_temp"] = None
    dog_prefetch(start_v)
    ov = loading_overlay(start_v, water_on)
    try:
        ov.set_preview([tuple(_unpack(start_v)[:2]), tuple(_unpack(ziel_v)[:2])])
    except Exception:
        pass
    with st.status("Route wird vorbereitet …", expanded=True) as status:
        try:
            view = make_ab_view_safe(start_v, ziel_v, log=ov.log)
            _rec_add(ziel_v)
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        san_gate(view, san_reach, ov)
    ov.empty()


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
        water_on = st.toggle("💧 Wassernähe bevorzugen", key="water_loop", value=False,
                             help="Führt die Route bevorzugt an Seen, Teichen und Flüssen entlang – für zusätzliche Entspannung.")
        san_reach = san_settings("loop")
        acc_on = access_toggle("loop")
        lit_on = lit_toggle("loop")
        go = st.button("Ründchen finden", type="primary", use_container_width=True, key="go_loop")
    if not go:
        return
    if not a_v or (not same and not b_v):
        st.warning("Bitte Start" + ("" if same else " und Ende") + " aus den Vorschlägen auswählen.")
        return
    view, holder = None, {}
    st.session_state["_water"] = bool(water_on)
    st.session_state["_access"] = bool(acc_on)
    st.session_state["_lit"] = bool(lit_on)
    st.session_state["_dog"] = bool(st.session_state.get("dog_mode", False))
    st.session_state["_temp"] = None
    dog_prefetch(a_v)

    def prog(f):
        txt = "🔄 Runden in verschiedene Richtungen probieren …"
        if "bar" not in holder:
            holder["bar"] = st.progress(0.0, text=txt)
        holder["bar"].progress(min(f, 1.0), text=txt)

    ov = loading_overlay(a_v, water_on)
    with st.status("Ründchen wird gesucht …", expanded=True) as status:
        try:
            view = make_loop_view_safe(a_v, None if same else b_v, km, level, log=ov.log, progress=prog)
            if "bar" in holder:
                holder["bar"].empty()
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        san_gate(view, san_reach, ov)
    ov.empty()


def _more_stops(d):
    st.session_state["n_stops"] = min(max(st.session_state.get("n_stops", 1) + d, 1), TOUR_MAX_STOPS)


def home_tour():
    with st.container(key="card_tour"):
        st.markdown("Grüne Runde mit Stopp unterwegs: Kiosk, Bäcker, Paket abgeben oder jemanden besuchen. "
                    "Die Reihenfolge wähle ich so, dass der Weg kurz bleibt.")
        a_v = start_input("Startpunkt", "tour_a_sb")
        n = st.session_state.setdefault("n_stops", 1)
        stops = [place_input(f"Zwischenziel {i + 1}", f"tour_s{i}_sb") for i in range(n)]
        c1, c2 = st.columns(2)
        if n < TOUR_MAX_STOPS:
            c1.button("＋ Weiteres Zwischenziel", key="tour_more", on_click=_more_stops, args=(1,),
                      use_container_width=True)
        if n > 1:
            c2.button("− Letztes entfernen", key="tour_less", on_click=_more_stops, args=(-1,),
                      use_container_width=True)
        same = st.checkbox("Runde endet wieder am Startpunkt", value=True, key="tour_same")
        b_v = None if same else place_input("Endpunkt", "tour_b_sb")
        st.caption(SEARCH_TIP)
        extra_km = st.slider("Extra Strecke (km, optional)", 0, 10, 0, 1, key="tour_extra_km",
                             help="Verlängert die Runde um diese Strecke, zusätzlich zum Weg über die Zwischenziele. 0 = keine.")
        level = st.select_slider("Wie grün soll es sein?", options=TOUR_LEVELS, value="Sehr grün", key="level_tour")
        water_on = st.toggle("💧 Wassernähe bevorzugen", key="water_tour", value=False,
                             help="Führt die Route bevorzugt an Seen, Teichen und Flüssen entlang – für zusätzliche Entspannung.")
        san_reach = san_settings("tour")
        acc_on = access_toggle("tour")
        lit_on = lit_toggle("tour")
        go = st.button("Ründchen mit Zwischenziel finden", type="primary", use_container_width=True, key="go_tour")
    if a_v and all(stops) and (same or b_v):
        prewarm_tour(a_v, stops, None if same else b_v, extra_km)  # Gebiet schon laden, während noch eingestellt wird
    if not go:
        return
    if not a_v or not all(stops) or (not same and not b_v):
        st.warning("Bitte Start, alle Zwischenziele" + ("" if same else " und Endpunkt") + " aus den Vorschlägen auswählen.")
        return
    view = None
    st.session_state["_water"] = bool(water_on)
    st.session_state["_access"] = bool(acc_on)
    st.session_state["_lit"] = bool(lit_on)
    st.session_state["_dog"] = bool(st.session_state.get("dog_mode", False))
    st.session_state["_temp"] = None
    dog_prefetch(a_v)
    ov = loading_overlay(a_v, water_on)
    try:
        ov.set_preview(tour_points(a_v, stops, None if same else b_v)[0], "Direkter Weg über die Zwischenziele")
    except Exception:
        pass
    with st.status("Runde mit Zwischenzielen wird berechnet …", expanded=True) as status:
        try:
            view = make_tour_view_safe(a_v, stops, None if same else b_v, level, log=ov.log, extra_km=extra_km)
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        san_gate(view, san_reach, ov)
    ov.empty()


def dog_home_info():
    """Hundemodus auf der Startseite: Wetter am Standort, Pfotenwarnung bei Hitze, Hinweis zu den Regeln."""
    lat = lon = None
    try:
        loc = st.session_state.get("loc_shared")
        b = st.session_state.get("bias")
        if loc:
            lat, lon = loc[0], loc[1]
        elif b:
            lat, lon = b[0], b[1]
    except Exception:
        pass
    w = get_weather(lat, lon, wait=1.5) if lat is not None else None
    parts = []
    if w:
        icon = {"sun": "☀️", "rain": "🌧️", "snow": "❄️", "cloud": "☁️"}[wx_kind(w["code"], w["temp"], w["day"])]
        parts.append(f"{icon} {w['temp']:.0f} °C am Standort")
        if w.get("wind") is not None:
            parts.append(f"💨 Wind {w['wind']:.0f} km/h" + (f" (Böen {w['gust']:.0f})" if w.get("gust", 0) >= w["wind"] + 8 else ""))
    parts.append("🚫 Friedhöfe, Spielplätze und Verbotszonen werden gemieden")
    parts.append("🐾 Belag wird angezeigt")
    st.markdown('<div class="dogchip"><span><img src="' + chay_icon_uri(22) + '" width="22" height="22" alt="" style="vertical-align:-6px;margin-right:6px">'
                f'{DOG_NAME} ist dabei</span>' + "".join(f"<span>{escape(p)}</span>" for p in parts) + "</div>", unsafe_allow_html=True)
    if w and w["temp"] >= HOT_C:
        st.markdown(f'<div class="dogwarn">🐾 Pfotenwarnung: Bei {w["temp"]:.0f} °C heizen Asphalt und Pflaster stark auf '
                    "und können die Pfoten verbrennen. Die Routen meiden harten Boden, wo es geht, und markieren ihn rot gepunktet. "
                    "Handrücken-Test: Hältst du ihn keine 5 Sekunden auf den Boden, ist es zu heiß. Besser früh morgens oder abends gehen.</div>",
                    unsafe_allow_html=True)
    wn = wind_note(w)
    if wn:
        st.markdown(f'<div class="dogwarn" style="background:#E3F0F8;border-color:#9CC4DC;color:#1E4A66">{escape(wn)}</div>', unsafe_allow_html=True)
    st.caption("Nicht jedes Verbot steht in den Kartendaten. Schilder vor Ort, Leinenpflicht sowie Brut- und Setzzeit gehen vor.")
    st.markdown('<div class="chayfact"><img src="' + chay_icon_uri(40) + '" width="40" height="40" alt="' + DOG_NAME + '">'
                f'<div><b>{DOG_NAME}</b>, auch Chayton Wakanda, war ein Black-Tri Australian Shepherd: schön, frech und clever. '
                'Er wird bis heute geliebt.</div></div>', unsafe_allow_html=True)


def wind_home_info():
    """Normaler Modus: Windhinweis auf der Startseite, wenn es am Standort windig ist."""
    try:
        loc, b = st.session_state.get("loc_shared"), st.session_state.get("bias")
        pt = (loc[0], loc[1]) if loc else (b[0], b[1]) if b else None
        w = get_weather(pt[0], pt[1], wait=0.8) if pt else None
        wn = wind_note(w)
        if wn:
            st.markdown(f'<div class="dogwarn" style="background:#E3F0F8;border-color:#9CC4DC;color:#1E4A66">{escape(wn)}</div>',
                        unsafe_allow_html=True)
    except Exception:
        pass


PRIVACY_TEXT = """
**Kurz:** GreenWalker braucht keinen Account, zeigt keine Werbung und enthält keine eigene Nutzungsanalyse. Es ist ein privates
Hobbyprojekt und keine Rechtsberatung. Der Betrieb läuft auf Streamlit Community Cloud, die technische Zugriffsdaten
(z. B. IP-Adresse) protokollieren kann.

**Standort**
- Dein Standort wird nur nach deiner Freigabe im Browser abgefragt und für die Routenberechnung an die App gesendet.
  Die App speichert ihn nicht dauerhaft. Zwischengespeichert werden nur Kartendaten für gerundete Gebiete.
- 📍 *Standort teilen* in der Karte erzeugt nur dann einen Link mit deiner Position, wenn du den Knopf tippst. Der Link geht
  an die Person oder App, die du im Teilen-Menü wählst. Es ist eine einmalige Momentaufnahme, kein Live-Tracking.

**Anfragen an Kartendienste (nötig für Karten, Suche und Routen)**
- Vom Server der App aus: Overpass-Server (Wege und Grünflächen eines Gebiets), Photon und Nominatim (Ortssuche, mit
  ungefährer Umgebung), FOSSGIS-Routing (Start, Ziel und Zwischenpunkte), OpenFreeMap (Kartenkacheln für den Grün-Scan),
  Open-Meteo (Wetter am Startpunkt) und Wikipedia/Wikimedia (Stadtbild und Namen im Ladebildschirm).
- Direkt von deinem Gerät aus: Kartenkacheln (Esri, OpenTopoMap, OpenStreetMap, EOX, Geobasis NRW) und die Kartenbibliothek
  Leaflet (cdnjs). Diese Dienste sehen deine IP-Adresse und den angezeigten Kartenausschnitt.
- 🚰 *Trinkwasser* in der Karte: Beim Antippen fragt dein Gerät direkt bei einem Overpass-Server nach Trinkbrunnen im Bereich
  der Route (Rechteck um die Route). Das Ergebnis wird 24 Stunden im Browser gemerkt.
- Nur wenn der Betreiber einen KI-Schlüssel hinterlegt hat: ein unbekannter Suchbegriff (ohne Standort) kann an einen
  KI-Dienst gehen, damit passende Orte gefunden werden.

**Auf deinem Gerät gespeichert (Browser, localStorage)**
- Favoriten und zuletzt genutzte Ziele. Sie liegen nur in diesem Browser. Entfernen: „Favoriten verwalten“ oder Browserdaten löschen.
- Eine Merkhilfe, welcher Dankestext zuletzt gezeigt wurde.

**Dateien, die du selbst erzeugst**
- GPX und Offline-Blatt entstehen im Browser und gehen nirgendwohin, solange du sie nicht selbst teilst.

Kartendaten © OpenStreetMap-Mitwirkende (ODbL).
"""


def render_home():
    restore_widgets()
    snap_widgets()
    back_fx()
    fav_sync()
    dog = bool(st.session_state.get("dog_mode", False))
    st.markdown(DOG_BASE_CSS + (DOG_CSS if dog else ""), unsafe_allow_html=True)
    logo = base64.b64encode(LOGO_SVG.encode()).decode()
    if dog:
        st.markdown('<div class="hero"><div class="ico"><img src="' + chay_icon_uri(52) + '" width="52" height="52" alt="' + DOG_NAME + '"></div>'
                    '<div><h1>GreenWalker · Hundemodus</h1>'
                    f'<p>Gassi-Runden im Grünen mit {DOG_NAME}: ohne Verbotszonen und mit Pfotenschutz.</p></div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="hero"><div class="ico"><img src="data:image/svg+xml;base64,' + logo +
                    '" width="46" height="46" alt=""></div><div><h1>GreenWalker</h1>'
                    '<p>Der grüne Weg statt nur der kürzeste.</p></div></div>',
                    unsafe_allow_html=True)
    st.toggle("🐕 Hundemodus", key="dog_mode", value=False,
              help="Für Spaziergänge mit Hund: meidet Friedhöfe, Spielplätze und Hunde-Verbotszonen, zeigt den Belag "
                   "(Asphalt, Pflaster, Erde) und warnt bei Hitze über 30 °C vor heißem Boden für die Pfoten.")
    if dog:
        dog_home_info()
    else:
        wind_home_info()
    # Nur der gewählte Reiter wird aufgebaut (st.tabs würde alle drei samt Suchfeldern laden = langsam)
    pages = {"route": "🧭 Route", "green": "🌳 Ins Grün", "loop": "🔄 Ründchen", "tour": "🛍️ Ründchen mit Zwischenziel"}
    home_favs()
    page = chips("", list(pages), "page_sel", default="route", fmt=pages.get)
    page = page or st.session_state.get("page_last", "route")
    st.session_state["page_last"] = page
    if page != "route":
        st.session_state.pop("_quick_dest", None)
    if st.session_state.get("_home_drawn") != page:  # Startseite oder Reiter wird neu aufgebaut: frische Suchfelder
        bump_sb_gen()
    st.session_state["_home_drawn"] = page
    {"route": home_route, "green": home_green, "loop": home_loop, "tour": home_tour}[page]()
    with st.expander("Datenquellen und Gewichtung"):
        st.markdown(sources_text(150, LEVELS["Ausgewogen"]))
    with st.expander("🔒 Datenschutz"):
        st.markdown(PRIVACY_TEXT)
    laps_ = st.session_state.get("_laps")
    if laps_ and len(laps_) > 1:
        with st.expander("⏱️ Zeiten der letzten Berechnung"):
            rows = []
            for (t0_, m0), (t1_, _m1) in zip(laps_[:-1], laps_[1:]):
                if m0 != "Start":
                    rows.append(f"- {re.sub(chr(10), ' ', m0)[:90]}: **{t1_ - t0_:.1f} s**")
            last_t, last_m = laps_[-1]
            end_t = st.session_state.get("_laps_end") or last_t
            rows.append(f"- {last_m[:90]}: **{max(end_t - last_t, 0):.1f} s**")
            rows.append(f"\n**Gesamt: {end_t - laps_[0][0]:.1f} s**")
            st.markdown("\n".join(rows))


if st.session_state.get("gw"):
    st.session_state.pop("_home_drawn", None)
    render_map_view()
elif st.session_state.get("san_pending"):
    st.session_state.pop("_home_drawn", None)
    render_san_pending()
elif st.session_state.get("green_pending"):
    st.session_state.pop("_home_drawn", None)
    render_green_pending()
else:
    render_home()
