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
def rank_mirrors():
    """Fragt den Status aller Overpass-Server kurz und parallel ab: Server mit freien Plätzen zuerst,
    nicht erreichbare Server werden übersprungen. Spart das lange Warten auf überlastete Server."""
    def probe(base):
        try:
            r = requests.get(base + "/status", headers=UA, timeout=4)
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
MAX_SPAN_M = 9000
M_PER_MIN = 80

LEVELS = {"Schnellste": 0.0, "Leicht grün": 1.0, "Ausgewogen": 3.0,
          "Sehr grün": 6.0, "Maximal grün": 12.0, "Grünster Umweg": 40.0}
# Hauptstraßen-Gewicht und Breiten-Gewicht je Stufe. „Maximal grün“ und „Grünster Umweg“ meiden
# Hauptstraßen und breite Straßen stark: bevorzugt kleine Gassen und Nebenstraßen.
LEVEL_QUIET = {"Maximal grün": (4.0, 6.0), "Grünster Umweg": (6.0, 10.0)}


def level_quiet(name):
    lam = LEVELS[name]
    return LEVEL_QUIET.get(name, (1.5 if lam > 0 else 0.0, 0.0))


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


@st.cache_data(ttl=3600, show_spinner=False)
def _photon(q, blat, blon, tag=None):
    params = {"q": q, "limit": 12, "lang": "de"}
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
    "bier": ["kneipe", "biergarten", "getraenke"], "kneipe": ["kneipe"], "pub": ["kneipe"], "bar": ["kneipe"],
    "biergarten": ["biergarten"], "feierabendbier": ["kneipe", "biergarten", "kiosk"], "cocktail": ["kneipe"],
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
KW_MAX_CATS = 2
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


@st.cache_data(ttl=600, show_spinner=False)
def _nearby(selectors, lat, lon, radius=KW_RADIUS_M):
    """Orte einer Kategorie rund um (lat, lon); Overpass mit kurzem Limit."""
    body = "".join(f"nwr{s}(around:{radius},{lat},{lon});" for s in selectors)
    q = f"[out:json][timeout:8];({body});out center 40;"
    last = None
    for base in OVERPASS_MIRRORS[:3]:
        try:
            r = requests.post(base + "/interpreter", data={"data": q}, headers=UA, timeout=7)
            r.raise_for_status()
            els = r.json().get("elements", [])
            res = []
            for e in els:
                c = e.get("center") or e
                if "lat" not in c:
                    continue
                t = e.get("tags", {})
                res.append((float(c["lat"]), float(c["lon"]), t.get("name") or t.get("brand") or t.get("operator") or "",
                            t.get("addr:street", ""), t.get("addr:housenumber", "")))
            return res
        except Exception as ex:
            last = ex
    raise last or RuntimeError("Overpass nicht erreichbar")


def keyword_places(term, bias):
    """Schlagwort -> nahe Orte als Vorschläge (nach Entfernung)."""
    if not bias:
        return []
    defs = _cat_defs(term)
    out = []
    for icon, label, sels in defs:
        try:
            items = _nearby(tuple(sels), round(bias[0], 3), round(bias[1], 3))
        except Exception:
            continue
        items = sorted((it for it in items if it[2]), key=lambda it: dist_m((bias[0], bias[1]), (it[0], it[1])))[:4]
        for la, lo, name, street, nr in items:
            title = name or label
            addr = (street + (" " + nr if nr else "")) if street else ""
            lab = f"{icon} {title} · {label}" + (f" · {addr}" if addr and name else "")
            out.append((lab, _pack(la, lo, f"{icon} {title}")))
    return out


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
    kwhits = []
    try:
        kwhits = keyword_places(term, bias)
    except Exception:
        kwhits = []
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
    if bias and out:
        # nahe Treffer zuerst: aufsteigend nach Entfernung vom Standort bzw. Startpunkt
        def dist_of(item):
            la, lo, _ = _unpack(item[1])
            return dist_m((bias[0], bias[1]), (la, lo))
        out.sort(key=dist_of)
        out = [(f"{lab}  ·  {_fmt_dist(dist_of((lab, val)))}", val) for lab, val in out]
    if kwhits:
        def dist_k(item):
            la, lo, _ = _unpack(item[1])
            return dist_m((bias[0], bias[1]), (la, lo))
        kwhits.sort(key=dist_k)
        kwhits = [(f"{lab}  ·  {_fmt_dist(dist_k((lab, val)))}", val) for lab, val in kwhits][:6]
        rest = [x for x in out if x[1] not in {v for _, v in kwhits}]
        return (kwhits + rest)[:10]
    return out[:8]


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
WALK_FILTER = ('["highway"]["area"!~"yes"]["access"!~"private"]'
               f'["highway"!~"{WALK_EXCLUDE}"]["foot"!~"no"]["service"!~"private"]'
               '["sidewalk"!~"separate"]["sidewalk:both"!~"separate"]'
               '["sidewalk:left"!~"separate"]["sidewalk:right"!~"separate"]')
_WALK_EXCLUDE_RE = re.compile(WALK_EXCLUDE)


def overpass_query(bbox):
    """Eine einzige Abfrage für Fußwege und Grünflächen (statt zwei getrennter)."""
    w, s, e, n = bbox
    b = f"({s},{w},{n},{e})"
    parts = [f"way{WALK_FILTER}{b};"]
    for k, vals in GREEN_TAGS.items():
        rx = "|".join(vals)
        parts += [f'way["{k}"~"^({rx})$"]{b};', f'relation["{k}"~"^({rx})$"]{b};']
    # Gewässer: Seen/Teiche (natural=water) und Flüsse/Bäche/Kanäle
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


def fetch_osm(bbox, deadline):
    """Lädt OSM-Daten mit harter Zeitgrenze. Kein endloses Warten: überlastete Server (429/50x)
    werden sofort übersprungen. Ergebnis wird einige Tage auf der Festplatte zwischengespeichert."""
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    path = os.path.join(OSM_CACHE_DIR, hashlib.md5((repr(bbox) + "|v2-wasser").encode()).hexdigest() + ".osm")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < OSM_CACHE_DAYS * 86400:
        return path
    query, last = overpass_query(bbox), "keine Antwort"
    tmp = path + f".{os.getpid()}.part"
    for rnd in range(2):
        for base in rank_mirrors():
            left = deadline - time.time()
            if left < 4:
                break
            try:
                with requests.post(base + "/interpreter", data={"data": query}, headers=UA,
                                   timeout=(4, min(25, left)), stream=True) as r:
                    if r.status_code != 200:
                        last = f"{base.split('/')[2]}: {r.status_code} {r.reason}"
                        continue
                    # Wächter: kappt die Verbindung an der Zeitgrenze, auch wenn Daten nur tröpfeln
                    guard = threading.Timer(max(0.5, deadline - time.time()), _cut, args=(r,))
                    guard.daemon = True
                    guard.start()
                    try:
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_content(1 << 14):
                                f.write(chunk)
                    finally:
                        guard.cancel()
                    if time.time() > deadline:
                        raise TimeoutError("Download dauert zu lange")
                if not _osm_complete(tmp):
                    last = f"{base.split('/')[2]}: Abfrage abgebrochen"
                    continue
                os.replace(tmp, path)
                return path
            except Exception as e:
                last = f"{base.split('/')[2]}: {type(e).__name__}"
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        if deadline - time.time() < 6:
            break
        time.sleep(2)
    raise RuntimeError(f"Alle Kartenserver sind gerade überlastet ({last}).")


def _walkable(el):
    tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
    hw = tags.get("highway")
    if not hw or tags.get("area") == "yes" or _WALK_EXCLUDE_RE.search(hw):
        return False
    return (tags.get("foot") != "no" and tags.get("access") != "private"
            and tags.get("service") != "private")


def walk_xml(path):
    """Schreibt eine zweite Datei nur mit Knoten und Fußwegen (osmnx macht aus jedem Weg eine Kante,
    Parkränder dürfen also nicht mit hinein). Arbeitet gestreamt, damit der Speicher klein bleibt."""
    out = path[:-4] + ".walk.osm"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    tmp = out + f".{os.getpid()}.part"
    depth = 0
    with open(tmp, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="green-walker">\n')
        for ev, el in ET.iterparse(path, events=("start", "end")):
            if ev == "start":
                depth += 1
                continue
            depth -= 1
            if depth != 1:
                continue
            if el.tag == "node" or (el.tag == "way" and _walkable(el)):
                el.tail = "\n"
                f.write(ET.tostring(el, encoding="unicode").encode("utf-8"))
            el.clear()
        f.write(b"</osm>\n")
    os.replace(tmp, out)
    return out


@st.cache_resource(show_spinner=False, max_entries=2, ttl=3600)
def load_area(bbox):
    path = fetch_osm(bbox, time.time() + LOAD_DEADLINE_S)
    try:
        G0 = ox.graph_from_xml(walk_xml(path), bidirectional=True, simplify=True, retain_all=False)
    except Exception as e:
        raise ValueError(f"In diesem Gebiet habe ich keine Fußwege gefunden ({type(e).__name__}).")
    if G0.number_of_edges() == 0:
        raise ValueError("In diesem Gebiet habe ich keine Fußwege gefunden.")

    parks, waters = None, None
    try:
        tags_all = {"leisure": GREEN_TAGS["leisure"], "landuse": GREEN_TAGS["landuse"] + ["reservoir", "basin"],
                    "natural": GREEN_TAGS["natural"] + ["water"], "waterway": WATER_WAYS}
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

    G = ox.project_graph(G0)
    crs = G.graph["crs"]
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    best = {}
    for u, v, d in G.edges(data=True):
        if (u, v) not in best or d["length"] < best[(u, v)][0]:
            hw = d.get("highway")
            hw = (hw[0] if hw else "") if isinstance(hw, (list, tuple)) else (hw or "")
            best[(u, v)] = (d["length"], d.get("geometry") or LineString([pos[u], pos[v]]), *edge_label(d), hw,
                            edge_wide(d, hw))
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

    # Hauptstraßen-Nähe je Kante (0 = ruhig, 1 = an/auf einer Hauptstraße)
    noise = np.zeros(len(keys), dtype=np.float32)
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

    wide = np.array([best[k][5] for k in keys], dtype=np.float32)

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

    # Gewässer-Nähe je Kante (Abstand zum nächsten See, Teich oder Fluss)
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

    node_list = list(pos)
    nidx = {n: i for i, n in enumerate(node_list)}
    return {"bbox": bbox, "crs": crs, "pos": pos, "H": H, "lengths": lengths, "lines": lines, "D": D, "DW": DW,
            "water_json": water_json,
            "node_list": node_list, "xy": np.array([pos[n] for n in node_list]),
            "eu": np.array([nidx[u] for u, v in keys]), "ev": np.array([nidx[v] for u, v in keys]),
            "names": names, "kinds": kinds, "noise": noise, "wide": wide,
            "parks_json": parks_json, "n_parks": n_parks, "n_edges": len(keys)}


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


def edge_water(area):
    """Wasser-Score je Kante: 1 direkt am See/Fluss, linear fallend auf 0 bei WATER_BUF Metern."""
    if "_w" not in area:
        dw = area.get("DW")
        area["_w"] = np.clip(1 - dw / WATER_BUF, 0, 1).mean(axis=0) if dw is not None else None
    return area["_w"]


def edge_green(area, buf, water=None):
    """Attraktivität je Kante: Grünscore; bei aktiver Wassernähe zählt Wasser genauso (das Bessere gilt)."""
    g = np.clip(1 - area["D"] / buf, 0, 1).mean(axis=0)
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


def ensure_water(area, log=None):
    """Zweite, unabhängige Gewässerquelle: Seen und Flüsse aus den Karten-Vektorkacheln. Ergänzt die OSM-Daten,
    falls dort ein großer See (z. B. als Multipolygon) fehlt oder unvollständig ankam. Nur bei aktiver Wassernähe."""
    if not _water() or area.get("_wt") or not area.get("bbox"):
        return
    area["_wt"] = True
    try:
        if log:
            log("💧 Gewässer prüfen und ergänzen")
        sa = scan_area(area["bbox"], True)
        feats = (sa.get("wgeo") or {}).get("features", [])
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
LOOP_MAX_KM = 10.0


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
    mu = (QUIET if lam > 0 else 0.0) if mu is None else mu
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
            c = L[i] * (1 + lam * (1 - g[i]) + mu * nz[i] + wq * wd[i])
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
    nz = area.get("noise")
    noise = float((l * nz[idx]).sum() / dist) if nz is not None else 0.0
    return dist, float((l * g[idx]).sum() / dist), float(rep / dist), noise


def find_loops(area, o, t, target, lam, buf, progress=None, quiet=None):
    """Probiert mehrere Richtungen, passt den Kreisradius an die Wunschlänge an und
    liefert bis zu 3 deutlich verschiedene Varianten (beste zuerst)."""
    g = edge_green(area, buf)
    ng = node_green(area, g)
    cands = []
    for k in range(LOOP_BEARINGS):
        theta = 2 * math.pi * k / LOOP_BEARINGS
        rho = target / (2 * math.pi * 1.35)
        for _ in range(3):
            try:
                path = build_loop_path(area, g, ng, lam, o, t, theta, rho, *(quiet or (None, 0.0)))
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
            progress((k + 1) / LOOP_BEARINGS)
    for c in cands:
        c["score"] = (c["green"] - 0.8 * c["rep"] - 0.4 * ((quiet[0] if quiet else QUIET) if lam > 0 else 0) * c["noise"]
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
#styletip{position:fixed;z-index:1200;left:50%;top:calc(var(--top) + 14px);transform:translateX(-50%);background:rgba(59,58,51,.88);color:#FBF7EE;padding:6px 12px;border-radius:10px;font-size:.85rem;opacity:0;transition:opacity .3s;pointer-events:none}
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
<div id="banner"><div id="ico">🧭</div><div><div id="dist"></div><div id="ins"></div></div></div>
<div id="side"><button id="layers" title="Kartenstil wechseln">🗺️</button><button id="sat" title="Satellit / Luftbild">🛰️</button><button id="fit" title="Ganze Route zeigen">⤢</button></div>
<div id="sheet">
 <div id="ttl"></div>
 <div id="pre"><div id="chips"></div><div id="stats"></div><div id="hint"></div></div>
 <div id="navbox"><div id="rem"></div><div id="prog"><div></div></div></div>
 <div id="btns"><button id="go">▶ Navigation starten</button><button class="sq on" id="vbtn" title="Sprachansagen">🔊</button><button class="sq on" id="fbtn" title="Karte folgt mir">🎯</button><button class="sq" id="lbtn" title="Alle Anweisungen">📋</button><button class="sq" id="demo" title="Demo ohne Gehen">🧪</button></div>
</div>
<div id="list"></div>
<div id="styletip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const WALK = D.speed || 80, TOP = D.topPad || 0, ky = 110540;
const GO = D.mode === 'ab' ? '▶ Route starten' : '▶ Ründchen starten';
document.documentElement.style.setProperty('--top', TOP + 'px');

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
function tip(t) {
  const el = $('styletip'); el.textContent = t; el.style.opacity = 1;
  clearTimeout(tipTimer); tipTimer = setTimeout(() => { el.style.opacity = 0; }, 1600);
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
const pin = e => L.divIcon({html: '<div style="font-size:30px;line-height:30px;filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))">' + e + '</div>',
  className: '', iconSize: [32, 32], iconAnchor: [16, 28]});
L.marker(D.p1, {icon: pin('🚶')}).addTo(map);
if (D.p1[0] !== D.p2[0] || D.p1[1] !== D.p2[1]) L.marker(D.p2, {icon: pin('🏁')}).addTo(map);
(D.stops || []).forEach(s => L.marker([s[0], s[1]], {icon: pin('📍')}).addTo(map).bindTooltip(s[2], {direction: 'top'}));
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
  map.fitBounds(L.latLngBounds(coords), {paddingTopLeft: [24, TOP + 24], paddingBottomRight: [24, 250], animate: false});
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
  const mainLine = add(L.polyline(coords, {color: '#3F6B3A', weight: 7, opacity: 1}));
  drawIn(mainLine, (window.INTRO && !INTRO_DONE) ? 1900 : 0);
  done = add(L.polyline([], {color: '#B9C9A8', weight: 7, opacity: 1}));
  steps.forEach((s, k) => { if (k > 0 && k < steps.length - 1)
    add(L.circleMarker([s.lat, s.lon], {radius: 5, color: '#fff', weight: 2, fillColor: '#4E6E45', fillOpacity: 1})); });
  document.querySelectorAll('.chip').forEach((c, k) => c.classList.toggle('on', k === i));
  $('stats').innerHTML = o.stats.map(s => '<div><span>' + esc(s[0]) + '</span><b>' + esc(s[1]) + '</b></div>').join('');
  $('hint').textContent = (o.hint || '') + (showShort ? ' · gestrichelt: kürzeste Route' : '');
  $('hint').className = o.warn ? 'warn' : '';
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
  document.body.classList.remove('nav'); $('go').textContent = GO;
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

    def run(w):
        path = nx.shortest_path(H, o, t, weight=w)
        idx = [H[a][b]["i"] for a, b in zip(path[:-1], path[1:])]
        dist = float(L[idx].sum()) if idx else 0.0
        share = float((L[idx] * g[idx]).sum() / dist) if dist > 0 else 0.0
        noise = float((L[idx] * nz[idx]).sum() / dist) if dist > 0 else 0.0
        narrow = float((L[idx] * (wd[idx] <= NARROW)).sum() / dist) if dist > 0 else 0.0
        return {"path": path, "dist": dist, "green": share, "noise": noise, "narrow": narrow}

    def cost(lam, mu, wq):
        return lambda u, v, d: L[d["i"]] * (1 + lam * (1 - g[d["i"]]) + mu * nz[d["i"]] + wq * wd[d["i"]])

    short = run(lambda u, v, d: L[d["i"]])
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
             ["Grünanteil", f"{r['green']:.0%}"], ["Hauptstraße", f"{r['noise']:.0%}"]]))
    if not options:
        raise ValueError("Start und Ziel liegen zu nah beieinander.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == "Ausgewogen"), 0)
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=parks_geo(area), water=water_geo(area),
                   options=options, default=default, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": dest_v, "area": area, "bbox": bbox, "buf": buf, "water": _water()}


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
    ensure_water(area, log)
    o, t = nearest_node(area, *p1), nearest_node(area, *p2)
    variants = find_loops(area, o, t, target, LEVELS[level], buf, progress=progress, quiet=level_quiet(level))
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
            f"Wunschlänge {fmt_km(target)} ({diff:+.0%}) · {1 - v['rep']:.0%} ohne Doppelweg"
            + (water_note(area, v['path']) if _water() else ""),
            [["Länge", fmt_km(v["dist"])], ["Dauer", f"{mins} min"],
             ["Grünanteil", f"{v['green']:.0%}"], ["Hauptstraße", f"{v.get('noise', 0):.0%}"]]))
    if not options:
        raise ValueError("Hier konnte ich keine passende Runde finden. Versuche eine andere Länge "
                         "oder einen anderen Startpunkt.")
    payload = dict(mode="loop", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=parks_geo(area), water=water_geo(area),
                   options=options, default=0, speed=M_PER_MIN, topPad=TOP_PAD)
    return {"mode": "loop", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "end": end_v, "km": km, "km_ok": km, "level": level, "buf": buf,
            "area": area, "bbox_km": bbox_km, "water": _water()}


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
            "start": start_v, "dest": dest_v, "area": None, "bbox": None, "buf": 150, "fallback": True}


def make_ab_view_safe(start_v, dest_v, buf=150, area=None, bbox=None, log=lambda *_: None):
    """Grüne Route; wenn die Kartenserver überlastet sind, automatisch die einfache Fußroute."""
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
SCAN_STEP = 10.0          # Abtastabstand entlang einer Route (m)
NOISE_DIST = 20.0         # bis zu diesem Abstand zählt eine Hauptstraße als laut (m)
QUIET = 1.5               # Gewicht "Hauptstraßen meiden" (bei "Schnellste" 0)
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
def scan_area(bbox, water=False):
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
                if gtype == 3 and _is_green(lname, cls, sub):
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
        return {"dist": 0.0, "green": 0.0, "noise": 0.0, "rep": 0.0}
    s = np.arange(SCAN_STEP / 2, total, SCAN_STEP)
    px, py = np.interp(s, cum, X), np.interp(s, cum, Y)
    r, c, ok = _cells(sa, px, py)
    gd = np.where(ok, sa["gdist"][r, c], np.inf)
    g = np.clip(1.0 - gd / buf, 0.0, 1.0)
    nz = np.where(ok, sa["noise"][r, c], 0.0)
    first, rep = {}, 0
    for i, k in enumerate(zip(np.round(px / 8).astype(int), np.round(py / 8).astype(int))):
        j = first.setdefault(k, i)
        if i - j > 5:
            rep += 1
    n = max(len(s), 1)
    return {"dist": total, "green": float(g.mean()), "noise": float(nz.mean()), "rep": rep / n}


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
    r = requests.get(OSRM_FOOT + coords, headers=UA, timeout=10, params=params)
    r.raise_for_status()
    js = r.json()
    if js.get("code") != "Ok" or not js.get("routes"):
        raise ValueError("Für diese Punkte habe ich keinen Fußweg gefunden.")
    return js["routes"]


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

    cands, first = [], []
    bearings = [2 * math.pi * k / 6 for k in range(6)]
    rho0 = target / (2 * math.pi * 1.3)
    total_steps = len(bearings) + 3
    for k, th in enumerate(bearings):
        try:
            c = build(th, rho0)
            c["theta"], c["rho"] = th, rho0
            cands.append(c)
            first.append(c)
        except Exception:
            pass
        if progress:
            progress((k + 1) / total_steps)
    # die drei vielversprechendsten Richtungen auf die Wunschlänge nachjustieren
    first.sort(key=lambda c: -(c["green"] - 0.8 * c["rep"] - abs(c["dist"] / target - 1)))
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


SCAN_NOTE = "🛰️ Grün-Scan (Ersatzdaten aus Kartenkacheln)"
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
    sa = scan_area(bbox, _water())
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
                                    f"{LEVEL_HINT[name]} · Umweg {detour:+.0%} · {SCAN_NOTE}", sc))
    if not options:
        raise ValueError("Start und Ziel liegen zu nah beieinander.")
    order = list(LEVELS)
    default = min(range(len(options)), key=lambda i: abs(order.index(options[i]["label"]) - 2))
    payload = dict(mode="ab", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=sa["geo"], water=sa.get("wgeo"),
                   options=options, default=default, speed=M_PER_MIN, topPad=TOP_PAD, attrib=SCAN_ATTRIB)
    return {"mode": "ab", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "dest": dest_v, "area": None, "bbox": None, "buf": buf, "fallback": "scan", "water": _water()}


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
    sa = scan_area(loop_bbox(p1, p2, target), _water())
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
        c["score"] = (w_green * c["green"] - 0.8 * c["rep"] - 0.25 * mu * c["noise"]
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
            f"Wunschlänge {fmt_km(target)} ({diff:+.0%}) · {1 - c['rep']:.0%} ohne Doppelweg · {SCAN_NOTE}", c))
    if not options:
        raise ValueError("Für diesen Startpunkt habe ich keine Runde gefunden.")
    payload = dict(mode="loop", p1=list(p1), p2=list(p2), t1=t1, t2=t2, parks=sa["geo"], water=sa.get("wgeo"),
                   options=options, default=0, speed=M_PER_MIN, topPad=TOP_PAD, attrib=SCAN_ATTRIB)
    return {"mode": "loop", "html": map_html(payload), "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "start": start_v, "end": end_v, "km": km, "km_ok": km, "level": level, "buf": buf,
            "area": None, "bbox_km": None, "fallback": "scan", "water": _water()}


def make_loop_view_safe(start_v, end_v, km, level, buf=150, area=None, bbox_km=None,
                        log=lambda *_: None, progress=None):
    """Ründchen mit Overpass-Daten; wenn die Server überlastet sind, per Grün-Scan."""
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


def tour_path(area, g, lam, mu, wq, nodes):
    """Teilstrecken Start -> Zwischenziele -> Ende, jede grün-optimal; schon gelaufene Wegstücke werden
    teurer, damit der Rückweg möglichst ein anderer ist. marks: Index der Zwischenziele im Pfad."""
    H, L = area["H"], area["lengths"]
    nz = area.get("noise")
    nz = np.zeros(len(L)) if nz is None else nz
    wd = area.get("wide")
    wd = np.zeros(len(L)) if wd is None else wd
    used, full, marks = set(), [nodes[0]], []
    for s_, e in zip(nodes[:-1], nodes[1:]):
        if s_ != e:
            def w(u, v, d, used=used):
                i = d["i"]
                c = L[i] * (1 + lam * (1 - g[i]) + mu * nz[i] + wq * wd[i])
                return c * LOOP_REUSE_PENALTY if ((u, v) in used or (v, u) in used) else c
            p = nx.shortest_path(H, s_, e, weight=w)
            full += p[1:]
            used.update(zip(p[:-1], p[1:]))
        marks.append(len(full) - 1)
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
        c["score"] = c["green"] - 0.8 * c["rep"] - 0.4 * mu * c["noise"] - 2.0 * abs(c["got"] / extra_m - 1)
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


def _tour_payload(mode_extra, p1, pe, t1, t2, stops, options, parks, default, attrib=None, water=None):
    payload = dict(mode="tour", p1=list(p1), p2=list(pe), t1=t1, t2=t2, parks=parks, options=options,
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
    bbox, span = points_bbox(pts, extra_m)
    if span > MAX_SPAN_M:
        raise ValueError(f"Die Punkte liegen {span / 1000:.1f} km auseinander. "
                         f"Bitte höchstens {MAX_SPAN_M // 1000} km wählen.")
    log("🗺️ Wegenetz und Grünflächen laden")
    t0 = time.time()
    area = load_area(bbox)
    log(f"✅ {fmt_m(area['n_edges'])} Wegstücke, {area['n_parks']} Grünflächen ({time.time() - t0:.0f} s)")
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
            f"{pre}{1 - rep:.0%} ohne Doppelweg" + (f" · {narrow:.0%} kleine Wege" if LEVELS[level if extra_m > 0 else name] >= 12 else "")
            + (water_note(area, path) if _water() else ""),
            [["Länge", fmt_km(dist)], ["Dauer", f"{mins} min"], ["Grünanteil", f"{green:.0%}"],
             ["Hauptstraße", f"{noise:.0%}"]]))
    if not options:
        raise ValueError("Ich konnte keine Runde mit diesen Zwischenzielen finden.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == level), 0)
    payload = _tour_payload(None, pts[0], pts[-1], names[0], names[-1], stops, options, parks_geo(area), default,
                            water=water_geo(area))
    return {"mode": "tour", "html": map_html(payload), "p1": pts[0], "p2": pts[-1], "t1": names[0],
            "t2": names[-1], "stops": names[1:-1], "start": start_v, "buf": buf, "stops_v": list(stops_v),
            "end_v": end_v, "level": level, "extra_km": extra_km, "water": _water()}


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
        c["score"] = (sc["green"] - 0.8 * sc["rep"] - 0.25 * mu * sc["noise"] - 2.0 * abs(c["got"] / extra_m - 1))
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
    sa = scan_area(bbox, _water())
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
                f"Extra {fmt_km(c['got'])} (Wunsch +{extra_km} km) · {1 - c['sc']['rep']:.0%} ohne Doppelweg · {SCAN_NOTE}",
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
                                    f"{LEVEL_HINT[name]} · {1 - sc['rep']:.0%} ohne Doppelweg · {SCAN_NOTE}", sc))
    if not options:
        raise ValueError("Ich konnte keine Runde mit diesen Zwischenzielen finden.")
    default = next((i for i, o_ in enumerate(options) if o_["label"] == level), 0)
    payload = _tour_payload(None, pts[0], pts[-1], names[0], names[-1], stops, options, sa["geo"], default,
                            attrib=SCAN_ATTRIB, water=sa.get("wgeo"))
    return {"mode": "tour", "html": map_html(payload), "p1": pts[0], "p2": pts[-1], "t1": names[0],
            "t2": names[-1], "stops": names[1:-1], "start": start_v, "buf": buf, "fallback": "scan",
            "stops_v": list(stops_v), "end_v": end_v, "level": level, "extra_km": extra_km, "water": _water()}


def make_tour_view_safe(start_v, stops_v, end_v, level, buf=150, log=lambda *_: None, extra_km=0):
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


# ------------------------------------------------------- Logo / App-Ikon (eingebettet, keine Zusatzdateien nötig)
# Blatt mit Zielpunkt: Kleinform fürs Ikon (Browser-Tab 64 px, Home-Bildschirm 180 px)
ICON_FAV_B64 = "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"
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
 [['apple-mobile-web-app-title',TITLE],['application-name',TITLE],['apple-mobile-web-app-capable','yes']].forEach(function(a){
  var m=d.querySelector('meta[name="'+a[0]+'"]');if(!m){m=d.createElement('meta');m.name=a[0];h.appendChild(m)}m.content=a[1]})
 }catch(e){}})}
var cur=[FAV,TOUCH];
patch(FAV,TOUCH);
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
        st_html(ICON_JS.replace("__FAV__", ICON_FAV_B64).replace("__TOUCH__", ICON_TOUCH_B64), height=0)
    except Exception:
        pass


# ------------------------------------------------------- UI
st.set_page_config(page_title="GreenWalker", page_icon=page_icon(), layout="centered",
                   initial_sidebar_state="collapsed")
install_icons()

BASE_CSS = """
<style>
:root{--moss:#4E6E45;--sage:#9DB38A;--sand:#E9E1D2;--paper:#FBF8F1;--line:#D9CFBC;--text:#3B3A33}
html,body,.stApp,.stApp input,.stApp textarea,.stApp button,.stApp label,.stApp p,.stApp li,.stApp div,
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp [data-baseweb]{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.block-container{padding-top:1.2rem;max-width:680px}
.hero{display:flex;gap:14px;align-items:center;border-radius:16px;padding:14px 16px;margin-bottom:14px;
 background:var(--moss);color:#FBF7EE;box-shadow:0 6px 18px rgba(60,50,30,.3)}
.hero .ico{font-size:36px;line-height:1;width:46px;text-align:center;flex:none}
.hero h1{margin:0;padding:0;font-size:1.5rem;font-weight:700;line-height:1.1;color:#FBF7EE;letter-spacing:0}
.hero p{margin:2px 0 0;font-size:1rem;color:#FBF7EE;opacity:.92}
.st-key-card_route,.st-key-card_loop,.st-key-card_tour{background:rgba(251,248,241,.98);border:1px solid var(--line);border-radius:20px;
 padding:14px 14px 12px;box-shadow:0 8px 28px rgba(60,50,30,.18);margin-top:12px}
.st-key-card_route label p,.st-key-card_loop label p,.st-key-card_tour label p{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
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
.st-key-card_route,.st-key-card_loop,.st-key-card_tour{animation:gwrise .55s cubic-bezier(.2,.8,.2,1) both}
.hero{animation:gwrise .6s cubic-bezier(.2,.8,.2,1) both}
.stButton button,[data-testid="stButtonGroup"] button{transition:transform .15s,background .25s,border-color .25s,box-shadow .25s}
.stButton button:active,[data-testid="stButtonGroup"] button:active{transform:scale(.97)}
.stButton button[kind="primary"]:hover{box-shadow:0 6px 18px rgba(78,110,69,.4);transform:translateY(-1px)}
[data-testid="stToggle"],[data-testid="stCheckbox"]{transition:opacity .2s}
@media (prefers-reduced-motion:reduce){.st-key-card_route,.st-key-card_loop,.st-key-card_tour,.hero{animation:none}}
</style>
"""


# Ladeansicht beim Berechnen: Wolken ziehen, Logo atmet, Sprüche wechseln. Gleiche Farbe wie die Wolkendecke der
# Karte, damit der Übergang ohne Bruch in die Kartenansicht läuft.
OVERLAY_HTML = """
<style>
#gwload{position:fixed;inset:0;z-index:99999;background:#F4EFE6;display:flex;flex-direction:column;align-items:center;
 justify-content:center;gap:16px;animation:gwfade .35s ease both;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;padding:0 22px}
@keyframes gwfade{from{opacity:0}to{opacity:1}}
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
@media (prefers-reduced-motion:reduce){#gwload *{animation-duration:0s!important}#gwload .fact p:first-of-type{opacity:1}}
__FACTCSS__
</style>
<div id="gwload"><i class="b"></i><i class="b"></i><i class="b"></i><i class="b"></i>
<div class="logo"><img src="data:image/svg+xml;base64,__LOGO__" alt=""></div>
<div class="st">__STAGE__</div><div class="bar"></div>
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
               "Rotkehlchen singen auch im Winter, denn sie verteidigen ihr Revier das ganze Jahr."],
    "spring": ["Im April und Mai singen die Vögel am lautesten: Der Morgenchor beginnt schon vor Sonnenaufgang.",
               "Bärlauch und Buschwindröschen färben im Frühjahr Auwälder und Parkränder weiß."],
    "summer": ["Linden blühen im Juni und Juli, und ihr Duft lockt Bienen und Hummeln an.",
               "An Hitzetagen sind Parks und Wälder oft mehrere Grad kühler als versiegelte Straßen."],
    "autumn": ["Das Herbstlaub wird nicht „bunt“: Blattgrün wird abgebaut, darunter kommen Gelb und Orange zum Vorschein.",
               "Eichelhäher vergraben im Herbst Tausende Eicheln. Viele vergessen sie, und so entstehen neue Eichen."],
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
]
WATER_FACTS = [
    "„Blaue Räume“: Wasser in Sichtweite macht Spaziergänge nach Studien erholsamer als reines Grün.",
    "Libellen gibt es seit über 300 Millionen Jahren. Ihre Larven leben im Wasser und zeigen saubere Gewässer an.",
    "Flüsse und Seen kühlen ihre Umgebung im Sommer spürbar. Am Ufer ist es oft ein paar Grad angenehmer.",
]


def region_facts(lat, lon):
    """Fakten zum Ort (nächstgelegene bekannte Stadt im Umkreis); leer, wenn keine passt."""
    best, bd = None, 1e9
    for name, la, lo, rad, facts in CITY_FACTS:
        d = dist_m((lat, lon), (la, lo)) / 1000
        if d <= rad and d < bd:
            best, bd = (name, facts), d
    return best


def pick_facts(lat=None, lon=None, water=False, month=None):
    month = month or time.localtime().tm_mon
    season = ("winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter")[month - 1]
    rnd = random.Random()
    out = []
    if lat is not None:
        reg = region_facts(lat, lon)
        if reg:
            local = list(reg[1])
            rnd.shuffle(local)
            out += local[:3]
    if water:
        out += rnd.sample(WATER_FACTS, 2)
    out += rnd.sample(SEASON_FACTS[season], 1)
    gen = list(GENERAL_FACTS)
    rnd.shuffle(gen)
    out += gen[: max(2, 6 - len(out))]
    return out[:7]


class Overlay:
    """Vollbild-Ladeansicht mit Arbeitsstand (aktueller Schritt) und wechselnden Natur-/Ortsfakten."""
    PER = 7.0  # Sekunden je Fakt

    def __init__(self, start=None, water=False):
        lat = lon = None
        try:
            lat, lon, _ = _unpack(start)
        except Exception:
            pass
        self.facts = pick_facts(lat, lon, water)
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
            html = (OVERLAY_HTML.replace("__FACTCSS__", css).replace("__LOGO__", logo)
                    .replace("__STAGE__", escape(self.stage)).replace("__FACTS__", facts)
                    .replace("__LABEL__", "Wusstest du?"))
            self.ph.markdown(html, unsafe_allow_html=True)
        except Exception:
            pass

    def log(self, msg):
        """Schritt anzeigen (und weiter ins versteckte Statusfeld schreiben)."""
        try:
            st.write(msg)
        except Exception:
            pass
        txt = re.sub(r"\s*\([^)]*\)\s*$", "", str(msg)).strip()
        if txt and txt != self.stage and len(txt) < 140:
            self.stage = txt
            self.render()

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

PLACEHOLDER = "Straße, Adresse, Café, Restaurant, Geschäft …"
SEARCH_TIP = ("Tipp: Tippe mindestens 3 Buchstaben, dann erscheinen Vorschläge. "
              "Tippfehler sind okay, es werden ähnliche Orte angezeigt.")


SEARCH_STYLE = {
    "searchbox": {
        "control": {"borderRadius": 14, "minHeight": 50, "borderColor": "#D9CFBC", "backgroundColor": "#FFFFFF",
                    "boxShadow": "0 1px 2px rgba(60,50,30,.06)"},
        "menu": {"borderRadius": 14, "overflow": "hidden", "border": "1px solid #D9CFBC",
                 "boxShadow": "0 8px 24px rgba(60,50,30,.18)"},
        "option": {"padding": "10px 14px"},
    },
}


def place_input(label, key):
    dot = "s" if label.startswith("Start") else "z"
    st.markdown(f'<div class="fld"><i class="dot {dot}"></i>{escape(label)}</div>', unsafe_allow_html=True)
    kw = dict(key=key, placeholder=PLACEHOLDER, clear_on_submit=False)
    try:
        return st_searchbox(search_places, style_overrides=SEARCH_STYLE, **kw)
    except TypeError:  # ältere Version ohne style_overrides
        return st_searchbox(search_places, **kw)


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
    use = st.toggle("📍 Von meinem Standort starten", key=f"{key}_useloc", value=True,
                    help="Der Browser fragt nach der Standortfreigabe. Der Standort wird nur für die Suche genutzt.")
    if not use:
        v = place_input(label, key)
        if v:
            st.session_state["bias"] = _unpack(v)[:2]
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
        "(bei „Schnellste“ ignoriert)"
    )


# ---------- Wechsel zwischen Startseite und Kartenansicht
def open_map(view):
    st.session_state["gw_sid"] = st.session_state.get("gw_sid", 0) + 1
    view["sid"] = st.session_state["gw_sid"]
    st.session_state["gw"] = view
    st.session_state["bias"] = view["p1"]


def close_map():
    st.session_state.pop("gw", None)


def set_retry():
    gw = st.session_state.get("gw")
    if gw:
        gw["retry"] = True


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
    st.session_state["_water"] = bool(gw.get("water", False))
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
                dest_new = st_searchbox(search_places, key=f"gw_dest_{gw['sid']}",
                                        placeholder=f"🏁 {gw['t2']} · Ziel ändern …", clear_on_submit=False)
        elif gw["mode"] == "tour":
            c1, c2 = st.columns([1, 6])
            c1.button("←", key="gw_back", use_container_width=True, on_click=close_map, help="Zurück zur Suche")
            n_ = len(gw.get("stops", []))
            with c2:
                if n_ < TOUR_MAX_MAP_STOPS:
                    add_new = st_searchbox(search_places, key=f"gw_add_{gw['sid']}", clear_on_submit=False,
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
        st.toast("🧭 Berechne neue Route …")
        try:
            if gw.get("fallback") == "scan":
                new = scan_ab_view(gw["start"], dest_new, gw["buf"])
            elif gw.get("fallback"):
                new = make_ab_view_safe(gw["start"], dest_new, gw["buf"])
            else:
                new = make_ab_view_safe(gw["start"], dest_new, gw["buf"], gw["area"], gw["bbox"])
        except Exception as e:
            gw["failed"] = dest_new
            st.toast(f"⚠️ {e}"[:220])
    elif gw["mode"] == "tour" and add_new and add_new != gw.get("failed"):
        st.toast("🛍️ Füge das Ziel hinzu und berechne neu …")
        try:
            new = make_tour_view_safe(gw["start"], gw["stops_v"] + [add_new], gw["end_v"], gw["level"], gw["buf"],
                                      extra_km=gw.get("extra_km", 0))
            new["sid"] = gw["sid"] + 1  # neues leeres Suchfeld
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
        st.toast("🔄 Suche neues Ründchen …")
        try:
            if gw.get("fallback") == "scan":
                new = scan_loop_view(gw["start"], gw["end"], gw["km"], gw["level"], gw["buf"])
            else:
                new = make_loop_view_safe(gw["start"], gw["end"], gw["km"], gw["level"], gw["buf"],
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
        water_on = st.toggle("💧 Wassernähe bevorzugen", key="water_route", value=False,
                             help="Führt die Route bevorzugt an Seen, Teichen und Flüssen entlang – für zusätzliche Entspannung.")
        go = st.button("Route finden", type="primary", use_container_width=True, key="go_ab")
    if not go:
        return
    if not (start_v and ziel_v):
        st.warning("Bitte Start und Ziel jeweils aus den Vorschlägen auswählen.")
        return
    view = None
    st.session_state["_water"] = bool(water_on)
    ov = loading_overlay(start_v, water_on)
    with st.status("Route wird vorbereitet …", expanded=True) as status:
        try:
            view = make_ab_view_safe(start_v, ziel_v, log=ov.log)
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        open_map(view)
        st.rerun()
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
        go = st.button("Ründchen finden", type="primary", use_container_width=True, key="go_loop")
    if not go:
        return
    if not a_v or (not same and not b_v):
        st.warning("Bitte Start" + ("" if same else " und Ende") + " aus den Vorschlägen auswählen.")
        return
    view, holder = None, {}
    st.session_state["_water"] = bool(water_on)

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
        open_map(view)
        st.rerun()
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
        extra_opt = st.selectbox("Extra Strecke (optional)", ["Keine"] + [f"+ {k} km" for k in range(1, 11)],
                                 key="tour_extra", help="Verlängert die Runde um diese Strecke, zusätzlich zum Weg über die Zwischenziele.")
        extra_km = 0 if extra_opt == "Keine" else int(extra_opt.replace("+", "").replace("km", "").strip())
        level = st.select_slider("Wie grün soll es sein?", options=TOUR_LEVELS, value="Sehr grün", key="level_tour")
        water_on = st.toggle("💧 Wassernähe bevorzugen", key="water_tour", value=False,
                             help="Führt die Route bevorzugt an Seen, Teichen und Flüssen entlang – für zusätzliche Entspannung.")
        go = st.button("Ründchen mit Zwischenziel finden", type="primary", use_container_width=True, key="go_tour")
    if not go:
        return
    if not a_v or not all(stops) or (not same and not b_v):
        st.warning("Bitte Start, alle Zwischenziele" + ("" if same else " und Endpunkt") + " aus den Vorschlägen auswählen.")
        return
    view = None
    st.session_state["_water"] = bool(water_on)
    ov = loading_overlay(a_v, water_on)
    with st.status("Runde mit Zwischenzielen wird berechnet …", expanded=True) as status:
        try:
            view = make_tour_view_safe(a_v, stops, None if same else b_v, level, log=ov.log, extra_km=extra_km)
            status.update(label="Fertig", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Fehler", state="error")
            show_error(e)
    if view:
        open_map(view)
        st.rerun()
    ov.empty()


def render_home():
    logo = base64.b64encode(LOGO_SVG.encode()).decode()
    st.markdown('<div class="hero"><div class="ico"><img src="data:image/svg+xml;base64,' + logo +
                '" width="46" height="46" alt=""></div><div><h1>GreenWalker</h1>'
                '<p>Der grüne Weg statt nur der kürzeste.</p></div></div>',
                unsafe_allow_html=True)
    # Nur der gewählte Reiter wird aufgebaut (st.tabs würde alle drei samt Suchfeldern laden = langsam)
    pages = {"route": "🧭 Route", "loop": "🔄 Ründchen", "tour": "🛍️ Ründchen mit Zwischenziel"}
    try:
        page = st.segmented_control("Ansicht", list(pages), format_func=pages.get, default="route",
                                    key="page_sel", label_visibility="collapsed")
    except Exception:
        page = st.radio("Ansicht", list(pages), format_func=pages.get, horizontal=True, key="page_sel_r",
                        label_visibility="collapsed")
    page = page or st.session_state.get("page_last", "route")
    st.session_state["page_last"] = page
    {"route": home_route, "loop": home_loop, "tour": home_tour}[page]()
    with st.expander("Datenquellen und Gewichtung"):
        st.markdown(sources_text(150, LEVELS["Ausgewogen"]))


if st.session_state.get("gw"):
    render_map_view()
else:
    render_home()
