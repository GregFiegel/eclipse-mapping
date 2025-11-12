from __future__ import annotations

import colorsys
import math
from datetime import datetime
from typing import List, Sequence, Tuple

import plotly.graph_objects as go

from shadow_mapper import CatalogEntry, normalize_date_key

SAROS_BASE_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (57, 59, 121),
    (82, 84, 163),
    (107, 174, 214),
    (49, 163, 84),
    (116, 196, 118),
    (166, 118, 29),
    (231, 186, 82),
    (173, 73, 74),
    (214, 97, 107),
    (142, 124, 195),
    (196, 156, 213),
    (140, 150, 198),
    (136, 86, 167),
    (199, 120, 186),
]

_COUNTRY_CODES_CACHE: List[str] | None = None

# ISO-3 codes covering modern sovereign states and territories. Plotly's gapminder
# sample omits several countries (e.g., DPRK, many former USSR republics), so we
# seed the outline set with this comprehensive list and union it with whatever
# the optional px dataset provides.
DEFAULT_COUNTRY_CODES = [
    "AFG",
    "ALA",
    "ALB",
    "DZA",
    "ASM",
    "AND",
    "AGO",
    "AIA",
    "ATA",
    "ATG",
    "ARG",
    "ARM",
    "ABW",
    "AUS",
    "AUT",
    "AZE",
    "BHS",
    "BHR",
    "BGD",
    "BRB",
    "BLR",
    "BEL",
    "BLZ",
    "BEN",
    "BMU",
    "BTN",
    "BOL",
    "BES",
    "BIH",
    "BWA",
    "BVT",
    "BRA",
    "IOT",
    "BRN",
    "BGR",
    "BFA",
    "BDI",
    "CPV",
    "KHM",
    "CMR",
    "CAN",
    "CYM",
    "CAF",
    "TCD",
    "CHL",
    "CHN",
    "CXR",
    "CCK",
    "COL",
    "COM",
    "COG",
    "COD",
    "COK",
    "CRI",
    "CIV",
    "HRV",
    "CUB",
    "CUW",
    "CYP",
    "CZE",
    "DNK",
    "DJI",
    "DMA",
    "DOM",
    "ECU",
    "EGY",
    "SLV",
    "GNQ",
    "ERI",
    "EST",
    "SWZ",
    "ETH",
    "FLK",
    "FRO",
    "FJI",
    "FIN",
    "FRA",
    "GUF",
    "PYF",
    "ATF",
    "GAB",
    "GMB",
    "GEO",
    "DEU",
    "GHA",
    "GIB",
    "GRC",
    "GRL",
    "GRD",
    "GLP",
    "GUM",
    "GTM",
    "GGY",
    "GIN",
    "GNB",
    "GUY",
    "HTI",
    "HMD",
    "VAT",
    "HND",
    "HKG",
    "HUN",
    "ISL",
    "IND",
    "IDN",
    "IRN",
    "IRQ",
    "IRL",
    "IMN",
    "ISR",
    "ITA",
    "JAM",
    "JPN",
    "JEY",
    "JOR",
    "KAZ",
    "KEN",
    "KIR",
    "PRK",
    "KOR",
    "KWT",
    "KGZ",
    "LAO",
    "LVA",
    "LBN",
    "LSO",
    "LBR",
    "LBY",
    "LIE",
    "LTU",
    "LUX",
    "MAC",
    "MDG",
    "MWI",
    "MYS",
    "MDV",
    "MLI",
    "MLT",
    "MHL",
    "MTQ",
    "MRT",
    "MUS",
    "MYT",
    "MEX",
    "FSM",
    "MDA",
    "MCO",
    "MNG",
    "MNE",
    "MSR",
    "MAR",
    "MOZ",
    "MMR",
    "NAM",
    "NRU",
    "NPL",
    "NLD",
    "NCL",
    "NZL",
    "NIC",
    "NER",
    "NGA",
    "NIU",
    "NFK",
    "MKD",
    "MNP",
    "NOR",
    "OMN",
    "PAK",
    "PLW",
    "PSE",
    "PAN",
    "PNG",
    "PRY",
    "PER",
    "PHL",
    "PCN",
    "POL",
    "PRT",
    "PRI",
    "QAT",
    "REU",
    "ROU",
    "RUS",
    "RWA",
    "BLM",
    "SHN",
    "KNA",
    "LCA",
    "MAF",
    "SPM",
    "VCT",
    "WSM",
    "SMR",
    "STP",
    "SAU",
    "SEN",
    "SRB",
    "SYC",
    "SLE",
    "SGP",
    "SXM",
    "SVK",
    "SVN",
    "SLB",
    "SOM",
    "ZAF",
    "SGS",
    "SSD",
    "ESP",
    "LKA",
    "SDN",
    "SUR",
    "SJM",
    "SWE",
    "CHE",
    "SYR",
    "TWN",
    "TJK",
    "TZA",
    "THA",
    "TLS",
    "TGO",
    "TKL",
    "TON",
    "TTO",
    "TUN",
    "TUR",
    "TKM",
    "TCA",
    "TUV",
    "UGA",
    "UKR",
    "ARE",
    "GBR",
    "USA",
    "UMI",
    "URY",
    "UZB",
    "VUT",
    "VEN",
    "VNM",
    "VGB",
    "VIR",
    "WLF",
    "ESH",
    "YEM",
    "ZMB",
    "ZWE",
]


def add_world_outline_trace(fig: go.Figure) -> None:
    codes = get_country_codes()
    if not codes:
        return
    transparent_scale = [[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]]
    fig.add_trace(
        go.Choropleth(
            locations=codes,
            locationmode="ISO-3",
            z=[0] * len(codes),
            zmin=0,
            zmax=1,
            colorscale=transparent_scale,
            autocolorscale=False,
            showscale=False,
            hoverinfo="skip",
            marker_line_color="rgba(15, 15, 15, 0.95)",
            marker_line_width=0.8,
        )
    )


def get_country_codes() -> List[str]:
    global _COUNTRY_CODES_CACHE
    if _COUNTRY_CODES_CACHE is not None:
        return _COUNTRY_CODES_CACHE

    codes = {code.strip().upper() for code in DEFAULT_COUNTRY_CODES if code}
    try:
        import plotly.express as px

        frame = px.data.gapminder()
        codes.update(
            {
                code.strip().upper()
                for code in frame["iso_alpha"]
                if isinstance(code, str) and code and code.upper() != "NA"
            }
        )
    except ImportError:
        print("plotly.express unavailable; using built-in ISO code list for outlines.")

    codes.discard("")
    _COUNTRY_CODES_CACHE = sorted(codes)
    return _COUNTRY_CODES_CACHE


def build_geojson_polygon(latitudes: List[float], longitudes: List[float]) -> dict | None:
    if len(latitudes) < 3 or len(longitudes) < 3:
        return None
    coords = [(lon, lat) for lon, lat in zip(longitudes, latitudes)]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {
        "type": "Polygon",
        "coordinates": [coords],
    }


def apply_geo_styling(fig: go.Figure) -> None:
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="rgba(20, 20, 20, 0.9)",
        countrywidth=0.7,
        showcoastlines=True,
        coastlinecolor="rgba(250, 250, 250, 0.9)",
        coastlinewidth=1.2,
        showsubunits=True,
        subunitcolor="rgba(10, 10, 10, 0.65)",
        subunitwidth=0.3,
        showland=True,
        landcolor="rgb(245, 245, 245)",
        showocean=True,
        oceancolor="rgb(200, 215, 238)",
        showlakes=False,
        showrivers=False,
        lataxis=dict(showgrid=True, dtick=30, gridcolor="rgba(255,255,255,0.4)"),
        lonaxis=dict(showgrid=True, dtick=30, gridcolor="rgba(255,255,255,0.4)"),
    )


def build_power_colorbar_ticks(max_years: float, exponent: float) -> Tuple[List[float], List[str]]:
    if max_years <= 0:
        return [0.0], ["0"]
    ticks: List[float] = [0.0]
    if max_years < 1:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            value = max_years * fraction
            if value > 0:
                ticks.append(value)
    else:
        scale = 1.0
        while scale <= max_years:
            for multiplier in (1, 2, 5):
                value = multiplier * scale
                if value > max_years:
                    break
                ticks.append(value)
            scale *= 10
        ticks.append(max_years)

    ticks = sorted({round(value, 10) for value in ticks if value >= 0})
    tickvals = [scale_years(value, exponent) for value in ticks]
    ticktext = [_format_colorbar_label(value) for value in ticks]
    return tickvals, ticktext


def _format_colorbar_label(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 1:
        label = f"{value:.2f}"
    elif value < 10:
        label = f"{value:.1f}"
    else:
        label = f"{value:,.0f}"
    return label.rstrip("0").rstrip(".") if "." in label else label


def scale_years(value: float, exponent: float) -> float:
    if value <= 0:
        return 0.0
    return value**exponent


def prepare_saros_coloring(
    series_ids: Sequence[int | None],
    *,
    full_opacity: bool = False,
) -> Tuple[
    dict[int, int], List[List[float | str]], float, float, List[float], List[str]
]:
    """Return color mapping plus a discrete colorscale and labeled tick marks."""
    if not series_ids:
        raise SystemExit("Saros coloring requested but no events were available.")
    if any(series is None for series in series_ids):
        raise SystemExit(
            "Saros coloring requested but one or more catalog entries lacked a Saros number."
        )
    unique_series = sorted({series for series in series_ids if series is not None})
    if not unique_series:
        raise SystemExit(
            "Saros coloring requested but no Saros numbers were found in the catalog."
        )
    mapping = {series: idx for idx, series in enumerate(unique_series)}
    palette = _generate_color_palette(
        len(unique_series), opacity=1.0 if full_opacity else 0.5
    )
    colorscale = _build_discrete_colorscale(palette)
    zmin = -0.5
    zmax = len(unique_series) - 0.5 if len(unique_series) > 1 else 0.5
    tick_pairs = sorted(mapping.items(), key=lambda item: item[1])
    tickvals = [float(idx) for _, idx in tick_pairs]
    ticktext = [f"Saros {series}" for series, _ in tick_pairs]
    return mapping, colorscale, zmin, zmax, tickvals, ticktext


def _generate_color_palette(count: int, *, opacity: float) -> List[str]:
    if count <= 0:
        return [f"rgba(136, 136, 136, {opacity})"]

    def _format_color(rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"rgba({r}, {g}, {b}, {opacity})"

    colors: List[str] = []
    base_total = len(SAROS_BASE_COLORS)
    if count <= base_total:
        colors.extend(_format_color(rgb) for rgb in SAROS_BASE_COLORS[:count])
        return colors

    colors.extend(_format_color(rgb) for rgb in SAROS_BASE_COLORS)
    remaining = count - base_total
    hue = 0.0
    golden_ratio = 0.6180339887498949
    for _ in range(remaining):
        hue = (hue + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append(
            f"rgba({int(r * 255):d}, {int(g * 255):d}, {int(b * 255):d}, {opacity})"
        )
    return colors


def _build_discrete_colorscale(colors: List[str]) -> List[List[float | str]]:
    if not colors:
        return [[0.0, "#888888"], [1.0, "#888888"]]
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]
    scale: List[List[float | str]] = []
    total = len(colors)
    for idx, color in enumerate(colors):
        start = idx / total
        end = (idx + 1) / total
        scale.append([start, color])
        scale.append([end, color])
    scale[0][0] = 0.0
    scale[-1][0] = 1.0
    return scale


def allow_eclipse_type(event: CatalogEntry, include_annular: bool) -> bool:
    """Return True when the event's eclipse_type should be rendered."""
    etype = (event.eclipse_type or "").strip().upper()
    total_like = {"T", "H"}  # Hybrids still produce a total track.
    if include_annular:
        total_like.add("A")
    return etype in total_like


def decimate_polygon(
    latitudes: List[float], longitudes: List[float], step: int
) -> Tuple[List[float], List[float]]:
    """Down-sample polygon points by keeping every nth vertex."""
    if step <= 1 or len(latitudes) < 4 or len(latitudes) != len(longitudes):
        return latitudes, longitudes
    indices = list(range(0, len(latitudes), step))
    if indices[-1] != len(latitudes) - 1:
        indices.append(len(latitudes) - 1)
    lat_filtered = [latitudes[i] for i in indices]
    lon_filtered = [longitudes[i] for i in indices]
    if len(lat_filtered) < 3:
        return latitudes, longitudes
    return lat_filtered, lon_filtered


def quantize_polygon(
    latitudes: List[float], longitudes: List[float], decimals: int = 4
) -> Tuple[List[float], List[float]]:
    """Round polygon coordinates to the requested precision."""
    if decimals < 0:
        return latitudes, longitudes
    return (
        [round(value, decimals) for value in latitudes],
        [round(value, decimals) for value in longitudes],
    )


def simplify_polygon_rdp(
    latitudes: List[float], longitudes: List[float], tolerance: float
) -> Tuple[List[float], List[float]]:
    """Simplify polygon via Ramer-Douglas-Peucker (tolerance in degrees)."""
    if tolerance <= 0 or len(latitudes) != len(longitudes) or len(latitudes) < 4:
        return latitudes, longitudes

    points = list(zip(latitudes, longitudes))
    simplified = _rdp(points, tolerance)
    if len(simplified) < 3:
        return latitudes, longitudes
    result_lat, result_lon = zip(*simplified)
    return list(result_lat), list(result_lon)


def _rdp(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    index = -1
    max_dist = -1.0
    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            index = i
            max_dist = dist
    if max_dist > epsilon and index != -1:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _perpendicular_distance(
    point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]
) -> float:
    if start == end:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    num = abs(
        (end[1] - start[1]) * point[0]
        - (end[0] - start[0]) * point[1]
        + end[0] * start[1]
        - end[1] * start[0]
    )
    den = math.hypot(end[1] - start[1], end[0] - start[0])
    return num / den


def date_sort_key(event) -> float:
    return event.year * 10000 + event.month * 100 + event.day


def date_to_float(event) -> float:
    return date_parts_to_float(event.year, event.month, event.day)


def format_event_date(event) -> str:
    year = event.year
    if year < 0:
        return f"{abs(year):04d} BCE-{event.month:02d}-{event.day:02d}"
    return f"{year:04d}-{event.month:02d}-{event.day:02d}"


def resolve_cutoff(
    max_date: str | None, include_future: bool
) -> Tuple[int | None, str | None, float | None]:
    if max_date and include_future:
        raise SystemExit("--include-future cannot be combined with --max-date.")
    if max_date:
        normalized = normalize_date_key(max_date)
        value = date_key_to_sort_value(normalized)
        return value, normalized, normalized_key_to_float(normalized)
    if include_future:
        return None, None, None
    today_key = current_date_key()
    value = date_key_to_sort_value(today_key)
    return value, today_key, normalized_key_to_float(today_key)


def resolve_future_start(min_date: str | None) -> Tuple[int, str, float]:
    if min_date:
        normalized = normalize_date_key(min_date)
    else:
        normalized = current_date_key()
    value = date_key_to_sort_value(normalized)
    return value, normalized, normalized_key_to_float(normalized)


def date_key_to_sort_value(key: str) -> int:
    year, month, day = parse_normalized_key(key)
    return year * 10000 + month * 100 + day


def normalized_key_to_float(key: str) -> float:
    year, month, day = parse_normalized_key(key)
    return date_parts_to_float(year, month, day)


def parse_normalized_key(key: str) -> Tuple[int, int, int]:
    if key.startswith("-"):
        parts = key[1:].split("-")
        if len(parts) != 3:
            raise SystemExit(f"Unexpected canonical date format: {key}")
        year = -int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    else:
        parts = key.split("-")
        if len(parts) != 3:
            raise SystemExit(f"Unexpected canonical date format: {key}")
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    return year, month, day


def date_parts_to_float(year: int, month: int, day: int) -> float:
    return year + (month - 1) / 12.0 + (day - 1) / 365.0


def current_date_key() -> str:
    today = datetime.now().date()
    return f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
