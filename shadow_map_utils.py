from __future__ import annotations

import math
from datetime import datetime
from typing import List, Tuple

import plotly.graph_objects as go

from shadow_mapper import normalize_date_key

_COUNTRY_CODES_CACHE: List[str] | None = None


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
    try:
        import plotly.express as px
    except ImportError:
        print("plotly.express unavailable; skipping outline overlay.")
        _COUNTRY_CODES_CACHE = []
        return _COUNTRY_CODES_CACHE

    frame = px.data.gapminder()
    codes = sorted(
        {
            code
            for code in frame["iso_alpha"]
            if isinstance(code, str) and code and code.upper() != "NA"
        }
    )
    _COUNTRY_CODES_CACHE = codes
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
