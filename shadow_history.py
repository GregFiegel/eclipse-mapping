from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import plotly.graph_objects as go

from shadow_mapper import (
    CATALOG_FILE,
    CatalogEntry,
    export_static_image,
    build_shadow_polygon,
    compute_centerline,
    load_catalog,
    normalize_date_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a global map showing when each location on Earth "
            "was most recently in the Moon's shadow."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_FILE,
        help="Path to the Besselian elements CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=160,
        help="Samples per eclipse track (higher => smoother, slower)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on the number of eclipses processed (oldest first) for testing.",
    )
    parser.add_argument(
        "--colorscale",
        default="Viridis",
        help="Plotly colorscale name used for the age gradient.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eclipse_shadow_history.html"),
        help="Destination HTML file for the combined map.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=None,
        help="Optional static image (png/svg/pdf) destination. Requires kaleido.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Scale factor for static image exports (larger values increase resolution).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML export (useful when only --image-output is needed).",
    )
    parser.add_argument(
        "--max-date",
        type=str,
        default=None,
        help="Only include eclipses on or before YYYY-MM-DD (prefix BCE years with '-').",
    )
    parser.add_argument(
        "--include-future",
        action="store_true",
        help="Process eclipses beyond today (overrides the default present-day cutoff).",
    )
    parser.add_argument(
        "--years-back",
        type=float,
        default=None,
        help=(
            "Only render eclipses within this many years before the effective end date "
            "(defaults to unlimited)."
        ),
    )
    parser.add_argument(
        "--outline-only",
        action="store_true",
        help="Skip eclipse rendering and output just the map outline for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_html and not args.image_output:
        raise SystemExit(
            "HTML output disabled but no --image-output provided. "
            "Either drop --no-html or supply an image path."
        )
    if args.image_scale <= 0:
        raise SystemExit("--image-scale must be positive.")
    if args.years_back is not None and args.years_back <= 0:
        raise SystemExit("--years-back must be positive.")

    if args.outline_only:
        figure = go.Figure()
        add_world_outline_trace(figure)
        figure.update_layout(
            title="World Outline Debug",
            margin=dict(l=0, r=0, t=60, b=0),
        )
        apply_geo_styling(figure)
        if not args.no_html:
            figure.write_html(args.output, include_plotlyjs="inline")
            print(f"Wrote {args.output.resolve()}")
        if args.image_output:
            export_static_image(figure, args.image_output, scale=args.image_scale)
        return

    catalog = load_catalog(args.catalog)
    events = sorted(catalog.values(), key=date_sort_key)
    if not events:
        raise SystemExit("Catalog did not yield any eclipses to plot.")
    cutoff_value, cutoff_label, cutoff_float = resolve_cutoff(
        args.max_date, args.include_future
    )
    if cutoff_value is not None:
        print(f"Using eclipses through {cutoff_label}.")
        events = [evt for evt in events if date_sort_key(evt) <= cutoff_value]
        if not events:
            raise SystemExit(f"No eclipses occur on or before {cutoff_label}.")

    years_window_label = None
    if args.years_back is not None:
        window_end_label = cutoff_label or format_event_date(events[-1])
        end_float = cutoff_float if cutoff_float is not None else date_to_float(events[-1])
        min_float = end_float - args.years_back
        events = [evt for evt in events if date_to_float(evt) >= min_float]
        if not events:
            raise SystemExit(
                f"No eclipses found within the last {args.years_back:g} years "
                f"ending {window_end_label}."
            )
        years_window_label = f"Last {args.years_back:g} years (ending {window_end_label})"
        print(f"Restricting to the {years_window_label}.")

    if args.max_events:
        events = events[: args.max_events]
    if not events:
        raise SystemExit("Catalog did not yield any eclipses to plot.")

    date_values = [date_to_float(evt) for evt in events]
    newest = max(date_values)
    oldest = min(date_values)
    span = newest - oldest if newest != oldest else 1.0

    features: List[dict] = []
    feature_ids: List[str] = []
    feature_values: List[float] = []
    hover_texts: List[str] = []
    skipped = 0

    for idx, (event, dv) in enumerate(zip(events, date_values), start=1):
        centerline = compute_centerline(event, args.samples)
        polygon = build_shadow_polygon(centerline)
        if not polygon[0]:
            skipped += 1
            continue
        latitudes, longitudes = polygon
        if len(latitudes) < 3:
            skipped += 1
            continue

        years_since = newest - dv
        hover = format_hover(event, years_since)
        geometry = build_geojson_polygon(latitudes, longitudes)
        if not geometry:
            continue
        feature_id = f"{event.year}_{event.month:02d}_{event.day:02d}_{idx}"
        features.append(
            {
                "type": "Feature",
                "id": feature_id,
                "properties": {"hover": hover},
                "geometry": geometry,
            }
        )
        feature_ids.append(feature_id)
        feature_values.append(years_since)
        hover_texts.append(hover)

        if idx % 100 == 0 or idx == len(events):
            print(f"Processed {idx}/{len(events)} eclipses (skipped {skipped})")

    if not features:
        raise SystemExit("No valid eclipse polygons to plot after filtering.")

    figure = go.Figure()
    figure.add_trace(
        go.Choropleth(
            geojson={"type": "FeatureCollection", "features": features},
            featureidkey="id",
            locations=feature_ids,
            z=feature_values,
            zmin=0,
            zmax=span,
            colorscale=args.colorscale,
            showscale=True,
            colorbar=dict(title="Years since last eclipse"),
            hoverinfo="text",
            text=hover_texts,
            marker=dict(line=dict(color="rgba(0, 0, 0, 0)", width=0)),
        )
    )
    add_world_outline_trace(figure)
    title_lines = [
        "Time Since Moon Shadow Coverage",
        "<sup>Brighter colors = longer since the last eclipse at that location</sup>",
    ]
    if cutoff_label:
        title_lines.append(f"<sup>Catalog through {cutoff_label}</sup>")
    if years_window_label:
        title_lines.append(f"<sup>{years_window_label}</sup>")
    figure.update_layout(
        title="<br>".join(title_lines),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    apply_geo_styling(figure)
    if not args.no_html:
        figure.write_html(args.output, include_plotlyjs="inline")
        print(f"Wrote {args.output.resolve()}")
    if args.image_output:
        export_static_image(figure, args.image_output, scale=args.image_scale)


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


_COUNTRY_CODES_CACHE: List[str] | None = None


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


def date_sort_key(event: CatalogEntry) -> float:
    return event.year * 10000 + event.month * 100 + event.day


def date_to_float(event: CatalogEntry) -> float:
    # Rough fractional year, sufficient for ordering and color scaling.
    return date_parts_to_float(event.year, event.month, event.day)


def format_hover(event: CatalogEntry, years_since: float) -> str:
    date_label = format_event_date(event)
    return (
        f"{date_label} ({event.eclipse_type})<br>"
        f"Years since: {years_since:,.1f}<br>"
        f"Path width: {event.path_width_km:.0f} km"
    )


def format_event_date(event: CatalogEntry) -> str:
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


if __name__ == "__main__":
    main()
