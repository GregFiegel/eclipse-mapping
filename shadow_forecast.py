from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import plotly.graph_objects as go

from shadow_mapper import (
    CATALOG_FILE,
    CatalogEntry,
    build_shadow_polygon,
    compute_centerline,
    export_static_image,
    load_catalog,
    normalize_date_key,
)
from shadow_map_utils import (
    add_world_outline_trace,
    allow_eclipse_type,
    apply_geo_styling,
    build_geojson_polygon,
    build_power_colorbar_ticks,
    date_key_to_sort_value,
    date_sort_key,
    date_to_float,
    decimate_polygon,
    format_event_date,
    resolve_future_start,
    scale_years,
)

TRANSPARENT_SCALE = [[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a global map showing how long every location must wait "
            "until its next visit from the Moon's shadow."
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
        help="Optional cap on the number of eclipses processed (farthest future first).",
    )
    parser.add_argument(
        "--colorscale",
        default="Viridis",
        help="Plotly colorscale name used for the wait-time gradient.",
    )
    parser.add_argument(
        "--color-exponent",
        type=float,
        default=0.5,
        help=(
            "Exponent applied to years-until values for color scaling. "
            "Values between 0 and 1 compress far-future eclipses, boosting near-term ones."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eclipse_shadow_forecast.html"),
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
        "--min-date",
        type=str,
        default=None,
        help="Earliest calendar date to consider (default: today). Prefix BCE years with '-'.",
    )
    parser.add_argument(
        "--max-date",
        type=str,
        default=None,
        help="Optional latest date to include (prefix BCE years with '-').",
    )
    parser.add_argument(
        "--years-forward",
        type=float,
        default=None,
        help=(
            "Only render eclipses within this many years after the start date "
            "(defaults to unlimited)."
        ),
    )
    parser.add_argument(
        "--include-annular",
        action="store_true",
        help="Include annular eclipses (default renders only total/hybrid events).",
    )
    parser.add_argument(
        "--polygon-step",
        type=int,
        default=1,
        help=(
            "Keep every Nth vertex when building polygons (set >1 to down-sample "
            "GeoJSON and shrink HTML file size)."
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
    if args.color_exponent <= 0:
        raise SystemExit("--color-exponent must be positive.")
    if args.years_forward is not None and args.years_forward <= 0:
        raise SystemExit("--years-forward must be positive.")
    if args.polygon_step <= 0:
        raise SystemExit("--polygon-step must be a positive integer.")

    if args.outline_only:
        outline_only(args)
        return

    catalog = load_catalog(args.catalog)
    events = sorted(catalog.values(), key=date_sort_key)
    if not events:
        raise SystemExit("Catalog did not yield any eclipses to plot.")
    events = [evt for evt in events if allow_eclipse_type(evt, args.include_annular)]
    if not events:
        raise SystemExit("No total eclipses available; pass --include-annular to include annular tracks.")

    start_value, start_label, start_float = resolve_future_start(args.min_date)
    events = [evt for evt in events if date_sort_key(evt) >= start_value]
    if not events:
        label = args.min_date or "today"
        raise SystemExit(f"No eclipses occur on or after {label}.")

    end_label = None
    if args.max_date:
        normalized_end = normalize_date_key(args.max_date)
        end_value = date_key_to_sort_value(normalized_end)
        if end_value < start_value:
            raise SystemExit("--max-date must be on or after the start date.")
        end_label = normalized_end
        events = [evt for evt in events if date_sort_key(evt) <= end_value]
        if not events:
            raise SystemExit(
                f"No eclipses occur between {start_label} and {end_label}."
            )

    years_window_label = None
    if args.years_forward is not None:
        max_float = start_float + args.years_forward
        events = [evt for evt in events if date_to_float(evt) <= max_float]
        if not events:
            raise SystemExit(
                f"No eclipses occur within the next {args.years_forward:g} years "
                f"starting {start_label}."
            )
        years_window_label = f"Next {args.years_forward:g} years (starting {start_label})"
        print(f"Restricting to the {years_window_label}.")

    # Draw distant eclipses first so near-term ones render on top.
    events.sort(key=date_sort_key, reverse=True)
    if args.max_events:
        events = events[: args.max_events]
    if not events:
        raise SystemExit("Catalog did not yield any eclipses to plot in the requested window.")

    date_values = [date_to_float(evt) for evt in events]
    furthest = max(date_values)
    max_years_ahead = max(furthest - start_float, 0.0)

    records: List[dict] = []
    skipped = 0

    for idx, (event, dv) in enumerate(zip(events, date_values), start=1):
        centerline = compute_centerline(event, args.samples)
        polygon = build_shadow_polygon(centerline)
        if not polygon[0]:
            skipped += 1
            continue
        latitudes, longitudes = polygon
        if args.polygon_step > 1:
            latitudes, longitudes = decimate_polygon(
                latitudes, longitudes, args.polygon_step
            )
        if len(latitudes) < 3:
            skipped += 1
            continue

        years_until = max(dv - start_float, 0.0)
        hover = format_forecast_hover(event, years_until)
        geometry = build_geojson_polygon(latitudes, longitudes)
        if not geometry:
            continue
        feature_id = f"{event.year}_{event.month:02d}_{event.day:02d}_{idx}"
        records.append(
            {
                "feature": {
                    "type": "Feature",
                    "id": feature_id,
                    "properties": {"hover": hover},
                    "geometry": geometry,
                },
                "id": feature_id,
                "value": years_until,
                "hover": hover,
            }
        )

        if idx % 100 == 0 or idx == len(events):
            print(f"Processed {idx}/{len(events)} eclipses (skipped {skipped})")

    if not records:
        raise SystemExit("No valid eclipse polygons to plot after filtering.")

    scaled_values = [scale_years(record["value"], args.color_exponent) for record in records]
    color_min = min(scaled_values)
    color_max = max(scaled_values)
    if color_max == color_min:
        color_max = color_min + 1e-9
    max_years = max(record["value"] for record in records) if records else 0.0
    tickvals, ticktext = build_power_colorbar_ticks(max_years, args.color_exponent)
    exponent_label = f"{args.color_exponent:g}"
    for record, scaled in zip(records, scaled_values):
        record["scaled"] = scaled

    color_records = sorted(records, key=lambda rec: rec["value"], reverse=True)
    hover_records = sorted(records, key=lambda rec: rec["value"])

    figure = go.Figure()
    figure.add_trace(
        go.Choropleth(
            geojson={
                "type": "FeatureCollection",
                "features": [rec["feature"] for rec in color_records],
            },
            featureidkey="id",
            locations=[rec["id"] for rec in color_records],
            z=[rec["scaled"] for rec in color_records],
            zmin=color_min,
            zmax=color_max,
            colorscale=args.colorscale,
            showscale=True,
            colorbar=dict(
                title=f"Years until next eclipse (exp {exponent_label})",
                tickvals=tickvals,
                ticktext=ticktext,
            ),
            hoverinfo="skip",
            marker=dict(line=dict(color="rgba(0, 0, 0, 0)", width=0)),
        )
    )
    figure.add_trace(
        go.Choropleth(
            geojson={
                "type": "FeatureCollection",
                "features": [rec["feature"] for rec in hover_records],
            },
            featureidkey="id",
            locations=[rec["id"] for rec in hover_records],
            z=[rec["scaled"] for rec in hover_records],
            zmin=color_min,
            zmax=color_max,
            colorscale=TRANSPARENT_SCALE,
            showscale=False,
            hoverinfo="text",
            text=[rec["hover"] for rec in hover_records],
            marker=dict(line=dict(color="rgba(0, 0, 0, 0)", width=0)),
        )
    )
    add_world_outline_trace(figure)

    title_lines = [
        "Time Until Next Total Solar Eclipse",
        "<sup>Darker colors = sooner eclipses for that location</sup>",
        f"<sup>Power color scale (exp {exponent_label}) highlights near-term coverage</sup>",
        f"<sup>Forecast window starts {start_label}</sup>",
    ]
    if end_label:
        title_lines.append(f"<sup>Forecast capped at {end_label}</sup>")
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


def outline_only(args: argparse.Namespace) -> None:
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


def format_forecast_hover(event: CatalogEntry, years_until: float) -> str:
    date_label = format_event_date(event)
    return (
        f"{date_label} ({event.eclipse_type})<br>"
        f"Years until: {years_until:,.1f}<br>"
        f"Path width: {event.path_width_km:.0f} km"
    )


if __name__ == "__main__":
    main()
