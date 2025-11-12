from __future__ import annotations

import argparse
import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import plotly.graph_objects as go

from plot_writer import write_interactive_html

CATALOG_FILE = Path("eclipse_besselian_from_mysqldump2.csv")
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1 / 298.257


@dataclass
class CatalogEntry:
    year: int
    month: int
    day: int
    saros: int | None
    eclipse_type: str
    td_ge: str
    td_ge_hours: float
    delta_t_seconds: float
    t0: float
    tmin: float
    tmax: float
    path_width_km: float
    sun_alt: float
    sun_azm: float
    greatest_lat: float
    greatest_lon: float
    coeffs: Dict[str, List[float]]


@dataclass
class ShadowPoint:
    lat: float
    lon: float
    width_km: float
    tt_hours: float
    ut_hours: float

    @property
    def ut_label(self) -> str:
        return format_hours(self.ut_hours, label="UT")

    @property
    def tt_label(self) -> str:
        return format_hours(self.tt_hours, label="TT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the shadow path of a solar eclipse on the requested date.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "date",
        help="Calendar date in YYYY-MM-DD (use a leading '-' for BCE years)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_FILE,
        help="Location of the eclipse Besselian elements CSV",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=400,
        help="Number of samples between tmin and tmax used for the track",
    )
    parser.add_argument(
        "--centerline-step",
        type=int,
        default=1,
        help="Keep every Nth point along the centerline when rendering (use >1 to shrink output size).",
    )
    parser.add_argument(
        "--polygon-step",
        type=int,
        default=1,
        help="Keep every Nth vertex along the corridor boundary to reduce polygon size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination HTML file for the interactive map",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=None,
        help="Optional static image (png/svg/pdf) destination. Requires kaleido.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML export (useful when only --image-output is needed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_html and not args.image_output:
        raise SystemExit(
            "HTML output disabled but no --image-output provided. "
            "Either drop --no-html or supply an image path."
        )
    if args.centerline_step <= 0:
        raise SystemExit("--centerline-step must be a positive integer.")
    if args.polygon_step <= 0:
        raise SystemExit("--polygon-step must be a positive integer.")

    catalog = load_catalog(args.catalog)
    target_key = normalize_date_key(args.date)
    if target_key not in catalog:
        raise SystemExit(
            f"No eclipse found on {args.date}. "
            "Check Key.txt for valid date ranges (−1999 through +3000)."
        )

    event = catalog[target_key]
    centerline = compute_centerline(event, args.samples)
    if args.centerline_step > 1:
        centerline = centerline[:: args.centerline_step]
    if len(centerline) < 2:
        raise SystemExit(
            "Not enough valid samples to form a track. "
            "This usually means the eclipse is purely partial."
        )

    shadow_polygon = build_shadow_polygon(centerline)
    shadow_polygon = simplify_polygon(shadow_polygon, args.polygon_step)
    shadow_polygon = quantize_polygon(shadow_polygon, decimals=4)
    title = build_title(event)

    fig = render_map(centerline, shadow_polygon, title, event)
    html_destination = args.output or Path(
        f"eclipse_shadow_{args.date.replace('-', '_')}.html"
    )
    if not args.no_html:
        write_interactive_html(
            fig,
            html_destination,
            title=title,
        )
        print(f"Wrote {html_destination.resolve()} (JSON saved beside HTML)")
    if args.image_output:
        export_static_image(fig, args.image_output)


def load_catalog(path: Path) -> Dict[str, CatalogEntry]:
    if not path.exists():
        raise SystemExit(f"Catalog not found: {path}")

    catalog: Dict[str, CatalogEntry] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entry = convert_row(row)
            key = f"{entry.year}-{entry.month:02d}-{entry.day:02d}"
            catalog[key] = entry
    return catalog


def convert_row(raw: Dict[str, str]) -> CatalogEntry:
    td_ge = raw["td_ge"].strip('"')
    td_hours = hms_to_hours(td_ge)
    saros_raw = (raw.get("saros") or "").strip()
    saros_value = int(float(saros_raw)) if saros_raw else None
    coeffs = {
        "x": [float(raw[f"x{i}"]) for i in range(4)],
        "y": [float(raw[f"y{i}"]) for i in range(4)],
        "d": [float(raw[f"d{i}"]) for i in range(3)],
        "mu": [float(raw[f"mu{i}"]) for i in range(3)],
        "l1": [float(raw[f"l1{i}"]) for i in range(3)],
        "l2": [float(raw[f"l2{i}"]) for i in range(3)],
    }

    return CatalogEntry(
        year=int(raw["year"]),
        month=int(raw["month"]),
        day=int(raw["day"]),
        saros=saros_value,
        eclipse_type=raw["eclipse_type"],
        td_ge=td_ge,
        td_ge_hours=td_hours,
        delta_t_seconds=float(raw["dt"]),
        t0=float(raw["t0"]),
        tmin=float(raw["tmin"]),
        tmax=float(raw["tmax"]),
        path_width_km=float(raw.get("path_width") or 0.0),
        sun_alt=float(raw.get("sun_alt") or 0.0),
        sun_azm=float(raw.get("sun_azm") or 0.0),
        greatest_lat=float(raw.get("lat_dd_ge") or 0.0),
        greatest_lon=float(raw.get("lng_dd_ge") or 0.0),
        coeffs=coeffs,
    )


def compute_centerline(event: CatalogEntry, samples: int) -> List[ShadowPoint]:
    if samples < 2:
        raise SystemExit("Need at least two samples for the path.")

    t_values = interpolate_times(event.tmin, event.tmax, samples)
    width_scale = calibrate_width_scale(event)

    points: List[ShadowPoint] = []
    for t in t_values:
        x = eval_poly(event.coeffs["x"], t)
        y = eval_poly(event.coeffs["y"], t)
        rho_sq = 1 - (x * x + y * y)
        if rho_sq < 0:
            continue  # axis misses Earth at this instant

        d = math.radians(eval_poly(event.coeffs["d"], t))
        mu = math.radians(eval_poly(event.coeffs["mu"], t))
        z = math.sqrt(max(rho_sq, 0.0))
        sin_phi = y * math.cos(d) + z * math.sin(d)
        if abs(sin_phi) > 1:
            continue
        phi_gc = math.asin(sin_phi)
        cos_phi = max(math.cos(phi_gc), 1e-9)

        sinH = x / cos_phi
        # Guard rounding error
        sinH = max(min(sinH, 1.0), -1.0)
        cosH = (z - math.sin(phi_gc) * math.sin(d)) / (cos_phi * math.cos(d))
        H = math.atan2(sinH, cosH)
        lon_east = wrap_radians(H - mu)

        lat_deg = geocentric_to_geodetic_degrees(phi_gc)
        lon_deg = math.degrees(lon_east)

        tt_hours = event.t0 + t
        ut_hours = tt_hours - event.delta_t_seconds / 3600.0
        width_km = width_scale * approximate_width_km(event, t, x, y, z)

        points.append(
            ShadowPoint(
                lat=lat_deg,
                lon=lon_deg,
                width_km=width_km,
                tt_hours=tt_hours,
                ut_hours=ut_hours,
            )
        )
    return points


def calibrate_width_scale(event: CatalogEntry) -> float:
    width_catalog = event.path_width_km
    if width_catalog <= 0:
        return 1.0

    t_ge = event.td_ge_hours - event.t0
    x = eval_poly(event.coeffs["x"], t_ge)
    y = eval_poly(event.coeffs["y"], t_ge)
    rho_sq = 1 - (x * x + y * y)
    if rho_sq <= 0:
        return 1.0
    z = math.sqrt(rho_sq)
    approx = approximate_width_km(event, t_ge, x, y, z)
    if approx <= 0:
        return 1.0
    return width_catalog / approx


def approximate_width_km(
    event: CatalogEntry, t: float, x: float, y: float, z: float
) -> float:
    l2 = abs(eval_poly(event.coeffs["l2"], t))
    if l2 <= 1e-6:
        return 0.0
    return 2 * EARTH_RADIUS_KM * l2


def build_shadow_polygon(
    centerline: List[ShadowPoint],
) -> Tuple[List[float], List[float]]:
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []

    for idx, point in enumerate(centerline):
        if point.width_km <= 0:
            continue
        bearing = select_bearing(centerline, idx)
        if bearing is None:
            continue
        half_width = point.width_km / 2
        left_pt = destination_point(point.lat, point.lon, bearing - math.pi / 2, half_width)
        right_pt = destination_point(point.lat, point.lon, bearing + math.pi / 2, half_width)
        left.append(left_pt)
        right.append(right_pt)

    if len(left) < 3 or len(right) < 3:
        return [], []

    lons = [pt[1] for pt in left] + [pt[1] for pt in reversed(right)]
    lats = [pt[0] for pt in left] + [pt[0] for pt in reversed(right)]
    return lats, lons


def select_bearing(points: List[ShadowPoint], idx: int) -> float | None:
    if len(points) < 2:
        return None

    if idx == 0:
        next_pt = points[idx + 1]
        curr = points[idx]
        return bearing_between(curr.lat, curr.lon, next_pt.lat, next_pt.lon)
    if idx == len(points) - 1:
        prev = points[idx - 1]
        curr = points[idx]
        return bearing_between(prev.lat, prev.lon, curr.lat, curr.lon)

    prev_pt = points[idx - 1]
    next_pt = points[idx + 1]
    b1 = bearing_between(prev_pt.lat, prev_pt.lon, points[idx].lat, points[idx].lon)
    b2 = bearing_between(points[idx].lat, points[idx].lon, next_pt.lat, next_pt.lon)
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    # Average bearings via vector sum to avoid wrap issues
    x = math.cos(b1) + math.cos(b2)
    y = math.sin(b1) + math.sin(b2)
    if x == 0 and y == 0:
        return b1
    return math.atan2(y, x)


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float | None:
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    if abs(lat1_r - lat2_r) < 1e-9 and abs(dlon) < 1e-9:
        return None
    y = math.sin(dlon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(
        dlon
    )
    return math.atan2(y, x)


def destination_point(lat: float, lon: float, bearing: float, distance_km: float) -> Tuple[float, float]:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    delta = distance_km / EARTH_RADIUS_KM
    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    sin_lat2 = sin_lat * cos_delta + cos_lat * sin_delta * math.cos(bearing)
    lat2 = math.asin(max(min(sin_lat2, 1.0), -1.0))
    y = math.sin(bearing) * sin_delta * cos_lat
    x = cos_delta - sin_lat * math.sin(lat2)
    lon2 = lon_r + math.atan2(y, x)
    return math.degrees(lat2), math.degrees(normalize_longitude(lon2))


def render_map(
    centerline: List[ShadowPoint],
    polygon: Tuple[List[float], List[float]],
    title: str,
    event: CatalogEntry,
) -> go.Figure:
    fig = go.Figure()
    line_lat = [round(pt.lat, 4) for pt in centerline]
    line_lon = [round(pt.lon, 4) for pt in centerline]

    if polygon[0]:
        fig.add_trace(
            go.Scattergeo(
                lat=polygon[0],
                lon=polygon[1],
                fill="toself",
                fillcolor="rgba(255, 99, 71, 0.25)",
                line=dict(color="rgba(0,0,0,0)", width=0),
                hoverinfo="skip",
                name="Approx. totality path",
            )
        )

    hover_text = [
        f"{pt.ut_label}<br>Width ~ {pt.width_km:0.1f} km<br>TT {pt.tt_label}"
        for pt in centerline
    ]
    fig.add_trace(
        go.Scattergeo(
            lat=line_lat,
            lon=line_lon,
            mode="lines",
            line=dict(color="crimson", width=2),
            name="Center line",
            hovertext=hover_text,
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[centerline[len(centerline) // 2].lat],
            lon=[centerline[len(centerline) // 2].lon],
            mode="markers",
            marker=dict(color="gold", size=8),
            name="Sample point",
            hoverinfo="skip",
        )
    )

    summary = textwrap.dedent(
        f"""
        Type: {event.eclipse_type} | Sun Alt: {event.sun_alt:.1f}°
        Greatest eclipse: lat {event.greatest_lat:.2f}°, lon {event.greatest_lon:.2f}°
        Catalog path width: {event.path_width_km:.1f} km | ΔT = {event.delta_t_seconds:.1f}s
        """
    ).strip()

    fig.update_layout(
        title=f"{title}<br><sup>{summary}</sup>",
        legend=dict(bgcolor="rgba(255,255,255,0.7)"),
    )
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="rgb(240, 240, 240)",
        oceancolor="rgb(215, 230, 245)",
        showocean=True,
        lataxis=dict(showgrid=True, dtick=30),
        lonaxis=dict(showgrid=True, dtick=30),
    )
    return fig


def simplify_polygon(
    polygon: Tuple[List[float], List[float]], step: int
) -> Tuple[List[float], List[float]]:
    latitudes, longitudes = polygon
    if step <= 1 or not latitudes or len(latitudes) != len(longitudes):
        return polygon
    if len(latitudes) < 4:
        return polygon
    indices = list(range(0, len(latitudes), step))
    if indices[-1] != len(latitudes) - 1:
        indices.append(len(latitudes) - 1)
    lat_filtered = [latitudes[i] for i in indices]
    lon_filtered = [longitudes[i] for i in indices]
    if len(lat_filtered) < 3:
        return polygon
    return lat_filtered, lon_filtered


def quantize_polygon(
    polygon: Tuple[List[float], List[float]], decimals: int = 4
) -> Tuple[List[float], List[float]]:
    latitudes, longitudes = polygon
    if decimals < 0 or not latitudes:
        return polygon
    return (
        [round(value, decimals) for value in latitudes],
        [round(value, decimals) for value in longitudes],
    )


def build_title(event: CatalogEntry) -> str:
    return f"{abs(event.year):04d}{' BCE' if event.year < 0 else ''} {event.month:02d}-{event.day:02d}"


def interpolate_times(start: float, stop: float, samples: int) -> List[float]:
    if samples == 1:
        return [start]
    step = (stop - start) / (samples - 1)
    return [start + i * step for i in range(samples)]


def eval_poly(coeffs: Iterable[float], t: float) -> float:
    total = 0.0
    power = 1.0
    for coeff in coeffs:
        total += coeff * power
        power *= t
    return total


def normalize_date_key(raw_date: str) -> str:
    if raw_date.startswith("-"):
        parts = raw_date[1:].split("-")
        if len(parts) != 3:
            raise SystemExit(f"Invalid date format: {raw_date}")
        year = -int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    else:
        parts = raw_date.split("-")
        if len(parts) != 3:
            raise SystemExit(f"Invalid date format: {raw_date}")
        year, month, day = map(int, parts)
    return f"{year}-{month:02d}-{day:02d}"


def geocentric_to_geodetic_degrees(phi_gc: float) -> float:
    sin_phi = math.sin(phi_gc)
    cos_phi = math.cos(phi_gc)
    numerator = sin_phi
    denominator = (1 - EARTH_FLATTENING) ** 2 * cos_phi
    return math.degrees(math.atan2(numerator, denominator))


def wrap_radians(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def normalize_longitude(lon_radians: float) -> float:
    return wrap_radians(lon_radians)


def hms_to_hours(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise SystemExit(f"Unexpected time format: {value}")
    hours, minutes, seconds = map(int, parts)
    return hours + minutes / 60 + seconds / 3600


def format_hours(raw_hours: float, *, label: str) -> str:
    day_offset = math.floor(raw_hours / 24)
    hours = raw_hours - 24 * day_offset
    if hours < 0:
        hours += 24
        day_offset -= 1
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(round((hours - h - m / 60) * 3600))
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        h = (h + 1) % 24
        if h == 0:
            day_offset += 1

    offset_label = ""
    if day_offset > 0:
        offset_label = f" (+{day_offset}d)"
    elif day_offset < 0:
        offset_label = f" ({day_offset}d)"
    return f"{h:02d}:{m:02d}:{s:02d} {label}{offset_label}"


def export_static_image(
    fig: go.Figure, destination: Path, *, scale: float | None = None
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        kwargs = {}
        if scale is not None:
            kwargs["scale"] = scale
        fig.write_image(str(destination), **kwargs)
    except ValueError as exc:  # Missing kaleido dependency.
        raise SystemExit(
            "Plotly static image export requires the 'kaleido' package. "
            "Install it via 'pip install kaleido' and retry."
        ) from exc
    print(f"Wrote {destination.resolve()}")


if __name__ == "__main__":
    main()
