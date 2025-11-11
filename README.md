# Eclipse Shadow Mapper

This repo contains a lightweight tool that reads NASA's Besselian elements
(`eclipse_besselian_from_mysqldump2.csv`) and plots the central shadow path for
any solar eclipse between −1999 and +3000.  The output is an interactive Plotly
map showing the center line plus an approximate totality/annularity corridor.

## Quick start

```sh
cd "/home/greg/Projects/Eclipse Mapping"
python -m venv .venv
. .venv/bin/activate
pip install plotly kaleido
python shadow_mapper.py 2024-04-08
```

An HTML file such as `eclipse_shadow_2024_04_08.html` will be written to the
current directory; open it with any browser.

## Command line reference

```
usage: shadow_mapper.py [-h] [--catalog CATALOG] [--samples SAMPLES]
                        [--output OUTPUT]
                        date

positional arguments:
  date                  Calendar date in YYYY-MM-DD (use a leading '-' for BCE years)

options:
  -h, --help            show this help message and exit
  --catalog CATALOG     Location of the eclipse Besselian elements CSV (default:
                        eclipse_besselian_from_mysqldump2.csv)
  --samples SAMPLES     Number of samples between tmin and tmax used for the track (default: 400)
  --output OUTPUT       Destination HTML file for the interactive map
```

### shadow_mapper.py option details

- `date` (positional): Calendar date in `YYYY-MM-DD`. Use a leading `-` for BCE, e.g. `-058-05-17`.
- `--catalog PATH`: Point to an alternate CSV containing Besselian elements.
- `--samples N`: Number of evaluation steps between `tmin` and `tmax`; higher values smooth the path but take longer.
- `--output FILE`: Explicit name for the HTML artifact (defaults to `eclipse_shadow_<date>.html`; ignored when `--no-html` is set).
- `--image-output FILE`: Write a static PNG/SVG/PDF image (requires the `kaleido` package).
- `--no-html`: Skip HTML export; combine with `--image-output` if you only want an image.

Examples:

```sh
# Total eclipse over North America
python shadow_mapper.py 2024-04-08 --samples 500

# Historic eclipse (BCE dates need a leading minus sign)
python shadow_mapper.py -099-05-15 --output ./ancient_eclipse.html
```

## How it works

* The tool parses the Besselian coefficients (x, y, d, μ, l₁, l₂) from the CSV.
* For each sample between `tmin` and `tmax` it solves the Besselian system to
  obtain the geocentric latitude/longitude of the shadow axis and converts them
  to geodetic coordinates.
* The umbral/antumbral corridor width is taken from `l₂` (scaled to match the
  catalogued path width at greatest eclipse) so the band keeps a realistic,
  consistent thickness along the track without flaring out at the ends.
* Left/right offsets are generated perpendicular to the instantaneous ground
  track and rendered as a filled polygon on a Plotly `Scattergeo` map.

This reproduces the catalog values for recent eclipses (e.g. 2017-08-21 and
2024-04-08) to within ~0.3° along the centerline.  The corridor width is an
approximation—the exact grazing limits would require solving the full Besselian
system with the w=0 constraint; the current approach is sufficient for visual
inspection but should not be used for precise local circumstances.

## Testing

Two quick sanity checks were run after implementation:

```sh
. .venv/bin/activate
python shadow_mapper.py 2017-08-21 --samples 300
python shadow_mapper.py 2024-04-08 --samples 400
```

Each command produced a map without runtime errors.

## Global coverage map

`shadow_history.py` now renders a single choropleth layer built from the union
of all eclipse polygons, which keeps file sizes manageable and guarantees a
consistent color scale. Each polygon is encoded in the GeoJSON payload so the
hover text still shows the individual event info, and a final country-outline
overlay is drawn to keep borders visible above the data. By default the script
ignores eclipses dated after the current day so the color scale reflects “years
since” relative to *now*; pass `--include-future` (or pick a custom
`--max-date`) if you want to push the end date farther out. Combine this with
`--years-back N` to focus on only the last `N` years relative to whatever end
date you chose. For fast, non-interactive deliverables favor `--image-output`
and increase `--image-scale` for higher resolution; use `--outline-only` if you
just need to verify the borders without plotting data.

```sh
. .venv/bin/activate
python shadow_history.py --samples 160 \
    --output eclipse_shadow_history.html
```

Expect a large HTML file and a lengthy runtime—the script loops through the
entire catalog (use `--max-events` while testing). The color bar shows the
number of years since a location last fell inside the Moon’s shadow according to
the dataset.

### shadow_history.py option details

- `--catalog PATH`: Alternate Besselian elements CSV (same format as the default).
- `--samples N`: Samples per eclipse track; increase for smoother polygons at the cost of runtime.
- `--max-events N`: Limit processing to the first `N` eclipses (oldest first) for testing.
- `--max-date YYYY-MM-DD`: Trim the catalog to eclipses on/before the supplied date (prefix BCE years with `-`). Supplying a future date intentionally extends the range.
- `--include-future`: Opt back into processing eclipses beyond today when no explicit `--max-date` is supplied (defaults to the current-day cutoff).
- `--years-back N`: Only include eclipses that occurred within the last `N` years relative to the effective end date (default current day unless you pass `--include-future` or `--max-date`).
- `--outline-only`: Emit just the base map outline (useful for confirming the overlay draws correctly).
- `--colorscale NAME`: Plotly colorscale to encode “years since” (e.g. `Viridis`, `Inferno`, `Turbo`).
- `--output FILE`: Destination HTML file for the combined map (default `eclipse_shadow_history.html`; ignored when `--no-html` is set).
- `--image-output FILE`: Write a static PNG/SVG/PDF snapshot (requires `kaleido`).
- `--image-scale N`: Multiply the static-image resolution by `N` (e.g., `2` doubles both width and height for a sharper export; defaults to `1.0`).
- `--no-html`: Suppress HTML output (useful when only emitting an image).
