from __future__ import annotations

import uuid
import shutil
from pathlib import Path
from typing import Optional

from plotly.graph_objs import Figure
from string import Template

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 1rem;
      background: #f6f6f6;
    }
    .plot-wrapper {
      min-height: 60vh;
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.12);
      padding: 0.5rem;
    }
    .placeholder {
      color: #666;
      text-align: center;
      margin-top: 2rem;
    }
  </style>
  <script defer src="${bundle}"></script>
  <script defer src="${geo_bundle}"></script>
</head>
<body>
  <main>
    <div class="plot-wrapper">
      <div id="${div_id}" class="plot" data-json="${json_name}">
        <p class="placeholder">Loading interactive map…</p>
      </div>
    </div>
  </main>
  <script>
    (function() {
      const container = document.getElementById("${div_id}");
      const jsonUrl = container.dataset.json;
      const waitForPlotly = () => new Promise((resolve) => {
        const check = () => {
          if (window.Plotly) {
            resolve();
            return;
          }
          setTimeout(check, 20);
        };
        check();
      });
      const waitForGeoAssets = () => new Promise((resolve, reject) => {
        const start = Date.now();
        const timeoutMs = 10000;
        const check = () => {
          if (window.PlotlyGeoAssets) {
            if (!window.__plotlyGeoRegistered) {
              try {
                window.Plotly.register(window.PlotlyGeoAssets);
                window.__plotlyGeoRegistered = true;
              } catch (err) {
                console.error("Failed to register Plotly geo assets", err);
                reject(err);
                return;
              }
            }
            resolve();
            return;
          }
          if (Date.now() - start > timeoutMs) {
            const err = new Error("Plotly geo assets were not found. Ensure the geo bundle is reachable.");
            console.error(err);
            reject(err);
            return;
          }
          setTimeout(check, 20);
        };
        check();
      });
      const loadPlot = async () => {
        if (container.dataset.loaded) return;
        container.dataset.loaded = "1";
        await waitForPlotly();
        await waitForGeoAssets();
        const response = await fetch(jsonUrl);
        if (!response.ok) throw new Error("Failed to load plot specification");
        const spec = await response.json();
        const frames = spec.frames || [];
        await Plotly.newPlot(container, spec.data, spec.layout, {
          responsive: true,
          displaylogo: false
        });
        if (frames.length) {
          Plotly.addFrames(container, frames);
        }
      };

      if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              observer.unobserve(entry.target);
              loadPlot();
            }
          });
        }, { rootMargin: "200px" });
        observer.observe(container);
      } else {
        loadPlot();
      }
    })();
  </script>
</body>
</html>
""")


def write_interactive_html(
    figure: Figure,
    html_path: Path,
    *,
    title: str,
    plotly_bundle: str = "plotly-2.27.0.min.js",
    geo_bundle: str = "plotly-geo-assets.min.js",
    json_path: Optional[Path] = None,
) -> Path:
    """Persist a Plotly figure as a JSON + lazy-loaded HTML pair."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = json_path or html_path.with_suffix(".json")
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_json(data_path, pretty=False)

    if _prepare_external_bundles(plotly_bundle, geo_bundle, html_path):
        div_id = f"plot_{uuid.uuid4().hex}"
        html_content = HTML_TEMPLATE.safe_substitute(
            title=title,
            bundle=Path(plotly_bundle).name,
            geo_bundle=Path(geo_bundle).name,
            div_id=div_id,
            json_name=data_path.name,
        )
        html_path.write_text(html_content, encoding="utf-8")
    else:
        figure.write_html(
            html_path,
            include_plotlyjs="inline",
            full_html=True,
            config={"responsive": True, "displaylogo": False},
        )
    return data_path


def _prepare_external_bundles(plotly_bundle: str, geo_bundle: str, html_path: Path) -> bool:
    bundle = Path(plotly_bundle)
    geo = Path(geo_bundle)
    for candidate in (bundle, geo):
        if not candidate.exists() or candidate.stat().st_size < 1024:
            return False
        head = candidate.read_bytes()[:128]
        if head.strip().startswith(b"<?xml"):
            return False

    for source in (bundle, geo):
        destination = html_path.parent / source.name
        try:
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        except FileNotFoundError:
            return False
    return True
