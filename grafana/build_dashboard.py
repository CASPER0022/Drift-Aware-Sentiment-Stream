"""Generate the provisioned Grafana dashboard (grafana/provisioning/dashboards/drift_stream.json).

The JSON is the artifact Grafana loads; this script is its readable source. Rerun it after
changing a panel:  python grafana/build_dashboard.py

Panels (all filtered by the selected producer run):
  accuracy (5-batch moving average) and cumulative accuracy per model, macro F1, lambda,
  model size, pipeline throughput, latency (mean / p95), driver memory
Annotations: drift changes and rebuilds, vocabulary detector in orange, ADWIN in blue.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "provisioning" / "dashboards" / "drift_stream.json"
DS = {"type": "influxdb", "uid": "influxdb-drift"}
BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
MODEL_COLORS = {"enhanced": BLUE, "baseline": ORANGE, "accumulative": GRAY, "fading": AQUA}

RANGE = 'from(bucket: "drift")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n'


def model_query(field: str, smooth: int = 1) -> str:
    """One series per model, named after the model."""
    q = (RANGE + f'  |> filter(fn: (r) => r._measurement == "model_metrics" and r.run == "${{run}}" '
         f'and r._field == "{field}")\n'
         '  |> map(fn: (r) => ({_time: r._time, _value: r._value, _field: r.model}))\n'
         '  |> group(columns: ["_field"])\n  |> sort(columns: ["_time"])\n')
    if smooth > 1:
        q += f"  |> movingAverage(n: {smooth})\n"
    return q


def pipeline_query(*fields: str) -> str:
    cond = " or ".join(f'r._field == "{f}"' for f in fields)
    return (RANGE + f'  |> filter(fn: (r) => r._measurement == "pipeline_metrics" and r.run == "${{run}}")\n'
            f"  |> filter(fn: (r) => {cond})\n"
            '  |> keep(columns: ["_time", "_value", "_field"])\n')


def color_overrides(names_colors: dict) -> list:
    return [{"matcher": {"id": "byName", "options": name},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
            for name, color in names_colors.items()]


def panel(pid, title, query, x, y, w, h, unit=None, minmax=(None, None), overrides=None,
          description="", draw="line", display=None, soft_max=None):
    defaults = {"custom": {"lineWidth": 2, "fillOpacity": 0, "showPoints": "never",
                           "drawStyle": draw, "lineInterpolation": "stepAfter" if draw == "step" else "linear"}}
    if draw == "step":
        defaults["custom"]["drawStyle"] = "line"
    if unit:
        defaults["unit"] = unit
    if minmax[0] is not None:
        defaults["min"] = minmax[0]
    if minmax[1] is not None:
        defaults["max"] = minmax[1]
    if display:
        defaults["displayName"] = display
    if soft_max is not None:  # keeps an all-zero series from auto-scaling to 0..100
        defaults["custom"]["axisSoftMax"] = soft_max
    return {"id": pid, "type": "timeseries", "title": title, "description": description,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "datasource": DS,
            "targets": [{"refId": "A", "datasource": DS, "query": query}],
            "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                        "tooltip": {"mode": "multi", "sort": "none"}}}


def annotation(name, detector, color):
    query = (RANGE + f'  |> filter(fn: (r) => r._measurement == "drift_event" and r.run == "${{run}}" '
             f'and r.detector == "{detector}" and r._field == "text")\n'
             '  |> filter(fn: (r) => r.kind == "change" and r.model == "${annotate_model}")\n'
             '  |> map(fn: (r) => ({_time: r._time, text: r._value}))\n'
             '  |> group()\n')
    return {"name": name, "datasource": DS, "enable": True, "iconColor": color,
            "target": {"refId": "Anno", "query": query}}


def build() -> dict:
    model_overrides = color_overrides(MODEL_COLORS)
    panels = [
        panel(1, "Accuracy per micro-batch (5-batch moving average)", model_query("accuracy", 5),
              0, 0, 16, 9, "percentunit", (0, 1), model_overrides,
              "Prequential accuracy: each tweet is predicted before the model learns it. "
              "Annotations mark detected drift (orange: vocabulary detector, blue: ADWIN)."),
        panel(2, "Cumulative accuracy", model_query("accuracy_cum"), 16, 0, 8, 9,
              "percentunit", (0, 1), model_overrides),
        panel(3, "Macro F1 per micro-batch (5-batch moving average)", model_query("f1_macro", 5),
              0, 9, 8, 7, "percentunit", (0, 1), model_overrides),
        panel(4, "Ageing factor lambda", model_query("lambda"), 8, 9, 8, 7, None, (0, None),
              model_overrides, "Rebuild strategies keep lambda at lambda0; rebuilds show in model size.",
              draw="step", soft_max=0.6),
        panel(5, "Model size ((word, class) entries)", model_query("model_entries"), 16, 9, 8, 7,
              "short", (0, None), model_overrides, "Drops mark rebuilds after a detected drift."),
        panel(6, "Throughput (tweets/s)", pipeline_query("throughput"), 0, 16, 8, 7,
              "short", (0, None), display="tweets per second",
              overrides=color_overrides({"throughput": BLUE})),
        panel(7, "End-to-end latency, producer to models", pipeline_query("latency_ms", "latency_p95_ms"),
              8, 16, 8, 7, "ms", (0, None),
              [{"matcher": {"id": "byName", "options": "latency_ms"},
                "properties": [{"id": "displayName", "value": "mean"},
                               {"id": "color", "value": {"mode": "fixed", "fixedColor": BLUE}}]},
               {"matcher": {"id": "byName", "options": "latency_p95_ms"},
                "properties": [{"id": "displayName", "value": "p95"},
                               {"id": "color", "value": {"mode": "fixed", "fixedColor": ORANGE}}]}]),
        panel(8, "Spark driver memory (RSS)", pipeline_query("driver_rss_mb"), 16, 16, 8, 7,
              "decmbytes", (0, None), display="driver RSS",
              overrides=color_overrides({"driver_rss_mb": BLUE})),
    ]
    run_var = {
        "name": "run", "label": "Run", "type": "query", "datasource": DS, "refresh": 2,
        "query": 'import "influxdata/influxdb/schema"\n'
                 'schema.tagValues(bucket: "drift", tag: "run", start: -30d)\n  |> sort(desc: true)',
        "includeAll": False, "multi": False,
    }
    anno_model_var = {
        "name": "annotate_model", "label": "Drift markers of", "type": "custom",
        "query": "enhanced,baseline", "current": {"text": "enhanced", "value": "enhanced"},
        "options": [{"text": m, "value": m, "selected": m == "enhanced"} for m in ("enhanced", "baseline")],
    }
    return {
        "uid": "drift-stream", "title": "Drift Stream", "tags": ["drift"], "timezone": "browser",
        "schemaVersion": 39, "version": 2, "editable": True, "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "templating": {"list": [run_var, anno_model_var]},
        "annotations": {"list": [annotation("Vocabulary detector", "vocab", ORANGE),
                                 annotation("ADWIN", "adwin", BLUE)]},
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")
