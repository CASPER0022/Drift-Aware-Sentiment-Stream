"""Screenshot the Grafana dashboard for one producer run (for the report).

Drives headless Microsoft Edge (or Chrome) over the DevTools protocol: logs in with HTTP
basic auth, opens the dashboard in kiosk mode with the time range zoomed to the run, and
saves a PNG. Credentials come from GRAFANA_USER / GRAFANA_PASSWORD (never stored).

Usage:
  GRAFANA_PASSWORD=... python analysis/grafana_screenshot.py --run <run id> \
      --out analysis/figures/grafana_label_flip.png [--annotate-model enhanced]
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from influxdb_client import InfluxDBClient

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9333


def run_span_ms(run: str, pad_s: int = 20) -> tuple[int, int]:
    with InfluxDBClient(url="http://localhost:8086", token=os.getenv("INFLUX_TOKEN", "dev-token-group18"),
                        org=os.getenv("INFLUX_ORG", "group18")) as client:
        flux = (f'from(bucket: "drift") |> range(start: -30d) '
                f'|> filter(fn: (r) => r._measurement == "pipeline_metrics" and r.run == "{run}" '
                f'and r._field == "seen") |> keep(columns: ["_time"])')
        times = [rec.get_time() for table in client.query_api().query(flux) for rec in table.records]
    if not times:
        raise SystemExit(f"no data for run {run!r}")
    return (int(min(times).timestamp() - pad_s) * 1000, int(max(times).timestamp() + pad_s) * 1000)


class Cdp:
    def __init__(self, ws_url: str) -> None:
        # suppress_origin: DevTools rejects unknown Origin headers; sending none is accepted.
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.next_id = 0

    def __call__(self, method: str, **params):
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotate-model", default="enhanced", choices=["enhanced", "baseline"])
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1060)
    ap.add_argument("--wait", type=float, default=12, help="seconds to let panels render")
    args = ap.parse_args()

    password = os.environ.get("GRAFANA_PASSWORD")
    if not password:
        raise SystemExit("set GRAFANA_PASSWORD (and GRAFANA_USER if not admin)")
    auth = base64.b64encode(f"{os.getenv('GRAFANA_USER', 'admin')}:{password}".encode()).decode()
    start, end = run_span_ms(args.run)
    url = (f"http://localhost:3000/d/drift-stream/drift-stream?orgId=1&var-run={args.run}"
           f"&var-annotate_model={args.annotate_model}&from={start}&to={end}&kiosk&theme=light")

    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        raise SystemExit("no Edge or Chrome found")
    profile = tempfile.mkdtemp(prefix="grafana-shot-")
    proc = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={profile}", "--no-first-run", "--disable-gpu",
                             f"--window-size={args.width},{args.height}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):  # wait for the DevTools endpoint
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(p for p in pages if p["type"] == "page")
                break
            except (OSError, StopIteration):
                time.sleep(0.2)
        else:
            raise SystemExit("browser did not start")
        cdp = Cdp(page["webSocketDebuggerUrl"])
        cdp("Network.enable")
        cdp("Network.setExtraHTTPHeaders", headers={"Authorization": f"Basic {auth}"})
        cdp("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
            deviceScaleFactor=1.5, mobile=False)
        cdp("Page.navigate", url=url)
        time.sleep(args.wait)
        shot = cdp("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(base64.b64decode(shot["data"]))
        print(f"wrote {args.out} ({args.run})")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
