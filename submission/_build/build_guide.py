"""Build the video + run guide PDF (submission/video/Video_and_Run_Guide.pdf).

Writes an HTML page (slide thumbnails embedded) and prints it to PDF with headless Microsoft
Edge or Chrome. Narration comes from build_slides.NARRATION, so slides, speaker notes and the
guide always say the same thing.

Usage:
  python submission/_build/build_slides.py        # first (deck + notes)
  # export slide thumbnails once with PowerPoint (see README of this folder), then:
  python submission/_build/build_guide.py
"""
import base64
import html
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_slides import NARRATION  # noqa: E402

ROOT = HERE.parents[1]
THUMBS = HERE / "thumbs"
OUT_DIR = ROOT / "submission" / "video"
GRAFANA_SHOT = HERE / "grafana_demo.png"  # the tested demo replay (tweets 25k-50k at 1,000/s)
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

DEMO_AFTER_SLIDE = 9  # the "Live Demo" slide introduces the demo
GRAFANA_URL = ("http://localhost:3000/d/drift-stream/drift-stream?orgId=1"
               "&from=now-2m&to=now&refresh=5s")
CONSUMER = ("docker exec -it spark-master /opt/spark/bin/spark-submit --master "
            "spark://spark-master:7077 --conf spark.driver.host=spark-master --conf "
            "spark.executorEnv.PYTHONPATH=/opt/project /opt/project/pipeline/spark_consumer.py")
PRODUCER = r".\.venv\Scripts\python.exe pipeline\producer.py --stream s_label_flip --rate 1000"
# The demo replays the 25,000 tweets around the flip (tweets 25,000-50,000; the flip is at 40,000):
# 25 s at 1,000 tweets/s, flip at second 15. Tested live on 16 Oct 2026: the consumer kept up
# (p95 latency ~1.5 s), ADWIN fired 47 tweets after the flip, the vocabulary detector ~5,000 after.
PRODUCER_DEMO = (r".\.venv\Scripts\python.exe pipeline\producer.py --stream s_label_flip "
                 "--start 25000 --limit 25000 --rate 1000")
OFFLINE = (r".\.venv\Scripts\python.exe experiments\runner_offline.py --stream s_label_flip "
           "--model enhanced --fusion or --strategy Rebuild --lam 0 --w 4000")

# (seconds, window, what to do on screen, narration). Speaker: Albin.
DEMO = [
    (10, "Terminal B",
     'Run <code>docker ps --format "table {{.Names}}\\t{{.Status}}"</code>. Six containers are listed; '
     "kafka and influxdb show <i>healthy</i>.",
     "All six services run in Docker: Kafka, Kafka UI, the Spark master and worker, InfluxDB and "
     "Grafana."),
    (10, "Terminal A",
     "Show the consumer that is already running (started before recording). The last line reads "
     "<code>consuming 'tweets' from kafka:29092 with models ['baseline', 'enhanced']</code>.",
     "The Spark consumer is subscribed to the Kafka topic and runs the baseline and our enhanced "
     "model side by side."),
    (8, "Terminal B",
     f"Run <code>{html.escape(PRODUCER_DEMO)}</code> and immediately switch to the browser.",
     "We replay the twenty-five thousand tweets around the label flip, at one thousand tweets per "
     "second. Fifteen seconds in, every label inverts while the words stay the same."),
    (17, "Grafana",
     "Press F5 once, then make sure the <b>Run</b> menu shows the newest "
     "<code>s_label_flip_seed0-...</code> run. Point at the accuracy panel, then throughput and "
     "latency.",
     "Both models learn quickly and reach about seventy-five percent. Throughput is about one "
     "thousand tweets per second, with a delay of one to two seconds."),
    (20, "Grafana",
     "About 15 s after the producer started: the flip. Point at the <b>blue ADWIN marker</b>, the "
     "drop in the <b>model size</b> panel (rebuild) and the <b>orange line staying near 25 %</b>.",
     "Here is the flip. The blue marker is ADWIN, firing within fifty tweets: the enhanced model "
     "rebuilds and stays near seventy percent, while the orange baseline drops to twenty-five "
     "percent and waits about five thousand tweets for its word detector, the orange marker."),
    (10, "Grafana",
     "When the run ends (25 s), point at the <b>Cumulative accuracy</b> panel (top right).",
     "Over these tweets, cumulative accuracy ends at about seventy-two percent for ours against "
     "sixty-three for the baseline. The gap is what the error-based detector buys us."),
    (20, "Terminal B",
     f"Run <code>{html.escape(OFFLINE)}</code> and scroll through the printed metrics. It takes "
     "10–40 s while Spark is busy: cut the wait in the editor, or run it once just before "
     "recording and only scroll through its output on camera.",
     "The same core code runs offline for the full experiment grid: one command prints accuracy, "
     "F1, detection delay, false alarms, recovery time and cost."),
]

SPEAKERS = {"Bichu": "Bichu Devnarayan", "Albin": "Albin John", "Sarthak": "Sarthak",
            "Aswin": "Aswin Paramesh"}

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; color: #111; line-height: 1.38; }
h1 { font-size: 21pt; margin: 0 0 2mm; }
h2 { font-size: 14.5pt; margin: 7mm 0 2.5mm; border-bottom: 1.5px solid #222; padding-bottom: 1mm; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 4mm 0 1.5mm; page-break-after: avoid; }
.sub { color: #555; font-size: 11pt; margin-bottom: 4mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; }
th, td { border: 0.75px solid #000; padding: 1.6mm 2mm; vertical-align: top; text-align: left; }
th { background: #f2f2f2; }
tr { page-break-inside: avoid; }
code, pre { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; }
code { background: #f2f2f2; padding: 0 1mm; border-radius: 1mm; overflow-wrap: anywhere; }
pre { background: #f6f6f6; border: 0.75px solid #bbb; padding: 2mm 3mm; white-space: pre-wrap; overflow-wrap: anywhere; margin: 1.5mm 0 3mm; }
.box { border: 1px solid #000; background: #f2f2f2; padding: 3mm 4mm; margin: 2mm 0 4mm; }
.warn { border-left: 3px solid #eb6834; padding: 1mm 3mm; margin: 2mm 0; background: #fff7f2; }
.time { white-space: nowrap; font-weight: bold; }
.thumb { width: 52mm; border: 0.75px solid #999; display: block; }
.say { font-style: italic; }
.who { color: #555; font-size: 9pt; }
.demo td { background: #f7fbff; }
ol li, ul li { margin: 0.8mm 0; }
.small { font-size: 9pt; color: #444; }
.pb { page-break-before: always; }
"""


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def mmss(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def route_rows() -> tuple[str, int]:
    rows, t = [], 0
    for i, (title, secs, who, say) in enumerate(NARRATION, start=1):
        thumb = THUMBS / f"Slide{i}.PNG"
        img = f'<img class="thumb" src="{b64(thumb)}">' if thumb.exists() else ""
        rows.append(f"<tr><td class='time'>{mmss(t)}<br><span class='who'>{secs} s</span></td>"
                    f"<td>{img}</td><td><b>Slide {i}: {html.escape(title)}</b><br>"
                    f"<span class='who'>Speaker: {SPEAKERS[who]}</span><br>"
                    f"<span class='say'>“{html.escape(say)}”</span></td></tr>")
        t += secs
        if i == DEMO_AFTER_SLIDE:
            demo_secs = sum(d[0] for d in DEMO)
            rows.append(f"<tr class='demo'><td class='time'>{mmss(t)}<br><span class='who'>{demo_secs} s"
                        f"</span></td><td><b>LIVE DEMO</b><br><span class='small'>terminals + Grafana"
                        f"</span></td><td><b>Live demo</b>, step by step in Section 2.<br>"
                        f"<span class='who'>Speaker: Albin John</span></td></tr>")
            t += demo_secs
    return "\n".join(rows), t


def demo_rows() -> str:
    rows, t = [], 0
    for secs, window, action, say in DEMO:
        rows.append(f"<tr><td class='time'>+{mmss(t)}</td><td><b>{window}</b></td><td>{action}</td>"
                    f"<td class='say'>“{html.escape(say)}”</td></tr>")
        t += secs
    return "\n".join(rows)


def build_html() -> str:
    route, total = route_rows()
    slide_secs = sum(n[1] for n in NARRATION)
    demo_secs = sum(d[0] for d in DEMO)
    grafana = f'<img src="{b64(GRAFANA_SHOT)}" style="width:100%;border:0.75px solid #999">'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Team 18 Video and Run Guide</title>
<style>{CSS}</style></head><body>

<h1>Team 18: Video and Run Guide</h1>
<div class="sub">Concept Drift Detection in Social Media Streams · IOE413 Streaming Data Analytics ·
IIIT Kottayam<br>Bichu Devnarayan (2023BCS0102) · Albin John (2023BCD0005) · Sarthak (2023BCS0068) ·
Aswin Paramesh (2023BCD0024)</div>

<div class="box"><b>What the video must be (LMS instructions)</b>
<ul>
<li><b>MP4</b> only, <b>at most 6:00</b> long, with <b>audio narration</b>.</li>
<li>Must contain the <b>PPT presentation</b> and a <b>demonstration of the complete project workflow</b>
(the pipeline actually running end to end on streaming data).</li>
<li>Every submitted file starts with the team number: <code>18_</code> (for example <code>18_Demo_Video.mp4</code>).</li>
<li>Only the <b>team lead</b> uploads to the LMS. Course timeline: final submission, presentation and Q&amp;A
in the week of <b>15–21 October 2026</b>.</li>
</ul>
This guide gives the <b>route</b> of the video (Section 1), the <b>demo steps with exact commands</b>
(Section 2), <b>how to record and export</b> it (Section 3), <b>how to set up and run the project from
scratch</b> on any machine (Section 4), troubleshooting (Section 5) and the final checklist (Section 6).
Planned length: slides {mmss(slide_secs)} + demo {mmss(demo_secs)} = <b>{mmss(total)}</b>, leaving
{6 * 60 - total} s of margin.</div>

<h2>1. Route of the video (what to show and what to say)</h2>
<p>Read the narration at a calm pace (about 140 words per minute). The same text is in the speaker
notes of <code>Presentation.pptx</code> (View → Notes, or Presenter View during the slide show).
The speaker split is a suggestion that gives each member a part; one narrator is also fine.</p>
<table><tr><th style="width:15mm">Start</th><th style="width:55mm">Screen</th><th>Slide, speaker and narration</th></tr>
{route}
</table>

<h2 class="pb">2. The live demo, step by step (about {mmss(demo_secs)})</h2>
<h3>Prepare before you press record (10 minutes earlier)</h3>
<ol>
<li>Start <b>Docker Desktop</b> and wait for <i>Engine running</i>. In PowerShell, in the project folder:
<pre>docker compose up -d
docker ps</pre>Six containers must be up: kafka, kafka-ui, spark-master, spark-worker, influxdb, grafana.</li>
<li>Open <b>Terminal A</b> (PowerShell, project folder) and start the Spark consumer. It keeps this window busy:
<pre>{html.escape(CONSUMER)}</pre>
Wait until it prints <code>consuming 'tweets' from kafka:29092 with models ['baseline', 'enhanced']</code>.</li>
<li>Open <b>Terminal B</b> (PowerShell, project folder) for the producer. Do one short dry run so Spark is warm:
<pre>.\\.venv\\Scripts\\python.exe pipeline\\producer.py --stream s_label_flip --limit 5000 --rate 1000</pre></li>
<li>Open the dashboard in the browser, logged in (user <code>admin</code> and your team's Grafana password):
<pre>{GRAFANA_URL}</pre>This shows the last 2 minutes and refreshes every 5 s. Close other tabs and notifications;
set the browser zoom to 100 %. Optional: add <code>&amp;kiosk</code> to the URL to hide Grafana's menus.</li>
<li>Arrange the windows so you can switch quickly: Terminal A, Terminal B, browser. Use a large terminal font (Ctrl + mouse wheel).</li>
</ol>

<h3>During the recording</h3>
<table><tr><th style="width:12mm">Time</th><th style="width:22mm">Window</th><th>Do</th><th style="width:62mm">Say</th></tr>
{demo_rows()}
</table>
<div class="warn"><b>Timing tip:</b> the flip happens about 15 s after the producer starts and the replay
takes 25 s. Grafana refreshes every 5 s, so the alarm appears a few seconds later on screen. Keep the rate
at <b>1,000 tweets/s</b>: on a freshly started stack the consumer sustains about 1,000/s with 1,000-tweet
micro-batches, and at 2,000/s it falls behind and the charts lag (tested). Pauses can be cut out later in
the editor (Section 3). To show the full 80,000-tweet stream instead, drop <code>--start</code> and
<code>--limit</code> (80 s; speed that part up 2x in the editor).</div>

<h3>What the dashboard looks like at the flip (orange = baseline, blue = enhanced)</h3>
{grafana}
<p class="small">Top left: accuracy per micro-batch, with the drift markers (dashed lines; blue = ADWIN,
orange = vocabulary detector). Middle right: model size; a sudden drop is a rebuild. Bottom: throughput,
latency, driver memory.</p>

<h2 class="pb">3. How to record, edit and export the MP4</h2>
<h3>Recommended: OBS Studio (free) + Clipchamp (built into Windows 11)</h3>
<ol>
<li>Install <b>OBS Studio</b> (obsproject.com). In <i>Settings → Output</i>: Recording format <b>MP4</b>
(or MKV and use <i>File → Remux Recordings</i> to get MP4). In <i>Settings → Video</i>: base and output
resolution <b>1920×1080</b>, 30 fps. In <i>Settings → Audio</i>: choose your microphone; check the level meter
moves when you speak and stays out of the red.</li>
<li>Create two scenes: <b>Slides</b> (source: <i>Window Capture</i> of the PowerPoint slide show, or
<i>Display Capture</i>) and <b>Demo</b> (source: <i>Display Capture</i> of the screen with the terminals and browser).</li>
<li>Record in <b>three clips</b> so a mistake never forces a full retake:
<ul><li><b>Clip 1:</b> slides 1–9 (PowerPoint: <i>Slide Show → From Beginning</i>; each speaker reads their part).</li>
<li><b>Clip 2:</b> the live demo (Section 2).</li>
<li><b>Clip 3:</b> slides 10–17 (<i>Slide Show</i>, start from slide 10).</li></ul></li>
<li>Open <b>Clipchamp</b> (Start menu), create a video, import the three clips, put them on the timeline in
order, cut long pauses and dead time (for example while Grafana refreshes), and check the total length is
under 6:00. <i>Export → 1080p</i>. Clipchamp exports MP4.</li>
<li>Name the file <code>18_Demo_Video.mp4</code> and put it in <code>submission\\18_LMS_Upload\\</code>.</li>
</ol>
<h3>Alternative without OBS</h3>
<ul>
<li><b>Slides:</b> PowerPoint → <i>Record</i> (Record from beginning); each speaker records the narration on
their slides; then <i>File → Export → Create a Video</i> (Full HD) gives an MP4.</li>
<li><b>Demo:</b> Xbox Game Bar (<b>Win + Alt + R</b> starts and stops recording of the active window, with the
microphone on via Win + Alt + M), or the Windows Snipping Tool's video mode.</li>
<li>Join the clips in Clipchamp as above.</li>
</ul>
<h3>Audio tips</h3>
<ul><li>Use a headset or earphone microphone in a quiet room; turn off fans and notifications.</li>
<li>Rehearse each part twice with a timer; the planned times are in Section 1.</li>
<li>Speak while the content is on screen; do not read the bullets word for word, the narration
already adds what the slide does not say.</li></ul>

<h2 class="pb">4. Set up and run the project from scratch (any Windows machine)</h2>
<p>All commands are for <b>PowerShell</b>, run in the project folder. Allow about 45 minutes the first time
(downloads of Python packages, the dataset and the Docker images).</p>
<h3>4.1 Install the prerequisites</h3>
<ul><li><b>Docker Desktop</b> (WSL 2 backend), with at least 8 GB memory available to Docker.</li>
<li><b>Python 3.12</b> (python.org; tick <i>Add to PATH</i>) and <b>Git</b>.</li></ul>
<h3>4.2 Get the code and the Python environment</h3>
<pre>git clone https://github.com/CASPER0022/Drift-Aware-Sentiment-Stream.git
cd Drift-Aware-Sentiment-Stream
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass     # only if activation is blocked
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pytest -q                                     # expect: 46 passed</pre>
<p class="small">If you use the submitted zip instead of git, unzip it and <code>cd</code> into the folder.</p>
<h3>4.3 Download Sentiment140 and build the streams</h3>
<pre>New-Item -ItemType Directory -Force data\\raw
Invoke-WebRequest https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip -OutFile data\\raw\\trainingandtestdata.zip
python pipeline\\build_ds1.py          # DS1: 1.6M tweets sorted by time (~15 s)
python pipeline\\build_ds2.py          # DS2: 1,073,065 tweets
python pipeline\\build_scenarios.py    # 6 drift scenarios x 5 seeds (~20 s)</pre>
<h3>4.4 Start the streaming stack</h3>
<pre>Copy-Item .env.example .env
docker compose up -d --build          # first time builds the Spark image (a few minutes)
docker ps                             # 6 containers up</pre>
<table><tr><th>Service</th><th>Address</th></tr>
<tr><td>Grafana dashboard “Drift Stream”</td><td><code>http://localhost:3000/d/drift-stream</code> — first login
<code>admin</code> / <code>admin</code>, then Grafana asks you to set a new password</td></tr>
<tr><td>Kafka UI</td><td><code>http://localhost:8085</code></td></tr>
<tr><td>Spark master UI / running consumer</td><td><code>http://localhost:8090</code> / <code>http://localhost:4040</code></td></tr>
<tr><td>InfluxDB</td><td><code>http://localhost:8086</code></td></tr></table>
<h3>4.5 Run the pipeline</h3>
<pre># Terminal A: Spark consumer (baseline + enhanced side by side); stop with Ctrl+C
{html.escape(CONSUMER)}

# Terminal B: replay a stream into Kafka
{html.escape(PRODUCER)}
.\\.venv\\Scripts\\python.exe pipeline\\producer.py --stream ds1 --start 1250000 --limit 150000 --rate 3000</pre>
<h3>4.6 Offline experiments (no Docker needed)</h3>
<pre>{html.escape(OFFLINE)}
python experiments\\run_config.py --config experiments\\configs\\baseline.yaml   # paper reproduction (~20 min)
python experiments\\run_grid.py                                                # full grid, 224 runs (~25 min)
python analysis\\make_report_figures.py                                        # all report figures and tables</pre>
<h3>4.7 Stop everything</h3>
<pre>Ctrl+C in Terminal A
docker compose stop          # keeps all data; "docker compose start" resumes</pre>

<h2>5. Troubleshooting</h2>
<table><tr><th style="width:55mm">Problem</th><th>Fix</th></tr>
<tr><td>Docker commands fail: “cannot find the file specified … dockerDesktopLinuxEngine”</td><td>Docker Desktop is not running. Start it, wait for <i>Engine running</i>, then <code>docker compose up -d</code>.</td></tr>
<tr><td>Grafana shows <i>No data</i></td><td>Pick the newest run in the <b>Run</b> menu and set the time range to <i>Last 5 minutes</i>; check Terminal A prints <code>batch ...</code> lines.</td></tr>
<tr><td>Consumer prints a Python traceback</td><td><code>docker restart spark-master spark-worker</code>, wait 10 s, start the consumer again (Section 2, step 2).</td></tr>
<tr><td>Latency keeps growing in Grafana</td><td>The consumer is behind: lower <code>--rate</code> to 1000, or restart the consumer with <code>--batch-size 5000</code> added at the end.</td></tr>
<tr><td><code>producer.py</code>: file not found under <code>data\\</code></td><td>Build the streams first (Section 4.3).</td></tr>
<tr><td>Activate.ps1 is blocked</td><td><code>Set-ExecutionPolicy -Scope Process Bypass</code>, then activate again.</td></tr>
<tr><td>Video is longer than 6:00</td><td>In Clipchamp trim pauses in the demo (Grafana refresh waits) and the gaps between slides; do not cut the flip.</td></tr>
</table>

<h2>6. Final checklist</h2>
<table><tr><th style="width:68mm">File in <code>submission\\18_LMS_Upload\\</code></th><th>What it is</th></tr>
<tr><td><code>18_Demo_Video.mp4</code></td><td>the recorded video: MP4, at most 6:00, slides + live demo + narration (add after recording)</td></tr>
<tr><td><code>18_Presentation.pptx</code></td><td>the slides, in the course template</td></tr>
<tr><td><code>18_Final_Report.pdf</code></td><td>IEEE two-column report, 8 pages, with names and roll numbers</td></tr>
<tr><td><code>18_Final_Report_Source.zip</code></td><td>editable LaTeX source of the report (main.tex, refs.bib, figures, tables)</td></tr>
<tr><td><code>18_Source_Code.zip</code></td><td>complete source code with the README (pipeline, setup, run instructions, tools and versions)</td></tr>
<tr><td><code>18_README.md</code></td><td>the README on its own, for quick reading</td></tr></table>
<ul>
<li>☐ Video plays from start to end, audio is clear, length ≤ 6:00, file is MP4.</li>
<li>☐ Every file name starts with <code>18_</code>.</li>
<li>☐ The team lead uploads all files to the LMS before the deadline.</li>
<li>☐ Everyone has rehearsed the Q&amp;A topics: why naive Bayes, why ADWIN, how delay is measured, why Spark
when the model is sequential, how the drifts were made.</li>
</ul>
</body></html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = HERE / "guide.html"
    html_path.write_text(build_html(), encoding="utf-8")
    pdf = OUT_DIR / "Video_and_Run_Guide.pdf"
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        raise SystemExit(f"no Edge/Chrome found; open {html_path} and print it to PDF")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", f"--print-to-pdf={pdf}", html_path.as_uri()],
                   check=True, timeout=120)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
