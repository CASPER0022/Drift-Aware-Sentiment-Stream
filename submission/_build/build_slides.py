"""Build the presentation from the course template (presentation template.pptx).

Keeps the template's master, logo, footer ("Indian Institute of Information Technology
Kottayam") and slide-number field, and recreates its look: Calibri, 36 pt titles, 20 pt
bullets, gray-bordered boxes and plain black-bordered tables. Every slide gets its narration
in the speaker notes; the same NARRATION list feeds the video guide (build_guide.py).

Usage (python-pptx needed only for this script):
  pip install python-pptx
  python submission/_build/build_slides.py
Writes submission/video/Presentation.pptx.
"""
import copy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "presentation template.pptx"
FIG = ROOT / "analysis" / "figures"
OUT = ROOT / "submission" / "video" / "Presentation.pptx"

BLACK, GRAY, DARK = RGBColor(0, 0, 0), RGBColor(0x59, 0x59, 0x59), RGBColor(0x7F, 0x7F, 0x7F)
LIGHT, WHITE = RGBColor(0xF2, 0xF2, 0xF2), RGBColor(0xFF, 0xFF, 0xFF)
BLUE, ORANGE = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34)
FONT = "Calibri"
LEFT = 0.92  # inches, the template's left margin

TEAM = [("2023BCS0102", "BICHU DEVNARAYAN"), ("2023BCD0005", "ALBIN JOHN"),
        ("2023BCS0068", "SARTHAK"), ("2023BCD0024", "ASWIN PARAMESH")]

# (slide title, seconds, suggested speaker, narration). Read at ~140 words per minute.
NARRATION = [
    ("Title", 12, "Bichu",
     "Hello, we are Team 18: Bichu Devnarayan, Albin John, Sarthak and Aswin Paramesh. Our "
     "project is concept drift detection in social media streams."),
    ("Introduction and Problem Statement", 22, "Bichu",
     "Tweets arrive as a fast, never-ending stream, and topics, words and opinions keep "
     "changing: concept drift. Some drift only changes the words, like a switch from work tweets "
     "to music tweets. Real drift changes the meaning: the same words now signal a different "
     "sentiment, and a model trained once quietly loses accuracy."),
    ("Literature Review", 18, "Bichu",
     "We built on three papers. Iosifidis and colleagues adapt a naive Bayes classifier when a "
     "word-based detector signals change; that is our baseline. Costa and colleagues show how to "
     "simulate drift with real tweets, and Bechini and colleagues show that real events cause it. "
     "The gap: the baseline only watches the words."),
    ("System Architecture", 22, "Albin",
     "Here is the pipeline. A Python producer replays Sentiment140's 1.6 million tweets, in time "
     "order, into Kafka. Spark Structured Streaming tokenizes each micro-batch, and the driver runs "
     "the baseline and our enhanced model side by side. Metrics and drift events go to InfluxDB "
     "and Grafana, all in Docker."),
    ("Baseline: Informed Ageing Naive Bayes", 20, "Albin",
     "The baseline is an ageing naive Bayes: old word counts fade with time. Every few thousand "
     "tweets a detector compares the words of recent tweets with the previous window, and when "
     "they change, the model forgets faster or is rebuilt. We had to fix two details the paper "
     "leaves open."),
    ("Enhancement: ADWIN on the Errors", 22, "Albin",
     "Our enhancement watches the model's own mistakes. Every tweet is predicted before it is "
     "learned, so each one gives a free right-or-wrong signal. ADWIN raises an alarm when the "
     "error rate rises, and the model is rebuilt on the tweets after the change. The OR mode, "
     "which combines both detectors, is our main model."),
    ("Data and Drift Scenarios", 15, "Sarthak",
     "We test on the real stream, the paper's more volatile variant, and six controlled "
     "scenarios with known drift points, each with five random seeds. The key one is the label "
     "flip: at tweet 40,000 every label inverts while the words stay the same."),
    ("Implementation: Tools and Versions", 10, "Sarthak",
     "These are the tools and their versions. The Python packages are pinned in the "
     "requirements file, and the Spark image is built from our Dockerfile."),
    ("Live Demo", 5, "Albin",
     "Now let us show the pipeline running end to end."),
    ("Results: Baseline Reproduction", 15, "Sarthak",
     "First, the baseline reproduces the paper: the model that never forgets is worst, fixed "
     "forgetting helps, and informed adaptation is best on both streams. One difference: in our "
     "runs, forgetting only on detected change beats constant forgetting."),
    ("Results: The Label Flip", 18, "Sarthak",
     "On the label flip, the word detector waits a full window, about four thousand tweets, while "
     "accuracy sits near twenty-five percent. ADWIN fires after thirty-nine tweets, recovery is two "
     "and a half times faster, and accuracy rises from seventy-one to almost seventy-four percent."),
    ("Results: All Streams and Real Data", 20, "Aswin",
     "ADWIN also catches a flip limited to one topic, which the baseline misses in two of five "
     "runs. On real data, accuracy improves on both streams, and Sentiment140's change point is "
     "caught after seven hundred tweets instead of nineteen thousand. The price is more false "
     "alarms, mostly from the word detector."),
    ("Performance", 12, "Aswin",
     "The cost is small. In our benchmark the pipeline kept up with two thousand tweets per "
     "second, and ADWIN adds only about ten milliseconds per thousand tweets."),
    ("Limitations and Future Work", 12, "Aswin",
     "Limitations: the labels come from emoticons, the controlled drifts are synthetic, and the "
     "model runs on the Spark driver. Next, we would try a context-aware classifier and recent, "
     "event-driven data."),
    ("Conclusion", 12, "Aswin",
     "To conclude: watch the errors, not only the words. Combining both detectors catches "
     "meaning drift about a hundred times sooner and improves accuracy, at almost no cost, in a "
     "real streaming pipeline."),
    ("References", 3, "Aswin", "These are our references."),
    ("Thank You", 4, "Aswin", "Thank you."),
]


# -- template plumbing ---------------------------------------------------------------------

def prepare(prs):
    """Grab logo, footer and slide-number elements from template slide 2, then drop all slides."""
    ref = prs.slides[1]
    logo = next(s for s in ref.shapes if s.shape_type == 13)
    parts = {"logo_blob": logo.image.blob, "logo_box": (logo.left, logo.top, logo.width, logo.height),
             "footer": [copy.deepcopy(s._element) for s in ref.shapes
                        if s.is_placeholder and s.placeholder_format.idx in (11, 12)]}
    sld_ids = prs.slides._sldIdLst
    for sld_id in list(sld_ids):
        prs.part.drop_rel(sld_id.rId)
        sld_ids.remove(sld_id)
    return parts


def new_slide(prs, parts, layout_idx, title=None, notes=None):
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    for el in parts["footer"]:
        slide.shapes._spTree.append(copy.deepcopy(el))
    left, top, width, height = parts["logo_box"]
    import io
    slide.shapes.add_picture(io.BytesIO(parts["logo_blob"]), left, top, width, height)
    if title is not None and layout_idx != 0:
        t = slide.shapes.title
        t.left, t.top, t.width, t.height = Emu(841248), Emu(274320), Emu(8595360), Emu(822960)
        t.text_frame.text = title
        run = t.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(36)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


# -- drawing helpers ------------------------------------------------------------------------

def style_run(run, size, bold=False, italic=False, color=BLACK):
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT
    f.color.rgb = color


def bullets(slide, x, y, w, h, items, size=20, space=12, bullet=True):
    """items: str, or list of (text, bold) runs, or ("  sub", ...) via a leading '- ' for level 2."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        level2 = isinstance(item, str) and item.startswith("- ")
        runs = [(item[2:] if level2 else item, False)] if isinstance(item, str) else item
        pPr = p._p.get_or_add_pPr()
        if bullet:
            pPr.set("marL", str(Emu(Inches(0.75 if level2 else 0.375))))
            pPr.set("indent", str(-Emu(Inches(0.3))))
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "–" if level2 else "•")
        if i:
            p.space_before = Pt(space if not level2 else space / 2)
        for text, bold in runs:
            r = p.add_run()
            r.text = text
            style_run(r, size - 2 if level2 else size, bold=bold)
    return box


def text(slide, x, y, w, h, s, size=14, bold=False, italic=False, color=BLACK,
         align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        style_run(r, size, bold=bold, italic=italic, color=color)
    return box


def caption(slide, x, y, w, s):
    return text(slide, x, y, w, 0.4, s, size=12, italic=True, color=GRAY)


def node(slide, x, y, w, h, s, size=14, bold=False, fill=WHITE, rounded=True, edge=BLACK,
         edge_w=1.0):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shape.adjustments[0] = 0.15
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge
    shape.line.width = Pt(edge_w)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        style_run(r, size if i == 0 else size - 2, bold=bold and i == 0,
                  color=BLACK if i == 0 else GRAY)
    return shape


def arrow(slide, x1, y1, x2, y2, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = BLACK
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def table(slide, x, y, w, rows, col_widths, size=14, row_h=0.42, bold_first_col=False,
          align=None):
    nrows, ncols = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow", "lastRow", "firstCol", "lastCol", "bandCol"):
        tblPr.attrib.pop(attr, None)
    for child in list(tblPr):
        tblPr.remove(child)
    total = sum(col_widths)
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = Inches(w * cw / total)
    for i in range(nrows):
        tbl.rows[i].height = Inches(row_h)
        for j in range(ncols):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if i == 0 else (align[j] if align else PP_ALIGN.LEFT)
            r = p.add_run()
            r.text = str(rows[i][j])
            style_run(r, size, bold=(i == 0) or (bold_first_col and j == 0))
            tcPr = cell._tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(side))
                ln.set("w", "9525")
                fill = etree.SubElement(ln, qn("a:solidFill"))
                etree.SubElement(fill, qn("a:srgbClr")).set("val", "000000")
            sf = etree.SubElement(tcPr, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", "F2F2F2" if i == 0 else "FFFFFF")
    return gf


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


# -- slides ---------------------------------------------------------------------------------

def build() -> None:
    prs = Presentation(str(TEMPLATE))
    parts = prepare(prs)
    notes = {title: f"[{secs} s, {who}] {say}" for title, secs, who, say in NARRATION}

    # 1 Title
    s = new_slide(prs, parts, 0, notes=notes["Title"])
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.75), Inches(0.95), Inches(11.83), Inches(2.0)
    t.text_frame.text = "Concept Drift Detection in Social Media Streams"
    t.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    t.text_frame.paragraphs[0].runs[0].font.size = Pt(40)
    text(s, 1.67, 3.0, 10.0, 0.9,
         "Informed ageing naive Bayes + an error-based ADWIN detector,\n"
         "live on Kafka, Spark, InfluxDB and Grafana", size=20, color=GRAY, align=PP_ALIGN.CENTER)
    text(s, 1.67, 3.95, 10.0, 0.4, "Streaming Data Analytics (IOE413)  ·  Semester 7  ·  Team 18",
         size=18, color=GRAY, align=PP_ALIGN.CENTER)
    team = "BY\n" + "\n".join(f"{roll} – {name}" for roll, name in TEAM)
    box = text(s, 1.67, 4.5, 10.0, 1.6, team, size=18, align=PP_ALIGN.CENTER)
    for p in box.text_frame.paragraphs[1:]:
        p.runs[0].font.italic = True

    # 2 Introduction and problem statement
    s = new_slide(prs, parts, 1, "Introduction and Problem Statement",
                  notes["Introduction and Problem Statement"])
    bullets(s, LEFT, 1.35, 6.4, 4.8, [
        "Tweets arrive as a fast, never-ending stream; topics, words and opinions keep changing: concept drift",
        "A sentiment model trained once quietly loses accuracy",
        [("Virtual drift: ", True), ("new words or topics, P(X) changes", False)],
        [("Real drift: ", True), ("same words, new meaning, P(y | X) changes", False)],
        [("Problem: ", True), ("the base paper's detector watches only the words, so it cannot see a change in meaning", False)],
        [("Goal: ", True), ("catch both kinds, early, in a live streaming pipeline", False)],
    ], size=19)
    node(s, 7.75, 1.55, 4.6, 1.75, "Virtual drift\nwork / school tweets  →  music / movie tweets\n"
         "the words change, the meaning does not", size=17, bold=True, fill=LIGHT, rounded=False)
    node(s, 7.75, 3.65, 4.6, 1.75, "Real drift\n\"so sick\" was negative, now means positive\n"
         "the words stay, the meaning changes", size=17, bold=True, fill=WHITE)
    caption(s, 7.75, 5.55, 4.6, "Only real drift necessarily hurts a classifier.")

    # 3 Literature review
    s = new_slide(prs, parts, 1, "Literature Review", notes["Literature Review"])
    bullets(s, LEFT, 1.3, 11.5, 4.9, [
        [("[1] Iosifidis, Oelschlager, Ntoutsi (2017): ", True),
         ("informed ageing naive Bayes on Sentiment140; a vocabulary detector tunes how fast the model forgets", False)],
        "- Our baseline: we reproduce the model, the detector and all 5 forgetting strategies",
        [("[2] Costa, Silva, Antunes, Ribeiro (ICMLA 2014): ", True),
         ("sudden, gradual and recurring drift simulated by re-timestamping real tweets", False)],
        "- Our method for controlled drift scenarios with known drift points",
        [("[3] Bechini et al. (IEEE Access 2021): ", True),
         ("real-world events cause drift in a Twitter stance stream; adaptive models recover", False)],
        "- Our motivation: drift on social media is event-driven and real",
        [("Gap: ", True), ("the baseline watches only the input words, reacts once per 24k-tweet window, "
                           "and was never run on a live streaming stack", False)],
    ], size=19, space=10)

    # 4 System architecture (native boxes, like the template)
    s = new_slide(prs, parts, 1, "System Architecture", notes["System Architecture"])
    rows = [
        ("Ingest", [("Sentiment140 → parquet", "pandas, pyarrow"), ("Producer", "confluent-kafka 2.15"),
                    ("Kafka 3.9.1 (KRaft)", "topic tweets, 1 partition")]),
        ("Process", [("Spark executors", "read Kafka, parse JSON, tokenize"),
                     ("Spark driver: foreachBatch", "predict, then learn, tweet by tweet")]),
        ("Models", [("Baseline", "ageing naive Bayes + vocabulary detector"),
                    ("Enhanced", "baseline + ADWIN on the errors (River)")]),
        ("Serve", [("InfluxDB 2.7", "per-model + pipeline metrics, drift events"),
                   ("Grafana 11.5.2", "live dashboard + drift markers")]),
    ]
    y0, rh, gap = 1.3, 0.88, 0.32
    for i, (label, boxes) in enumerate(rows):
        y = y0 + i * (rh + gap)
        node(s, LEFT, y, 2.0, rh, label, size=17, bold=True, fill=LIGHT, rounded=False)
        n = len(boxes)
        bw = (9.2 - 0.3 * (n - 1)) / n
        for k, (title, sub) in enumerate(boxes):
            edge = ORANGE if title == "Baseline" else BLUE if title == "Enhanced" else BLACK
            node(s, 3.22 + k * (bw + 0.3), y, bw, rh, f"{title}\n{sub}", size=16, bold=True,
                 edge=edge, edge_w=1.75 if edge != BLACK else 1.0)
            if label == "Ingest" and k < n - 1:
                arrow(s, 3.22 + k * (bw + 0.3) + bw, y + rh / 2, 3.22 + (k + 1) * (bw + 0.3), y + rh / 2)
        if i < len(rows) - 1:
            arrow(s, 7.82, y + rh, 7.82, y + rh + gap)
    caption(s, LEFT, 6.12, 11.5, "Kafka, Kafka UI, Spark master + worker, InfluxDB and Grafana run in "
            "Docker Compose; one Python package (core/) holds the models for both the live and the offline runner.")

    # 5 Baseline
    s = new_slide(prs, parts, 1, "Baseline: Informed Ageing Naive Bayes",
                  notes["Baseline: Informed Ageing Naive Bayes"])
    bullets(s, LEFT, 1.35, 6.6, 4.8, [
        "Multinomial naive Bayes whose counts fade with age:",
        "- weight of a word in a class  ∝  count × e^(−λ · hours since last seen)",
        "Vocabulary detector, every w tweets, per class:",
        "- overlap of the latest window's words with the previous window's",
        "- alarm if it leaves mean ± 1.8 standard deviations",
        "On an alarm: forget faster (raise λ) or rebuild the model; 5 strategies from the paper",
    ], size=19, space=10)
    node(s, 7.95, 1.45, 4.4, 0.62, "Details we had to fix", size=17, bold=True, fill=LIGHT, rounded=False)
    bullets(s, 7.95, 2.2, 4.4, 3.8, [
        "Time measured in hours: per tweet, the model forgot everything within ~100 tweets",
        "Compare with the previous window, not all words so far: otherwise the score drifts down and the detector fires constantly",
        "Mean / deviation over 20 checks instead of 10: removes false alarms",
    ], size=16, space=10)

    # 6 Enhancement
    s = new_slide(prs, parts, 1, "Enhancement: ADWIN on the Errors",
                  notes["Enhancement: ADWIN on the Errors"])
    flow = [("Predict the tweet", ""), ("Learn its label", ""), ("Error: right = 0, wrong = 1", ""),
            ("ADWIN (River, δ = 0.01)", "alarm if the error rate rises"),
            ("Rebuild on the tweets after the change", "only the new concept")]
    for i, (a, b) in enumerate(flow):
        y = 1.3 + i * 0.95
        node(s, LEFT, y, 4.6, 0.68, a + (f"\n{b}" if b else ""), size=16, bold=i == 3,
             edge=BLUE if i >= 3 else BLACK, edge_w=1.75 if i >= 3 else 1.0)
        if i < len(flow) - 1:
            arrow(s, LEFT + 2.3, y + 0.68, LEFT + 2.3, y + 0.95)
    bullets(s, 6.1, 1.3, 6.3, 2.6, [
        "Every tweet is tested before it is learned, so the error comes for free",
        "Sees changes in meaning even when the words stay the same",
        "Checks every tweet, not once per 24k-tweet window",
    ], size=18, space=10)
    table(s, 6.1, 3.85, 6.3, [
        ("Fusion mode", "Model reacts when"),
        ("vocab only", "the word detector fires (= baseline)"),
        ("ADWIN only", "ADWIN fires"),
        ("OR (main)", "either detector fires"),
        ("AND", "both fire within one window"),
    ], [1.5, 3.2], size=15, row_h=0.4, bold_first_col=False)

    # 7 Data and scenarios
    s = new_slide(prs, parts, 1, "Data and Drift Scenarios", notes["Data and Drift Scenarios"])
    table(s, LEFT, 1.3, 11.5, [
        ("Stream", "What changes", "Size"),
        ("DS1: Sentiment140", "real data, sorted by time; all tweets negative from #1,324,775", "1.6M tweets"),
        ("DS2: volatile variant", "positive share swings 68 / 27 / 70 / 17 / 56 / 28 / 0 %", "1.07M tweets"),
        ("Sudden / gradual / recurring", "topic switch: work / school ↔ music / movies", "80k × 5 seeds"),
        ("Prior shift", "50 % → 10 % positive", "80k × 5 seeds"),
        ("Label flip", "every label inverts at 40k; the words stay the same", "80k × 5 seeds"),
        ("Label flip (topic)", "only work / school labels invert", "80k × 5 seeds"),
    ], [2.8, 6.3, 1.6], size=15, row_h=0.46, bold_first_col=True)
    caption(s, LEFT, 4.75, 11.5, "Scenarios use real tweets re-timestamped as in Costa et al. [2]; "
            "every stream has a file of its true drift points.")
    bullets(s, LEFT, 5.2, 11.5, 1.0, [
        [("Metrics: ", True), ("accuracy, macro precision / recall / F1, detection delay, false alarms, "
                               "missed drifts, recovery time, throughput, latency, memory", False)],
    ], size=17)

    # 8 Tools and versions
    s = new_slide(prs, parts, 1, "Implementation: Tools and Versions",
                  notes["Implementation: Tools and Versions"])
    table(s, LEFT, 1.3, 11.5, [
        ("Tool", "Version", "Role"),
        ("Apache Kafka (KRaft) + confluent-kafka", "3.9.1 + 2.15.1", "ordered, replayable tweet stream"),
        ("Apache Spark Structured Streaming", "3.5.3", "parse, tokenize, run both models"),
        ("River", "0.26.1 / 0.22.0 in Spark", "ADWIN drift detector"),
        ("InfluxDB + Grafana OSS", "2.7 + 11.5.2", "metrics store, live dashboard"),
        ("Docker + Compose", "28.3.2 + 2.39.1", "runs all six services"),
        ("Python, pandas, pyarrow", "3.12.4, 3.0.6, 25.0.1", "data preparation, offline runner"),
        ("scikit-learn, pytest", "1.9.1, 9.1.1", "reference model, 46 unit tests"),
    ], [4.2, 2.8, 3.8], size=15, row_h=0.5, bold_first_col=True)
    caption(s, LEFT, 5.45, 11.5, "Exact pins: requirements.txt, docker/spark/Dockerfile, docker-compose.yml "
            "(see the README).")

    # 9 Live demo
    s = new_slide(prs, parts, 1, "Live Demo", notes["Live Demo"])
    steps = [("1", "Six services up in Docker"), ("2", "Spark consumer: baseline + enhanced"),
             ("3", "Producer replays the label-flip stream"), ("4", "Grafana: the flip, ADWIN's alarm, recovery"),
             ("5", "Offline runner: one command, full metrics")]
    for i, (n, label) in enumerate(steps):
        y = 1.4 + i * 0.9
        node(s, LEFT, y, 0.7, 0.66, n, size=20, bold=True, fill=LIGHT, rounded=False)
        node(s, LEFT + 0.95, y, 7.2, 0.66, label, size=18)
    caption(s, LEFT, 6.0, 11.0, "Pipeline: Kafka → Spark Structured Streaming → InfluxDB → Grafana, on real tweets.")

    # 10 Baseline reproduction
    s = new_slide(prs, parts, 1, "Results: Baseline Reproduction", notes["Results: Baseline Reproduction"])
    picture(s, FIG / "baseline_overall_accuracy.png", LEFT, 1.3, w=7.6)
    bullets(s, 8.75, 1.4, 3.75, 4.6, [
        "Same ordering as the paper on both streams",
        "Never forgetting is worst: 78.3 % / 80.3 %",
        "Informed adaptation best: up to 81.4 % / 83.0 %",
        "After DS1's change point: 93–99.9 % vs 77 %",
        "Difference: forgetting only on alarms (Zero) beats constant forgetting (Init)",
    ], size=16, space=9)
    caption(s, LEFT, 5.0, 7.6, "Overall accuracy of all baseline strategies (Table 1 parameters of [1]).")

    # 11 Label flip
    s = new_slide(prs, parts, 1, "Results: The Label Flip", notes["Results: The Label Flip"])
    picture(s, FIG / "fig2_label_flip_seeds.png", LEFT, 1.25, w=11.5)
    table(s, LEFT, 5.35, 11.5, [
        ("", "Detection delay", "Recovery to 95 %", "Accuracy (5 seeds)"),
        ("Baseline (word detector)", "3,999 tweets", "8,204 tweets", "71.0 ± 1.0 %"),
        ("Enhanced (OR)", "39 tweets", "3,183 tweets", "73.8 ± 0.3 %"),
    ], [3.0, 2.6, 2.6, 2.6], size=15, row_h=0.36, bold_first_col=True,
        align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3)

    # 12 All streams and real data
    s = new_slide(prs, parts, 1, "Results: All Streams and Real Data",
                  notes["Results: All Streams and Real Data"])
    table(s, LEFT, 1.3, 7.0, [
        ("Metric", "Baseline", "Enhanced"),
        ("Topic-only flip: drifts detected", "3 / 5", "5 / 5"),
        ("DS1 accuracy", "80.8 %", "81.3–81.7 %"),
        ("DS1 change point: delay", "19,224", "704"),
        ("DS2 accuracy", "83.0 %", "83.7–84.1 %"),
        ("Drifts detected (all 47)", "36", "39"),
        ("False alarms (all runs)", "70", "102"),
        ("≥ baseline accuracy", "–", "28 of 32 runs"),
    ], [3.4, 1.6, 1.9], size=15, row_h=0.5, bold_first_col=True,
        align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
    bullets(s, 8.35, 1.35, 4.1, 4.8, [
        "ADWIN catches meaning drift; the word detector catches topic and prior shifts",
        "OR keeps both: never more than 0.04 points below the baseline",
        "Extra false alarms come mostly from the word detector; a rebuild makes them cheap",
        "Topic and prior shifts barely change accuracy for any model",
    ], size=16, space=10)

    # 13 Performance
    s = new_slide(prs, parts, 1, "Performance", notes["Performance"])
    table(s, LEFT, 1.4, 6.4, [
        ("Metric", "Result"),
        ("Throughput, 1k-tweet batches", "2,000 tweets/s"),
        ("Latency at 2,000 tweets/s", "0.42 s median, 0.76 s p95"),
        ("Throughput, 5k-tweet batches", "5,593 tweets/s"),
        ("Model time per 1k tweets", "22.6 ms → 32.9 ms"),
        ("ADWIN update", "< 1 µs"),
        ("Offline runner", "37–46k tweets/s"),
        ("Unit tests", "46 passing"),
    ], [3.3, 3.1], size=15, row_h=0.48, bold_first_col=False)
    bullets(s, 7.75, 1.45, 4.7, 4.5, [
        "Live accuracy = offline accuracy (72 % vs 74 % on the label flip)",
        "The bottleneck is Spark's ~170 ms per micro-batch, not the models",
        "ADWIN adds ~10 ms per 1,000 tweets",
        "Spark driver memory: 200–330 MB",
    ], size=17, space=10)

    # 14 Limitations and future work
    s = new_slide(prs, parts, 1, "Limitations and Future Work", notes["Limitations and Future Work"])
    text(s, LEFT, 1.3, 5.6, 0.5, "Limitations", size=20, bold=True)
    bullets(s, LEFT, 1.85, 5.6, 4.4, [
        "Labels come from emoticons and are noisy",
        "Controlled drifts are synthetic; topics defined by keywords",
        "Real streams list only designed drift points: false alarms are upper bounds",
        "Model state lives on the Spark driver",
        "Naive Bayes cannot learn a topic-only flip (~64 % ceiling)",
    ], size=17, space=9)
    text(s, 6.85, 1.3, 5.6, 0.5, "Future work", size=20, bold=True)
    bullets(s, 6.85, 1.85, 5.6, 4.4, [
        "A context-aware classifier for topic-specific meaning changes",
        "Drift-specific reactions, e.g. re-weight the class prior on a prior shift",
        "Keyed state so Spark can partition the model",
        "Recent, event-driven Twitter / X data",
    ], size=17, space=9)

    # 15 Conclusion
    s = new_slide(prs, parts, 1, "Conclusion", notes["Conclusion"])
    bullets(s, LEFT, 1.4, 11.5, 4.6, [
        "Reproduced the paper's informed ageing naive Bayes on Sentiment140",
        "Its word-based detector misses or delays drift that changes meaning",
        "ADWIN on the errors detects it ~100× sooner and recovers 2.6× faster",
        "Accuracy improves on both real streams, at ~10 ms per 1,000 tweets",
        "Runs end to end on Kafka, Spark Structured Streaming, InfluxDB and Grafana",
        [("Watch the errors, not only the words.", True)],
    ], size=20)

    # 16 References
    s = new_slide(prs, parts, 1, "References", notes["References"])
    refs = [
        "[1] V. Iosifidis, A. Oelschlager and E. Ntoutsi, “Sentiment classification over opinionated data streams through informed model adaptation,” in Proc. TPDL, Springer LNCS, 2017.",
        "[2] J. Costa, C. Silva, M. Antunes and B. Ribeiro, “Concept drift awareness in Twitter streams,” in Proc. 13th ICMLA, IEEE, 2014.",
        "[3] A. Bechini, A. Bondielli, P. Ducange, F. Marcelloni and A. Renda, “Addressing event-driven concept drift in Twitter stream: a stance detection application,” IEEE Access, vol. 9, 2021.",
        "[4] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” in Proc. SIAM SDM, 2007.",
        "[5] S. Wagner, M. Zimmermann, E. Ntoutsi and M. Spiliopoulou, “Ageing-based multinomial naive Bayes classifiers over opinionated data streams,” in Proc. ECML PKDD, 2015.",
        "[6] A. Go, R. Bhayani and L. Huang, “Twitter sentiment classification using distant supervision,” Stanford CS224N report, 2009.",
        "[7] J. Montiel et al., “River: machine learning for streaming data in Python,” JMLR, vol. 22, 2021.",
        "[8] M. Armbrust et al., “Structured Streaming: a declarative API for real-time applications in Apache Spark,” in Proc. ACM SIGMOD, 2018.",
    ]
    bullets(s, LEFT, 1.3, 11.5, 5.0, refs, size=14, space=6, bullet=False)

    # 17 Thank you
    s = new_slide(prs, parts, 6, notes=notes["Thank You"])
    text(s, 0.75, 2.6, 11.83, 1.0, "Thank You", size=44, align=PP_ALIGN.CENTER)
    text(s, 0.75, 3.9, 11.83, 0.5, "Questions are welcome", size=18, color=GRAY, align=PP_ALIGN.CENTER)
    text(s, 0.75, 4.5, 11.83, 0.5, "github.com/CASPER0022/Drift-Aware-Sentiment-Stream", size=16,
         color=GRAY, align=PP_ALIGN.CENTER)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
