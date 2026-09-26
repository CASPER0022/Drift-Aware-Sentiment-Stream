# Project Timeline: Concept Drift Detection & Adaptive Learning in Social Media Streams

**Group 18** · **Start:** Sat 26 Sep 2026 · **Hard deadline:** Sat 10 Oct 2026 (15 working days)

The course timeline runs to 21 Oct (demo + in-class Q&A). This plan finishes **everything** by 10 Oct: code, experiments, IEEE report, README, and the recorded demo. That leaves 11–21 Oct for rehearsal and slack.

---

## 1. Project in one paragraph

We replay **Sentiment140** (1.6M tweets, Apr–Jun 2009) in timestamp order through **Kafka**. **Spark Structured Streaming** preprocesses the tweets and feeds an online sentiment classifier, and metrics go to **InfluxDB → Grafana**.

- **Baseline** (Iosifidis et al.): *ageing Multinomial Naive Bayes*. Word and class counts are down-weighted by `e^(-λ·age)`. A **vocabulary-based change detector** watches vocabulary overlap per class and retunes λ when it fires.
- **Enhancement (ours):** add an **ADWIN detector on the prediction-error stream** (River). The system can then react to changes in P(y|X), where the words stay the same but the meaning or label changes. The vocabulary detector cannot see this kind of change; the paper admits the gap in §5.3.
- **Evaluation:** we build controlled drift scenarios (sudden, gradual, recurring) following Costa et al., so the true drift points are known. This lets us measure detection delay, false alarms and recovery time, plus accuracy, P, R, F1, throughput, latency and memory.

## 2. Key technical facts from the papers (pin these)

| Item | Value / definition | Source |
|---|---|---|
| Class prior | `P(c) = N_c · e^(-λ(t - t_lo^c)) / |S|` | Iosifidis Eq. 1 |
| Word likelihood | `P(w|c) = N_wc·e^(-λ(t - t_lo^(w,c))) / Σ_j N_jc·e^(-λ(t - t_lo^(j,c)))` (add Laplace smoothing) | Eq. 2 |
| Detector statistic | `precision = |V_sl ∩ V| / |V|`, computed **per class** (pos and neg) | Eq. 3 |
| Change / warning | change if outside `μ ± α·σ`, α = **1.8**; warning at β·σ, β = **0.334**; warning ⇒ start buffering instances | Eq. 4, §5.1 |
| Check frequency | every `w` instances (not per tweet) | §4.2 |
| λ strategies | SlowIncreaseUpToALimit, SlowIncreaseFastReset, FastSetFastReset, FastSetSlowDecrease, Rebuild; reference models: accumulativeMNB (λ=0) and fadingMNB (fixed λ) | §4.2 |
| Best params (DS1) | fadingMNB λ0=0.2; FastSetFastReset-Init λ0=0.1, w=24k, λmax=0.5; SlowIncreaseUpToALimit-Init λ0=0.2, c=0.1; Rebuild-Init λ0=0.2 | Table 1 |
| Targets to reproduce | DS1 overall accuracy: accumulative ≈ **73%**, ageing/informed ≈ **78%**; natural change point at instance **≈1,326,000** (stream turns all-negative) | Fig. 2, §5.1 |
| Evaluation | **Prequential** (test-then-train), accuracy over window `evalW` | §5.1 |
| Drift types | sudden, gradual, incremental, recurring, simulated by **re-timestamping real tweets** into 24 time windows | Costa et al., Table I |
| Motivation | real events cause drift; a static model decays, retraining or adaptation helps; report accuracy and macro-F1 | Bechini et al. |

**Sentiment140 gotchas:**
- The CSV is sorted **by label**, not by time. Parse `Mon Apr 06 22:19:45 PDT 2009` and sort before replaying.
- The file uses `latin-1` encoding.
- Collection has large time gaps, which is expected (see paper Fig. 1).
- Labels are 0 and 4. Map them to 0 and 1.

## 3. Architecture decisions (decide on Day 1, don't revisit)

1. **Run infrastructure in Docker** (Kafka in KRaft mode, Spark, InfluxDB 2, Grafana) via `docker-compose.yml`. Spark and Kafka on bare Windows cause trouble, so use Docker Desktop or WSL2.
2. **One model library, two runners:**
   - `core/`: pure-Python models, detectors and metrics, with no Spark dependency.
   - `runner_offline.py` iterates the parquet stream directly. Use it for the large experiment grid, where 1.6M tweets × many configs has to be fast.
   - `runner_streaming.py`: Kafka → Spark `foreachBatch` → the same `core/` model. Use it for the live demo and the throughput and latency numbers.

   The ageing MNB is inherently sequential and stateful, so it runs on the driver inside `foreachBatch`. Micro-batch = the detector check window `w`. Say this explicitly in the report.
3. **Time unit `t`:** instance index, with an hourly-timestamp variant as an optional extra. Keep it consistent, because λ values from Table 1 only make sense if the unit matches. Tune λ if the numbers don't line up.
4. **Features:** lowercase, strip URLs, @mentions and `#`, and collapse repeated characters. Tokenize to unigrams. Skip stemming at first, because MNB counts need fast dictionaries.

### Repo layout
```
project/
├── docker-compose.yml
├── data/            (raw/, processed/, scenarios/)   ← gitignored
├── core/            mnb.py, vocab_detector.py, adwin_detector.py, strategies.py, metrics.py
├── pipeline/        producer.py, spark_consumer.py, influx_sink.py
├── experiments/     run_grid.py, configs/*.yaml, results/*.csv
├── analysis/        plots.ipynb, figures/
├── grafana/         dashboard.json, provisioning/
├── report/          IEEE LaTeX (Overleaf)
├── tests/
└── README.md
```

---

## 4. Day-by-day plan

> ☐ = task · ✅ **Done when** = exit criterion for the day. If a day's exit criterion isn't met, fix it before moving on. The checkpoints on Day 6 and Day 11 are the go/no-go gates.

### Phase A: Ingestion + baseline (Sep 26 – Oct 1)

#### Day 1 · Sat 26 Sep: Environment and data
- ☐ Create the repo and folder structure above. Set up `requirements.txt` (`kafka-python` or `confluent-kafka`, `pyspark`, `river`, `influxdb-client`, `pandas`, `pyarrow`, `pytest`, `matplotlib`, `psutil`).
- ☐ Write `docker-compose.yml` with Kafka (KRaft), Spark master + worker, InfluxDB 2 and Grafana. Run `docker compose up` and check that each UI or port responds.
- ☐ Download Sentiment140 (`training.1600000.processed.noemoticon.csv`).
- ☐ EDA notebook: parse dates, sort by time, plot the **hourly class distribution** (reproduce paper Fig. 1a), and locate the all-negative tail near instance ≈1.326M. Note the time gaps.
- ✅ **Done when:** all containers are up, and a sorted `data/processed/ds1.parquet` exists with an EDA plot that matches the paper's Fig. 1a.

#### Day 2 · Sun 27 Sep: Preprocessing and drift scenarios
- ☐ `preprocess.py`: clean and tokenize (rules in §3.4). Apply it identically in the offline and streaming paths through a shared function.
- ☐ Build **DS2** (volatile variant): remove fractions of the positive or negative class in chosen periods to create extra prior shifts. Aim for roughly 1.07M tweets and about 35/65 pos/neg, as in the paper.
- ☐ Build **controlled scenarios** with a known ground truth (`scenarios/*.parquet` + `*_drift_points.json`):
  - **S-sudden:** topic A → topic B at instance *k*. Split topics with keyword filters, e.g. work/school vs. music/movies, as with Obama/Adele.
  - **S-gradual:** P(topic B) rises linearly from 0 to 1 over a transition window.
  - **S-recurring:** A → B → A → B (Costa-style windows).
  - **S-label-flip (real drift, same vocabulary):** at instance *k*, flip the labels of one sub-topic. The vocabulary detector should be blind to this and ADWIN should catch it. This is the key scenario for showing the enhancement's value.
  - **S-prior-shift:** the class balance swings from 50/50 to 10/90.
- ✅ **Done when:** 2 real streams and 5 synthetic streams are saved, each with a drift-point JSON and a sanity plot.

#### Day 3 · Mon 28 Sep: Streaming pipeline end-to-end (dummy model)
- ☐ `producer.py`: read the parquet, send JSON `{id, ts, text, label}` to topic `tweets`, and support `--rate` (tweets/sec) and `--scenario`.
- ☐ `spark_consumer.py`: `readStream` from Kafka → parse → preprocess → `foreachBatch(process_batch)`.
- ☐ `influx_sink.py`: write per-batch points for `accuracy`, `n`, `latency_ms`, `throughput`, `lambda` and `drift_event`.
- ☐ Plug in a trivial majority-class model and run 50k tweets end to end.
- ✅ **Done when:** tweets flow Kafka → Spark → InfluxDB, and a bare Grafana panel shows accuracy over time.

#### Day 4 · Tue 29 Sep: MNB models
- ☐ `core/mnb.py`: `AccumulativeMNB`, plus `AgeingMNB(λ)` implementing Eq. 1–2. Store per-(word, class) `count` and `t_last`, compute in log space, and apply Laplace smoothing. Add `predict_one`, `learn_one` and `set_lambda`.
- ☐ Make the normalizer in Eq. 2 efficient. Either cache the per-class decayed total and update it lazily, or recompute it once per micro-batch. Record which one you chose.
- ☐ `core/metrics.py`: prequential loop with windowed accuracy (`evalW`), plus P, R and F1 (macro).
- ☐ Unit tests: λ=0 must equal AccumulativeMNB, which must match sklearn `MultinomialNB.partial_fit` predictions on a small sample.
- ✅ **Done when:** tests pass, and offline runs of accumulative and fading (λ=0.2) complete on the full DS1 in a reasonable time (under about 15 min).

#### Day 5 · Wed 30 Sep: Vocabulary detector and λ strategies
- ☐ `core/vocab_detector.py`: per class, reference vocabulary `V` (the accumulated stream) vs. sliding-window vocabulary `V_sl` (last *w* instances). Compute precision (Eq. 3), keep a running μ and σ, emit `WARNING` at β=0.334 and `CHANGE` at α=1.8, and buffer instances during a warning.
- ☐ `core/strategies.py`: at minimum **FastSetFastReset**, **SlowIncreaseUpToALimit** and **Rebuild**. SlowIncreaseFastReset and FastSetSlowDecrease are small variants, so add them if time permits.
- ☐ Wire everything into `InformedAgeingMNB(detector, strategy)`, which logs every warning, change and λ value.
- ☐ Sanity check on S-sudden: the detector fires shortly after *k*.
- ✅ **Done when:** the baseline runs on S-sudden and DS1 and outputs change points and a λ trace.

#### Day 6 · Thu 1 Oct: Baseline reproduction ★ Checkpoint 1
- ☐ Run the full DS1 and DS2 with accumulative, fading, and the 3–5 informed strategies, using Table 1 parameters. Tune λ briefly if your time unit differs.
- ☐ Compare against paper Fig. 2: the ordering should hold (accumulative worst; ageing and informed about 5 points better) and there should be a clear accuracy jump after the change point at ≈1.326M.
- ☐ Save `results/baseline_*.csv` and **freeze the baseline** (git tag `baseline-v1`).
- ✅ **Gate:** the baseline reproduces the paper's qualitative findings. If the numbers are off but the ordering holds, document it and move on; don't sink more than half a day here.

### Phase B: Enhancement (Oct 2 – Oct 4)

#### Day 7 · Fri 2 Oct: ADWIN error detector
- ☐ `core/adwin_detector.py`: wrap `river.drift.ADWIN(delta=…)` and feed `err = int(y_pred != y_true)` per instance during prequential evaluation.
- ☐ `EnhancedMNB`: combine the two signals, and implement these fusion modes as a config switch (the ablations become free):
  - `vocab_only` (= baseline)
  - `adwin_only`
  - `OR`: either detector fires, then apply the strategy
  - `confirm`: a vocabulary WARNING plus ADWIN within *m* instances counts as CHANGE; ADWIN alone also triggers
- ☐ Decide the reaction to each signal, for example ADWIN → FastSet λmax or Rebuild from the warning buffer, and vocabulary change → the existing strategy. Add a cooldown so both detectors firing doesn't double-trigger.
- ✅ **Done when:** on **S-label-flip**, ADWIN detects the drift and the vocabulary detector does not (or does so much later). This single result is the headline plot.

#### Day 8 · Sat 3 Oct: Evaluation metrics and tuning
- ☐ `metrics.py` additions, using the ground-truth points and a tolerance window *T*:
  - **Detection delay:** first detection after the true point.
  - **False alarms:** detections outside every [p, p+T] window.
  - **Missed detections.**
  - **Recovery time:** instances after drift until windowed accuracy is back to ≥95% of the pre-drift level.
  - **System metrics:** throughput (tweets/s), per-batch latency (p50/p95) and memory (`tracemalloc` / `psutil` RSS, plus model size = number of (word, class) entries).
- ☐ Small sensitivity sweep on synthetic scenarios: ADWIN `delta ∈ {0.002, 0.01, 0.05}` and α ∈ {1.5, 1.8, 2.5}. Pick defaults, then freeze them.
- ✅ **Done when:** one command outputs a complete metrics row for any (model, scenario) pair.

#### Day 9 · Sun 4 Oct: Grafana and full streaming run
- ☐ Grafana dashboard (export it to `grafana/dashboard.json` and provision it):
  - rolling accuracy for baseline vs. enhanced
  - F1
  - λ over time
  - **drift events as annotations**, colored by detector (vocab vs. ADWIN)
  - throughput
  - latency p95
  - memory
- ☐ Run the enhanced and baseline models side by side through Kafka + Spark, either as two consumer groups or both models in one `foreachBatch`, on DS1 and S-label-flip.
- ☐ Measure streaming throughput and latency at 2–3 producer rates.
- ☐ **Buffer:** use any leftover time to fix issues from Days 1–8.
- ✅ **Done when:** a live dashboard shows both models, the drift markers and the system metrics. Take screenshots for the report.

### Phase C: Experiments and analysis (Oct 5 – Oct 6)

#### Day 10 · Mon 5 Oct: Full experiment grid
- ☐ `run_grid.py` covering:
  - **Models:** accumulative, fading, baseline-informed (best 2 strategies), enhanced (`adwin_only`, `OR`, `confirm`)
  - **Streams:** DS1, DS2, S-sudden, S-gradual, S-recurring, S-label-flip, S-prior-shift
- ☐ Run synthetic scenarios with **3–5 random seeds** (different tweet samples) and report the mean ± std.
- ☐ Write everything to `results/grid.csv`. Log each config as YAML for reproducibility.
- ✅ **Done when:** the grid is complete with no missing cells.

#### Day 11 · Tue 6 Oct: Analysis and inferences ★ Checkpoint 2
- ☐ Figures:
  1. Accuracy over time with drift markers for DS1 and DS2 (paper Fig. 3 style).
  2. Headline S-label-flip plot: baseline vs. enhanced accuracy, with detector firings.
  3. Detection delay and false-alarm bar chart per scenario.
  4. Recovery time per scenario.
  5. Throughput, latency and memory table for baseline vs. enhanced (the cost of adding ADWIN).
  6. Ablation across fusion modes.
- ☐ Main results table: accuracy, P, R, F1, delay, false alarms, recovery, throughput, latency and memory.
- ☐ Write 6–10 bullet **inferences**, for example: which drift types each detector catches, whether the enhancement costs anything on the real DS1, and the false-alarm trade-off.
- ✅ **Gate:** all figures and tables are final. No more experiments after today, except a re-run of something broken.

### Phase D: Report, README, demo (Oct 7 – Oct 10)

#### Day 12 · Wed 7 Oct: Report part 1 (IEEE conference template, Overleaf)
- ☐ Abstract and Introduction: motivate with event-driven drift (Bechini) and state the contributions.
- ☐ Related Work: the 3 base papers, blind vs. informed adaptation, ADWIN (Bifet & Gavaldà 2007) and the Gama et al. 2014 drift survey.
- ☐ Methodology: ageing MNB equations, the vocabulary detector, the λ strategies, **the ADWIN enhancement and fusion logic** (with a pseudocode algorithm box), and the scenario construction.
- ☐ System architecture figure: Kafka → Spark → model → InfluxDB → Grafana.
- ✅ **Done when:** sections I–IV are drafted.

#### Day 13 · Thu 8 Oct: Report part 2 and README
- ☐ Experimental Setup (datasets, parameters, metrics), Results (figures and tables from Day 11), Discussion, Limitations, Conclusion and Future Work, References.
- ☐ Limitations to state honestly: Sentiment140 labels are noisy (emoticon-derived), synthetic drift is artificial, the model is sequential so Spark parallelism is limited, and the dataset covers only 2009.
- ☐ `README.md`: overview, architecture diagram, prerequisites, `docker compose up`, how to run the producer and consumer, how to reproduce the grid and figures, the results summary table, repo layout and team contributions.
- ✅ **Done when:** the full report PDF compiles (check the page limit) and the README lets someone reproduce the results.

#### Day 14 · Fri 9 Oct: Demo and slides
- ☐ Demo script (about 5–7 min):
  1. Architecture overview
  2. `docker compose up`
  3. Start the producer on S-label-flip at a visible rate
  4. Grafana: baseline accuracy drops while the vocabulary detector stays silent
  5. Enhanced: ADWIN fires, λ jumps and accuracy recovers
  6. The DS1 natural change point
  7. Results table
- ☐ Record the demo (OBS). Also keep a pre-recorded backup in case live infrastructure fails in class.
- ☐ Slides (10–12): problem, base papers, baseline, enhancement, setup, key results, limitations, Q&A backup slides.
- ✅ **Done when:** the video is recorded and the slides are done.

#### Day 15 · Sat 10 Oct: Final QA and submission 🎯
- ☐ **Fresh-clone test** on a second machine or a clean folder: follow only the README and confirm everything runs.
- ☐ Proofread the report: check that numbers in the text match the tables, that figures are readable in print, and that references are complete.
- ☐ Clean the repo: remove secrets, keep large data out of git, pin `requirements.txt`, add `LICENSE` if needed.
- ☐ Package the submission: report PDF, repo link or zip, demo video and slides.
- ✅ **Done when:** everything is submitted. ✔

---

## 5. After the deadline (11 – 21 Oct)
- Rehearse the presentation 2–3 times and time it.
- Prepare Q&A answers:
  - Why MNB?
  - Why ADWIN and not DDM or Page-Hinkley?
  - How is detection delay measured?
  - Why use Spark if the model is sequential?
  - How were the synthetic drifts made?

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Spark/Kafka setup eats Day 1–3 | Use Bitnami images. If Spark still fails, a plain Python Kafka consumer is acceptable for the demo; keep Spark in the report as the design. Don't let the infrastructure block Phase B. |
| AgeingMNB too slow on 1.6M | Use lazy decay: store `(count, t_last)` and decay on read. Run the grid with the offline runner. Subsample to every 2nd tweet only as a last resort, and document it. |
| Baseline numbers ≠ paper | Match the **qualitative** ordering and document the differences (time unit, preprocessing, smoothing). |
| Enhancement shows no gain on DS1 | Expected: DS1's drift is a prior shift, which both detectors catch. The value shows on S-label-flip, S-gradual and in recovery time, so frame the result that way. |
| Falling behind | Cut in this order: extra λ strategies → S-gradual seeds → multiple streaming rates → Grafana polish. **Never cut:** the baseline, the ADWIN enhancement, the S-label-flip result, the report or the README. |

## 7. Daily habit
- Push to git at the end of every day.
- Write a 3-line log in `NOTES.md` (done / blocked / next). It becomes the raw material for the report.
