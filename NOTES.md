# Daily log

## Day 1 · Sat 26 Sep
- **Done:** Docker stack up (Kafka, Kafka UI, Spark master/worker, InfluxDB, Grafana); DS1 = Sentiment140 sorted by time (`pipeline/build_ds1.py`); EDA figure reproduces paper Fig. 1a; natural change point at instance 1,324,775 (paper ~1,326,000).
- **Blocked:** nothing.
- **Next:** preprocessing, DS2, controlled drift scenarios.

## Day 2 · Sun 27 Sep (finished 30 Sep)
- **Done:** shared tokenizer `core/preprocess.py` (+7 tests); DS2 (`pipeline/build_ds2.py`) hits the paper's exact counts (1,073,065 tweets, 378,288 pos / 694,777 neg) with 6 known prior-shift points + natural tail; 5 controlled scenarios x 5 seeds (`pipeline/build_scenarios.py`, 80k tweets each) with drift-point JSONs; sanity plots (`analysis/plot_streams.py`).
- **Findings:** DS1 holds 1,685 tweet ids twice with *opposite* labels (label noise) - kept in DS1/DS2, excluded from scenarios. Topic pool B (music/movies) has only ~21.5k negatives, which caps balanced scenarios at 80k and makes seeds overlap ~50% of tweets. The paper does not say how DS2 was cut, so our segment pattern is our own design (state this in the report).
- **Next:** Day 3 - Kafka producer, Spark consumer, InfluxDB sink, dummy model end to end.

## Day 3 · Mon 28 Sep (done 30 Sep)
- **Done:** custom Spark image (`docker/spark/Dockerfile`: Kafka connector jars + influxdb-client + numpy); `pipeline/producer.py` (paced replay, one partition, idempotent); `pipeline/spark_consumer.py` (Kafka -> JSON -> tokenizer UDF on executors -> foreachBatch prequential loop on the driver, one model per run); `pipeline/influx_sink.py`; `core/majority.py` placeholder model; provisioned Grafana dashboard "Drift Stream" (accuracy, throughput, latency, run picker). 50k DS1 tweets at 2,000/s went end to end: 50,000 seen, majority accuracy 58.5% (= early DS1 positive share), p95 latency ~1.5 s once warm (~3.5 s in the first batches).
- **Notes:** the model lives in driver memory, so restarting the consumer restarts the model (each start uses a fresh checkpoint for that reason). Docker Desktop must be running before `docker compose up -d`.
- **Next:** Day 4 - AccumulativeMNB / AgeingMNB, prequential metrics, tests against sklearn.

## Day 4 · Tue 29 Sep (done 30 Sep)
- **Done:** `core/mnb.py` - AccumulativeMNB and AgeingMNB (Eq. 1-2, ageing on read from count + last-seen time, Laplace alpha=1, log space, `set_lambda`). Eq. 2's normaliser is a cached per-class sum rebased against a reference time (O(1) per word, exact rebuild on `set_lambda`). `core/metrics.py` (prequential loop, windowed accuracy, macro P/R/F1), `core/timeunit.py`, `experiments/runner_offline.py`. Tests: matches sklearn MultinomialNB to 1e-9, lam=0 == accumulative, cached normaliser == literal equations (19 tests). Spark consumer now runs the MNB models.
- **Decision change:** model time `t` is **hours** (from the tweet timestamp), not the instance index. With lam=0.2 per instance the model forgets everything within ~100 tweets (DS1 pre-change accuracy 68%).
- **DS1 results (offline, full 1.6M):** accumulative 78.3% (paper ~73%), fading lam=0.2/h 80.2% (paper ~78%), fading lam=0.2/instance 73.5%. Fading reaches 99.2% on the all-negative tail vs 77.2% for accumulative; before the change it is 2.3 pts worse (cost of forgetting in stable periods). Ordering matches the paper; our accumulative baseline is ~5 pts stronger (preprocessing/smoothing?) - document, don't chase. Runtime 17-29 s per model on DS1 (~55-95k tweets/s).
- **Streaming:** 60k tweets across the change point through Kafka+Spark with fading: tail batches reach 100%. At 3,000/s latency grew 2.8 -> 6.3 s (consumer falls behind) - measure on Day 9.
- **Next:** Day 5 - vocabulary change detector + lambda strategies.

## Day 5 · Wed 30 Sep
- **Done:** `core/vocab_detector.py` (Eq. 3-4 per class, warning buffer), `core/strategies.py` (all 5 strategies: SlowIncreaseUpToALimit, SlowIncreaseFastReset, FastSetFastReset, FastSetSlowDecrease, Rebuild), `core/informed.py` (InformedAgeingMNB: logs checks, warnings, changes, rebuilds and the lambda trace; `drain_events` for Grafana). Runner saves `results/events/<run>.json`. 29 tests.
- **Finding 1 - reference vocabulary:** an *accumulated* reference (whole stream so far) makes precision fall steadily (0.45 -> 0.10 on S-sudden) because V grows while V_sl does not, so the detector fires constantly and misses the real switch. The paper says "reference window of past data": we use the **previous window** (same size w). Keep the accumulated mode as an option for the report.
- **Finding 2 - history length:** mu/sigma over 10 checks gives many false alarms; 20 checks fixes it. S-sudden with w=4,000, history 20: exactly one change at 43,999 (true switch 40,000, first check after it). w=2,000 reacts sooner (41,999) but adds 4 false alarms.
- **Finding 3 - topic drift is weak for this statistic:** topic A/B differ in a few keywords, so precision dips only ~2-3 sigma at the switch. Accuracy barely moves on S-sudden either (virtual drift: P(y|X) unchanged).
- **DS1, Table 1 params, w=24k:** 7 changes incl. 1,343,999 (first check after the natural change at 1,324,775). Accuracy: Rebuild 80.52%, FastSetFastReset 80.45%, SlowIncreaseFastReset 80.14%, FastSetSlowDecrease 80.05%, SlowIncreaseUpToALimit 79.66% vs fading 80.17%, accumulative 78.31%. Ordering as in the paper.
- **Open choices (document):** SlowIncreaseUpToALimit lam_max not given in Table 1 - used 0.5; SlowIncreaseFastReset c=0.4 read from Table 1.
- **Next:** Day 6 - baseline reproduction on DS1 + DS2, figures vs paper Fig. 2/3, tag `baseline-v1`.

## Day 6 · Thu 1 Oct (done 30 Sep) - Checkpoint 1
- **Done:** `experiments/configs/baseline.yaml` (Table 1 params, DS1 + DS2, all 12 configs incl. -Zero-/-Init-), `experiments/run_baseline.py` -> `results/baseline_{summary,windows,events,lambda}.csv`, `analysis/plot_baseline.py` -> Fig. 2/3-style figures.
- **Results (overall accuracy):** DS1 accumulative 78.3%, fading 80.2%, informed 79.4-81.4% (best SlowIncreaseFastReset-Zero 81.4%, Rebuild-Zero 80.8%). DS2 accumulative 80.3%, fading 82.5%, informed 81.6-83.0% (best SlowIncreaseFastReset-Zero / Rebuild-Zero 83.0%). After the natural change point adaptive models reach 93-99.9% vs 77% (DS1) / 87% (DS2) for accumulative.
- **Detector on DS2:** fires near all 7 true drift points (6 segment prior shifts + natural tail); on DS1 first change after the natural point at 1,343,999.
- **Gate: PASSED (qualitative).** Matches paper: accumulative worst on both streams; adaptive ~2-3 pts better (paper ~5); clear jump after the change point. **Differs:** paper finds -Init- (lambda0 > 0) better; we find -Zero- better, because constant ageing costs ~1-2 pts in stable periods here. Lambda sweep for fadingMNB on DS1 (0.02/0.05/0.1/0.2 per hour -> 80.2/80.5/80.4/80.2%) shows no lambda removes that trade-off (pre-change 77.8 -> 76.3% as lambda grows; tail 92 -> 99%), so we keep Table 1 params and report the difference. It supports the paper's own argument for informed adaptation: forget only when the detector says so.
- **Frozen:** tag `baseline-v1`.
- **Next:** Day 7 - ADWIN on the prediction error + fusion modes; headline result on S-label-flip.
