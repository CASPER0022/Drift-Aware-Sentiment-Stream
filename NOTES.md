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
