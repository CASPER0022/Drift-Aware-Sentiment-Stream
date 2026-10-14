# Demo script (5-7 minutes)

Commands are for **PowerShell** in the project folder (`cd "E:\Downloads\SEM 7\Streaming Data Analysis\Project"`).
Use three windows: **A** = Spark consumer, **B** = producer, **C** = browser (Grafana).

## Before the session (10 minutes earlier)

1. Start Docker Desktop and wait until it says *Engine running*.
2. `docker compose up -d` and check all six containers: `docker ps` (kafka and influxdb show *healthy*).
3. Open <http://localhost:3000/d/drift-stream> and log in. Open the slides (`report/slides.pdf`).
4. Start the consumer in window **A** (it keeps the window busy):
   ```powershell
   docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --conf spark.driver.host=spark-master --conf spark.executorEnv.PYTHONPATH=/opt/project /opt/project/pipeline/spark_consumer.py
   ```
   Wait for `consuming 'tweets' from kafka:29092 with models ['baseline', 'enhanced']`.
5. Dry run: `.\.venv\Scripts\python.exe pipeline\producer.py --stream s_label_flip --limit 5000 --rate 1000`
   and check that the dashboard shows a new run. Then set the dashboard time range to *Last 5 minutes*.
6. Keep the backup ready: the pre-recorded video, and `analysis/figures/grafana_label_flip.png` /
   `grafana_ds1.png` (screenshots of real runs).

## Script

| Time | Show | Say |
|---|---|---|
| 0:00-1:00 | Slides 1-3 | Problem: sentiment streams drift. The paper's detector watches the *words*; we ask whether it sees changes in *meaning*, and add ADWIN on the model's *errors*. |
| 1:00-1:45 | Slide 7 (System) | Producer replays tweets in time order into Kafka; Spark executors tokenise; the driver runs the **baseline** and the **enhanced** model side by side on the same micro-batches; metrics go to InfluxDB and Grafana. |
| 1:45-2:00 | `docker ps` in window B | Everything runs in Docker: Kafka, Spark master + worker, InfluxDB, Grafana. |
| 2:00-2:15 | Window B: start the producer | `.\.venv\Scripts\python.exe pipeline\producer.py --stream s_label_flip --rate 1000`. This is the label-flip scenario: 80,000 real tweets, and at tweet 40,000 every label inverts while the words stay the same. It takes 80 seconds. |
| 2:15-2:55 | Grafana, Run = the newest run | Both models learn; accuracy ~75%. Point out the panels: accuracy, cumulative accuracy, F1, model size, throughput (~1,000/s), latency (~0.4 s), driver memory. Window A prints one line per batch. |
| 2:55-3:40 | Grafana at ~40 s (tweet 40k) | **The flip.** Both models crash. The **blue ADWIN marker** appears within ~40 tweets; the enhanced model rebuilds (model size drops) and climbs back. The **orange baseline stays near 25%** for ~4,000 tweets until its vocabulary detector fires (orange marker). |
| 3:40-4:10 | Cumulative accuracy panel | Final: enhanced ~74% vs baseline ~72%, same numbers as the offline grid. |
| 4:10-5:00 | Window B: DS1 change point | `.\.venv\Scripts\python.exe pipeline\producer.py --stream ds1 --start 1250000 --limit 150000 --rate 3000` (50 s). Real data: Sentiment140 turns all negative at tweet 1,324,775 (~25 s in). ADWIN fires ~650 tweets after it; the enhanced model reaches 100% first. Macro F1 drops to 0.5 for both: only one class is left, so this is expected. |
| 5:00-6:30 | Slides 9-12 | Results over 224 runs: detection 3,999 -> 39 tweets, recovery 8,204 -> 3,183, accuracy 71.0 -> 73.8%; topic-only flip: vocabulary detector misses 2/5, ADWIN catches 5/5; DS1/DS2 +0.5-1.1 pt; cost +10 ms per 1k tweets. |
| 6:30-7:00 | Slide 13 | Takeaways and limitations. Questions (backup slide 15). |

## If something fails live

- **No new run in Grafana:** check window A for a traceback; restart with `docker restart spark-master spark-worker` and start the consumer again (step 4). The producer can be rerun at any time; every replay gets a new run id.
- **Docker will not start:** switch to the pre-recorded video, then the two Grafana screenshots.
- **Dashboard shows "No data":** set the time range to *Last 15 minutes* and pick the newest run in the **Run** menu.
- **Consumer falls behind** (latency climbing): lower `--rate`, or restart the consumer with `--batch-size 5000`.

## After the demo

`docker compose stop` (keeps all data; `docker compose start` to resume).

## Recording (OBS)

Record the full script once as the backup video: 1920x1080, browser at 100% zoom, Grafana in
kiosk mode (`...&kiosk` added to the URL hides the menus), slides in a second scene.
