#!/usr/bin/env bash
# Start the streaming consumer on the Docker Spark cluster (driver in spark-master,
# executors on spark-worker). Arguments are passed to spark_consumer.py, e.g.
#   bash pipeline/submit_consumer.sh --batch-size 1000 --model majority
# Stop with Ctrl+C. Driver UI: http://localhost:4040
set -euo pipefail
export MSYS_NO_PATHCONV=1  # keep Git Bash on Windows from rewriting /opt/... paths

tty_flags=-i
[ -t 0 ] && tty_flags=-it

exec docker exec $tty_flags spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --conf spark.driver.host=spark-master \
  --conf spark.executorEnv.PYTHONPATH=/opt/project \
  --conf spark.sql.shuffle.partitions=2 \
  /opt/project/pipeline/spark_consumer.py "$@"
