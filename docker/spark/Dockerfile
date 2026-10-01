# Spark 3.5.3 + the Kafka source for Structured Streaming + the InfluxDB client.
# The connector jars are baked in so spark-submit needs no --packages download at run time.
FROM apache/spark:3.5.3-scala2.12-java17-python3-ubuntu

USER root

ARG MAVEN=https://repo1.maven.org/maven2
ADD --chmod=644 ${MAVEN}/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.3/spark-sql-kafka-0-10_2.12-3.5.3.jar /opt/spark/jars/
ADD --chmod=644 ${MAVEN}/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.3/spark-token-provider-kafka-0-10_2.12-3.5.3.jar /opt/spark/jars/
ADD --chmod=644 ${MAVEN}/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar /opt/spark/jars/
ADD --chmod=644 ${MAVEN}/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar /opt/spark/jars/

# The driver runs the model inside foreachBatch and writes metrics to InfluxDB.
# river provides ADWIN for the enhanced model (driver side).
RUN python3 -m pip install --no-cache-dir "influxdb-client==1.48.0" "numpy==1.26.4" "river==0.22.0"

# Project code is bind-mounted here; executors import core/ from it for the tokenizer UDF.
ENV PYTHONPATH=/opt/project

USER spark
