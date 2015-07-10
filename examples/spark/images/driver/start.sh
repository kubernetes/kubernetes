#!/bin/bash

echo "$SPARK_MASTER_SERVICE_HOST spark-master" >> /etc/hosts
echo "SPARK_LOCAL_HOSTNAME=$(hostname -i)" >> /opt/spark/conf/spark-env.sh
echo "MASTER=spark://spark-master:$SPARK_MASTER_SERVICE_PORT" >> /opt/spark/conf/spark-env.sh

while true; do
sleep 100
done
