#!/bin/bash
# This script runs the docker-fluentd image which sets up the
# ingestion of Docker log files into Elasticsearch. This configuration
# depends on a container running Elasticsearch which is linked.
# Such a container can be run with:
#   docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -v /data:/data dockerfile/elasticsearch

set -e

docker run --name docker-fluentd -d --link elasticsearch:elasticsearch -v /var/lib/docker/containers:/var/lib/docker/containers satnam6502/docker-fluentd