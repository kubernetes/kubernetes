# Collecting Docker Log Files with Fluentd and sending to GCP.
This directory contains the source files needed to make a Docker image
that collects Docker container log files using [Fluentd](http://www.fluentd.org/)
and sends them to GCP.
This image is designed to be used as part of the [LMKTFY](https://github.com/GoogleCloudPlatform/lmktfy)
cluster bring up process. The image resides at DockerHub under the name
[lmktfy/fluentd-gcp](https://registry.hub.docker.com/u/lmktfy/fluentd-gcp/).

