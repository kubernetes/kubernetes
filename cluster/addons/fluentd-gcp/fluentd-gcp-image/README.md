# Collecting Docker Log Files with Fluentd and sending to GCP.
This directory contains the source files needed to make a Docker image
that collects Docker container log files using [Fluentd](http://www.fluentd.org/)
and sends them to GCP.
This image is designed to be used as part of the [Kubernetes](https://github.com/kubernetes/kubernetes)
cluster bring up process. The image resides at DockerHub under the name
[kubernetes/fluentd-gcp](https://registry.hub.docker.com/u/kubernetes/fluentd-gcp/).



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/fluentd-gcp/fluentd-gcp-image/README.md?pixel)]()
