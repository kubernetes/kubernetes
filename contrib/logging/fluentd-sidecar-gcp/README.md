# Collecting log files from within containers with Fluentd and sending to the Google Cloud Logging service.
This directory contains the source files needed to make a Docker image that collects log files from arbitrary files within a container using [Fluentd](http://www.fluentd.org/) and sends them to GCP.
This image is designed to be used as a sidecar container as part of a [Kubernetes](https://github.com/GoogleCloudPlatform/kubernetes) pod.
The image resides at DockerHub under the name
[kubernetes/fluentd-sidecar-gcp](https://registry.hub.docker.com/u/kubernetes/fluentd-sidecar-gcp/).

# TODO: Add example pod config.
# TODO: say that it resides at gcr.io instead?
