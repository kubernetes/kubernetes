# Collecting Docker Log Files with Fluentd and Elasticsearch
This directory contains the source files needed to make a Docker image
that collects Docker container log files using [Fluentd][fluentd]
and sends them to an instance of [Elasticsearch][elasticsearch].
This image is designed to be used as part of the [Kubernetes][kubernetes]
cluster bring up process. The image resides at GCR under the name
[k8s.gcr.io/fluentd-elasticsearch][image].

[fluentd]: http://www.fluentd.org/
[elasticsearch]: https://www.elastic.co/products/elasticsearch
[kubernetes]: https://kubernetes.io
[image]: https://k8s.gcr.io/fluentd-elasticsearch

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/fluentd-elasticsearch/fluentd-es-image/README.md?pixel)]()
