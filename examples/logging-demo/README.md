# Elasticsearch/Kibana Logging Demonstration
This directory contains two [pod](../../docs/pods.md) specifications which can be used as synthetic
logging sources. The pod specification in [synthetic_0_25lps.yaml](synthetic_0_25lps.yaml)
describes a pod that just emits a log message once every 4 seconds. The pod specification in
[synthetic_10lps.yaml](synthetic_10lps.yaml)
describes a pod that just emits 10 log lines per second.

To observe the ingested log lines when using Google Cloud Logging please see the getting
started instructions
at [Cluster Level Logging to Google Cloud Logging](/docs/getting-started-guides/logging.md).
To observe the ingested log lines when using Elasticsearch and Kibana please see the getting
started instructions
at [Cluster Level Logging with Elasticsearch and Kibana](/docs/getting-started-guides/logging-elasticsearch.md).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/logging-demo/README.md?pixel)]()
