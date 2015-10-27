<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Elasticsearch/Kibana Logging Demonstration

This directory contains two [pod](../../../docs/user-guide/pods.md) specifications which can be used as synthetic
logging sources. The pod specification in [synthetic_0_25lps.yaml](synthetic_0_25lps.yaml)
describes a pod that just emits a log message once every 4 seconds. The pod specification in
[synthetic_10lps.yaml](synthetic_10lps.yaml)
describes a pod that just emits 10 log lines per second.

See [logging document](../logging.md) for more details about logging. To observe the ingested log lines when using Google Cloud Logging please see the getting
started instructions
at [Cluster Level Logging to Google Cloud Logging](../../../docs/getting-started-guides/logging.md).
To observe the ingested log lines when using Elasticsearch and Kibana please see the getting
started instructions
at [Cluster Level Logging with Elasticsearch and Kibana](../../../docs/getting-started-guides/logging-elasticsearch.md).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/logging-demo/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
