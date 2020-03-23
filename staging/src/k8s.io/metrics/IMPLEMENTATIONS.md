# Implementations

## Resource Metrics API

- [Heapster](https://github.com/kubernetes/heapster): a application which
  gathers metrics, writes them to metrics storage "sinks", and exposes the
  resource metrics API from in-memory storage.

- [Metrics Server](https://github.com/kubernetes-incubator/metrics-server):
  a lighter-weight in-memory server specifically for the resource metrics
  API.

## Custom Metrics API

***NB: None of the below implementations are officially part of Kubernetes.
They are listed here for convenience.***

- [Prometheus
  Adapter](https://github.com/directxman12/k8s-prometheus-adapter).  An
  implementation of the custom metrics API that attempts to support
  arbitrary metrics following a set label and naming scheme.

- [Microsoft Azure Adapter](https://github.com/Azure/azure-k8s-metrics-adapter). An implementation of the custom metrics API that allows you to retrieve arbitrary metrics from Azure Monitor.

- [Google Stackdriver (coming
  soon)](https://github.com/GoogleCloudPlatform/k8s-stackdriver)

- [Datadog Cluster Agent](https://github.com/DataDog/datadog-agent/blob/c4f38af1897bac294d8fed6285098b14aafa6178/docs/cluster-agent/CUSTOM_METRICS_SERVER.md).
  Implementation of the external metrics provider, using Datadog as a backend for the metrics.
  Coming soon: Implementation of the custom metrics provider to support in-cluster metrics collected by the Datadog Agents.

- [Kube Metrics Adapter](https://github.com/zalando-incubator/kube-metrics-adapter). A general purpose metrics adapter for Kubernetes that can collect and serve custom and external metrics for Horizontal Pod Autoscaling.
  Provides the ability to scrape pods directly or from Prometheus through user defined queries.
  Also capable of serving external metrics from a number of sources including AWS' SQS and [ZMON monitoring](https://github.com/zalando/zmon).