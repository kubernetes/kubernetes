# Implementations

## Resource Metrics API

- [Heapster](https://github.com/kubernetes/heapster): a application which
  gathers metrics, writes them to metrics storage "sinks", and exposes the
  resource metrics API from in-memory storage.

- [Metrics Server](https://github.com/kubernetes-incubator/metrics-server):
  a lighter-weight in-memory server specifically for the resource metrics
  API.

## Custom Metrics API

***NB: None of the below implemenations are officially part of Kubernetes.
They are listed here for convenience.***

- [Prometheus
  Adapter](https://github.com/directxman12/k8s-prometheus-adapter).  An
  implementation of the custom metrics API that attempts to support
  arbitrary metrics following a set label and naming scheme.

- [Google Stackdriver (coming
  soon)](https://github.com/GoogleCloudPlatform/k8s-stackdriver)
