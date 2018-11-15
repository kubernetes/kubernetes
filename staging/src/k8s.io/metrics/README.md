# metrics

Kubernetes metrics API type definitions and clients.

## Purpose

This repository contains type definitions and client code for the metrics
APIs that Kubernetes makes use of.  Depending on the API, the actual
implementations live elsewhere.

Consumers of the metrics APIs can make use of this repository to access
implementations of the APIs, while implementors should make use of this
library when implementing their API servers.

## APIs

This repository contains types and clients for several APIs.  For more
details on implementations of these apis, see
[IMPLEMENTATIONS.md](IMPLEMENTATIONS.md).

### Custom Metrics API

This API allows consumers to access arbitrary metrics which describe
Kubernetes resources.

The API is intended to be implemented by monitoring pipeline vendors, on
top of their metrics storage solutions.

If you want to implement this an API server for this API, please see the
[kubernetes-incubator/custom-metrics-apiserver](https://github.com/kubernetes-incubator/custom-metrics-apiserver)
library, which contains the basic infrastructure required to set up such
an API server.

Import Path: `k8s.io/metrics/pkg/apis/custom_metrics`.

### Resource Metrics API

This API allows consumers to access resource metrics (CPU and memory) for
pods and nodes.

The API is implemented by Heapster
(https://github.com/kubernetes/heapster) and metrics-server
(https://github.com/kubernetes-incubator/metrics-server).

Import Path: `k8s.io/metrics/pkg/apis/metrics`.

## Compatibility

The APIs in this repository follow the standard guarantees for Kubernetes
APIs, and will follow Kubernetes releases.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community
page](http://kubernetes.io/community/).

You can reach the maintainers of this repository at:

- Slack: #sig-instrumentation (on https://kubernetes.slack.com -- get an
  invite at slack.kubernetes.io)
- Mailing List:
  https://groups.google.com/forum/#!forum/kubernetes-sig-instrumentation

### Code of Conduct

Participation in the Kubernetes community is governed by the [Kubernetes
Code of Conduct](code-of-conduct.md).

### Contibution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
