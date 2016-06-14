<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Resource Usage Metrics plumbing in Kubernetes

**Author**: Vishnu Kannan (@vishh)

**Status**: Draft proposal; some parts are already implemented

* This document presents a design for handling container metrics in Kubernetes clusters*

## Motivation

Resource usage metrics are critical for various reasons:
* Monitor and maintain the health of the cluster and user applications.
* Improve the efficiency of the cluster by making more optimal scheduling decisions and enabling components like auto-scalers.

There are multiple types of metrics that describe the state of a container.
Numerous strategies exist to aggregate these metrics from containers.
There are a variety of storage backends that can handle metrics.

This document presents a design to abstract out collection and storage backends, and provide stable Kubernetes APIs that can be consumed by users and other cluster components.

## Introduction

Container metrics can be of two types.

1. `Compute resource metrics` refers to compute resources being used by a container. Ex.: CPU, Memory, Network, File-system
2. `Service metrics` refers to container app specific metrics. Ex: QPS, query latency, etc.

Metrics can be collected either for cluster components or for user containers.

[cAdvisor](https://github.com/google/cadvisor) is a node level container metrics aggregator that is built into the kubelet. cAdvisor can collect both types of metrics, although the support for service metrics is limited at this point. cAdvisor collects metrics for both system components and user containers.

[heapster](https://github.com/kubernetes/heapster) is a cluster level metrics aggregator that is run by default on most Kubernetes cluster. Heapster aggregates all the metrics exposed by cAdvisor from the nodes. Heapster has a pluggable storage backend. It supports the following timeseries storage backends - InfluxDB, Google Cloud Monitoring and Hawkular.
Heapster builds a model of the cluster and can aggregate metrics across pods, nodes, namespaces and the entire cluster. It exposes this data via [REST endpoints](https://github.com/kubernetes/heapster/blob/master/docs/model.md#api-documentation).

Metrics data will be consumed by many different clients - scheduler, horizontal and vertical pod auto scalers, initial pod limits controller, kubectl, web consoles, cluster management software, etc.

Storage backends can be shared for both monitoring of clusters and powering advanced cluster features.

## Goals

* Abstract out timeseries storage backends from Kubernetes components.
* Provide stable Kubernetes Metrics APIs that other components can consume.

#### Non Goals

* Requiring users to run a specific storage backend.
* Compatibility with other node level metrics aggregator. cAdvisor should be able to provide all the metrics.
* Support for service metrics at the cluster level is out of scope for this document.
Once the use cases for service metrics, other than monitoring, are clear, we can explore adding support for service metrics.

## Design

The basic idea is to evolve heapster to serve Metrics APIs which can then be consumed by other cluster components.
Heapster will be run in all clusters by default. Heapster's memory usage is proportional to the number of containers in the cluster and so it should be possible to run heapster by default even on small development or test clusters.
A cluster administrator will have to either run one of the supported storage backends or write a new storage plugin in heapster to support custom storage backends.
Heapster will manage versioning and storage schema for the various storage backends it supports.
Heapster APIs will be exposed as Kubernetes APIs once the apiserver supports [dynamic API plugins](https://github.com/kubernetes/kubernetes/issues/991).

Heapster stores a days worth of historical metrics. Heapster will fetch data from storage backends on-demand to serve metrics that are older than a day. Setting [initial pod resources](initial-resources.md) requires access to metrics from the past 30 days.

To make heapster APIs compatible with Kubernetes API requirements, heapster will have to incorporate the API server library. Until that is possible, we will run a secondary API server binary that supports the metrics APIs being consumed by other components. The initial plan is to use etcd to store the most recent metrics. Eventually, we would like to get rid of etcd for metrics and make heapster act as a backend to the api-server.

This is the current plan for supporting node and pod metrics API as described in this [proposal](resource-metrics-api.md).

There will be proposals in the future for adding more heapster metrics APIs in Kubernetes.

## Implementation plan

Heapster has an in-build model of a cluster and can expose the average, 95%ile and max of compute resource metrics for containers, pods, nodes, namespaces and entire cluster.
However the existing APIs are not suitable for Kubernetes components.
The metrics are stored in a rolling window. Adding support for other percentiles should be straightforward.
Heapster is currently stateless and so it will loose its history upon restart.
Some of the specific work items include,

1. Improve the existing API schema to be Kubernetes compatible ([Related issue](https://github.com/kubernetes/heapster/issues/476))
2. Add support for fetching historical data from storage backends.
3. Fetch historical metrics from storage backends upon restarts to pre-populate the internal model.
4. Add support for image based aggregation.
5. Add support for label queries.
6. Expose heapster APIs via a Kubernetes service until the primary API server can handle plugins.

### Non Goals

### Known issues

* Running other metrics aggregators

  An example here would be running collectd in-place of cadvisor and storing metrics to a custom database or running prometheus. We can let cluster admins run their own aggregation and storage stack as long as the storage backend is supported in heapster and the storage schema is versioned. Compatibility can be guaranteed by explicitly specifying the versions of different components that are supported in a specific Kubernetes release.

* Heapster scalability

  Heapster's resource utilization is proportional to the number of containers running in the cluster. A fair amount of effort has gone into optimizing heapster's memory usage. As our cluster size increases, we can shard heapster. We believe the existing heapster design should scale for fairly large clusters with reasonable amount of compute resources.

### How can you contribute?

We are tracking heapster work items using [milestones](https://github.com/kubernetes/heapster/milestones) in the heapster repo.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/metrics-plumbing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
