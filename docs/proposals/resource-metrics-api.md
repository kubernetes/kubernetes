<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Resource Metrics API

*This proposal is based on and supersedes [compute-resource-metrics-api.md](compute-resource-metrics-api.md).*

This document describes API part of MVP version of Resource Metrics API effort in Kubernetes.
Once the agreement will be made the document will be extended to also cover implementation details.
The shape of the effort may be also a subject of changes once we will have more well-defined use cases.

## Goal

The goal for the effort is to provide resource usage metrics for pods and nodes through the API server.
This will be a stable, versioned API which core Kubernetes components can rely on.
In the first version only the well-defined use cases will be handled,
although the API should be easily extensible for potential future use cases.

## Main use cases

This section describes well-defined use cases which should be handled in the first version.
Use cases which are not listed below are out of the scope of MVP version of Resource Metrics API.

#### Horizontal Pod Autoscaler

HPA uses the latest value of cpu usage as an average aggregated across 1 minute
(the window may change in the future). The data for a given set of pods
(defined either by pod list or label selector) should be accesible in one request
due to performance issues.

#### Scheduler

Scheduler in order to schedule best-effort pods requires node level resource usage metrics
as an average aggreated across 1 minute (the window may change in the future).
The metrics should be available for all resources supported in the scheduler.
Currently the scheduler does not need this information, because it schedules best-effort pods
without considering node usage. But having the metrics available in the API server is a blocker
for adding the ability to take node usage into account when scheduling best-effort pods.

## Other considered use cases

This section describes the other considered use cases and explains why they are out
of the scope of the MVP version.

#### Custom metrics in HPA

HPA requires the latest value of application level metrics.

The design of the pipeline for collecting application level metrics should
be revisited and it's not clear whether application level metrics should be
available in API server so the use case initially won't be supported.

#### Ubernetes

Ubernetes might want to consider cluster-level usage (in addition to cluster-level request)
of running pods when choosing where to schedule new pods. Although Ubernetes is still in design,
we expect the metrics API described here to be sufficient. Cluster-level usage can be
obtained by summing over usage of all nodes in the cluster.

#### kubectl top

This feature is not yet specified/implemented although it seems reasonable to provide users information
about resource usage on pod/node level.

Since this feature has not been fully specified yet it will be not supported initally in the API although
it will be probably possible to provide a reasonable implementation of the feature anyway.

#### Kubernetes dashboard

[Kubernetes dashboard](https://github.com/kubernetes/dashboard) in order to draw graphs requires resource usage
in timeseries format from relatively long period of time. The aggreations should be also possible on various levels
including replication controllers, deployments, services, etc.

Since the use case is complicated it will not be supported initally in the API and they will query Heapster
directly using some custom API there.

## Proposed API

Initially the metrics API will be in a separate [API group](api-group.md) called ```metrics```.
Later if we decided to have Node and Pod in different API groups also
NodeMetrics and PodMetrics should be in different API groups.

#### Schema

The proposed schema is as follow. Each top-level object has `TypeMeta` and `ObjectMeta` fields
to be compatible with Kubernetes API standards.

```go
type NodeMetrics struct {
  unversioned.TypeMeta
  ObjectMeta

  // The following fields define time interval from which metrics were
  // collected in the following format [Timestamp-Window, Timestamp].
  Timestamp unversioned.Time
  Window    unversioned.Duration

  // The memory usage is the memory working set.
  Usage v1.ResourceList
}

type PodMetrics struct {
  unversioned.TypeMeta
  ObjectMeta

  // The following fields define time interval from which metrics were
  // collected in the following format [Timestamp-Window, Timestamp].
  Timestamp unversioned.Time
  Window    unversioned.Duration

  // Metrics for all containers are collected within the same time window.
  Containers []ContainerMetrics
}

type ContainerMetrics struct {
  // Container name corresponding to the one from v1.Pod.Spec.Containers.
  Name string
  // The memory usage is the memory working set.
  Usage v1.ResourceList
}
```

By default `Usage` is the mean from samples collected within the returned time window.
The default time window is 1 minute.

#### Endpoints

All endpoints are GET endpoints, rooted at `/apis/metrics/v1alpha1/`.
There won't be support for the other REST methods.

The list of supported endpoints:
- `/nodes` - all node metrics; type `[]NodeMetrics`
- `/nodes/{node}` - metrics for a specified node; type `NodeMetrics`
- `/namespaces/{namespace}/pods` - all pod metrics within namespace with support for `all-namespaces`; type `[]PodMetrics`
- `/namespaces/{namespace}/pods/{pod}` - metrics for a specified pod; type `PodMetrics`

The following query parameters are supported:
- `labelSelector` - restrict the list of returned objects by labels (list endpoints only)

In the future we may want to introduce the following params:
`aggreator` (`max`, `min`, `95th`, etc.) and `window` (`1h`, `1d`, `1w`, etc.)
which will allow to get the other aggregates over the custom time window.

## Further improvements

Depending on the further requirements the following features may be added:
- support for more metrics
- support for application level metrics
- watch for metrics
- possibility to query for window sizes and aggreation functions (though single window size/aggregation function per request)
- cluster level metrics

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/resource-metrics-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
