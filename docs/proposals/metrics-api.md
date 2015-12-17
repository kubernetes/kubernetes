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
This document may be also a subject of changes once we will have more well-defined use cases.

## Goal

The goal for the effort is to provide resource usage metrics for pods and nodes through the API server.
This will be a stable, versioned API which core Kubernetes components can rely on.
In the first version only the well-defined use cases will be handled,
altough the API should be easily extensible for potential future use cases.

## Use cases

This paragraph describes well-defined use cases which should be handled in the first version.
Use cases which are not listed below are out of the scope of MVP version of Resource Metrics API.

#### Horizontal Pod Autoscaler

HPA needs the latest value of cpu usage (as an mean from some time window)
for the given set of pods (both pod list or label selector are fine for this purpose).

#### Scheduler

Scheduler in order to schedule best-effort pods requires node level resource usage metrics.

#### kubectl top

This feature is not yet specified/implemented although it seems reasonable to support the following options:
- ```kubectl top node <id>``` - list all pods running on the given node accompanied with theirs cpu/mem usage
- ```kubectl top pod <id>``` - print cpu usage stats for the given pod
- ```kubectl top pod --selector=<label-query>``` - list all pods defined by the label query accompanied with theirs cpu/mem usage

#### Kubernetes dashboard

[Kubernetes dashboard](https://github.com/kubernetes/dashboard) in order to draw graps requires resource usage
in timeseries format form the last 15 minutes. Alternatively just the latest usage value is fine.

## Proposed API

Initially the metrics API will be in a separate [API group](api-group.md) called ```metrics```.
Later Node level metrics and Pod level metrics should be in different API groups.

#### Schema

The proposed schema is as follow. Each object has `TypeMeta` and `ObjectMeta` fields
to be compatible with Kubernetes API standards.

```go
type Node struct {
  TypeMeta
  ObjectMeta
  Metrics Metrics
}

type Pod struct {
  TypeMeta
  ObjectMeta
  Metrics Metrics
}

// The latest available metrics for the appriopriate resource.
type Metrics struct {
  // StartTime and EndTime specified the time window from which the response was returned.
  StartTime unversioned.Time `json:"start"`
  EndTime   unversioned.Time `json:"end"`
  Usage     v1.ResourceList `json:"usage"`
}
```

By default `Usage` is the mean from samples collected within the returned time window.
In the future we may want to introduce the following params:
`aggreator` (`max`, `min`, `95th`, etc.) and `window` (`1h`, `1d`, `1w`, etc.)
which will allow to get the other aggregates over the custom time window.

The proposed API can be easily extended to support also containers level metrics if needed.

#### Endpoints

All endpoints are GET endpoints, rooted at `/apis/metrics/v1alpha1/`.

The list of supported endpoints:
- `/` - discovery endpoint; type resource list
- `/nodes` - host metrics; type `[]metrics.Node`
- `/nodes/{node}` - metrics for a specific node; type `metrics.Node`
- `/namespaces/{namespace}/pods` - all pod metrics within namespace with support for `all-namespaces`; type `[]metrics.Pod`
- `/namespaces/{namespace}/pods/{pod}` - metrics for specific pod; type `metrics.Pod`

The following query parameters are supported:
- `pretty` - pretty print the response
- `labelSelector` - restrict the list of returned objects by labels (list endpoints only)
- `fieldSelector` - restrict the list of returned objects by fields (list endpoints only)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/metrics-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
