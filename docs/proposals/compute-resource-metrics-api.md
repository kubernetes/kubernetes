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

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/compute-resource-metrics-api.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes compute resource metrics API

## Goals

Provide resource usage metrics on pods and nodes through the API server to be
used by the scheduler to improve pod placement, utilization, etc. and by end
users to understand the resource utilization of their jobs. Horizontal and
vertical auto-scaling are also near-term uses. Additionally, a subset of the
metrics API should be served directly from the kubelet.

### API Requirements

- Provide machine level metrics, all pod metrics (in single request), specific
  pod metrics
- Ability to authenticate machine & pod metrics independently from each other
- Support multiple kinds of metrics (e.g. raw & derived types)
- Follow existing API conventions, fully compatible types able to eventually be
  served by apiserver library
- Maximum common ground between cluster and Kubelet API.

## Current state

Kubelet currently exposes raw container metrics through the `/stats/` endpoint
that serves raw container stats. However, this endpoint serves individual
container stats, and applications like heapster, which aggregates metrics across
the cluster, must repeatedly query for each container. The high QPS combined
with the potential data size of raw stats puts unnecessary load on the system
that could be avoided with an aggregate API. This information is not gathered
nor served by the Kubernetes API server.

## Use cases

The first user will be kubectl. The resource usage data can be shown to the
user via a periodically refreshing interface similar to `top` on Unix-like
systems. This info could let users assign resource limits more efficiently.

```
$ kubectl top kubernetes-minion-abcd
POD                        CPU         MEM
monitoring-heapster-abcde  0.12 cores  302 MB
kube-ui-v1-nd7in           0.07 cores  130 MB
```

A second user will be the scheduler. To assign pods to nodes efficiently, the
scheduler needs to know the current free resources on each node.

The Kubelet API will be used by heapster to provide metrics at the
cluster-level. The Kubelet API will also be useful for debugging individual
nodes, and stand-alone kubelets.

## Proposed endpoints

The metrics API will be its own [API group](api-group.md), and is shared by the
kubelet and cluster API. The derived metrics include the mean, max and a few
percentiles of the list of values, and will initially only be available through
the API server. The raw metrics include the stat samples from cAdvisor, and will
only be available through the kubelet. The types of metrics are detailed
[below](#schema). All endpoints are GET endpoints, rooted at
`/apis/metrics/v1alpha1/`

- `/` - discovery endpoint; type resource list
- `/rawNodes` - raw host metrics; type `[]metrics.RawNode`
  - `/rawNodes/localhost` - The only node provided is `localhost`; type
    metrics.Node
- `/derivedNodes` - host metrics; type `[]metrics.DerivedNode`
  - `/nodes/{node}` - derived metrics for a specific node
- `/rawPods` - All raw pod metrics across all namespaces; type
  `[]metrics.RawPod`
- `/derivedPods` - All derived pod metrics across all namespaces; type
  `[]metrics.DerivedPod`
- `/namespaces/{namespace}/rawPods` - All raw pod metrics within namespace; type
  `[]metrics.RawPod`
  - `/namespaces/{namespace}/rawPods/{pod}` - raw metrics for specific pod
- `/namespaces/{namespace}/derivedPods` - All derived pod metrics within
  namespace; type `[]metrics.DerivedPod`
  - `/namespaces/{namespace}/derivedPods/{pod}` - derived metrics for specific
  pod
- Unsupported paths return status not found (404)
  - `/namespaces/`
  - `/namespaces/{namespace}`

Additionally, all endpoints (except root discovery endpoint) support the
following optional query parameters:

- `start` - start time to return metrics from; type json encoded
  `time.Time`; since samples are retrieved at discrete intervals, the first
  sample after the start time is the actual beginning.
- `end` - end time to return metrics to; type json encoded `time.Time`
- `step` - the time step between each stats sample; type int (seconds), default
  10s, must be a multiple of 10s
- `count` - maximum number of stats to return in each ContainerMetrics instance;
  type int

As well as the common query parameters:

- `pretty` - pretty print the response
- `labelSelector` - restrict the list of returned objects by labels (list endpoints only)
- `fieldSelector` - restrict the list of returned objects by fields (list endpoints only)

### Rationale

We are not adding new methods to pods and nodes, e.g.
`/api/v1/namespaces/myns/pods/mypod/metrics`, for a number of reasons. For
example, having a separate endpoint allows fetching all the pod metrics in a
single request. The rate of change of the data is also too high to include in
the pod resource.

In the future, if any uses cases are found that would benefit from RC,
namespace or service aggregation, metrics at those levels could also be
exposed taking advantage of the fact that Heapster already does aggregation
and metrics for them.

## Schema

Types are colocated with other API groups in `/pkg/apis/metrics`, and follow api
groups conventions there.

```go
// Raw metrics are only available through the kubelet API.
type RawNode struct {
  TypeMeta
  ObjectMeta              // Should include node name
  Machine ContainerMetrics
  SystemContainers []ContainerMetrics
}
type RawPod struct {
  TypeMeta
  ObjectMeta              // Should include pod name
  Containers []Container
}
type RawContainer struct {
  TypeMeta
  ObjectMeta              // Should include container name
  Spec ContainerSpec      // Mirrors cadvisorv2.ContainerSpec
  Stats []ContainerStats  // Mirrors cadvisorv2.ContainerStats
}

// Derived metrics are (initially) only available through the API server.
type DerivedNode struct {
  TypeMeta
  ObjectMeta              // Should include node name
  Machine MetricsWindow
  SystemContainers []DerivedContainer
}
type DerivedPod struct {
  TypeMeta
  ObjectMeta              // Should include pod name
  Containers []DerivedContainer
}
type DerivedContainer struct {
  TypeMeta
  ObjectMeta              // Should include container name
  Metrics DerivedWindows
}

// Last overlapping 10s, 1m, 1h and 1d as a start
// Updated every 10s, so the 10s window is sequential and the rest are
// rolling.
type DerivedWindows map[time.Duration]DerivedMetrics

type DerivedMetrics struct {
	// End time of all the time windows in Metrics
	EndTime unversioned.Time `json:"endtime"`

	Mean       ResourceUsage `json:"mean"`
	Max        ResourceUsage `json:"max"`
	NinetyFive ResourceUsage `json:"95th"`
}

type ResourceUsage map[resource.Type]resource.Quantity
```

See
[cadvisor/info/v2](https://github.com/google/cadvisor/blob/master/info/v2/container.go)
for `ContainerSpec` and `ContainerStats` definitions.

## Implementation

### Cluster

We will use a push based system. Each kubelet will periodically - every 10s -
POST its derived metrics to the API server. Then, any users of the metrics can
register as watchers to receive the new metrics when they are available.

Users of the metrics may also periodically poll the API server instead of
registering as a watcher, having in mind that new data may only be available
every 10 seconds. If any user requires metrics that are either more specific
(e.g. last 1s) or updated more often, they should use the metrics pipeline via
Heapster.

The API server will not hold any of this data directly. For our initial
purposes, it will hold the most recent metrics obtained from each node in
etcd. Then, when polled for metrics, the API server would only serve said most
recent data per node.

Benchmarks will be run with etcd to see if it can keep up with the frequent
writes of data. If it turns out that etcd doesn't scale well enough, we will
have to switch to a different storage system.

If a pod gets deleted, the API server will get rid of any metrics it may
currently be holding for it.

The clients watching the metrics data may cache it for longer periods of time.
The clearest example would be Heapster.

In the future, we might want to store the metrics differently:

* via heapster - Since heapster keeps data for a period of time, we could
  redirect requests to the API server to heapster instead of using etcd. This
  would also allow serving metrics other than the latest ones.

An edge case that this proposal doesn't take into account is kubelets being
restarted. If any of them are, with a simple implementation they would lose
historical data and thus take hours to gather enough information to provide
relevant metrics again. We might want to use persistent storage directly or in
the future to improve that situation.

More information on kubelet checkpoints can be read on
[#489](https://issues.k8s.io/489).

### Kubelet

The eventual goal is to use the `apiserver` library to serve kubelet versioned
APIs. Since the apiserver library is not currently reuseable at the kubelet and
we do not want to block on it, we will write a simple 1-off solution for this
API. The 1-off code should be an implementation detail, and the exposed API
should match the expectations of the API server, so that we can throw away the
initial implementation when the apiserver is ready to serve the kubelet API. We
should prioritize replacing it before the API becomes too large or complicated.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/compute-resource-metrics-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
