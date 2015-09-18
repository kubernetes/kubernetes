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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/proposals/compute-resource-metrics-api.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes compute resource metrics API

## Goals

Provide resource usage metrics on pods and nodes on the API server to be used
by the scheduler to improve job placement, utilization, etc. and by end users
to understand the resource utilization of their jobs. Horizontal and vertical
auto-scaling are also near-term uses.

## Current state

Right now, the Kubelet exports container metrics via an API endpoint. This
information is not gathered nor served by the Kubernetes API server.

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

## Proposed endpoints

	/api/v1/namespaces/myns/podMetrics/mypod
	/api/v1/nodeMetrics/myNode

The derived metrics include the mean, max and a few percentiles of the list of
values.

We are not adding new methods to pods and nodes, e.g.
`/api/v1/namespaces/myns/pods/mypod/metrics`, for a number of reasons. For
example, having a separate endpoint allows fetching all the pod metrics in a
single request. The rate of change of the data is also too high to include in
the pod resource.

In the future, if any uses cases are found that would benefit from RC,
namespace or service aggregation, metrics at those levels could also be
exposed taking advantage of the fact that Heapster already does aggregation
and metrics for them.

Initially, this proposal included raw metrics alongside the derived metrics.
After revising the use cases, it was clear that raw metrics could be left out
of this proposal. They can be dealt with in a separate proposal, exposing them
in the Kubelet API via proper versioned endpoints for Heapster to poll
periodically.

This also means that the amount of data pushed by each Kubelet to the API
server will be much smaller.

## Data gathering

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

## Data structure

```Go
type DerivedPodMetrics struct {
	TypeMeta
	ObjectMeta // should have pod name
	// the key is the container name
	Containers []struct {
		ContainerReference *Container
		Metrics            MetricsWindows
	}
}

type DerivedNodeMetrics struct {
	TypeMeta
	ObjectMeta // should have node name
	NodeMetrics      MetricsWindows
	SystemContainers []struct {
		ContainerReference *Container
		Metrics            MetricsWindows
	}
}

// Last overlapping 10s, 1m, 1h and 1d as a start
// Updated every 10s, so the 10s window is sequential and the rest are
// rolling.
type MetricsWindows map[time.Duration]DerivedMetrics

type DerivedMetrics struct {
	// End time of all the time windows in Metrics
	EndTime unversioned.Time `json:"endtime"`

	Mean       ResourceUsage `json:"mean"`
	Max        ResourceUsage `json:"max"`
	NinetyFive ResourceUsage `json:"95th"`
}

type ResourceUsage map[resource.Type]resource.Quantity
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/compute-resource-metrics-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
