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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/custom-metrics.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Specifying Metrics in the HPA
=============================

Introduction
------------

Currently, custom metrics are specified via a list of metric names and
target values.  Each metric name is prefixed with "custom/", and then is
requested at the pod level for each pod of the target scalable.

Several use cases are not supported by this design:

### Referring to Non-Custom Metrics ###

Several users have asked for the ability to scale on existing built-in metrics
(e.g. memory usage).  While the current design does allow the user to specify
a metric name, all metrics are prefixed with "custom/", meaning built-in
metrics cannot be passed this way.  A new design should not have completely
separate mechanisms for custom and non-custom metrics.  This also allows
referring to raw CPU usage rate, instead of referring to CPU usage as a
percentage of the CPU request (this has been a commonly asked-for feature
by users and operators).

### Referring to Metrics Not Associated With Pods ###

Heapster currently can track metrics which describe with pods and
namespaces. With Heapster push metrics, it becomes possible to define
custom metrics on the namespace level.  However, the custom metrics
annotation in the HPA currently only allows referring to custom metrics
describing pods.

Referring to custom metrics at the namespace level allows us to support certain
use cases in which an application wishes to expose a certain special metric
to an entire namespace (e.g. queue length) and scale based on that.
Additionally, until Heapster has the capability to track metrics from service
objects, RCs, etc, namespace-level metrics can be used with push metrics to
"fake" this type of metrics.  Metrics associated with services could be
extracted from a load balancer or reverse proxy, for instance.

Heapster will, most likely, support metrics describing services,
replicationcontrollers, etc, in the future, so it becomes adventageous to
be able to refer to metrics describing those objects as well (such as HTTP
requests, network throughput, etc).

Proposed New Design
-------------------

```go
// refers to relative sources (current namespace, controller, or pod)
type SourceType string
var (
    // the current namespace
    SourceTypeNamespace SourceType = "namespace"
    // each pod that would be considered
    SourceTypePod SourceType = "pod"
    // the target of the HPA
    SourceTypeController SourceType = "controller"
)

// indicates how to combine the replica counts from
// multiple metrics into a single replica count for the HPA
// (this allows the user to choose whether to err on the side of
// overscaling or underscaling)
type AggregationType string
var (
	AggregationTypeMax AggregationType = "max"
	AggregationTypeMin AggregationType = "min"
	AggregationTypeAverage AggregationType = "average"
)

// indicates which object a metric describes (and thus where to find it in
// the metrics source)
type MetricSource struct {
    // use a metric describing the current namespace, pod, or controller
    CurrentSource SourceType `json:"current,omitempty`

    // use a metric describing a different object in the current namespace
    SourceRef *CrossVersionObjectReference `json:"object,omitempty"`
}

// describes one metric to scale on
type MetricTarget struct {
    // the name of the of target metric
    Name string `json:"name,omitempty"`

    // the target (raw) value of the given metric aim for
    TargetValue resource.Quantity `json:"targetValue,omitempty"`

    // the target (as a percentage of requests) of the given metric to aim for
    // (only works when given a metric name that is a resource)
    TargetRequestPercentage *int32 `json:targetRequestPercentage,omitempty"`

    // specifies the object that the metric describes (and thus where to
    // find the metric in the metric source).  Defaults to the current pod.
    SourceObject *MetricSource `json:"from,omitempty"`
}

type HorizontalPodAutoscalerSpec struct {

    // a reference to the scaled resource
    ScaleTargetRef CrossVersionObjectReference `json:"scaleTargetRef"`

    // the minimum replica count
    MinReplicas *int32 `json:"minReplicas,omitempty"`

    // the maximum replica count
    MaxReplicas int32 `json:"maxReplicas"`

    // the list of metrics and target values to scale on
    Metrics []MetricTarget `json:"metrics"`

	// the aggregation function used to combine the computed replica counts
	// when multiple metrics are specified.  Defaults to max.
	ReplicasAggregator AggregationType `json:"replicasAggregator"`
}

type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	ObservedGeneration *int64 `json:"observedGeneration,omitempty"`

	// last time the autoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	LastScaleTime *unversioned.Time `json:"lastScaleTime,omitempty"`

	// the last observed number of replicas from the target object.
	CurrentReplicas int32 `json:"currentReplicas"`

	// the desired number of replicas as last computed by the autoscaler
	DesiredReplicas int32 `json:"desiredReplicas"`

	// the last read state of the metrics used by this autoscaler
	CurrentMetrics []MetricStatus `json:"currentMetrics" protobuf:"bytes,5,rep,name=currentMetrics"`
}

// indicates the status of one metric
type MetricStatus struct {
	// the name of the metric in use
	Name string `json:"name,omitempty"`

	// the current raw value of the metric in use
    // (always populated, unlike the field in spec)
	CurrentValue resource.Quantity `json:"currentValue"`

	// the current percentage of the pods' request for the resource
	// corresponding to the given metric.  It is only valid in the same situations where
	// targetRequestPercentage is valid in the corresponding target.
	CurrentRequestPercentage *int32 `json:"currentResourcePercentage,omitempty"`

	// the object which the metric describes
    // (always populated, unlike the field in spec)
	SourceObject MetricSource `json:"from"`
}
```

This design allows us to refer to arbitrarily named metrics (such as any
built-in metrics, plus any custom metrics stored in the metric source),
and allows referring to metrics from the current namespace, metrics from
each pod individually (the default, and current setup), as well as metrics
associated with other objects in the namespace.

The design also allows the user to specify multiple metrics (instead of
requiring multiple horizontal pod autoscalers, which could potentially
fight between themeselves).  In the case of multiple metrics, it allows
users to indicate whether they would like the HPA to err on the side of
overscaling (max), underscaling (min), or to simply average the replica
counts together (average).

An HPA under this design would look like:

```yaml
apiVersion: autoscaling/v2alpha1
kind: HorizontalPodAutoscaler
metadata:
    ...
spec:
    scaleTargetRef:
        ...
    maxReplicas: 10
    metrics:
    - name: memory/usage
      targetValue: 256Mi
    - name: cpu
      targetRequestPercentage: 75
    - name: custom/queue-length
      targetValue: 20
      from:
        current: "namespace"
    - name: router/http_requests_rate
      targetValue: 1K
      from:
        object:
            kind: Service
            apiVersion: v1
            name: "my-service"
```

This would result in the following queries to Heapster (or a similar API):

- `/api/v1/model/namespace/$NS/pod-list/$POD1,$POD2/metrics/memory/usage`
- `/apis/metrics/v1alpha1/namespaces/$NS/pods?labelSelector=$SEL` (for CPU usage)
- `/api/v1/model/namespace/$NS/metrics/custom/queue-length`
- `/api/v1/model/namespace/$NS/service/my-service/metrics/router/http_requests_rate`
  (not currently supported by Heapster, will be in the future)

Alternatives
------------

It has been proposed that users who wish to use custom metrics should
simply write their own autoscaler.  This is problematic in multiple ways.
If the user does not have access to Heapster and wished to scale on both
CPU and a custom metric, it would require running both the internal
autoscaler and the custom autoscaler, which could fight over desired
replica counts and thus cause issues.  If the user does have access to
Heapster, it requires duplicating the logic of our internal autoscaler
simply to have access to the same mechanisms, but using different metric
names.  Finally, if the user does not have access to Heapster, they have
no way to scale on built-in metrics besides CPU.

Open Questions
--------------

### How should CPU usage percentage be integrated? ###

In the proposal, it is expressed as a potential field in a given
`MetricTarget`.  This has the advantage that it can be extended to work
with different resources (for instance, memory).  The downside is that it
*only* works with certain metrics, and could require a slightly different
metric name, depending on whether a percentage is used or not (e.g. `cpu`
vs `cpu/usage_rate`, although we could "guess" at the resource name by
simply taking the part before the slash).

It could also be a completely separate field, like it is in the v1 API:

```go
type HorizontalPodAutoscalerSpec struct {
    ScaleTargetRef CrossVersionObjectReference `json:"scaleTargetRef"`
    MinReplicas *int32 `json:"minReplicas,omitempty"`
    MaxReplicas int32 `json:"maxReplicas"`
    Metrics []MetricTarget `json:"metrics"`
    TargetCPUUtilizationPercentage *int32 `json:targetCPUUtilizationPercentage,omitempty"`
}
```

However, this makes CPU a special snowflake, and thus seems like a strange
API design.

An additional concern is whether or not scaling on a percentage of limits
should be possible.  In high-density cluster situations where request may
be automatically set to a small number, specifying percentage of limit
makes much more sense.  If we were to introduce this feature, it should be
taken into account in the API design.

### Should this be an annotation, or an alpha API version? ###

Doing this as an annotation introduces a whole host of additional issues,
but is "easier" to get up and running, since it does not require the
creation of a new API version.  However, it would most likely be very
unweildy to use, and introduces a new set of problems defining how the
annotation interacts with existing API fields. A new API version would be
easier for users to use, but is more work to initially set up.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/custom-metrics.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
