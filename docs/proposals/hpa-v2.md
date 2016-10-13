<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
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

Horizontal Pod Autoscaler with Arbitary Metrics
===============================================

The current Horizontal Pod Autoscaler object only has support for CPU as
a percentage of requested CPU.  While this is certainly a common case, one
of the most frequently sought-after features for the HPA is the ability to
scale on different metrics (be they custom metrics, memory, etc).

The current HPA controller supports targeting "custom" metrics (metrics
with a name prefixed with "custom/") via an annotation, but this is
suboptimal for a number of reasons: it does not allow for arbitrary
"non-custom" metrics (e.g. memory), it does not allow for metrics
describing other objects (e.g. scaling based on metrics on services), and
carries the various downsides of annotations (not be typed/validated,
being hard for a user to hand-construct, etc).

Object Design
-------------

### Requirements ###

This proposal describes a new version of the Horizontal Pod Autoscaler
object with the following requirements kept in mind:

1. The HPA should continue to support scaling based on percentage of CPU
   request

2. The HPA should support scaling on arbitrary metrics associated with
   pods

3. The HPA should support scaling on arbitrary metrics associated with
   other Kubernetes objects in the same namespace as the HPA (and the
   namespace itself)

4. The HPA should make scaling on multiple metrics in a single HPA
   possible and explicit (splitting metrics across multiple HPAs leads to
   the possibility of fighting between HPAs)

### Specification ###

```go
type HorizontalPodAutoscalerSpec struct {
    // the target scalable object to autoscale
    ScaleTargetRef CrossVersionObjectReference `json:"scaleTargetRef"`

    // the minimum number of replicas to which the autoscaler may scale
    // +optional
    MinReplicas *int32 `json:"minReplicas,omitempty"`
    // the maximum number of replicas to which the autoscaler may scale
    MaxReplicas int32 `json:"maxReplicas"`

    // the metrics to use to calculate the desired replica count (the
    // maximum replica count across all metrics will be used).  The
    // desired replica count is calculated multiplying the ratio between
    // the target value and the current value by the current number of
    // pods.  Ergo, metrics used must decrease as the pod count is
    // increased, and vice-versa.  See the individual metric source
    // types for more information about how each type of metric
    // must respond.
    // +optional
    Metrics []MetricSpec `json:"metrics,omitempty"`
}

// a type of metric source
type MetricSourceType string
var (
    // a metric describing a kubernetes object (for example, hits-per-second on an Ingress object)
    ObjectSourceType MetricSourceType = "object"
    // a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
    // The values will be averaged together before being compared to the target value
    PodsSourceType MetricSourceType = "pods"
    // a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
    // in the current scale target (e.g. CPU or memory).  Such metrics are built in to Kubernetes,
    // and have special scaling options on top of those available to normal per-pod metrics (the "pods" source)
    ResourceSourceType MetricSourceType = "resource"
)

// a specification for how to scale based on a single metric
// (only `type` and one other matching field should be set at once)
type MetricSpec struct {
    // the type of metric source (should match one of the fields below)
    Type MetricSourceType `json:"type"`

    // a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
    Object *ObjectMetricSource `json:"object,omitempty"`
    // a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
    // The values will be averaged together before being compared to the target value
    Pods *PodsMetricSource `json:"pods,omitemtpy"`
    // a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
    // in the current scale target (e.g. CPU or memory).  Such metrics are built in to Kubernetes,
    // and have special scaling options on top of those available to normal per-pod metrics using the "pods" source.
    Resource *ResourceMetricSource `json:"resource,omitempty"`
}

// a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
type ObjectMetricSource struct {
    // the described Kubernetes object
    Target CrossVersionObjectReference `json:"target"`

    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the target value of the metric (as a quantity)
    TargetValue resource.Quantity `json:"targetValue"`
}

// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
// The values will be averaged together before being compared to the target value
type PodsMetricSource struct {
    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the target value of the metric (as a quantity)
    TargetAverageValue resource.Quantity `json:"targetAverageValue"`
}

// a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
// in the current scale target (e.g. CPU or memory).  The values will be averaged together before
// being compared to the target.  Such metrics are built in to Kubernetes, and have special
// scaling options on top of those available to normal per-pod metrics using the "pods" source.
// Only one "target" type should be set.
type ResourceMetricSource struct {
    // the name of the resource in question
    Name api.ResourceName `json:"name"`
    // the target value of the resource metric, represented as
    // a percentage of the requested value of the resource on the pods.
    // +optional
    TargetAverageRequestUtilization *int32 `json:"targetAverageRequestUtilization,omitempty"`
    // the target value of the resource metric as a raw value, similarly
    // to the "pods" metric source type.
    // +optional
    TargetAverageRawValue *resource.Quantity `json:"targetAverageRawValue,omitempty"`
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

// the status of a single metric
type MetricStatus struct {
    // the type of metric source
    Type MetricSourceType `json:"type"`

    // a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
    Object *ObjectMetricStatus `json:"object,omitemtpy"`
    // a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
    // The values will be averaged together before being compared to the target value
    Pods *PodsMetricStatus `json:"pods,omitemtpy"`
    // a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
    // in the current scale target (e.g. CPU or memory).  Such metrics are built in to Kubernetes,
    // and have special scaling options on top of those available to normal per-pod metrics using the "pods" source.
    Resource *ResourceMetricStatus `json:"resource,omitempty"`
}

// a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
type ObjectMetricStatus struct {
    // the described Kubernetes object
    Target CrossVersionObjectReference `json:"target"`

    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the current value of the metric (as a quantity)
    CurrentValue resource.Quantity `json:"currentValue"`
}

// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
// The values will be averaged together before being compared to the target value
type PodsMetricStatus struct {
    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the current value of the metric (as a quantity)
    CurrentAverageValue resource.Quantity `json:"currentAverageValue"`
}

// a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
// in the current scale target (e.g. CPU or memory).  The values will be averaged together before
// being compared to the target.  Such metrics are built in to Kubernetes, and have special
// scaling options on top of those available to normal per-pod metrics using the "pods" source.
// Only one "target" type should be set.
type ResourceMetricStatus struct {
    // the name of the resource in question
    Name api.ResourceName `json:"name"`
    // the target value of the resource metric, represented as
    // a percentage of the requested value of the resource on the pods
    // (only populated if the corresponding request target was set)
    // +optional
    CurrentAverageRequestUtilization *int32 `json:"currentAverageRequestUtilization,omitempty"`
    // the current value of the resource metric as a raw value
    CurrentAverageRawValue resource.Quantity `json:"currentAverageRawValue"`
}
```

### Example ###

In this example, we scale based on the `hits-per-second` value recorded as
describing a service in our namespace, plus the CPU usage of the pods in
the ReplicationController being autoscaled.

```yaml
kind: HorizontalPodAutoscaler
apiVersion: autoscaling/v2alpha1
spec:
  scaleTargetRef:
    kind: ReplicationController
    name: WebFrontend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: resource
    resource:
      name: cpu
      targetAverageRequestUtilization: 80
  - type: object
    object:
      target:
        kind: Service
        name: Frontend
      metricName: hits-per-second
      targetValue: 1k
```

### Alternatives and Future Considerations ###

Since the new design mirrors volume plugins (and similar APIs), it makes
it relatively easy to introduce new fields in a backwards-compatible way:
we simply introduce a new field in `MetricSpec` as a new "metric type".

#### External ####

It was discussed adding a source type of `External` which has a single
opaque metric field and target value.  This would indicate that the HPA
was under control of an external autoscaler, which would allow external
autoscalers to be present in the cluster while still indicating to tooling
that autoscaling is taking place.

However, since this raises a number of questions and complications about
interaction with the existing autoscaler, it was decided to exclude this
feature.  We may reconsider in the future.

### Limit Percentages ###

In cluster environments where request is automatically set for scheduling
purposes, it is advantageous to be able to autoscale on percentage of
limit for resource metrics.  We may wish to consider adding
a `targetPercentageOfLimit` to the `ResourceMetricSource` type.

#### Referring to the current Namespace ####

It is beneficial to be able to refer to a metric on the current namespace,
similarly to the `ObjectMetricSource` source type, but without an explicit
name.  Because of the similarity to `ObjectMetricSource`, it may simply be
sufficient to allow specificying a `kind` of "Namespace" without a name.
Alternatively, a similar source type to `PodsMetricSource` could be used.

#### Calculating Final Desired Replica Count ####

Since we have multiple replica counts (one from each metric), we must have
a way to aggregated them into a final replica count.  In this iteration of
the proposal, we simply take the maximum of all the computed replica
counts.  However, in certain cases, it could be useful to allow the user
to specify that they wanted the minimum or average instead.

In the general case, maximum should be sufficient, but if the need arises,
it should be fairly easy to add such a field in.

Mechanical Concerns
-------------------

The HPA will derive metrics from two sources: resource metrics (i.e. CPU
request percentage) will come from the
[master metrics API](resource-metrics-api.md), while other metrics will
come from the custom metrics API (currently proposed as #34586), which is
an adapter API which sources metrics directly from the monitoring
pipeline.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/hpa-v2.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
