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

    // the target CPU percentage
    // +optional
    TargetCPUUtilizationPercentage `json:"targetCPUUtilizationPercentage,omitemtpy"`
    // the non-CPU percentage metrics (the maximum desired replica count
    // across all metrics is used as desired replica count for the HPA)
    // +optional
    Metrics []MetricSpec `json:"metrics,omitempty"`
}

// a type of metric source
type MetricSourceType string
var (
    // a metric describing a kubernetes object
    ObjectSourceType MetricSourceType = "object"
    // a metric describing pods in the scale target
    PodsSourceType MetricSourceType = "pods"
)

// a specification for how to scale based on a single metric
// (only `type` and one other matching field should be set at once)
type MetricSpec struct {
    // the type of metric source (should match one of the fields below)
    Type MetricSourceType `json:"type"`

    // metric describing a single Kubernetes object
    Object *ObjectMetricSource `json:"object,omitempty"`
    // metric describing pods in the scale target
    Pods *PodsMetricSource `json:"pods,omitemtpy"`
}

// a metric describing a Kubernetes object
type ObjectMetricSource struct {
    // the described Kubernetes object
    Target CrossVersionObjectReference `json:"target"`

    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the target value of the metric (as a quantity)
    TargetValue resource.Quantity `json:"targetValue"`
}

// metric describing pods in the scale target
type PodsMetricSource struct {
    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the target value of the metric (as a quantity)
    TargetValue resource.Quantity `json:"targetValue"`
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

    // current average CPU utilization percentage, set if a target CPU was requested
    CurrentCPUUtilizationPercentage *int32 `json:"currentCPUUtilizationPercentage,omitempty"`
    // the last read state of the metrics used by this autoscaler
    CurrentMetrics []MetricStatus `json:"currentMetrics" protobuf:"bytes,5,rep,name=currentMetrics"`
}

// the status of a single metric
type MetricStatus struct {
    // the type of metric source
    Type MetricSourceType `json:"type"`

    // metric describing a single Kubernetes object
    Object *ObjectMetricStatus `json:"object,omitemtpy"`
    // metric describing pods in the scale target
    Pods *PodsMetricStatus `json:"pods,omitemtpy"`
}

// a metric describing a Kubernetes object
type ObjectMetricStatus struct {
    // the described Kubernetes object
    Target CrossVersionObjectReference `json:"target"`

    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the current value of the metric (as a quantity)
    CurrentValue resource.Quantity `json:"targetValue"`
}

// metric describing pods in the scale target
type PodsMetricStatus struct {
    // the name of the metric in question
    MetricName string `json:"metricName"`
    // the current value of the metric (as a quantity)
    CurrentValue resource.Quantity `json:"targetValue"`
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
  targetCPUUtilizationPercentage: 80
  metrics:
  - object:
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

As a potential future consideration, we may wish to include a metric
source type of `External`, which simply has an opaque metric type and
target value (as strings).  This would allow custom autoscalers with
custom source mechanics to be written, while still providing the same
indication that a controller was being autoscaled.

#### Resource Percentages ####

In the current design, CPU percentage metrics are a completely separate
field from the rest of the metrics.  While this simplifies the common case
of scaling on CPU, it makes the overall design a bit more clunky.  It also
doesn't provide a method for scaling based on memory percentage.

Two alternatives to the method proposed above are shown the object below.
The former is more compact, but precludes scaling memory as a percentage
of request.

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
  - cpuRequest:
      targetPercentage: 80
  - resourceRequest:
      resource: cpu
      targetPercentage: 80
```

#### Referring to the current Namespace ####

It is beneficial to be able to refer to a metric on the current namespace,
similarly to the `ObjectMetricSource` source type, but without an explicit
name.  Because of the similarity to `ObjectMetricSource`, it may simply be
sufficient to allow specificying a `kind` of "Namespace" without a name.
Alternatively, a similar source type to `PodsMetricSource` could be used.

#### Aggregating Desired Replica Counts ####

In this iteration of the proposal, we simply take the maximum of all the
desired replica counts across the specified metrics to figure out the
final replica count for the HPA.  However, in certain cases, it may be
advantageous to instead take the minimum or average.  It could be
desirable to add a field for specifying this (with "max" then becoming the
default).

Mechanical Concerns
-------------------

The HPA will derive metrics from two sources: resource metrics (i.e. CPU
request percentage) will come from the
[master metrics API](resource-metrics-api.md), while other metrics will
come from the [custom/arbitrary metrics API](custom-metrics-api.md)
(currently proposed as #34586).

