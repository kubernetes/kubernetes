Custom Metrics API
==================

The new [metrics monitoring vision](../design/monitoring_architecture.md)
proposes an API that the Horizontal Pod Autoscaler can use to access
arbitrary metrics.

Similarly to the [master metrics API](resource-metrics-api.md), the new
API should be structured around accessing metrics by referring to
kubernetes objects (or groups thereof) and a metric name.  For this
reason, the API could be useful for other consumers (most likely
controllers) that want to consume custom metrics (similarly to how the
master metrics API is generally useful to multiple cluster components).

The HPA can currently refer to all pods selected by a label selector, and
is planned to be able to support referring to metrics describing
a namespace and other non-pod objects within the namespace.

API Paths
---------

The root API path will look like `/apis/general-metrics/v1alpha1`.  For
brevity, this will be left off below.

- `/nodes/{node-name}/metrics/{metric-name...}`: retrieve the given metric
  on the given node

- `/namespaces/{namespace-name}/metrics/{metric-name...}`: retrieve the
  given metric on the given namespace

- `/namespaces/{namespace-name}/object-metrics/{object-type}/{metric-name...}`:
  retrieve the given metric for all objects of the given type in the given
  namespace.

- `/namespaces/{namespace-name}/object-metrics/{object-type}/{metric-name...}?labelSelector=foo`:
  retrieve the given metric for all objects of the given type matching the
  given label selector in the given namespace.

- `/namespaces/{namespace-name}/object-metrics/{object-type}/{metric-name...}?names=foo,bar`:
  retrieve the given metric for the objects of the given type with the
  given names in the given namespace

For example, to retrieve the custom metric "hits-per-second" for all pods
matching "app=frontend` in the namespaces "webapp", the request might look
like:

`/apis/general-metrics/v1alpha1/namespaces/webapp/object-metrics/pods/hits-per-second?labelSelector=app%3Dfrontend`.

API Objects
-----------

The request URLs listed above will return objects of the `Metrics`
type, described below:

```go

// a list of values for a given metric for some set of objects
type Metrics struct {
    unversioned.TypeMeta `json:",inline"`
    unversioned.ListMeta `json:"metadata,omitempty"`

    // the name of the metric
    MetricName string `json:"metricName"`

    // the value of the metric across the described objects
    MetricValues []MetricValue `json:"metricValues"`
}

// a metric value for some object
type MetricValue struct {
    // a reference to the described object
    DescribedObject ObjectReference `json:"describedObject"`

    // indicates the end of the time window containing these metrics (i.e.
    // these metrics come from some time in [Timestamp-Window, Timestamp])
    Timestamp unversioned.Time `json:"timestamp"`

    // indicates the duration of the time window containing these metrics
    Window    unversioned.Duration `json:"window"`

    // the value of the metric for this
    Value resource.Quantity
}
```

For instance, the example request above would yield the following object:

```json
{
    "kind": "Metrics",
    "apiVersion": "general-metrics/v1alpha1",
    "metricName": "hits-per-second",
    "metricValues": [
        {
            "describedObject": {
                "kind": "Pod",
                "name": "server1",
                "namespace": "webapp"
            },
            "timestamp": SOME_TIMESTAMP_HERE,
            "window": "10s",
            "value": "10"
        },
        {
            "describedObject": {
                "kind": "Pod",
                "name": "server2",
                "namespace": "webapp"
            },
            "timestamp": SOME_TIMESTAMP_HERE,
            "window": "10s",
            "value": "15"
        }
    ]
}
```

Semantics
---------

The `object-type` parameter should be the string form of
`unversioned.GroupKind`.  Note that we do not include version in this; we
simply wish to uniquely identify all the different types of objects in
Kubernetes.

In the case of cross-group object renames, the adapter should maintain
a list of "equivalent versions" that the monitoring system uses. This is
monitoring-system dependent (for instance, the monitoring system might
record all HorizontalPodAutoscalers as in `autoscaling`, but should be
aware that HorizontalPodAutoscaler also exist in `extensions`).

The returned metrics should be the most recenly available metrics, as with
the resource metrics API.  The timestamp and window should indicate to the
consumer what timeframe the metric has come from.  The timestamp indicates
the "batch" of metrics, while the window indicates the length of time
between batches.

For metrics systems that support differentiating metrics beyond the Kubernetes
object hierarchy (such as using additional labels), the metrics systems should
have a metric which represents all such series aggregated together.
Additionally, implementors may choose to the individual "sub-metrics" via
the metric name, but this is expected to be fairly rare, since it most
likely requires specific knowledge of individual metrics.  For instance,
suppose we record filesystem usage by filesystem inside the container.
There should then be a metric `filesystem/usage`, and the implementors of
the API may choose to expose more detailed metrics like
`filesystem/usage/my-first-filesystem`.

Relationship to HPA v2
----------------------

The URL paths in this API are designed to correspond to different source
types in the [HPA v2](hpa-v2.md).  Specifially, the `pods` source type
corresponds to a URL of the form
`/namespaces/$NS/object-metrics/pod/$METRIC_NAME?labelSelector=foo`, while
the `object` source type corresponds to a URL of the form
`/namespaces/$NS/object-metrics/$KIND.$GROUP/$METRIC_NAME?names=$OBJECT_NAME`.

The HPA then takes the results, aggregates them together (in the case of
the former source type), and uses the resulting value to produce a usage
ratio.

Mechanical Concerns
-------------------

This API is intended to be implemented by monitoring pipelines (e.g.
inside Heapster, or as an adapter on top of a solution like Prometheus).
It shares many mechanical requirements with normal Kubernetes APIs, such
as needed to support encoding different versions of objects in both JSON
and protobuf, as well as acting as a discoverable API server.  For these
reasons, it is expected that implemenators will make use of the Kubernetes
genericapiserver code.  If implementors choose not to use this, they must
still follow all of the Kubernetes API server conventions in order to work
properly with consumers of the API.

Location
--------

The types and clients for this API will live in a separate repository
under the Kubernetes organization (e.g. `kubernetes/metrics`).  This
respository will most likely also house other metrics-related APIs for
Kubernetes (e.g. historical metrics API definitions, the resource metrics
API definitions, etc).

Note that there will not be a canonical implemenation of the custom
metrics API under Kubernetes, just the types and clients.  Implementations
will be left up to the monitoring pipelines.

Alternative Considerations
--------------------------

### Pods vs Objects API ###

Since the HPA itself is only interested in groups of pods (by name or
label selector) or in individual objects, one could potentially argue that
it would be better to have separate endpoints for pods vs other objects.
The complicates the API structure a bit, but does make it possible to
return container-level metrics in the pod results.  Container-level
metrics are generally less useful for custom metrics, since the smallest
abstraction that the HPA cares about is pods (for custom metrics, at
least).

### Quantity vs Float ###

In the past, custom metrics were represented as floats.  In general,
however, Kubernetes APIs are not supposed to use floats. The API proposed
above thus uses `resource.Quantity`.  This adds a bit of encoding
overhead, but makes the API line up nicely with other Kubernetes APIs.

### Labeled Metrics ###

Many metric systems support labeled metrics, allowing for dimenisionality
beyond the Kubernetes object hierarchy.  Since the HPA currently doesn't
support specifying metric labels, this is not supported via this API.  We
may wish to explore this in the future.
