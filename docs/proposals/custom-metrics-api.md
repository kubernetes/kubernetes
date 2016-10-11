Custom Metrics API
==================

The new [metrics monitoring vision]() proposes an API that the Horizontal
Pod Autoscaler can use to access arbitrary metrics.

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
  given name in the given namespace

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
type GenericMetrics struct {
    // the type and object metadata describes the object in question
    unversioned.TypeMeta `json:",inline"`
    v1.ObjectMeta        `json:"metadata,omitempty"`

    // indicates the end of the time window containing these metrics (i.e.
    // these metrics come from some time in [Timestamp-Window, Timestamp])
    Timestamp unversioned.Time `json:"timestamp"`

    // indicates the duration of the time window containing these metrics
    Window    unversioned.Duration `json:"window"`

    // the value of the metric for this
    Value float64
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
            "kind": "Pod",
            "metadata": { "name": "server1", "namespace": "webapp" },
            "timestamp": SOME_TIMESTAMP_HERE,
            "window": "10s",
            "value": 10.0
        },
        {
            "kind": "Pod",
            "metadata": { "name": "server2", "namespace": "webapp" },
            "timestamp": SOME_TIMESTAMP_HERE,
            "window": "10s",
            "value": 15.0
        }
    ]
}
```

Mechanical Concerns
-------------------

This API is intended to be implemented by monitoring pipelines (e.g.
inside Heapster, or as an adapter on top of a solution like Prometheus).
It shares many mechanical requirements with normal Kubernetes APIs.

The API should support encoding the result in both JSON and Protobuf,
similarly to how core Kubernetes APIs work.  Most likely, the Protobuf
encoding will be used by most consumers for efficiency's sake.

Additionally, the API should support encoding the output result in
different versions, if needed, as is currently the case for normal
Kubernetes APIs.

### Discoverability ###

While the initial plan for the API is for consumption by the HPA
controller, the API could be more broadly useful -- custom autoscaler or
scheduler implementations could make use of it's structured presentation
of metrics.  Broadly speaking, any consumer that might wish to
consume "core" metrics could potentially benefit from also being able to
consume general metrics.

While initially the API seems like a good fit for webhook-style
consumption (similarly to authentication today, for example), the lack of
discoverability makes it more difficult for multiple consumers to make use
of it (each has to be specifically configured to find it at a particular
location, instead of being able to make use of the normal API discovery
and summarization plans, as with the master metrics API).

The API should thus be discoverable and accessible as if it were just
another API server, similarly to the master metrics API.

Implementation Concerns
-----------------------

This API is intended to be implemented by monitoring pipelines (e.g.
Heapster, or as an adapter on top of a solution like Prometheus).  Because
it requires implementing multiple Kubernetes API server concepts (e.g.
protobuf and JSON encoding, output versioning, discoverability),
implementing it using the genericapiserver code initially seems
reasonable.  However, there is some concern that the internals of the
generic API server are ill-suited for a metrics API, since it's more aimed
read-write APIs for manipulating stored objects.

Should the generic API server prove too cumbersome (and/or insufficiently
performant), it may be advantageous to consider producing a slimmed-down
version of the generic API server for read-only APIs (like metrics APIs);
the master metrics API has roughly the same requirements as this API, and
would thus share many of the same concerns about the use of the generic
API server library.

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

Additionally, the ability to query for metrics for multiple objects in
a given namespace is useful for dashboard-like interfaces (e.g.
a dashboard UI, a top-like CLI, etc).

### Quantity vs Float ###

In general, Kubernetes APIs are not supposed to use floats.  The API
proposed above uses floats, since this is what most metric systems work
with.  However, the use of floats could be replaced with resources, which
fits better with the rest of the APIs in Kube, but does add some
encode/decode overhead.
