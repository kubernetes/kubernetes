# Heapster Push Metrics

## Overview and Motivation

Currently, Heapster supports pulling metrics from kubelet, and defines an
interface for pulling from other sources.  However, in certain cases, it is
more useful to be able to have services push metrics into Heapster, instead
of having Heapster pull metrics.

For instance, supporting push metrics makes it easy for cluster admins to add
custom metrics into Heapster with relatively minimal effort: they can simply
write a program or script which collects and processes the information, and
then add a recurring Job or cron job which pushes the metrics.  This also
enables existing tooling designed with the push model in mind to be adapted to
provide metrics through Heapster.

### Target Audience and Metrics

Like the existing custom metrics pull mechanism, the metrics pushed through the
push mechanism are intended to be those metrics that are useful for consumption
by system components, such as metrics intended for use with autoscaling.  The
custom metrics pushed through the push mechanisms are still intended to follow
the overall guidelines for Heapster custom metrics (keep the number of
different metric names relatively limited, etc).

The current custom metrics mechanism can only be used to collect custom metrics
which describe the producing pod.  This proposal is designed to provide a
method of collecting metrics which describe multiple resources (other pods,
services, etc) across the cluster.  Such producers are generally add-on cluster
infrastructure components deployed by the cluster admins.  The push mechanisms
are not, in general, intended for use by arbitrary cluster users (although the
metrics transfered would probably apply to the users' pods).  It is up to
cluster admins to decide which applications are permitted to use the push
mechanims (see the "Authentication" section below for more information on how
cluster admins can control access).

For these reasons, the push metrics mechanism is not designed to replace the
existing pull mechanism for custom metrics, but instead to support additional
use cases not already supported.

## API

### Authentication, Segregation, and Flow Control

Producers will be authenticated similarly to the way Heapster currently
authenticates clients: producers will present a certificate signed by a CA, and
Heapster will be configurable to only allow certain names to push metrics, or
to allow any certificate signed by the CA.

Additionally, the name presented during authentication will be used as a prefix
for all metrics added.  This will prevent two different metrics producers from
accidentally overwriting each other's custom metrics (otherwise, push metrics
will be stored and retrieved identically to pull-based custom metrics).

In order to prevent push metrics from overwhelming the Heapster instance, it
will be possible to limit the total number of custom metrics each producer is
allowed to add, and the frequency at which producers are allowed to push new
sets of custom metrics.  By default, no limits will be enforced unless
explicitly set via command line arguments.

### Paths

To add metrics, metrics producers will `POST` new metrics to
`/api/v1/push/{format}/{subpath}` in the format specified by `{format}`.  The
metrics are pushed in bulk -- it is up to the format to support a way to name
specific metrics, namespaces, pods, and containers (for instance, the
Prometheus format uses metric names and labels for this purpose).  The
`{subpath}` option enables different formats to have format specific sub-paths.

### Metrics Format

The underlying design can support multiple format "backends".  The initial
backend, detailed here, will be based on the Prometheus format, and will be
available at `/api/v1/push/prometheus/`,
`/api/v1/push/prometheus/metrics/job/{producer_name}`, or
`/api/v1/push/promethus/metrics/jobs/{producer_name}`.  If either of the latter
two paths are used, `{producer_name}` should match the name used when
authenticating with Heapster.

Both the Prometheus text and protobuf formats will be supported.

The metric name specified on each metric line will be used as the custom metric
name in Heapster (except prefixed as discussed in the "Authentication"
section).  The following labels will be used to determine which object a metric
is associated with:

- For kubernetes-related metrics, the `namespace` label will indicate
  namespace, the `pod` label will indicate pod, and the `container` label will
  indicate container.  If only `namespace` is present, then the metric will be
  considered namespace-level.  If `namespace` and `pod` are present, then the
  metric will be considered pod-level.  If all three are present, the metric
  will be considered container-level.

- For kubernetes-related metrics, the `service` label can be used in
  conjunction with the `namespace` label to indicate a service-level metric.
  Currently, this will be stored in Heapster as a namespace-level metric
  prefixed with the service name.

- For non-kubernetes-related metrics, the `node` label will indicate node, and
  the `container` label will indicate free container.   If only `node` is
  present, the metric will be considered node-level.  Otherwise, the metric
  will be considered free-container-level.

If the `node`, `namespace`, `pod`, and `container`, and `service` labels are
not present in one of the configurations listed above, the metric line is
invalid and the batch should be rejected.

Any additional labels will be treated the same as labels on existing custom
metrics (currently multiple custom metrics with the same name, but different
labels, are ignored, but this seems like an oversight and should probably be
fixed).

Timestamps will be assigned by using the next Heapster metrics batch timestamp
after the time at which the metrics are received.  If a timestamp is provided
as part of the metric line, this may be stored as a separate field for
posterity, but the "official" timestamp will be that of the assigned batch.

As with normal Prometheus metrics, the `TYPE` line should be used to provide
the type of the metric.

#### Example

Suppose a producer with the name "http_gatherer" sent the following metrics to
`/api/v1/push/prometheus`:

```
# This is a pod-level metric (it might be used for autoscaling)
# TYPE http_requests_per_second guage
http_requests_per_minute{namespace="webapp",pod="frontend-server-a-1"} 20
http_requests_per_minute{namespace="webapp",pod="frontend-server-a-2"} 5
http_requests_per_minute{namespace="webapp",pod="frontend-server-b-1"} 25

# This is a service-level metric, which will be stored as frontend_hits_total
# and restapi_hits_total (these might be used for auto-idling)
# TYPE hits_total counter
hits_total{namespace="webapp",service="frontend"} 5000
hits_total{namespace="webapp",service="restapi"} 6000
```

This would result in the metrics being available at:

```
/api/v1/model/namespaces/webapp/pods/frontend-server-a-1/metrics/custom/http_gatherer/http_requests_per_minute
/api/v1/model/namespaces/webapp/pods/frontend-server-a-2/metrics/custom/http_gatherer/http_requests_per_minute
/api/v1/model/namespaces/webapp/pods/frontend-server-b-1/metrics/custom/http_gatherer/http_requests_per_minute
/api/v1/model/namespaces/webapp/metrics/custom/http_gatherer/frontend/hits_total
/api/v1/model/namespaces/webapp/metrics/custom/http_gatherer/restapi/hits_total
```

## Discussed Alternatives

A number of alternatives came up during the discussion of this proposal.  They
are discussed briefly below.  Note that most of these alternatives do not deal
particularly well with a case where metrics need to come from a source that is
not running as a pod on the cluster.  While it is expected that many of the
producers will be running as components on the cluster (e.g. as DaemonSets or
PetSets), it could still be adventageous to support metrics coming from
components that are not in the form of pods.

### Writing directly into sinks

This alternative would have producers write directly into sinks in the Heapster
storage schema, and then use a mechanism similar to the Oldtimer API to read
the metrics back.

This would require every producer to know how to talk to every sink, would make
configuring the sinks more complicated, and would most likely lead to software
only being able to talk to one of the sinks supported by Heapster.
Additionally, you lose the benefits of the Heapster model, and either have to
adapt the existing Heapster model to fall back to an Oldtimer-like approach, or
teach all cluster components to be able to read from both the Heapster model
and Oldtimer simultaneously.

### Reworking the existing cAdvisor-Kubelet-Heapster Pull Mechanism

This alternative would involve reworking the existing pull mechanism to allow
certain pods to produce metrics that describe other resources besides the
themselves, as opposed to the current situation, where all custom metrics
collected via the current pull mechanism are marked as describing the producer
pod.

This would require a mechanism for indicating to Heapster which pod names were
allowed to produce metrics that describe other resources, since admins would
generally want most pods producing metrics to continue to just have metrics
which describe only the producer pod.  It would also conceptually blend
together pods producing metrics about themselves versus pods producing metrics
about others.  Additionally, the current cAdvisor-based custom metrics
collection is not secured, so all metrics would be available to anyone with
knowlege of the appropriate port, but this may change in the future.

### Using a new daemon per node to produce metrics

This alternative would involve running a daemon on each node that aggregated
all the separate custom metrics producers' results together.

It was suggested that an approach similar to the Prometheus Node Exporter
Textfile Collector could be used, in which sources would write their metrics to
files in a directly, which would later be read by the collector when polled for
metrics.  When the producers are containerized, you'd need to use a hostPath
volume, have the daeamon look for specific emptyDir mounts in containers (and
use one director per container), or something similar.

Alternatively, a new daemon could be run on each node that was responsible for
collecting metrics from producers who produce bulk metrics describing other
resources.

This would still require some sort of auth to limit which pods where allowed to
do so (while the scoping above prevents collision, cluster admins would most
likely still want to limit which pods are allowed to post metrics which appear
in another pod's list of custom metrics).  Unlike the proposal above, admins
could not simply rely on "whoever is allowed to authenticate" rule, since
cAdvisor does not check certificates like the normal Heapster auth mechanism.

Additionally, this adds a bit of complexity on the producer's side, since it
requires continuously serving the metrics (this could be made easier by
providing a tool like the Prometheus Node Exporter Textfile Collector, which
just serves up metrics based on text files in a directory).

### Adding an additional standard pull mechanism

This alternative would involve writing a pull mechanim which, for instance, was
just able to read Prometheus metrics directly.  This would either require
admins to configure the Heapster instance to know about every custom metrics
source (and restart Heapster when a new source needed to be added), or would
require teaching Heapster how to look for an annotation on certain pods to
determine which pods to query (Heapster would have a list-watch on the pods,
and look for pods added/removed/changed with the appropriate annotation).

When used in the latter form, this mechanism would still likely require a
similar auth setup to the one proposed above, in order to allow the admin to
restrict which pods actually were allowed to produce the metrics.  It also has
similar restrictions/disadvantages as the "Using a new daemon per node" method
discussed above.
