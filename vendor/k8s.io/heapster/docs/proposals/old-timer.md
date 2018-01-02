# Heapster Oldtimer

## Overview

Prior to the Heapster refactor, the Heapster model presented aggregations of
metrics over certain time periods (the last hour and day).  Post-refactor, the
concern of presenting an interface for historical metrics was to be split into
a separate Heapster component: Oldtimer.

Oldtimer will run as part of the main Heapster executable, and will present
common interfaces for retrieving historical metrics over longer periods of time
than the Heapster model, and will allow fetching aggregations of metrics (e.g.
averages, 95 percentile, etc) over different periods of time.  It will do this
by querying the sink to which it is storing metrics.

Note: even though we are retrieving metrics, this document refers to the
metrics storage locations as "sinks" to be consistent with the rest
of Heapster.

## Motivation

There are two major motivations for exposing historical metrics information:

1. Using aggregated historical data to make size-related decisions
   (for example, idling requires looking for traffic over a long time period)

2. Providing a common interface for users to view historical metrics

Before the Heapster refactoring (see the
[Heapster Long Term Vision Proposal](https://github.com/kubernetes/heapster/blob/master/docs/proposals/vision.md)),
Heapster supported querying metrics aggregated over certain extended time
periods (the last hour and day) via the Heapster model.

However, since the Heapster model is stored in-memory, and not persisted to
disk, this historical data would be "lost" whenever Heapster was restarted.
This made it unreliable for use by system components which need a historical
view.

Since we already persist metrics into a sink, it does not make sense for
Heapster itself to persist long-term metrics to disk itself.  Instead, we can
just query the sink directly.

## API

Oldtimer will present an api somewhat similar to the normal Heapster model.
The structure of the URLs is designed to mirror those exposed by the model API.
When used simply to retrieve historical data points, Oldtimer will return the
same types as the model API.  When the used to retrieve aggregations, Oldtimer
will return special data types detailed under the "Return Types" section.

### Paths

`/api/v1/historical/{prefix}/metrics/`: Returns a list of all available
metrics.

`/api/v1/historical{prefix}/metrics/{metric-name}?start=X&end=Y`: Returns a set
of (Timestamp, Value) pairs for the requested {prefix}-level metric, over the
given time range.

`/api/v1/historical{prefix}/metrics-aggregated/{aggregations}/{metric-name}?start=X&end=Y&bucket=B`
Returns the requested {prefix}-level metric, aggregated with the given
aggregation over the requested time period (potentially split into several
different bucket of duration `B`).  `{aggregations}` may be a comma-separated
list of aggregations to retrieve multiple at once.

Where `{prefix}` is normally either empty (cluster-level),
`/namespaces/{namespace}` (namespace-level),
`/namespaces/{namespace}/pods/{pod-name}` (pod-level),
`/namespaces/{namespace}/pod-list/{pod-list}` (multi-pod-level), or
`/namespaces/{namespace}/pods/{pod-name}/containers/{container-name}`
(container-level).

Additionally, since pod names are not temporally unique (i.e. it is possible to
delete a pod, and then create a new, completely different pod with the same
name), `{prefix}` may also be `/pod-id/{pod-id}` (pod-level metrics),
`/pod-id-list/{pod-id-list}` (multi-pod-level), or
`/pod-id/{pod-id}/containers/{container-name}` (container-level metrics).

In addition, when `{prefix}` is not empty, there will be a url of the form:
`/api/v1/historical/{prefix-without-final-element}` which allows fetching the
list of available nodes/namespaces/pods/containers.

Note that queries by pod name will return metrics from the latest pod with the
given name.  This may require an extra trip to the database in some cases, in
order to determine which pod id that actually is.  For this reason, if a
component knows the pod ids for which it is querying, using these is preferred
to using the pod names.  The pod-name-based API is retained for the sake of
easy queries and to match up with the model API.

### Parameter Types

The `start` and `end` parameters are defined the same way as for the model:
each should be a timestamp formatted according to RFC 3339, if no start time is
specified, it defaults to zero in Unix epoch time, and if no end time is
specified, all data after the start time will be considered.

The `bucket` (bucket duration) parameter is a number followed by any of the
following suffixes:

- `ms`: milliseconds
- `s`: seconds
- `m`: minutes
- `h`: hours
- `d`: days

### Return Types

For requests which simply fetch data points or list available objects, the
return format will be the same as that used in the Heapster model API.

The the case of aggregations, a different set of types is used: each bucket is
represented by a `MetricAggregationBucket`, which contains the timestamp for
that bucket (the start of the bucket), the count of entries in that bucket (if
requested) as an unsigned integer, as well as each of the other requested
aggregations, in the form of a `MetricValue` (which just holds an unsigned int
or a float).

All buckets for a particular metric are grouped together in a
`MetricAggregationResult`, which also holds the bucket size (duration) for the
buckets.  If multiple pods are requested, the result will be returned as a
`MetricAggregationResultList`, similarly to the `MetricResultList` for the
model API.

```go
type MetricValue struct {
    IntValue *uint64
    FloatValue *float64
}

type MetricAggregationBucket struct {
    Timestamp time.Time
    Count *uint64

    Average *MetricValue
    Maximum *MetricValue
    Minimum *MetricValue
    Median *MetricValue
    Percentiles map[uint64]MetricValue
}

type MetricAggregationResult struct {
    Buckets []MetricAggregationBucket
    BucketSize time.Duration
}

type MetricAggregationResultList struct {
    Items []MetricAggregationResult
}
```

### Aggregations

Several different aggregations will be supported.  Aggregations should be
performed in the metrics sink.  If more aggregations later become supported
across all metrics sinks, the list can be expanded.

- Average (arithmetic mean): `/metrics-aggregated/average`
- Maximum: `/metrics-aggregated/max`
- Minimum: `/metrics-aggregated/min`
- Percentile: `/metrics-aggregated/{number}-perc`
- Median: `/metrics-aggregated/median`
- Count: `/metrics-aggregated/count`

Note: to support all the existing sinks, the supported percentiles will be
limitted to 50, 95, and 99.  If additional percentile values later become
supported by other sinks, this list may be expanded (see the Sink Support
section below).

### Example

Suppose that one wanted to retrieve the 95th percentile of CPU usage for a
given pod over the past 30 days, in 1 hour intervals, along with the maximum
usage for each interval.  Call the pod "somepod", in the namespace "somens".
To fetch the results, you'd perform:

```
GET /api/v1/historical/namespaces/somens/pods/somepod/metrics-aggregated/95-perc,average/cpu/usage?start=2016-03-20T10:57:37-04:00&bucket=1h
```

Which would then return:

```json
{
    "bucketSize": "3600000000000",
    "buckets": [
        {
            "timestamp": "2016-03-20T10:57:37-04:00",
            "average": "32",
            "percentiles": {
                "95": "27"
            }
        },
        ...
    ]
}
```

## Sink Support and Functionality

When Oldtimer receives a request, it will compose a query to the sink, send the
query to the sink, and the transform the results into the appropriate API
formats.  Note that Oldtimer is designed to retrieve information that was
originally written by Heapster itself.  Any information read by Oldtimer must
have been stored according to the Heapster storage schema.

All computations, filtering, etc should be performed in the sink.  Oldtimer
should only be composing queries.  Ergo, the feature set of Oldtimer must
represent the lowest-common-denominator of features supported by the sinks.
Oldtimer is meant to be an API for performing basic aggregations supported by
all of the sinks, and is not meant to be a general purpose query tool.

At the time of writing of this proposal, the following sinks were considered:
Hawkular, InfluxDB, GCM, and OpenTSDB.  However, the aggregations supported are
fairly basic, so if new sinks are added, it should be fairly likely that they
support the required Oldtimer features.

## Scaling and Performance Considerations

Since Oldtimer itself does not store any data, it should have a fairly low
memory footprint.  The current plan is to have Oldtimer run as part of the main
Heapster executable.  However, in the future it may be advantageous to have the
ability to split Oldtimer out into a separate executable in order to scale it
independently of Heapster.

The metrics sinks themselves should already have clustering support, and thus
can be scaled if needed.  Since Oldtimer queries the metrics sinks themselves,
response latency should depend mainly on how quickly the sinks can respond to
queries.
