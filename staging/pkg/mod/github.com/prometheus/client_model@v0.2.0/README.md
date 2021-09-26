# Deprecation note

This repository used to contain the [protocol
buffer](https://developers.google.com/protocol-buffers) code that defined both
the data model and the exposition format of Prometheus metrics.

Starting with v2.0.0, the [Prometheus
server](https://github.com/prometheus/prometheus) does not ingest the
protobuf-based exposition format anymore. Currently, all but one of the
[official instrumentation
libraries](https://prometheus.io/docs/instrumenting/clientlibs/) do not expose
the protobuf-based exposition format. The [Go instrumentation
library](https://github.com/prometheus/client_golang), however, has been built
around the protobuf-based data model. As a byproduct thereof, it is still able
to expose the protobuf-based exposition format. The Go instrumentation library
is the only remaining repository within the [Prometheus GitHub
org](https://github.com/prometheus) directly using the prometheus/client_model
repository.

Therefore, formerly existing support for languages other than Go (namely C++,
Java, Python, Ruby) has been removed from this repository. If you are a 3rd
party user of those languages, you can go back to [commit
14fe0d1](https://github.com/prometheus/client_model/commit/14fe0d1b01d4d5fc031dd4bec1823bd3ebbe8016)
to keep using the old code, or you can consume
[`metrics.proto`](https://github.com/prometheus/client_model/blob/master/metrics.proto)
directly with your own protobuf tooling. Note, however, that changes of
`metrics.proto` after [commit
14fe0d1](https://github.com/prometheus/client_model/commit/14fe0d1b01d4d5fc031dd4bec1823bd3ebbe8016)
are solely informed by requirements of the Go instrumentation library and will
not take into account any requirements of other languages or stability concerns
for the protobuf-based exposition format.

Check out the [OpenMetrics project](https://openmetrics.io/) for the future of
the data model and exposition format used by Prometheus and others.
