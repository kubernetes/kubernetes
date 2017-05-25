[![GoDoc][doc-img]][doc] [![Build Status][ci-img]][ci] [![Coverage Status][cov-img]][cov] [![OpenTracing 1.0 Enabled][ot-img]][ot-url]

# Jaeger Bindings for Go OpenTracing API

This is a client side library that implements an
[OpenTracing](http://opentracing.io) Tracer,
with Zipkin-compatible data model.

## Initialization

See [tracer initialization](https://godoc.org/github.com/uber/jaeger-client-go/config#pkg-examples).

### Metrics & Monitoring

The tracer emits a number of different metrics, defined in
[metrics.go](metrics.go). The monitoring backend is expected to support
tag-based metric names, e.g. instead of `statsd`-style string names
like `counters.my-service.jaeger.spans.started.sampled`, the metrics
are defined by a short name and a collection of key/value tags, for
example: `name:traces, state:started, sampled:true`.

The monitoring backend is represented by the
[StatsReporter](stats_reporter.go) interface. An implementation
of that interface should be passed to the `New` method during
tracer initialization:

```go
    stats := // create StatsReporter implementation
    tracer := config.Tracing.New("your-service-name", stats)
```

By default, a no-op `NullStatsReporter` is used.

### Logging

The tracer can be configured with an optional logger, which will be
used to log communication errors, or log spans if a logging reporter
option is specified in the configuration. The logging API is abstracted
by the [Logger](logger.go) interface. A logger instance implementing
this interface can be set on the `Config` object before calling the
`New` method.

## Instrumentation for Tracing

Since this tracer is fully compliant with OpenTracing API 1.0,
all code instrumentation should only use the API itself, as described
in the [opentracing-go]
(https://github.com/opentracing/opentracing-go) documentation.

## Features

### Reporters

A "reporter" is a component receives the finished spans and reports
them to somewhere. Under normal circumstances, the Tracer
should use the default `RemoteReporter`, which sends the spans out of
process via configurable "transport". For testing purposes, one can
use an `InMemoryReporter` that accumulates spans in a buffer and
allows to retrieve them for later verification. Also available are
`NullReporter`, a no-op reporter that does nothing, a `LoggingReporter`
which logs all finished spans using their `String()` method, and a
`CompositeReporter` that can be used to combine more than one reporter
into one, e.g. to attach a logging reporter to the main remote reporter.

### Span Reporting Transports

The remote reporter uses "transports" to actually send the spans out
of process. Currently two supported transports are Thrift over UDP
and Thrift over TChannel. More transports will be added in the future.

The only data format currently used is Zipkin Thrift 1.x span format,
which allows easy integration of the tracer with Zipkin backend.

### Sampling

The tracer does not record all spans, but only those that have the
sampling bit set in the `flags`. When a new trace is started and a new
unique ID is generated, a sampling decision is made whether this trace
should be sampled. The sampling decision is propagated to all downstream
calls via the `flags` field of the trace context. The following samplers
are available:
  1. `RemotelyControlledSampler` uses one of the other simpler samplers
     and periodically updates it by polling an external server. This
     allows dynamic control of the sampling strategies.
  1. `ConstSampler` always makes the same sampling decision for all
     trace IDs. it can be configured to either sample all traces, or
     to sample none.
  1. `ProbabilisticSampler` uses a fixed sampling rate as a probability
     for a given trace to be sampled. The actual decision is made by
     comparing the trace ID with a random number multiplied by the
     sampling rate.
  1. `RateLimitingSampler` can be used to allow only a certain fixed
     number of traces to be sampled per second.

### Baggage Injection

The OpenTracing spec allows for [baggage](https://github.com/opentracing/specification/blob/master/specification.md#set-a-baggage-item),
which are key value pairs that are added to the span context and propagated
throughout the trace.
An external process can inject baggage by setting the special
HTTP Header `jaeger-baggage` on a request

```sh
curl -H "jaeger-baggage: key1=value1, key2=value2" http://myhost.com
```

Baggage can also be programatically set inside your service by doing
the following

```go
if span := opentracing.SpanFromContext(ctx); span != nil {
    span.SetBaggageItem("key", "value")
}
```

Another service downstream of that can retrieve the baggage in a similar way:

```go
if span := opentracing.SpanFromContext(ctx); span != nil {
    val := span.BaggageItem("key")
    println(val)
}
```

### Debug Traces (Forced Sampling)

#### Programmatically

The OpenTracing API defines a `sampling.priority` standard tag that
can be used to affect the sampling of a span and its children:

```go
import (
    "github.com/opentracing/opentracing-go"
    "github.com/opentracing/opentracing-go/ext"
)

span := opentracing.SpanFromContext(ctx)
ext.SamplingPriority.Set(span, 1)    
```

#### Via HTTP Headers

Jaeger Tracer also understands a special HTTP Header `jaeger-debug-id`,
which can be set in the incoming request, e.g.

```sh
curl -H "jaeger-debug-id: some-correlation-id" http://myhost.com
```

When Jaeger sees this header in the request that otherwise has no
tracing context, it ensures that the new trace started for this
request will be sampled in the "debug" mode (meaning it should survive
all downsampling that might happen in the collection pipeline), and the
root span will have a tag as if this statement was executed:

```go
span.SetTag("jaeger-debug-id", "some-correlation-id")
```

This allows using Jaeger UI to find the trace by this tag.

### Zipkin HTTP B3 compatible header propagation

Jaeger Tracer supports Zipkin B3 Propagation HTTP headers, which are used
by a lot of Zipkin tracers. This means that you can use Jaeger in conjunction with e.g. [these OpenZipkin tracers](https://github.com/openzipkin).

However it is not the default propagation format, see [here](zipkin/README.md#NewZipkinB3HTTPHeaderPropagator) how to set it up.

## License

  [The MIT License](LICENSE).


[doc-img]: https://godoc.org/github.com/uber/jaeger-client-go?status.svg
[doc]: https://godoc.org/github.com/uber/jaeger-client-go
[ci-img]: https://travis-ci.org/uber/jaeger-client-go.svg?branch=master
[ci]: https://travis-ci.org/uber/jaeger-client-go
[cov-img]: https://coveralls.io/repos/uber/jaeger-client-go/badge.svg?branch=master&service=github
[cov]: https://coveralls.io/github/uber/jaeger-client-go?branch=master
[ot-img]: https://img.shields.io/badge/OpenTracing--1.0-enabled-blue.svg
[ot-url]: http://opentracing.io
