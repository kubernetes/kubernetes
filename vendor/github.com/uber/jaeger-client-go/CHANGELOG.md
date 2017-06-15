Changes by Version
==================

2.6.1 (unreleased)
------------------

- Nothing yet


2.6.0 (2017-03-28)
------------------

- Add config option to initialize RPC Metrics feature


2.5.0 (2017-03-23)
------------------

- Split request latency metric by success/failure [#123](https://github.com/uber/jaeger-client-go/pull/123)
- Add mutex to adaptive sampler and fix race condition [#124](https://github.com/uber/jaeger-client-go/pull/124)
- Fix rate limiter panic [#125](https://github.com/uber/jaeger-client-go/pull/125)


2.4.0 (2017-03-21)
------------------

- Remove `_ms` suffix from request latency metric name [#121](https://github.com/uber/jaeger-client-go/pull/121)
- Rename all metrics to "request" and "http_request" and use tags for other dimensions [#121](https://github.com/uber/jaeger-client-go/pull/121)


2.3.0 (2017-03-20)
------------------

- Make Span type public to allow access to non-std methods for testing [#117](https://github.com/uber/jaeger-client-go/pull/117)
- Add a structured way to extract traces for logging with zap [#118](https://github.com/uber/jaeger-client-go/pull/118)


2.2.1 (2017-03-14)
------------------

- Fix panic caused by updating the remote sampler from adaptive sampler to any other sampler type (https://github.com/uber/jaeger-client-go/pull/111)


2.2.0 (2017-03-10)
------------------

- Introduce Observer and SpanObserver (https://github.com/uber/jaeger-client-go/pull/94)
- Add RPC metrics emitter as Observer/SpanObserver (https://github.com/uber/jaeger-client-go/pull/103)


2.1.2 (2017-02-27)
-------------------

- Fix leaky bucket bug (https://github.com/uber/jaeger-client-go/pull/99)
- Fix zap logger Infof (https://github.com/uber/jaeger-client-go/pull/100)
- Add tracer initialization godoc examples


2.1.1 (2017-02-21)
-------------------

- Fix inefficient usage of zap.Logger


2.1.0 (2017-02-17)
-------------------

- Add adapter for zap.Logger (https://github.com/uber-go/zap)
- Move logging API to ./log/ package


2.0.0 (2017-02-08)
-------------------

- Support Adaptive Sampling
- Support 128bit Trace IDs
- Change trace/span IDs from uint64 to strong types TraceID and SpanID
- Add Zipkin HTTP B3 Propagation format support #72
- Rip out existing metrics and use github.com/uber/jaeger-lib/metrics
- Change API for tracer, reporter, sampler initialization


1.6.0 (2016-10-14)
-------------------

- Add Zipkin HTTP transport
- Support external baggage via jaeger-baggage header
- Unpin Thrift version, keep to master


1.5.1 (2016-09-27)
-------------------

- Relax dependency on opentracing to ^1


1.5.0 (2016-09-27)
-------------------

- Upgrade to opentracing-go 1.0
- Support KV logging for Spans


1.4.0 (2016-09-14)
-------------------

- Support debug traces via HTTP header "jaeger-debug-id"
