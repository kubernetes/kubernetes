[![Gitter chat](http://img.shields.io/badge/gitter-join%20chat%20%E2%86%92-brightgreen.svg)](https://gitter.im/opentracing/public) [![Build Status](https://travis-ci.org/opentracing/opentracing-go.svg?branch=master)](https://travis-ci.org/opentracing/opentracing-go) [![GoDoc](https://godoc.org/github.com/opentracing/opentracing-go?status.svg)](http://godoc.org/github.com/opentracing/opentracing-go)

# OpenTracing API for Go

This package is a Go platform API for OpenTracing.

## Required Reading

In order to understand the Go platform API, one must first be familiar with the
[OpenTracing project](http://opentracing.io) and
[terminology](http://opentracing.io/documentation/pages/spec.html) more specifically.

## API overview for those adding instrumentation

Everyday consumers of this `opentracing` package really only need to worry
about a couple of key abstractions: the `StartSpan` function, the `Span`
interface, and binding a `Tracer` at `main()`-time. Here are code snippets
demonstrating some important use cases.

#### Singleton initialization

The simplest starting point is `./default_tracer.go`. As early as possible, call

```go
    import "github.com/opentracing/opentracing-go"
    import ".../some_tracing_impl"

    func main() {
        opentracing.InitGlobalTracer(
            // tracing impl specific:
            some_tracing_impl.New(...),
        )
        ...
    }
```

##### Non-Singleton initialization

If you prefer direct control to singletons, manage ownership of the
`opentracing.Tracer` implementation explicitly.

#### Creating a Span given an existing Go `context.Context`

If you use `context.Context` in your application, OpenTracing's Go library will
happily rely on it for `Span` propagation. To start a new (blocking child)
`Span`, you can use `StartSpanFromContext`.

```go
    func xyz(ctx context.Context, ...) {
        ...
        span, ctx := opentracing.StartSpanFromContext(ctx, "operation_name")
        defer span.Finish()
        span.LogFields(
            log.String("event", "soft error"),
            log.String("type", "cache timeout"),
            log.Int("waited.millis", 1500))
        ...
    }
```

#### Starting an empty trace by creating a "root span"

It's always possible to create a "root" `Span` with no parent or other causal
reference.

```go
    func xyz() {
        ...
        sp := opentracing.StartSpan("operation_name")
        defer sp.Finish()
        ...
    }
```

#### Creating a (child) Span given an existing (parent) Span

```go
    func xyz(parentSpan opentracing.Span, ...) {
        ...
        sp := opentracing.StartSpan(
            "operation_name",
            opentracing.ChildOf(parentSpan.Context()))
        defer sp.Finish()
        ...
    }
```

#### Serializing to the wire

```go
    func makeSomeRequest(ctx context.Context) ... {
        if span := opentracing.SpanFromContext(ctx); span != nil {
            httpClient := &http.Client{}
            httpReq, _ := http.NewRequest("GET", "http://myservice/", nil)

            // Transmit the span's TraceContext as HTTP headers on our
            // outbound request.
            opentracing.GlobalTracer().Inject(
                span.Context(),
                opentracing.HTTPHeaders,
                opentracing.HTTPHeadersCarrier(httpReq.Header))

            resp, err := httpClient.Do(httpReq)
            ...
        }
        ...
    }
```

#### Deserializing from the wire

```go
    http.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
        var serverSpan opentracing.Span
        appSpecificOperationName := ...
        wireContext, err := opentracing.GlobalTracer().Extract(
            opentracing.HTTPHeaders,
            opentracing.HTTPHeadersCarrier(req.Header))
        if err != nil {
            // Optionally record something about err here
        }

        // Create the span referring to the RPC client if available.
        // If wireContext == nil, a root span will be created.
        serverSpan = opentracing.StartSpan(
            appSpecificOperationName,
            ext.RPCServerOption(wireContext))

        defer serverSpan.Finish()

        ctx := opentracing.ContextWithSpan(context.Background(), serverSpan)
        ...
    }
```

#### Goroutine-safety

The entire public API is goroutine-safe and does not require external
synchronization.

## API pointers for those implementing a tracing system

Tracing system implementors may be able to reuse or copy-paste-modify the `basictracer` package, found [here](https://github.com/opentracing/basictracer-go). In particular, see `basictracer.New(...)`.

## API compatibility

For the time being, "mild" backwards-incompatible changes may be made without changing the major version number. As OpenTracing and `opentracing-go` mature, backwards compatibility will become more of a priority.
