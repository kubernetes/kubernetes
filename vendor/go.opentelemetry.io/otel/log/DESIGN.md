# Logs API

## Abstract

`go.opentelemetry.io/otel/log` provides
[Logs API](https://opentelemetry.io/docs/specs/otel/logs/api/).

The prototype was created in
[#4725](https://github.com/open-telemetry/opentelemetry-go/pull/4725).

## Background

The key challenge is to create a performant API compliant with the [specification](https://opentelemetry.io/docs/specs/otel/logs/api/)
with an intuitive and user friendly design.
Performance is seen as one of the most important characteristics of logging libraries in Go.

## Design

This proposed design aims to:

- be specification compliant,
- be similar to Trace and Metrics API,
- take advantage of both OpenTelemetry and `slog` experience to achieve acceptable performance.

### Module structure

The API is published as a single `go.opentelemetry.io/otel/log` Go module.

The package structure is similar to Trace API and Metrics API.
The Go module consists of the following packages:

- `go.opentelemetry.io/otel/log`
- `go.opentelemetry.io/otel/log/embedded`
- `go.opentelemetry.io/otel/log/logtest`
- `go.opentelemetry.io/otel/log/noop`

Rejected alternative:

- [Reuse slog](#reuse-slog)

### LoggerProvider

The [`LoggerProvider` abstraction](https://opentelemetry.io/docs/specs/otel/logs/api/#loggerprovider)
is defined as `LoggerProvider` interface in [provider.go](provider.go).

The specification may add new operations to `LoggerProvider`.
The interface may have methods added without a package major version bump.
This embeds `embedded.LoggerProvider` to help inform an API implementation
author about this non-standard API evolution.
This approach is already used in Trace API and Metrics API.

#### LoggerProvider.Logger

The `Logger` method implements the [`Get a Logger` operation](https://opentelemetry.io/docs/specs/otel/logs/api/#get-a-logger).

The required `name` parameter is accepted as a `string` method argument.

The `LoggerOption` options are defined to support optional parameters.

Implementation requirements:

- The [specification requires](https://opentelemetry.io/docs/specs/otel/logs/api/#concurrency-requirements)
  the method to be safe to be called concurrently.

- If the passed name is empty, this should be reported as invalid as specified
  by the [Logs SDK specification](https://opentelemetry.io/docs/specs/otel/logs/sdk/#logger-creation).
  The method should still use it as the instrumentation scope name and return
  a working logger.

`Logger` can be extended by adding new `LoggerOption` options
and adding new exported fields to the `LoggerConfig` struct.
This design is already used in Trace API for getting tracers
and in Metrics API for getting meters.

Rejected alternative:

- [Passing struct as parameter to LoggerProvider.Logger](#passing-struct-as-parameter-to-loggerproviderlogger).

### Logger

The [`Logger` abstraction](https://opentelemetry.io/docs/specs/otel/logs/api/#logger)
is defined as `Logger` interface in [logger.go](logger.go).

The specification may add new operations to `Logger`.
The interface may have methods added without a package major version bump.
This embeds `embedded.Logger` to help inform an API implementation
author about this non-standard API evolution.
This approach is already used in Trace API and Metrics API.

### Logger.Emit

The `Emit` method implements the [`Emit a LogRecord` operation](https://opentelemetry.io/docs/specs/otel/logs/api/#emit-a-logrecord).

[`Context` associated with the `LogRecord`](https://opentelemetry.io/docs/specs/otel/context/)
is accepted as a `context.Context` method argument.

Calls to `Emit` are supposed to be on the hot path.
Therefore, in order to reduce the number of heap allocations,
the [`LogRecord` abstraction](https://opentelemetry.io/docs/specs/otel/logs/api/#emit-a-logrecord),
is defined as `Record` struct  in [record.go](record.go).

[`Timestamp`](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-timestamp)
is accessed using following methods:

```go
func (r *Record) Timestamp() time.Time
func (r *Record) SetTimestamp(t time.Time)
```

[`ObservedTimestamp`](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-observedtimestamp)
is accessed using following methods:

```go
func (r *Record) ObservedTimestamp() time.Time
func (r *Record) SetObservedTimestamp(t time.Time)
```

[`EventName`](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-eventname)
is accessed using following methods:

```go
func (r *Record) EventName() string
func (r *Record) SetEventName(s string)
```

[`SeverityNumber`](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-severitynumber)
is accessed using following methods:

```go
func (r *Record) Severity() Severity
func (r *Record) SetSeverity(s Severity)
```

`Severity` type is defined in [severity.go](severity.go).
The constants are are based on
[Displaying Severity recommendation](https://opentelemetry.io/docs/specs/otel/logs/data-model/#displaying-severity).
Additionally, `Severity[Level]` constants are defined to make the API more readable and user friendly.

[`SeverityText`](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-severitytext)
is accessed using following methods:

```go
func (r *Record) SeverityText() string
func (r *Record) SetSeverityText(s string)
```

[`Body`](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-body)
is accessed using following methods:

```go
func (r *Record) Body() attribute.Value
func (r *Record) SetBody(v attribute.Value)
```

[Log record attributes](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-attributes)
are accessed using following methods:

```go
func (r *Record) WalkAttributes(f func(attribute.KeyValue) bool)
func (r *Record) AddAttributes(attrs ...attribute.KeyValue)
```

`Record` has a `AttributesLen` method that returns
the number of attributes to allow slice preallocation
when converting records to a different representation:

```go
func (r *Record) AttributesLen() int
```

The records attributes design and implementation is based on
[`slog.Record`](https://pkg.go.dev/log/slog#Record).
It allows achieving high-performance access and manipulation of the attributes
while keeping the API user friendly.
It relieves the user from making his own improvements
for reducing the number of allocations when passing attributes.

Log body and attributes use the common
[`attribute.Value`](https://pkg.go.dev/go.opentelemetry.io/otel/attribute#Value)
and
[`attribute.KeyValue`](https://pkg.go.dev/go.opentelemetry.io/otel/attribute#KeyValue)
types.
These types cover the Logs Data Model `any` value shape, including empty
values, byte slices, generic slices, and maps.
Reusing the attribute package keeps the API consistent across signals and
avoids maintaining a second value model with conversion helpers.

The zero value of `attribute.Value` represents an empty body.
Log maps use `attribute.MAP`, which may contain duplicate keys when callers
construct one that way. Duplicate-key normalization is an SDK policy, not an
API behavior.

The caller must not subsequently mutate the record passed to `Emit`.
This would allow the implementation to not clone the record,
but simply retain, modify or discard it.
The implementation may still choose to clone the record or copy its attributes
if it needs to retain or modify it,
e.g. in case of asynchronous processing to eliminate the possibility of data races,
because the user can technically reuse the record and add new attributes
after the call (even when the documentation says that the caller must not do it).

Implementation requirements:

- The [specification requires](https://opentelemetry.io/docs/specs/otel/logs/api/#concurrency-requirements)
  the method to be safe to be called concurrently.

- The method must not interrupt the record processing if the context is canceled
  per ["ignoring context cancellation" guideline](../CONTRIBUTING.md#ignoring-context-cancellation).

- The [specification requires](https://opentelemetry.io/docs/specs/otel/logs/api/#emit-a-logrecord)
  use the current time as observed timestamp if the passed is empty.

- The method should handle the trace context passed via `ctx` argument in order to meet the
  [specification's SDK requirement](https://opentelemetry.io/docs/specs/otel/logs/sdk/#readablelogrecord)
  to populate the trace context fields from the resolved context.

`Emit` can be extended by adding new exported fields to the `Record` struct.

Rejected alternatives:

- [Record as interface](#record-as-interface)
- [Options as parameter to Logger.Emit](#options-as-parameter-to-loggeremit)
- [Passing record as pointer to Logger.Emit](#passing-record-as-pointer-to-loggeremit)
- [Logger.WithAttributes](#loggerwithattributes)
- [Record attributes as slice](#record-attributes-as-slice)
- [Use any instead of attribute.Value](#use-any-instead-of-attributevalue)
- [Severity type encapsulating number and text](#severity-type-encapsulating-number-and-text)
- [Define log-specific value types](#define-log-specific-value-types)
- [Mix receiver types for Record](#mix-receiver-types-for-record)
- [Add XYZ method to Logger](#add-xyz-method-to-logger)

### Logger.Enabled

The `Enabled` method implements the [`Enabled` operation](https://opentelemetry.io/docs/specs/otel/logs/api/#enabled).

[`Context` associated with the `LogRecord`](https://opentelemetry.io/docs/specs/otel/context/)
is accepted as a `context.Context` method argument.

Calls to `Enabled` are supposed to be on the hot path and the list of arguments
can be extendend in future. Therefore, in order to reduce the number of heap
allocations and make it possible to handle new arguments, `Enabled` accepts
a `EnabledParameters` struct, defined in [logger.go](logger.go), as the second
method argument.

The `EnabledParameters` uses fields, instead of getters and setters, to allow
simpler usage which allows configuring the `EnabledParameters` in the same line
where `Enabled` is called.

### noop package

The `go.opentelemetry.io/otel/log/noop` package provides
[Logs API No-Op Implementation](https://opentelemetry.io/docs/specs/otel/logs/noop/).

### Trace context correlation

The bridge implementation should do its best to pass
the `ctx` containing the trace context from the caller
so it can later be passed via `Logger.Emit`.

It is not expected that users (caller or bridge implementation) reconstruct
a `context.Context`. Reconstructing a `context.Context` with
[`trace.ContextWithSpanContext`](https://pkg.go.dev/go.opentelemetry.io/otel/trace#ContextWithSpanContext)
and [`trace.NewSpanContext`](https://pkg.go.dev/go.opentelemetry.io/otel/trace#NewSpanContext)
would usually involve more memory allocations.

The logging libraries which have recording methods that accepts `context.Context`,
such us [`slog`](https://pkg.go.dev/log/slog),
[`logrus`](https://pkg.go.dev/github.com/sirupsen/logrus),
[`zerolog`](https://pkg.go.dev/github.com/rs/zerolog),
makes passing the trace context trivial.

However, some libraries do not accept a `context.Context` in their recording methods.
Structured logging libraries,
such as [`logr`](https://pkg.go.dev/github.com/go-logr/logr)
and [`zap`](https://pkg.go.dev/go.uber.org/zap),
offer passing `any` type as a log attribute/field.
Therefore, their bridge implementations can define a "special" log attributes/field
that will be used to capture the trace context.

[The prototype](https://github.com/open-telemetry/opentelemetry-go/pull/4725)
has bridge implementations that handle trace context correlation efficiently.

## Benchmarking

The benchmarks take inspiration from [`slog`](https://pkg.go.dev/log/slog),
because for the Go team it was also critical to create API that would be fast
and interoperable with existing logging packages.[^2][^3]

The benchmark results can be found in [the prototype](https://github.com/open-telemetry/opentelemetry-go/pull/4725).

## Rejected alternatives

### Reuse slog

The API must not be coupled to [`slog`](https://pkg.go.dev/log/slog),
nor any other logging library.

The API needs to evolve orthogonally to `slog`.

`slog` is not compliant with the [Logs API](https://opentelemetry.io/docs/specs/otel/logs/api/).
and we cannot expect the Go team to make `slog` compliant with it.

The interoperability can be achieved using [a log bridge](https://opentelemetry.io/docs/specs/otel/glossary/#log-appender--bridge).

You can read more about OpenTelemetry Logs design on [opentelemetry.io](https://opentelemetry.io/docs/concepts/signals/logs/).

### Record as interface

`Record` is defined as a `struct` because of the following reasons.

Log record is a value object without any behavior.
It is used as data input for Logger methods.

The log record resembles the instrument config structs like [metric.Float64CounterConfig](https://pkg.go.dev/go.opentelemetry.io/otel/metric#Float64CounterConfig).

Using `struct` instead of `interface` improves the performance as e.g.
indirect calls are less optimized,
usage of interfaces tend to increase heap allocations.[^3]

### Options as parameter to Logger.Emit

One of the initial ideas was to have:

```go
type Logger interface{
	embedded.Logger
	Emit(ctx context.Context, options ...RecordOption)
}
```

The main reason was that design would be similar
to the [Meter API](https://pkg.go.dev/go.opentelemetry.io/otel/metric#Meter)
for creating instruments.

However, passing `Record` directly, instead of using options,
is more performant as it reduces heap allocations.[^4]

Another advantage of passing `Record` is that API would not have functions like `NewRecord(options...)`,
which would be used by the SDK and not by the users.

Finally, the definition would be similar to [`slog.Handler.Handle`](https://pkg.go.dev/log/slog#Handler)
that was designed to provide optimization opportunities.[^2]

### Passing record as pointer to Logger.Emit

So far the benchmarks do not show differences that would
favor passing the record via pointer (and vice versa).

Passing via value feels safer because of the following reasons.

The user would not be able to pass `nil`.
Therefore, it reduces the possibility to have a nil pointer dereference.

It should reduce the possibility of a heap allocation.

It follows the design of [`slog.Handler`](https://pkg.go.dev/log/slog#Handler).

If follows one of Google's Go Style Decisions
to prefer [passing values](https://google.github.io/styleguide/go/decisions#pass-values).

### Passing struct as parameter to LoggerProvider.Logger

Similarly to `Logger.Emit`, we could have something like:

```go
type LoggerProvider interface{
	embedded.LoggerProvider
	Logger(name string, config LoggerConfig)
}
```

The drawback of this idea would be that this would be
a different design from Trace and Metrics API.

The performance of acquiring a logger is not as critical
as the performance of emitting a log record. While a single
HTTP/RPC handler could write hundreds of logs, it should not
create a new logger for each log entry.
The bridge implementation should reuse loggers whenever possible.

### Logger.WithAttributes

We could add `WithAttributes` to the `Logger` interface.
Then `Record` could be a simple struct with only exported fields.
The idea was that the SDK would implement the performance improvements
instead of doing it in the API.
This would allow having different optimization strategies.

During the analysis[^5], it occurred that the main problem of this proposal
is that the variadic slice passed to an interface method is always heap allocated.

Moreover, the logger returned by `WithAttribute` was allocated on the heap.

Lastly, the proposal was not specification compliant.

### Record attributes as slice

One of the proposals[^6] was to have `Record` as a simple struct:

```go
type Record struct {
	Timestamp         time.Time
	ObservedTimestamp time.Time
	EventName         string
	Severity          Severity
	SeverityText      string
	Body              Value
	Attributes        []KeyValue
}
```

The bridge implementations could use [`sync.Pool`](https://pkg.go.dev/sync#Pool)
for reducing the number of allocations when passing attributes.

The benchmarks results were better.

In such a design, most bridges would have a `sync.Pool`
to reduce the number of heap allocations.
However, the `sync.Pool` will not work correctly with API implementations
that would take ownership of the record
(e.g. implementations that do not copy records for asynchronous processing).
The current design, even in case of improper API implementation,
has lower  chances of encountering a bug as most bridges would
create a record, pass it, and forget about it.

For reference, here is the reason why `slog` does not use `sync.Pool`[^3]
as well:

> We can use a sync pool for records though we decided not to.
You can but it's a bad idea for us. Why?
Because users have control of Records.
Handler writers can get their hands on a record
and we'd have to ask them to free it
or try to free it magically at some some point.
But either way, they could get themselves in trouble by freeing it twice
or holding on to one after they free it.
That's a use after free bug and that's why `zerolog` was problematic for us.
`zerolog` as as part of its speed exposes a pool allocated value to users
if you use `zerolog` the normal way, that you'll see in all the examples,
you will never encounter a problem.
But if you do something a little out of the ordinary you can get
use after free bugs and we just didn't want to put that in the standard library.

Therefore, we decided to not follow the proposal as it is
less user friendly (users and bridges would use e.g. a `sync.Pool` to reduce
the number of heap allocation), less safe (more prone to use after free bugs
and race conditions), and the benchmark differences were not significant.

### Use any instead of attribute.Value

[Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-body)
defines Body to be `any`.
One could propose to define `Body` (and attribute values) as `any`
instead of using `attribute.Value`.

First of all, [`any` type defined in the specification](https://opentelemetry.io/docs/specs/otel/logs/data-model/#type-any)
is not the same as `any` (`interface{}`) in Go.

Moreover, using `any` as a field would decrease the performance.[^7]

Using `attribute.Value` preserves a typed, allocation-conscious representation
of the Logs Data Model `any` values while avoiding unconstrained `interface{}`
handling in bridge implementations.

### Severity type encapsulating number and text

We could combine severity into a single field defining a type:

```go
type Severity struct {
	Number SeverityNumber
	Text string
}
```

However, the [Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/#log-and-event-record-definition)
define it as independent fields.
It should be more user friendly to have them separated.
Especially when having getter and setter methods, setting one value
when the other is already set would be unpleasant.

### Define log-specific value types

The original design defined `Kind`, `Value`, and `KeyValue` in
`go.opentelemetry.io/otel/log`.
That avoided coupling log bodies to the common attribute package while logs were
still exploring structured value support.

The Logs Data Model and common attribute value model now share the structured
`any` shapes needed by Go: empty, bool, int64, float64, string, byte slice,
homogeneous slices, generic slices, and maps.
The specification direction is to reuse these value shapes across signals.
Keeping log-specific types would duplicate API surface, require conversion
helpers, and make bridge code choose between two equivalent value models.

Therefore log records now use `attribute.Value` for body values and
`attribute.KeyValue` for attributes and map entries.

### Mix receiver types for Record

Methods of [`slog.Record`](https://pkg.go.dev/log/slog#Record)
have different receiver types.

In `log/slog` GitHub issue we can only find that the reason is:[^8]

>> some receiver of Record struct is by value
> Passing Records by value means they incur no heap allocation.
> That improves performance overall, even though they are copied.

However, the benchmarks do not show any noticeable differences.[^9]

The compiler is smart-enough to not make a heap allocation for any of these methods.
The use of a pointer receiver does not cause any heap allocation.
From Go FAQ:[^10]

> In the current compilers, if a variable has its address taken,
> that variable is a candidate for allocation on the heap.
> However, a basic escape analysis recognizes some cases
> when such variables will not live past the return from the function
> and can reside on the stack.

The [Understanding Allocations: the Stack and the Heap](https://www.youtube.com/watch?v=ZMZpH4yT7M0)
presentation by Jacob Walker describes the escape analysis with details.

Moreover, also from Go FAQ:[^10]

> Also, if a local variable is very large,
> it might make more sense to store it on the heap rather than the stack.

Therefore, even if we use a value receiver and the value is very large
it may be heap allocated.

Both [Go Code Review Comments](https://go.dev/wiki/CodeReviewComments#receiver-type)
and [Google's Go Style Decisions](https://google.github.io/styleguide/go/decisions#receiver-type)
highly recommend making the methods for a type either all pointer methods
or all value methods. Google's Go Style Decisions even goes further and says:

> There is a lot of misinformation about whether passing a value or a pointer
> to a function can affect performance.
> The compiler can choose to pass pointers to values on the stack
> as well as copying values on the stack,
> but these considerations should not outweigh the readability
> and correctness of the code in most circumstances.
> When the performance does matter, it is important to profile both approaches
> with a realistic benchmark before deciding that one approach outperforms the other.

Because, the benchmarks[^9] do not proof any performance difference
and the general recommendation is to not mix receiver types,
we decided to use pointer receivers for all `Record` methods.

### Add XYZ method to Logger

The `Logger` does not have methods like `SetSeverity`, etc.
as the Logs API needs to follow (be compliant with)
the [specification](https://opentelemetry.io/docs/specs/otel/logs/api/)

[^1]: [Handle structured body and attributes](https://github.com/pellared/opentelemetry-go/pull/7)
[^2]: Jonathan Amsterdam, [The Go Blog: Structured Logging with slog](https://go.dev/blog/slog)
[^3]: Jonathan Amsterdam, [GopherCon Europe 2023: A Fast Structured Logging Package](https://www.youtube.com/watch?v=tC4Jt3i62ns)
[^4]: [Emit definition discussion with benchmarks](https://github.com/open-telemetry/opentelemetry-go/pull/4725#discussion_r1400869566)
[^5]: [Logger.WithAttributes analysis](https://github.com/pellared/opentelemetry-go/pull/3)
[^6]: [Record attributes as field and use sync.Pool for reducing allocations](https://github.com/pellared/opentelemetry-go/pull/4) and [Record attributes based on slog.Record](https://github.com/pellared/opentelemetry-go/pull/6)
[^7]: [Record.Body as any](https://github.com/pellared/opentelemetry-go/pull/5)
[^8]: [log/slog: structured, leveled logging](https://github.com/golang/go/issues/56345#issuecomment-1302563756)
[^9]: [Record with pointer receivers only](https://github.com/pellared/opentelemetry-go/pull/8)
[^10]: [Go FAQ: Stack or heap](https://go.dev/doc/faq#stack_or_heap)
