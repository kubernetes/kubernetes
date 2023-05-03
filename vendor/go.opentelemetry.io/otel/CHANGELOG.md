# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.15.1/0.38.1] 2023-05-02

### Fixed

- Remove unused imports from `sdk/resource/host_id_bsd.go` which caused build failures. (#4040, #4041)

## [1.15.0/0.38.0] 2023-04-27

### Added

- The `go.opentelemetry.io/otel/metric/embedded` package. (#3916)
- The `Version` function to `go.opentelemetry.io/otel/sdk` to return the SDK version. (#3949)
- Add a `WithNamespace` option to `go.opentelemetry.io/otel/exporters/prometheus` to allow users to prefix metrics with a namespace. (#3970)
- The following configuration types were added to `go.opentelemetry.io/otel/metric/instrument` to be used in the configuration of measurement methods. (#3971)
  - The `AddConfig` used to hold configuration for addition measurements
    - `NewAddConfig` used to create a new `AddConfig`
    - `AddOption` used to configure an `AddConfig`
  - The `RecordConfig` used to hold configuration for recorded measurements
    - `NewRecordConfig` used to create a new `RecordConfig`
    - `RecordOption` used to configure a `RecordConfig`
  - The `ObserveConfig` used to hold configuration for observed measurements
    - `NewObserveConfig` used to create a new `ObserveConfig`
    - `ObserveOption` used to configure an `ObserveConfig`
- `WithAttributeSet` and `WithAttributes` are added to `go.opentelemetry.io/otel/metric/instrument`.
  They return an option used during a measurement that defines the attribute Set associated with the measurement. (#3971)
- The `Version` function to `go.opentelemetry.io/otel/exporters/otlp/otlpmetric` to return the OTLP metrics client version. (#3956)
- The `Version` function to `go.opentelemetry.io/otel/exporters/otlp/otlptrace` to return the OTLP trace client version. (#3956)

### Changed

- The `Extrema` in `go.opentelemetry.io/otel/sdk/metric/metricdata` is redefined with a generic argument of `[N int64 | float64]`. (#3870)
- Update all exported interfaces from `go.opentelemetry.io/otel/metric` to embed their corresponding interface from `go.opentelemetry.io/otel/metric/embedded`.
  This adds an implementation requirement to set the interface default behavior for unimplemented methods. (#3916)
- Move No-Op implementation from `go.opentelemetry.io/otel/metric` into its own package `go.opentelemetry.io/otel/metric/noop`. (#3941)
  - `metric.NewNoopMeterProvider` is replaced with `noop.NewMeterProvider`
- Add all the methods from `"go.opentelemetry.io/otel/trace".SpanContext` to `bridgeSpanContext` by embedding `otel.SpanContext` in `bridgeSpanContext`. (#3966)
- Wrap `UploadMetrics` error in `go.opentelemetry.io/otel/exporters/otlp/otlpmetric/` to improve error message when encountering generic grpc errors. (#3974)
- The measurement methods for all instruments in `go.opentelemetry.io/otel/metric/instrument` accept an option instead of the variadic `"go.opentelemetry.io/otel/attribute".KeyValue`. (#3971)
  - The `Int64Counter.Add` method now accepts `...AddOption`
  - The `Float64Counter.Add` method now accepts `...AddOption`
  - The `Int64UpDownCounter.Add` method now accepts `...AddOption`
  - The `Float64UpDownCounter.Add` method now accepts `...AddOption`
  - The `Int64Histogram.Record` method now accepts `...RecordOption`
  - The `Float64Histogram.Record` method now accepts `...RecordOption`
  - The `Int64Observer.Observe` method now accepts `...ObserveOption`
  - The `Float64Observer.Observe` method now accepts `...ObserveOption`
- The `Observer` methods in `go.opentelemetry.io/otel/metric` accept an option instead of the variadic `"go.opentelemetry.io/otel/attribute".KeyValue`. (#3971)
  - The `Observer.ObserveInt64` method now accepts `...ObserveOption`
  - The `Observer.ObserveFloat64` method now accepts `...ObserveOption`
- Move global metric back to `go.opentelemetry.io/otel/metric/global` from `go.opentelemetry.io/otel`. (#3986)

### Fixed

- `TracerProvider` allows calling `Tracer()` while it's shutting down.
  It used to deadlock. (#3924)
- Use the SDK version for the Telemetry SDK resource detector in `go.opentelemetry.io/otel/sdk/resource`. (#3949)
- Fix a data race in `SpanProcessor` returned by `NewSimpleSpanProcessor` in `go.opentelemetry.io/otel/sdk/trace`. (#3951)
- Automatically figure out the default aggregation with `aggregation.Default`. (#3967)

### Deprecated

- The `go.opentelemetry.io/otel/metric/instrument` package is deprecated.
  Use the equivalent types added to `go.opentelemetry.io/otel/metric` instead. (#4018)

## [1.15.0-rc.2/0.38.0-rc.2] 2023-03-23

This is a release candidate for the v1.15.0/v0.38.0 release.
That release will include the `v1` release of the OpenTelemetry Go metric API and will provide stability guarantees of that API.
See our [versioning policy](VERSIONING.md) for more information about these stability guarantees.

### Added

- The `WithHostID` option to `go.opentelemetry.io/otel/sdk/resource`. (#3812)
- The `WithoutTimestamps` option to `go.opentelemetry.io/otel/exporters/stdout/stdoutmetric` to sets all timestamps to zero. (#3828)
- The new `Exemplar` type is added to `go.opentelemetry.io/otel/sdk/metric/metricdata`.
  Both the `DataPoint` and `HistogramDataPoint` types from that package have a new field of `Exemplars` containing the sampled exemplars for their timeseries. (#3849)
- Configuration for each metric instrument in `go.opentelemetry.io/otel/sdk/metric/instrument`. (#3895)
- The internal logging introduces a warning level verbosity equal to `V(1)`. (#3900)
- Added a log message warning about usage of `SimpleSpanProcessor` in production environments. (#3854)

### Changed

- Optimize memory allocation when creation a new `Set` using `NewSet` or `NewSetWithFiltered` in `go.opentelemetry.io/otel/attribute`. (#3832)
- Optimize memory allocation when creation new metric instruments in `go.opentelemetry.io/otel/sdk/metric`. (#3832)
- Avoid creating new objects on all calls to `WithDeferredSetup` and `SkipContextSetup` in OpenTracing bridge. (#3833)
- The `New` and `Detect` functions from `go.opentelemetry.io/otel/sdk/resource` return errors that wrap underlying errors instead of just containing the underlying error strings. (#3844)
- Both the `Histogram` and `HistogramDataPoint` are redefined with a generic argument of `[N int64 | float64]` in `go.opentelemetry.io/otel/sdk/metric/metricdata`. (#3849)
- The metric `Export` interface from `go.opentelemetry.io/otel/sdk/metric` accepts a `*ResourceMetrics` instead of `ResourceMetrics`. (#3853)
- Rename `Asynchronous` to `Observable` in `go.opentelemetry.io/otel/metric/instrument`. (#3892)
- Rename `Int64ObserverOption` to `Int64ObservableOption` in `go.opentelemetry.io/otel/metric/instrument`. (#3895)
- Rename `Float64ObserverOption` to `Float64ObservableOption` in `go.opentelemetry.io/otel/metric/instrument`. (#3895)
- The internal logging changes the verbosity level of info to `V(4)`, the verbosity level of debug to `V(8)`. (#3900)

### Fixed

- `TracerProvider` consistently doesn't allow to register a `SpanProcessor` after shutdown. (#3845)

### Removed

- The deprecated `go.opentelemetry.io/otel/metric/global` package is removed. (#3829)
- The unneeded `Synchronous` interface in `go.opentelemetry.io/otel/metric/instrument` was removed. (#3892)
- The `Float64ObserverConfig` and `NewFloat64ObserverConfig` in `go.opentelemetry.io/otel/sdk/metric/instrument`.
  Use the added `float64` instrument configuration instead. (#3895)
- The `Int64ObserverConfig` and `NewInt64ObserverConfig` in `go.opentelemetry.io/otel/sdk/metric/instrument`.
  Use the added `int64` instrument configuration instead. (#3895)
- The `NewNoopMeter` function in `go.opentelemetry.io/otel/metric`, use `NewMeterProvider().Meter("")` instead. (#3893)

## [1.15.0-rc.1/0.38.0-rc.1] 2023-03-01

This is a release candidate for the v1.15.0/v0.38.0 release.
That release will include the `v1` release of the OpenTelemetry Go metric API and will provide stability guarantees of that API.
See our [versioning policy](VERSIONING.md) for more information about these stability guarantees.

This release drops the compatibility guarantee of [Go 1.18].

### Added

- Support global `MeterProvider` in `go.opentelemetry.io/otel`. (#3818)
  - Use `Meter` for a `metric.Meter` from the global `metric.MeterProvider`.
  - Use `GetMeterProivder` for a global `metric.MeterProvider`.
  - Use `SetMeterProivder` to set the global `metric.MeterProvider`.

### Changed

- Dropped compatibility testing for [Go 1.18].
  The project no longer guarantees support for this version of Go. (#3813)

### Fixed

- Handle empty environment variable as it they were not set. (#3764)
- Clarify the `httpconv` and `netconv` packages in `go.opentelemetry.io/otel/semconv/*` provide tracing semantic conventions. (#3823)

### Deprecated

- The `go.opentelemetry.io/otel/metric/global` package is deprecated.
  Use `go.opentelemetry.io/otel` instead. (#3818)

### Removed

- The deprecated `go.opentelemetry.io/otel/metric/unit` package is removed. (#3814)

## [1.14.0/0.37.0/0.0.4] 2023-02-27

This release is the last to support [Go 1.18].
The next release will require at least [Go 1.19].

### Added

- The `event` type semantic conventions are added to `go.opentelemetry.io/otel/semconv/v1.17.0`. (#3697)
- Support [Go 1.20]. (#3693)
- The `go.opentelemetry.io/otel/semconv/v1.18.0` package.
  The package contains semantic conventions from the `v1.18.0` version of the OpenTelemetry specification. (#3719)
  - The following `const` renames from `go.opentelemetry.io/otel/semconv/v1.17.0` are included:
    - `OtelScopeNameKey` -> `OTelScopeNameKey`
    - `OtelScopeVersionKey` -> `OTelScopeVersionKey`
    - `OtelLibraryNameKey` -> `OTelLibraryNameKey`
    - `OtelLibraryVersionKey` -> `OTelLibraryVersionKey`
    - `OtelStatusCodeKey` -> `OTelStatusCodeKey`
    - `OtelStatusDescriptionKey` -> `OTelStatusDescriptionKey`
    - `OtelStatusCodeOk` -> `OTelStatusCodeOk`
    - `OtelStatusCodeError` -> `OTelStatusCodeError`
  - The following `func` renames from `go.opentelemetry.io/otel/semconv/v1.17.0` are included:
    - `OtelScopeName` -> `OTelScopeName`
    - `OtelScopeVersion` -> `OTelScopeVersion`
    - `OtelLibraryName` -> `OTelLibraryName`
    - `OtelLibraryVersion` -> `OTelLibraryVersion`
    - `OtelStatusDescription` -> `OTelStatusDescription`
- A `IsSampled` method is added to the `SpanContext` implementation in `go.opentelemetry.io/otel/bridge/opentracing` to expose the span sampled state.
  See the [README](./bridge/opentracing/README.md) for more information. (#3570)
- The `WithInstrumentationAttributes` option to `go.opentelemetry.io/otel/metric`. (#3738)
- The `WithInstrumentationAttributes` option to `go.opentelemetry.io/otel/trace`. (#3739)
- The following environment variables are supported by the periodic `Reader` in `go.opentelemetry.io/otel/sdk/metric`. (#3763)
  - `OTEL_METRIC_EXPORT_INTERVAL` sets the time between collections and exports.
  - `OTEL_METRIC_EXPORT_TIMEOUT` sets the timeout an export is attempted.

### Changed

- Fall-back to `TextMapCarrier` when it's not `HttpHeader`s in `go.opentelemetry.io/otel/bridge/opentracing`. (#3679)
- The `Collect` method of the `"go.opentelemetry.io/otel/sdk/metric".Reader` interface is updated to accept the `metricdata.ResourceMetrics` value the collection will be made into.
  This change is made to enable memory reuse by SDK users. (#3732)
- The `WithUnit` option in `go.opentelemetry.io/otel/sdk/metric/instrument` is updated to accept a `string` for the unit value. (#3776)

### Fixed

- Ensure `go.opentelemetry.io/otel` does not use generics. (#3723, #3725)
- Multi-reader `MeterProvider`s now export metrics for all readers, instead of just the first reader. (#3720, #3724)
- Remove use of deprecated `"math/rand".Seed` in `go.opentelemetry.io/otel/example/prometheus`. (#3733)
- Do not silently drop unknown schema data with `Parse` in  `go.opentelemetry.io/otel/schema/v1.1`. (#3743)
- Data race issue in OTLP exporter retry mechanism. (#3755, #3756)
- Wrapping empty errors when exporting in `go.opentelemetry.io/otel/sdk/metric`. (#3698, #3772)
- Incorrect "all" and "resource" definition for schema files in `go.opentelemetry.io/otel/schema/v1.1`. (#3777)

### Deprecated

- The `go.opentelemetry.io/otel/metric/unit` package is deprecated.
  Use the equivalent unit string instead. (#3776)
  - Use `"1"` instead of `unit.Dimensionless`
  - Use `"By"` instead of `unit.Bytes`
  - Use `"ms"` instead of `unit.Milliseconds`

## [1.13.0/0.36.0] 2023-02-07

### Added

- Attribute `KeyValue` creations functions to `go.opentelemetry.io/otel/semconv/v1.17.0` for all non-enum semantic conventions.
  These functions ensure semantic convention type correctness. (#3675)

### Fixed

- Removed the `http.target` attribute from being added by `ServerRequest` in the following packages. (#3687)
  - `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`
  - `go.opentelemetry.io/otel/semconv/v1.14.0/httpconv`
  - `go.opentelemetry.io/otel/semconv/v1.15.0/httpconv`
  - `go.opentelemetry.io/otel/semconv/v1.16.0/httpconv`
  - `go.opentelemetry.io/otel/semconv/v1.17.0/httpconv`

### Removed

- The deprecated `go.opentelemetry.io/otel/metric/instrument/asyncfloat64` package is removed. (#3631)
- The deprecated `go.opentelemetry.io/otel/metric/instrument/asyncint64` package is removed. (#3631)
- The deprecated `go.opentelemetry.io/otel/metric/instrument/syncfloat64` package is removed. (#3631)
- The deprecated `go.opentelemetry.io/otel/metric/instrument/syncint64` package is removed. (#3631)

## [1.12.0/0.35.0] 2023-01-28

### Added

- The `WithInt64Callback` option to `go.opentelemetry.io/otel/metric/instrument`.
  This options is used to configure `int64` Observer callbacks during their creation. (#3507)
- The `WithFloat64Callback` option to `go.opentelemetry.io/otel/metric/instrument`.
  This options is used to configure `float64` Observer callbacks during their creation. (#3507)
- The `Producer` interface and `Reader.RegisterProducer(Producer)` to `go.opentelemetry.io/otel/sdk/metric`.
  These additions are used to enable external metric Producers. (#3524)
- The `Callback` function type to `go.opentelemetry.io/otel/metric`.
  This new named function type is registered with a `Meter`. (#3564)
- The `go.opentelemetry.io/otel/semconv/v1.13.0` package.
  The package contains semantic conventions from the `v1.13.0` version of the OpenTelemetry specification. (#3499)
  - The `EndUserAttributesFromHTTPRequest` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is merged into `ClientRequest` and `ServerRequest` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `HTTPAttributesFromHTTPStatusCode` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is merged into `ClientResponse` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `HTTPClientAttributesFromHTTPRequest` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is replaced by `ClientRequest` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `HTTPServerAttributesFromHTTPRequest` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is replaced by `ServerRequest` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `HTTPServerMetricAttributesFromHTTPRequest` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is replaced by `ServerRequest` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `NetAttributesFromHTTPRequest` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is split into `Transport` in `go.opentelemetry.io/otel/semconv/v1.13.0/netconv` and `ClientRequest` or `ServerRequest` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `SpanStatusFromHTTPStatusCode` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is replaced by `ClientStatus` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `SpanStatusFromHTTPStatusCodeAndSpanKind` function in `go.opentelemetry.io/otel/semconv/v1.12.0` is split into `ClientStatus` and `ServerStatus` in `go.opentelemetry.io/otel/semconv/v1.13.0/httpconv`.
  - The `Client` function is included in `go.opentelemetry.io/otel/semconv/v1.13.0/netconv` to generate attributes for a `net.Conn`.
  - The `Server` function is included in `go.opentelemetry.io/otel/semconv/v1.13.0/netconv` to generate attributes for a `net.Listener`.
- The `go.opentelemetry.io/otel/semconv/v1.14.0` package.
  The package contains semantic conventions from the `v1.14.0` version of the OpenTelemetry specification. (#3566)
- The `go.opentelemetry.io/otel/semconv/v1.15.0` package.
  The package contains semantic conventions from the `v1.15.0` version of the OpenTelemetry specification. (#3578)
- The `go.opentelemetry.io/otel/semconv/v1.16.0` package.
  The package contains semantic conventions from the `v1.16.0` version of the OpenTelemetry specification. (#3579)
- Metric instruments to `go.opentelemetry.io/otel/metric/instrument`.
  These instruments are use as replacements of the deprecated `go.opentelemetry.io/otel/metric/instrument/{asyncfloat64,asyncint64,syncfloat64,syncint64}` packages.(#3575, #3586)
  - `Float64ObservableCounter` replaces the `asyncfloat64.Counter`
  - `Float64ObservableUpDownCounter` replaces the `asyncfloat64.UpDownCounter`
  - `Float64ObservableGauge` replaces the `asyncfloat64.Gauge`
  - `Int64ObservableCounter` replaces the `asyncint64.Counter`
  - `Int64ObservableUpDownCounter` replaces the `asyncint64.UpDownCounter`
  - `Int64ObservableGauge` replaces the `asyncint64.Gauge`
  - `Float64Counter` replaces the `syncfloat64.Counter`
  - `Float64UpDownCounter` replaces the `syncfloat64.UpDownCounter`
  - `Float64Histogram` replaces the `syncfloat64.Histogram`
  - `Int64Counter` replaces the `syncint64.Counter`
  - `Int64UpDownCounter` replaces the `syncint64.UpDownCounter`
  - `Int64Histogram` replaces the `syncint64.Histogram`
- `NewTracerProvider` to `go.opentelemetry.io/otel/bridge/opentracing`.
  This is used to create `WrapperTracer` instances from a `TracerProvider`. (#3116)
- The `Extrema` type to `go.opentelemetry.io/otel/sdk/metric/metricdata`.
  This type is used to represent min/max values and still be able to distinguish unset and zero values. (#3487)
- The `go.opentelemetry.io/otel/semconv/v1.17.0` package.
  The package contains semantic conventions from the `v1.17.0` version of the OpenTelemetry specification. (#3599)

### Changed

- Jaeger and Zipkin exporter use `github.com/go-logr/logr` as the logging interface, and add the `WithLogr` option. (#3497, #3500)
- Instrument configuration in `go.opentelemetry.io/otel/metric/instrument` is split into specific options and configuration based on the instrument type. (#3507)
  - Use the added `Int64Option` type to configure instruments from `go.opentelemetry.io/otel/metric/instrument/syncint64`.
  - Use the added `Float64Option` type to configure instruments from `go.opentelemetry.io/otel/metric/instrument/syncfloat64`.
  - Use the added `Int64ObserverOption` type to configure instruments from `go.opentelemetry.io/otel/metric/instrument/asyncint64`.
  - Use the added `Float64ObserverOption` type to configure instruments from `go.opentelemetry.io/otel/metric/instrument/asyncfloat64`.
- Return a `Registration` from the `RegisterCallback` method of a `Meter` in the `go.opentelemetry.io/otel/metric` package.
  This `Registration` can be used to unregister callbacks. (#3522)
- Global error handler uses an atomic value instead of a mutex. (#3543)
- Add `NewMetricProducer` to `go.opentelemetry.io/otel/bridge/opencensus`, which can be used to pass OpenCensus metrics to an OpenTelemetry Reader. (#3541)
- Global logger uses an atomic value instead of a mutex. (#3545)
- The `Shutdown` method of the `"go.opentelemetry.io/otel/sdk/trace".TracerProvider` releases all computational resources when called the first time. (#3551)
- The `Sampler` returned from `TraceIDRatioBased` `go.opentelemetry.io/otel/sdk/trace` now uses the rightmost bits for sampling decisions.
  This fixes random sampling when using ID generators like `xray.IDGenerator` and increasing parity with other language implementations. (#3557)
- Errors from `go.opentelemetry.io/otel/exporters/otlp/otlptrace` exporters are wrapped in errors identifying their signal name.
  Existing users of the exporters attempting to identify specific errors will need to use `errors.Unwrap()` to get the underlying error. (#3516)
- Exporters from `go.opentelemetry.io/otel/exporters/otlp` will print the final retryable error message when attempts to retry time out. (#3514)
- The instrument kind names in `go.opentelemetry.io/otel/sdk/metric` are updated to match the API. (#3562)
  - `InstrumentKindSyncCounter` is renamed to `InstrumentKindCounter`
  - `InstrumentKindSyncUpDownCounter` is renamed to `InstrumentKindUpDownCounter`
  - `InstrumentKindSyncHistogram` is renamed to `InstrumentKindHistogram`
  - `InstrumentKindAsyncCounter` is renamed to `InstrumentKindObservableCounter`
  - `InstrumentKindAsyncUpDownCounter` is renamed to `InstrumentKindObservableUpDownCounter`
  - `InstrumentKindAsyncGauge` is renamed to `InstrumentKindObservableGauge`
- The `RegisterCallback` method of the `Meter` in `go.opentelemetry.io/otel/metric` changed.
  - The named `Callback` replaces the inline function parameter. (#3564)
  - `Callback` is required to return an error. (#3576)
  - `Callback` accepts the added `Observer` parameter added.
    This new parameter is used by `Callback` implementations to observe values for asynchronous instruments instead of calling the `Observe` method of the instrument directly. (#3584)
  - The slice of `instrument.Asynchronous` is now passed as a variadic argument. (#3587)
- The exporter from `go.opentelemetry.io/otel/exporters/zipkin` is updated to use the `v1.16.0` version of semantic conventions.
  This means it no longer uses the removed `net.peer.ip` or `http.host` attributes to determine the remote endpoint.
  Instead it uses the `net.sock.peer` attributes. (#3581)
- The `Min` and `Max` fields of the `HistogramDataPoint` in `go.opentelemetry.io/otel/sdk/metric/metricdata` are now defined with the added `Extrema` type instead of a `*float64`. (#3487)

### Fixed

- Asynchronous instruments that use sum aggregators and attribute filters correctly add values from equivalent attribute sets that have been filtered. (#3439, #3549)
- The `RegisterCallback` method of the `Meter` from `go.opentelemetry.io/otel/sdk/metric` only registers a callback for instruments created by that meter.
  Trying to register a callback with instruments from a different meter will result in an error being returned. (#3584)

### Deprecated

- The `NewMetricExporter` in `go.opentelemetry.io/otel/bridge/opencensus` is deprecated.
  Use `NewMetricProducer` instead. (#3541)
- The `go.opentelemetry.io/otel/metric/instrument/asyncfloat64` package is deprecated.
  Use the instruments from `go.opentelemetry.io/otel/metric/instrument` instead. (#3575)
- The `go.opentelemetry.io/otel/metric/instrument/asyncint64` package is deprecated.
  Use the instruments from `go.opentelemetry.io/otel/metric/instrument` instead. (#3575)
- The `go.opentelemetry.io/otel/metric/instrument/syncfloat64` package is deprecated.
  Use the instruments from `go.opentelemetry.io/otel/metric/instrument` instead. (#3575)
- The `go.opentelemetry.io/otel/metric/instrument/syncint64` package is deprecated.
  Use the instruments from `go.opentelemetry.io/otel/metric/instrument` instead. (#3575)
- The `NewWrappedTracerProvider` in `go.opentelemetry.io/otel/bridge/opentracing` is now deprecated.
  Use `NewTracerProvider` instead. (#3116)

### Removed

- The deprecated `go.opentelemetry.io/otel/sdk/metric/view` package is removed. (#3520)
- The `InstrumentProvider` from `go.opentelemetry.io/otel/sdk/metric/asyncint64` is removed.
  Use the new creation methods of the `Meter` in `go.opentelemetry.io/otel/sdk/metric` instead. (#3530)
  - The `Counter` method is replaced by `Meter.Int64ObservableCounter`
  - The `UpDownCounter` method is replaced by `Meter.Int64ObservableUpDownCounter`
  - The `Gauge` method is replaced by `Meter.Int64ObservableGauge`
- The `InstrumentProvider` from `go.opentelemetry.io/otel/sdk/metric/asyncfloat64` is removed.
  Use the new creation methods of the `Meter` in `go.opentelemetry.io/otel/sdk/metric` instead. (#3530)
  - The `Counter` method is replaced by `Meter.Float64ObservableCounter`
  - The `UpDownCounter` method is replaced by `Meter.Float64ObservableUpDownCounter`
  - The `Gauge` method is replaced by `Meter.Float64ObservableGauge`
- The `InstrumentProvider` from `go.opentelemetry.io/otel/sdk/metric/syncint64` is removed.
  Use the new creation methods of the `Meter` in `go.opentelemetry.io/otel/sdk/metric` instead. (#3530)
  - The `Counter` method is replaced by `Meter.Int64Counter`
  - The `UpDownCounter` method is replaced by `Meter.Int64UpDownCounter`
  - The `Histogram` method is replaced by `Meter.Int64Histogram`
- The `InstrumentProvider` from `go.opentelemetry.io/otel/sdk/metric/syncfloat64` is removed.
  Use the new creation methods of the `Meter` in `go.opentelemetry.io/otel/sdk/metric` instead. (#3530)
  - The `Counter` method is replaced by `Meter.Float64Counter`
  - The `UpDownCounter` method is replaced by `Meter.Float64UpDownCounter`
  - The `Histogram` method is replaced by `Meter.Float64Histogram`

## [1.11.2/0.34.0] 2022-12-05

### Added

- The `WithView` `Option` is added to the `go.opentelemetry.io/otel/sdk/metric` package.
   This option is used to configure the view(s) a `MeterProvider` will use for all `Reader`s that are registered with it. (#3387)
- Add Instrumentation Scope and Version as info metric and label in Prometheus exporter.
  This can be disabled using the `WithoutScopeInfo()` option added to that package.(#3273, #3357)
- OTLP exporters now recognize: (#3363)
  - `OTEL_EXPORTER_OTLP_INSECURE`
  - `OTEL_EXPORTER_OTLP_TRACES_INSECURE`
  - `OTEL_EXPORTER_OTLP_METRICS_INSECURE`
  - `OTEL_EXPORTER_OTLP_CLIENT_KEY`
  - `OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY`
  - `OTEL_EXPORTER_OTLP_METRICS_CLIENT_KEY`
  - `OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE`
  - `OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE`
  - `OTEL_EXPORTER_OTLP_METRICS_CLIENT_CERTIFICATE`
- The `View` type and related `NewView` function to create a view according to the OpenTelemetry specification are added to `go.opentelemetry.io/otel/sdk/metric`.
  These additions are replacements for the `View` type and `New` function from `go.opentelemetry.io/otel/sdk/metric/view`. (#3459)
- The `Instrument` and `InstrumentKind` type are added to `go.opentelemetry.io/otel/sdk/metric`.
  These additions are replacements for the `Instrument` and `InstrumentKind` types from `go.opentelemetry.io/otel/sdk/metric/view`. (#3459)
- The `Stream` type is added to `go.opentelemetry.io/otel/sdk/metric` to define a metric data stream a view will produce. (#3459)
- The `AssertHasAttributes` allows instrument authors to test that datapoints returned have appropriate attributes. (#3487)

### Changed

- The `"go.opentelemetry.io/otel/sdk/metric".WithReader` option no longer accepts views to associate with the `Reader`.
   Instead, views are now registered directly with the `MeterProvider` via the new `WithView` option.
   The views registered with the `MeterProvider` apply to all `Reader`s. (#3387)
- The `Temporality(view.InstrumentKind) metricdata.Temporality` and `Aggregation(view.InstrumentKind) aggregation.Aggregation` methods are added to the `"go.opentelemetry.io/otel/sdk/metric".Exporter` interface. (#3260)
- The `Temporality(view.InstrumentKind) metricdata.Temporality` and `Aggregation(view.InstrumentKind) aggregation.Aggregation` methods are added to the `"go.opentelemetry.io/otel/exporters/otlp/otlpmetric".Client` interface. (#3260)
- The `WithTemporalitySelector` and `WithAggregationSelector` `ReaderOption`s have been changed to `ManualReaderOption`s in the `go.opentelemetry.io/otel/sdk/metric` package. (#3260)
- The periodic reader in the `go.opentelemetry.io/otel/sdk/metric` package now uses the temporality and aggregation selectors from its configured exporter instead of accepting them as options. (#3260)

### Fixed

- The `go.opentelemetry.io/otel/exporters/prometheus` exporter fixes duplicated `_total` suffixes. (#3369)
- Remove comparable requirement for `Reader`s. (#3387)
- Cumulative metrics from the OpenCensus bridge (`go.opentelemetry.io/otel/bridge/opencensus`) are defined as monotonic sums, instead of non-monotonic. (#3389)
- Asynchronous counters (`Counter` and `UpDownCounter`) from the metric SDK now produce delta sums when configured with delta temporality. (#3398)
- Exported `Status` codes in the `go.opentelemetry.io/otel/exporters/zipkin` exporter are now exported as all upper case values. (#3340)
- `Aggregation`s from `go.opentelemetry.io/otel/sdk/metric` with no data are not exported. (#3394, #3436)
- Re-enabled Attribute Filters in the Metric SDK. (#3396)
- Asynchronous callbacks are only called if they are registered with at least one instrument that does not use drop aggragation. (#3408)
- Do not report empty partial-success responses in the `go.opentelemetry.io/otel/exporters/otlp` exporters. (#3438, #3432)
- Handle partial success responses in `go.opentelemetry.io/otel/exporters/otlp/otlpmetric` exporters. (#3162, #3440)
- Prevent duplicate Prometheus description, unit, and type. (#3469)
- Prevents panic when using incorrect `attribute.Value.As[Type]Slice()`. (#3489)

### Removed

- The `go.opentelemetry.io/otel/exporters/otlp/otlpmetric.Client` interface is removed. (#3486)
- The `go.opentelemetry.io/otel/exporters/otlp/otlpmetric.New` function is removed. Use the `otlpmetric[http|grpc].New` directly. (#3486)

### Deprecated

- The `go.opentelemetry.io/otel/sdk/metric/view` package is deprecated.
  Use `Instrument`, `InstrumentKind`, `View`, and `NewView` in `go.opentelemetry.io/otel/sdk/metric` instead. (#3476)

## [1.11.1/0.33.0] 2022-10-19

### Added

- The Prometheus exporter in `go.opentelemetry.io/otel/exporters/prometheus` registers with a Prometheus registerer on creation.
   By default, it will register with the default Prometheus registerer.
   A non-default registerer can be used by passing the `WithRegisterer` option. (#3239)
- Added the `WithAggregationSelector` option to the `go.opentelemetry.io/otel/exporters/prometheus` package to change the default `AggregationSelector` used. (#3341)
- The Prometheus exporter in `go.opentelemetry.io/otel/exporters/prometheus` converts the `Resource` associated with metric exports into a `target_info` metric. (#3285)

### Changed

- The `"go.opentelemetry.io/otel/exporters/prometheus".New` function is updated to return an error.
   It will return an error if the exporter fails to register with Prometheus. (#3239)

### Fixed

- The URL-encoded values from the `OTEL_RESOURCE_ATTRIBUTES` environment variable are decoded. (#2963)
- The `baggage.NewMember` function decodes the `value` parameter instead of directly using it.
   This fixes the implementation to be compliant with the W3C specification. (#3226)
- Slice attributes of the `attribute` package are now comparable based on their value, not instance. (#3108 #3252)
- The `Shutdown` and `ForceFlush` methods of the `"go.opentelemetry.io/otel/sdk/trace".TraceProvider` no longer return an error when no processor is registered. (#3268)
- The Prometheus exporter in `go.opentelemetry.io/otel/exporters/prometheus` cumulatively sums histogram buckets. (#3281)
- The sum of each histogram data point is now uniquely exported by the `go.opentelemetry.io/otel/exporters/otlpmetric` exporters. (#3284, #3293)
- Recorded values for asynchronous counters (`Counter` and `UpDownCounter`) are interpreted as exact, not incremental, sum values by the metric SDK. (#3350, #3278)
- `UpDownCounters` are now correctly output as Prometheus gauges in the `go.opentelemetry.io/otel/exporters/prometheus` exporter. (#3358)
- The Prometheus exporter in `go.opentelemetry.io/otel/exporters/prometheus` no longer describes the metrics it will send to Prometheus on startup.
   Instead the exporter is defined as an "unchecked" collector for Prometheus.
   This fixes the `reader is not registered` warning currently emitted on startup. (#3291 #3342)
- The `go.opentelemetry.io/otel/exporters/prometheus` exporter now correctly adds `_total` suffixes to counter metrics. (#3360)
- The `go.opentelemetry.io/otel/exporters/prometheus` exporter now adds a unit suffix to metric names.
   This can be disabled using the `WithoutUnits()` option added to that package. (#3352)

## [1.11.0/0.32.3] 2022-10-12

### Added

- Add default User-Agent header to OTLP exporter requests (`go.opentelemetry.io/otel/exporters/otlptrace/otlptracegrpc` and `go.opentelemetry.io/otel/exporters/otlptrace/otlptracehttp`). (#3261)

### Changed

- `span.SetStatus` has been updated such that calls that lower the status are now no-ops. (#3214)
- Upgrade `golang.org/x/sys/unix` from `v0.0.0-20210423185535-09eb48e85fd7` to `v0.0.0-20220919091848-fb04ddd9f9c8`.
  This addresses [GO-2022-0493](https://pkg.go.dev/vuln/GO-2022-0493). (#3235)

## [0.32.2] Metric SDK (Alpha) - 2022-10-11

### Added

- Added an example of using metric views to customize instruments. (#3177)
- Add default User-Agent header to OTLP exporter requests (`go.opentelemetry.io/otel/exporters/otlpmetric/otlpmetricgrpc` and `go.opentelemetry.io/otel/exporters/otlpmetric/otlpmetrichttp`). (#3261)

### Changed

- Flush pending measurements with the `PeriodicReader` in the `go.opentelemetry.io/otel/sdk/metric` when `ForceFlush` or `Shutdown` are called. (#3220)
- Update histogram default bounds to match the requirements of the latest specification. (#3222)
- Encode the HTTP status code in the OpenTracing bridge (`go.opentelemetry.io/otel/bridge/opentracing`) as an integer.  (#3265)

### Fixed

- Use default view if instrument does not match any registered view of a reader. (#3224, #3237)
- Return the same instrument every time a user makes the exact same instrument creation call. (#3229, #3251)
- Return the existing instrument when a view transforms a creation call to match an existing instrument. (#3240, #3251)
- Log a warning when a conflicting instrument (e.g. description, unit, data-type) is created instead of returning an error. (#3251)
- The OpenCensus bridge no longer sends empty batches of metrics. (#3263)

## [0.32.1] Metric SDK (Alpha) - 2022-09-22

### Changed

- The Prometheus exporter sanitizes OpenTelemetry instrument names when exporting.
   Invalid characters are replaced with `_`. (#3212)

### Added

- The metric portion of the OpenCensus bridge (`go.opentelemetry.io/otel/bridge/opencensus`) has been reintroduced. (#3192)
- The OpenCensus bridge example (`go.opentelemetry.io/otel/example/opencensus`) has been reintroduced. (#3206)

### Fixed

- Updated go.mods to point to valid versions of the sdk. (#3216)
- Set the `MeterProvider` resource on all exported metric data. (#3218)

## [0.32.0] Revised Metric SDK (Alpha) - 2022-09-18

### Changed

- The metric SDK in `go.opentelemetry.io/otel/sdk/metric` is completely refactored to comply with the OpenTelemetry specification.
  Please see the package documentation for how the new SDK is initialized and configured. (#3175)
- Update the minimum supported go version to go1.18. Removes support for go1.17 (#3179)

### Removed

- The metric portion of the OpenCensus bridge (`go.opentelemetry.io/otel/bridge/opencensus`) has been removed.
  A new bridge compliant with the revised metric SDK will be added back in a future release. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/aggregator/aggregatortest` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/aggregator/histogram` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/aggregator/lastvalue` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/aggregator/sum` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/aggregator` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/controller/basic` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/controller/controllertest` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/controller/time` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/export/aggregation` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/export` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/metrictest` package is removed.
  A replacement package that supports the new metric SDK will be added back in a future release. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/number` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/processor/basic` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/processor/processortest` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/processor/reducer` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/registry` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/sdkapi` package is removed, see the new metric SDK. (#3175)
- The `go.opentelemetry.io/otel/sdk/metric/selector/simple` package is removed, see the new metric SDK. (#3175)
- The `"go.opentelemetry.io/otel/sdk/metric".ErrUninitializedInstrument` variable was removed. (#3175)
- The `"go.opentelemetry.io/otel/sdk/metric".ErrBadInstrument` variable was removed. (#3175)
- The `"go.opentelemetry.io/otel/sdk/metric".Accumulator` type was removed, see the `MeterProvider`in the new metric SDK. (#3175)
- The `"go.opentelemetry.io/otel/sdk/metric".NewAccumulator` function was removed, see `NewMeterProvider`in the new metric SDK. (#3175)
- The deprecated `"go.opentelemetry.io/otel/sdk/metric".AtomicFieldOffsets` function was removed. (#3175)

## [1.10.0] - 2022-09-09

### Added

- Support Go 1.19. (#3077)
  Include compatibility testing and document support. (#3077)
- Support the OTLP ExportTracePartialSuccess response; these are passed to the registered error handler. (#3106)
- Upgrade go.opentelemetry.io/proto/otlp from v0.18.0 to v0.19.0 (#3107)

### Changed

- Fix misidentification of OpenTelemetry `SpanKind` in OpenTracing bridge (`go.opentelemetry.io/otel/bridge/opentracing`).  (#3096)
- Attempting to start a span with a nil `context` will no longer cause a panic. (#3110)
- All exporters will be shutdown even if one reports an error (#3091)
- Ensure valid UTF-8 when truncating over-length attribute values. (#3156)

## [1.9.0/0.0.3] - 2022-08-01

### Added

- Add support for Schema Files format 1.1.x (metric "split" transform) with the new `go.opentelemetry.io/otel/schema/v1.1` package. (#2999)
- Add the `go.opentelemetry.io/otel/semconv/v1.11.0` package.
  The package contains semantic conventions from the `v1.11.0` version of the OpenTelemetry specification. (#3009)
- Add the `go.opentelemetry.io/otel/semconv/v1.12.0` package.
  The package contains semantic conventions from the `v1.12.0` version of the OpenTelemetry specification. (#3010)
- Add the `http.method` attribute to HTTP server metric from all `go.opentelemetry.io/otel/semconv/*` packages. (#3018)

### Fixed

- Invalid warning for context setup being deferred in `go.opentelemetry.io/otel/bridge/opentracing` package. (#3029)

## [1.8.0/0.31.0] - 2022-07-08

### Added

- Add support for `opentracing.TextMap` format in the `Inject` and `Extract` methods
of the `"go.opentelemetry.io/otel/bridge/opentracing".BridgeTracer` type. (#2911)

### Changed

- The `crosslink` make target has been updated to use the `go.opentelemetry.io/build-tools/crosslink` package. (#2886)
- In the `go.opentelemetry.io/otel/sdk/instrumentation` package rename `Library` to `Scope` and alias `Library` as `Scope` (#2976)
- Move metric no-op implementation form `nonrecording` to `metric` package. (#2866)

### Removed

- Support for go1.16. Support is now only for go1.17 and go1.18 (#2917)

### Deprecated

- The `Library` struct in the `go.opentelemetry.io/otel/sdk/instrumentation` package is deprecated.
  Use the equivalent `Scope` struct instead. (#2977)
- The `ReadOnlySpan.InstrumentationLibrary` method from the `go.opentelemetry.io/otel/sdk/trace` package is deprecated.
  Use the equivalent `ReadOnlySpan.InstrumentationScope` method instead. (#2977)

## [1.7.0/0.30.0] - 2022-04-28

### Added

- Add the `go.opentelemetry.io/otel/semconv/v1.8.0` package.
  The package contains semantic conventions from the `v1.8.0` version of the OpenTelemetry specification. (#2763)
- Add the `go.opentelemetry.io/otel/semconv/v1.9.0` package.
  The package contains semantic conventions from the `v1.9.0` version of the OpenTelemetry specification. (#2792)
- Add the `go.opentelemetry.io/otel/semconv/v1.10.0` package.
  The package contains semantic conventions from the `v1.10.0` version of the OpenTelemetry specification. (#2842)
- Added an in-memory exporter to metrictest to aid testing with a full SDK. (#2776)

### Fixed

- Globally delegated instruments are unwrapped before delegating asynchronous callbacks. (#2784)
- Remove import of `testing` package in non-tests builds of the `go.opentelemetry.io/otel` package. (#2786)

### Changed

- The `WithLabelEncoder` option from the `go.opentelemetry.io/otel/exporters/stdout/stdoutmetric` package is renamed to `WithAttributeEncoder`. (#2790)
- The `LabelFilterSelector` interface from `go.opentelemetry.io/otel/sdk/metric/processor/reducer` is renamed to `AttributeFilterSelector`.
  The method included in the renamed interface also changed from `LabelFilterFor` to `AttributeFilterFor`. (#2790)
- The `Metadata.Labels` method from the `go.opentelemetry.io/otel/sdk/metric/export` package is renamed to `Metadata.Attributes`.
  Consequentially, the `Record` type from the same package also has had the embedded method renamed. (#2790)

### Deprecated

- The `Iterator.Label` method in the `go.opentelemetry.io/otel/attribute` package is deprecated.
  Use the equivalent `Iterator.Attribute` method instead. (#2790)
- The `Iterator.IndexedLabel` method in the `go.opentelemetry.io/otel/attribute` package is deprecated.
  Use the equivalent `Iterator.IndexedAttribute` method instead. (#2790)
- The `MergeIterator.Label` method in the `go.opentelemetry.io/otel/attribute` package is deprecated.
  Use the equivalent `MergeIterator.Attribute` method instead. (#2790)

### Removed

- Removed the `Batch` type from the `go.opentelemetry.io/otel/sdk/metric/metrictest` package. (#2864)
- Removed the `Measurement` type from the `go.opentelemetry.io/otel/sdk/metric/metrictest` package. (#2864)

## [0.29.0] - 2022-04-11

### Added

- The metrics global package was added back into several test files. (#2764)
- The `Meter` function is added back to the `go.opentelemetry.io/otel/metric/global` package.
  This function is a convenience function equivalent to calling `global.MeterProvider().Meter(...)`. (#2750)

### Removed

- Removed module the `go.opentelemetry.io/otel/sdk/export/metric`.
  Use the `go.opentelemetry.io/otel/sdk/metric` module instead. (#2720)

### Changed

- Don't panic anymore when setting a global MeterProvider to itself. (#2749)
- Upgrade `go.opentelemetry.io/proto/otlp` in `go.opentelemetry.io/otel/exporters/otlp/otlpmetric` from `v0.12.1` to `v0.15.0`.
  This replaces the use of the now deprecated `InstrumentationLibrary` and `InstrumentationLibraryMetrics` types and fields in the proto library with the equivalent `InstrumentationScope` and `ScopeMetrics`. (#2748)

## [1.6.3] - 2022-04-07

### Fixed

- Allow non-comparable global `MeterProvider`, `TracerProvider`, and `TextMapPropagator` types to be set. (#2772, #2773)

## [1.6.2] - 2022-04-06

### Changed

- Don't panic anymore when setting a global TracerProvider or TextMapPropagator to itself. (#2749)
- Upgrade `go.opentelemetry.io/proto/otlp` in `go.opentelemetry.io/otel/exporters/otlp/otlptrace` from `v0.12.1` to `v0.15.0`.
  This replaces the use of the now deprecated `InstrumentationLibrary` and `InstrumentationLibrarySpans` types and fields in the proto library with the equivalent `InstrumentationScope` and `ScopeSpans`. (#2748)

## [1.6.1] - 2022-03-28

### Fixed

- The `go.opentelemetry.io/otel/schema/*` packages now use the correct schema URL for their `SchemaURL` constant.
  Instead of using `"https://opentelemetry.io/schemas/v<version>"` they now use the correct URL without a `v` prefix, `"https://opentelemetry.io/schemas/<version>"`. (#2743, #2744)

### Security

- Upgrade `go.opentelemetry.io/proto/otlp` from `v0.12.0` to `v0.12.1`.
  This includes an indirect upgrade of `github.com/grpc-ecosystem/grpc-gateway` which resolves [a vulnerability](https://nvd.nist.gov/vuln/detail/CVE-2019-11254) from `gopkg.in/yaml.v2` in version `v2.2.3`. (#2724, #2728)

## [1.6.0/0.28.0] - 2022-03-23

### ⚠️ Notice ⚠️

This update is a breaking change of the unstable Metrics API.
Code instrumented with the `go.opentelemetry.io/otel/metric` will need to be modified.

### Added

- Add metrics exponential histogram support.
  New mapping functions have been made available in `sdk/metric/aggregator/exponential/mapping` for other OpenTelemetry projects to take dependencies on. (#2502)
- Add Go 1.18 to our compatibility tests. (#2679)
- Allow configuring the Sampler with the `OTEL_TRACES_SAMPLER` and `OTEL_TRACES_SAMPLER_ARG` environment variables. (#2305, #2517)
- Add the `metric/global` for obtaining and setting the global `MeterProvider`. (#2660)

### Changed

- The metrics API has been significantly changed to match the revised OpenTelemetry specification.
  High-level changes include:

  - Synchronous and asynchronous instruments are now handled by independent `InstrumentProvider`s.
    These `InstrumentProvider`s are managed with a `Meter`.
  - Synchronous and asynchronous instruments are grouped into their own packages based on value types.
  - Asynchronous callbacks can now be registered with a `Meter`.

  Be sure to check out the metric module documentation for more information on how to use the revised API. (#2587, #2660)

### Fixed

- Fallback to general attribute limits when span specific ones are not set in the environment. (#2675, #2677)

## [1.5.0] - 2022-03-16

### Added

- Log the Exporters configuration in the TracerProviders message. (#2578)
- Added support to configure the span limits with environment variables.
  The following environment variables are supported. (#2606, #2637)
  - `OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT`
  - `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`
  - `OTEL_SPAN_EVENT_COUNT_LIMIT`
  - `OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT`
  - `OTEL_SPAN_LINK_COUNT_LIMIT`
  - `OTEL_LINK_ATTRIBUTE_COUNT_LIMIT`

  If the provided environment variables are invalid (negative), the default values would be used.
- Rename the `gc` runtime name to `go` (#2560)
- Add resource container ID detection. (#2418)
- Add span attribute value length limit.
  The new `AttributeValueLengthLimit` field is added to the `"go.opentelemetry.io/otel/sdk/trace".SpanLimits` type to configure this limit for a `TracerProvider`.
  The default limit for this resource is "unlimited". (#2637)
- Add the `WithRawSpanLimits` option to `go.opentelemetry.io/otel/sdk/trace`.
  This option replaces the `WithSpanLimits` option.
  Zero or negative values will not be changed to the default value like `WithSpanLimits` does.
  Setting a limit to zero will effectively disable the related resource it limits and setting to a negative value will mean that resource is unlimited.
  Consequentially, limits should be constructed using `NewSpanLimits` and updated accordingly. (#2637)

### Changed

- Drop oldest tracestate `Member` when capacity is reached. (#2592)
- Add event and link drop counts to the exported data from the `oltptrace` exporter. (#2601)
- Unify path cleaning functionally in the `otlpmetric` and `otlptrace` configuration. (#2639)
- Change the debug message from the `sdk/trace.BatchSpanProcessor` to reflect the count is cumulative. (#2640)
- Introduce new internal `envconfig` package for OTLP exporters. (#2608)
- If `http.Request.Host` is empty, fall back to use `URL.Host` when populating `http.host` in the `semconv` packages. (#2661)

### Fixed

- Remove the OTLP trace exporter limit of SpanEvents when exporting. (#2616)
- Default to port `4318` instead of `4317` for the `otlpmetrichttp` and `otlptracehttp` client. (#2614, #2625)
- Unlimited span limits are now supported (negative values). (#2636, #2637)

### Deprecated

- Deprecated `"go.opentelemetry.io/otel/sdk/trace".WithSpanLimits`.
  Use `WithRawSpanLimits` instead.
  That option allows setting unlimited and zero limits, this option does not.
  This option will be kept until the next major version incremented release. (#2637)

## [1.4.1] - 2022-02-16

### Fixed

- Fix race condition in reading the dropped spans number for the `BatchSpanProcessor`. (#2615)

## [1.4.0] - 2022-02-11

### Added

- Use `OTEL_EXPORTER_ZIPKIN_ENDPOINT` environment variable to specify zipkin collector endpoint. (#2490)
- Log the configuration of `TracerProvider`s, and `Tracer`s for debugging.
  To enable use a logger with Verbosity (V level) `>=1`. (#2500)
- Added support to configure the batch span-processor with environment variables.
  The following environment variables are used. (#2515)
  - `OTEL_BSP_SCHEDULE_DELAY`
  - `OTEL_BSP_EXPORT_TIMEOUT`
  - `OTEL_BSP_MAX_QUEUE_SIZE`.
  - `OTEL_BSP_MAX_EXPORT_BATCH_SIZE`

### Changed

- Zipkin exporter exports `Resource` attributes in the `Tags` field. (#2589)

### Deprecated

- Deprecate module the `go.opentelemetry.io/otel/sdk/export/metric`.
  Use the `go.opentelemetry.io/otel/sdk/metric` module instead. (#2382)
- Deprecate `"go.opentelemetry.io/otel/sdk/metric".AtomicFieldOffsets`. (#2445)

### Fixed

- Fixed the instrument kind for noop async instruments to correctly report an implementation. (#2461)
- Fix UDP packets overflowing with Jaeger payloads. (#2489, #2512)
- Change the `otlpmetric.Client` interface's `UploadMetrics` method to accept a single `ResourceMetrics` instead of a slice of them. (#2491)
- Specify explicit buckets in Prometheus example, fixing issue where example only has `+inf` bucket. (#2419, #2493)
- W3C baggage will now decode urlescaped values. (#2529)
- Baggage members are now only validated once, when calling `NewMember` and not also when adding it to the baggage itself. (#2522)
- The order attributes are dropped from spans in the `go.opentelemetry.io/otel/sdk/trace` package when capacity is reached is fixed to be in compliance with the OpenTelemetry specification.
  Instead of dropping the least-recently-used attribute, the last added attribute is dropped.
  This drop order still only applies to attributes with unique keys not already contained in the span.
  If an attribute is added with a key already contained in the span, that attribute is updated to the new value being added. (#2576)

### Removed

- Updated `go.opentelemetry.io/proto/otlp` from `v0.11.0` to `v0.12.0`. This version removes a number of deprecated methods. (#2546)
  - [`Metric.GetIntGauge()`](https://pkg.go.dev/go.opentelemetry.io/proto/otlp@v0.11.0/metrics/v1#Metric.GetIntGauge)
  - [`Metric.GetIntHistogram()`](https://pkg.go.dev/go.opentelemetry.io/proto/otlp@v0.11.0/metrics/v1#Metric.GetIntHistogram)
  - [`Metric.GetIntSum()`](https://pkg.go.dev/go.opentelemetry.io/proto/otlp@v0.11.0/metrics/v1#Metric.GetIntSum)

## [1.3.0] - 2021-12-10

### ⚠️ Notice ⚠️

We have updated the project minimum supported Go version to 1.16

### Added

- Added an internal Logger.
  This can be used by the SDK and API to provide users with feedback of the internal state.
  To enable verbose logs configure the logger which will print V(1) logs. For debugging information configure to print V(5) logs. (#2343)
- Add the `WithRetry` `Option` and the `RetryConfig` type to the `go.opentelemetry.io/otel/exporter/otel/otlpmetric/otlpmetrichttp` package to specify retry behavior consistently. (#2425)
- Add `SpanStatusFromHTTPStatusCodeAndSpanKind` to all `semconv` packages to return a span status code similar to `SpanStatusFromHTTPStatusCode`, but exclude `4XX` HTTP errors as span errors if the span is of server kind. (#2296)

### Changed

- The `"go.opentelemetry.io/otel/exporter/otel/otlptrace/otlptracegrpc".Client` now uses the underlying gRPC `ClientConn` to handle name resolution, TCP connection establishment (with retries and backoff) and TLS handshakes, and handling errors on established connections by re-resolving the name and reconnecting. (#2329)
- The `"go.opentelemetry.io/otel/exporter/otel/otlpmetric/otlpmetricgrpc".Client` now uses the underlying gRPC `ClientConn` to handle name resolution, TCP connection establishment (with retries and backoff) and TLS handshakes, and handling errors on established connections by re-resolving the name and reconnecting. (#2425)
- The `"go.opentelemetry.io/otel/exporter/otel/otlpmetric/otlpmetricgrpc".RetrySettings` type is renamed to `RetryConfig`. (#2425)
- The `go.opentelemetry.io/otel/exporter/otel/*` gRPC exporters now default to using the host's root CA set if none are provided by the user and `WithInsecure` is not specified. (#2432)
- Change `resource.Default` to be evaluated the first time it is called, rather than on import. This allows the caller the option to update `OTEL_RESOURCE_ATTRIBUTES` first, such as with `os.Setenv`. (#2371)

### Fixed

- The `go.opentelemetry.io/otel/exporter/otel/*` exporters are updated to handle per-signal and universal endpoints according to the OpenTelemetry specification.
  Any per-signal endpoint set via an `OTEL_EXPORTER_OTLP_<signal>_ENDPOINT` environment variable is now used without modification of the path.
  When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, if it contains a path, that path is used as a base path which per-signal paths are appended to. (#2433)
- Basic metric controller updated to use sync.Map to avoid blocking calls (#2381)
- The `go.opentelemetry.io/otel/exporter/jaeger` correctly sets the `otel.status_code` value to be a string of `ERROR` or `OK` instead of an integer code. (#2439, #2440)

### Deprecated

- Deprecated the `"go.opentelemetry.io/otel/exporter/otel/otlpmetric/otlpmetrichttp".WithMaxAttempts` `Option`, use the new `WithRetry` `Option` instead. (#2425)
- Deprecated the `"go.opentelemetry.io/otel/exporter/otel/otlpmetric/otlpmetrichttp".WithBackoff` `Option`, use the new `WithRetry` `Option` instead. (#2425)

### Removed

- Remove the metric Processor's ability to convert cumulative to delta aggregation temporality. (#2350)
- Remove the metric Bound Instruments interface and implementations. (#2399)
- Remove the metric MinMaxSumCount kind aggregation and the corresponding OTLP export path. (#2423)
- Metric SDK removes the "exact" aggregator for histogram instruments, as it performed a non-standard aggregation for OTLP export (creating repeated Gauge points) and worked its way into a number of confusing examples. (#2348)

## [1.2.0] - 2021-11-12

### Changed

- Metric SDK `export.ExportKind`, `export.ExportKindSelector` types have been renamed to `aggregation.Temporality` and `aggregation.TemporalitySelector` respectively to keep in line with current specification and protocol along with built-in selectors (e.g., `aggregation.CumulativeTemporalitySelector`, ...). (#2274)
- The Metric `Exporter` interface now requires a `TemporalitySelector` method instead of an `ExportKindSelector`. (#2274)
- Metrics API cleanup. The `metric/sdkapi` package has been created to relocate the API-to-SDK interface:
  - The following interface types simply moved from `metric` to `metric/sdkapi`: `Descriptor`, `MeterImpl`, `InstrumentImpl`, `SyncImpl`, `BoundSyncImpl`, `AsyncImpl`, `AsyncRunner`, `AsyncSingleRunner`, and `AsyncBatchRunner`
  - The following struct types moved and are replaced with type aliases, since they are exposed to the user: `Observation`, `Measurement`.
  - The No-op implementations of sync and async instruments are no longer exported, new functions `sdkapi.NewNoopAsyncInstrument()` and `sdkapi.NewNoopSyncInstrument()` are provided instead. (#2271)
- Update the SDK `BatchSpanProcessor` to export all queued spans when `ForceFlush` is called. (#2080, #2335)

### Added

- Add the `"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc".WithGRPCConn` option so the exporter can reuse an existing gRPC connection. (#2002)
- Added a new `schema` module to help parse Schema Files in OTEP 0152 format. (#2267)
- Added a new `MapCarrier` to the `go.opentelemetry.io/otel/propagation` package to hold propagated cross-cutting concerns as a `map[string]string` held in memory. (#2334)

## [1.1.0] - 2021-10-27

### Added

- Add the `"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc".WithGRPCConn` option so the exporter can reuse an existing gRPC connection. (#2002)
- Add the `go.opentelemetry.io/otel/semconv/v1.7.0` package.
  The package contains semantic conventions from the `v1.7.0` version of the OpenTelemetry specification. (#2320)
- Add the `go.opentelemetry.io/otel/semconv/v1.6.1` package.
  The package contains semantic conventions from the `v1.6.1` version of the OpenTelemetry specification. (#2321)
- Add the `go.opentelemetry.io/otel/semconv/v1.5.0` package.
  The package contains semantic conventions from the `v1.5.0` version of the OpenTelemetry specification. (#2322)
  - When upgrading from the `semconv/v1.4.0` package note the following name changes:
    - `K8SReplicasetUIDKey` -> `K8SReplicaSetUIDKey`
    - `K8SReplicasetNameKey` -> `K8SReplicaSetNameKey`
    - `K8SStatefulsetUIDKey` -> `K8SStatefulSetUIDKey`
    - `k8SStatefulsetNameKey` -> `K8SStatefulSetNameKey`
    - `K8SDaemonsetUIDKey` -> `K8SDaemonSetUIDKey`
    - `K8SDaemonsetNameKey` -> `K8SDaemonSetNameKey`

### Changed

- Links added to a span will be dropped by the SDK if they contain an invalid span context (#2275).

### Fixed

- The `"go.opentelemetry.io/otel/semconv/v1.4.0".HTTPServerAttributesFromHTTPRequest` now correctly only sets the HTTP client IP attribute even if the connection was routed with proxies and there are multiple addresses in the `X-Forwarded-For` header. (#2282, #2284)
- The `"go.opentelemetry.io/otel/semconv/v1.4.0".NetAttributesFromHTTPRequest` function correctly handles IPv6 addresses as IP addresses and sets the correct net peer IP instead of the net peer hostname attribute. (#2283, #2285)
- The simple span processor shutdown method deterministically returns the exporter error status if it simultaneously finishes when the deadline is reached. (#2290, #2289)

## [1.0.1] - 2021-10-01

### Fixed

- json stdout exporter no longer crashes due to concurrency bug. (#2265)

## [Metrics 0.24.0] - 2021-10-01

### Changed

- NoopMeterProvider is now private and NewNoopMeterProvider must be used to obtain a noopMeterProvider. (#2237)
- The Metric SDK `Export()` function takes a new two-level reader interface for iterating over results one instrumentation library at a time. (#2197)
  - The former `"go.opentelemetry.io/otel/sdk/export/metric".CheckpointSet` is renamed `Reader`.
  - The new interface is named `"go.opentelemetry.io/otel/sdk/export/metric".InstrumentationLibraryReader`.

## [1.0.0] - 2021-09-20

This is the first stable release for the project.
This release includes an API and SDK for the tracing signal that will comply with the stability guarantees defined by the projects [versioning policy](./VERSIONING.md).

### Added

- OTLP trace exporter now sets the `SchemaURL` field in the exported telemetry if the Tracer has `WithSchemaURL` option. (#2242)

### Fixed

- Slice-valued attributes can correctly be used as map keys. (#2223)

### Removed

- Removed the `"go.opentelemetry.io/otel/exporters/zipkin".WithSDKOptions` function. (#2248)
- Removed the deprecated package `go.opentelemetry.io/otel/oteltest`. (#2234)
- Removed the deprecated package `go.opentelemetry.io/otel/bridge/opencensus/utils`. (#2233)
- Removed deprecated functions, types, and methods from `go.opentelemetry.io/otel/attribute` package.
  Use the typed functions and methods added to the package instead. (#2235)
  - The `Key.Array` method is removed.
  - The `Array` function is removed.
  - The `Any` function is removed.
  - The `ArrayValue` function is removed.
  - The `AsArray` function is removed.

## [1.0.0-RC3] - 2021-09-02

### Added

- Added `ErrorHandlerFunc` to use a function as an `"go.opentelemetry.io/otel".ErrorHandler`. (#2149)
- Added `"go.opentelemetry.io/otel/trace".WithStackTrace` option to add a stack trace when using `span.RecordError` or when panic is handled in `span.End`. (#2163)
- Added typed slice attribute types and functionality to the `go.opentelemetry.io/otel/attribute` package to replace the existing array type and functions. (#2162)
  - `BoolSlice`, `IntSlice`, `Int64Slice`, `Float64Slice`, and `StringSlice` replace the use of the `Array` function in the package.
- Added the `go.opentelemetry.io/otel/example/fib` example package.
  Included is an example application that computes Fibonacci numbers. (#2203)

### Changed

- Metric instruments have been renamed to match the (feature-frozen) metric API specification:
  - ValueRecorder becomes Histogram
  - ValueObserver becomes Gauge
  - SumObserver becomes CounterObserver
  - UpDownSumObserver becomes UpDownCounterObserver
  The API exported from this project is still considered experimental. (#2202)
- Metric SDK/API implementation type `InstrumentKind` moves into `sdkapi` sub-package. (#2091)
- The Metrics SDK export record no longer contains a Resource pointer, the SDK `"go.opentelemetry.io/otel/sdk/trace/export/metric".Exporter.Export()` function for push-based exporters now takes a single Resource argument, pull-based exporters use `"go.opentelemetry.io/otel/sdk/metric/controller/basic".Controller.Resource()`. (#2120)
- The JSON output of the `go.opentelemetry.io/otel/exporters/stdout/stdouttrace` is harmonized now such that the output is "plain" JSON objects after each other of the form `{ ... } { ... } { ... }`. Earlier the JSON objects describing a span were wrapped in a slice for each `Exporter.ExportSpans` call, like `[ { ... } ][ { ... } { ... } ]`. Outputting JSON object directly after each other is consistent with JSON loggers, and a bit easier to parse and read. (#2196)
- Update the `NewTracerConfig`, `NewSpanStartConfig`, `NewSpanEndConfig`, and `NewEventConfig` function in the `go.opentelemetry.io/otel/trace` package to return their respective configurations as structs instead of pointers to the struct. (#2212)

### Deprecated

- The `go.opentelemetry.io/otel/bridge/opencensus/utils` package is deprecated.
  All functionality from this package now exists in the `go.opentelemetry.io/otel/bridge/opencensus` package.
  The functions from that package should be used instead. (#2166)
- The `"go.opentelemetry.io/otel/attribute".Array` function and the related `ARRAY` value type is deprecated.
  Use the typed `*Slice` functions and types added to the package instead. (#2162)
- The `"go.opentelemetry.io/otel/attribute".Any` function is deprecated.
  Use the typed functions instead. (#2181)
- The `go.opentelemetry.io/otel/oteltest` package is deprecated.
  The `"go.opentelemetry.io/otel/sdk/trace/tracetest".SpanRecorder` can be registered with the default SDK (`go.opentelemetry.io/otel/sdk/trace`) as a `SpanProcessor` and used as a replacement for this deprecated package. (#2188)

### Removed

- Removed metrics test package `go.opentelemetry.io/otel/sdk/export/metric/metrictest`. (#2105)

### Fixed

- The `fromEnv` detector no longer throws an error when `OTEL_RESOURCE_ATTRIBUTES` environment variable is not set or empty. (#2138)
- Setting the global `ErrorHandler` with `"go.opentelemetry.io/otel".SetErrorHandler` multiple times is now supported. (#2160, #2140)
- The `"go.opentelemetry.io/otel/attribute".Any` function now supports `int32` values. (#2169)
- Multiple calls to `"go.opentelemetry.io/otel/sdk/metric/controller/basic".WithResource()` are handled correctly, and when no resources are provided `"go.opentelemetry.io/otel/sdk/resource".Default()` is used. (#2120)
- The `WithoutTimestamps` option for the `go.opentelemetry.io/otel/exporters/stdout/stdouttrace` exporter causes the exporter to correctly omit timestamps. (#2195)
- Fixed typos in resources.go. (#2201)

## [1.0.0-RC2] - 2021-07-26

### Added

- Added `WithOSDescription` resource configuration option to set OS (Operating System) description resource attribute (`os.description`). (#1840)
- Added `WithOS` resource configuration option to set all OS (Operating System) resource attributes at once. (#1840)
- Added the `WithRetry` option to the `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp` package.
  This option is a replacement for the removed `WithMaxAttempts` and `WithBackoff` options. (#2095)
- Added API `LinkFromContext` to return Link which encapsulates SpanContext from provided context and also encapsulates attributes. (#2115)
- Added a new `Link` type under the SDK `otel/sdk/trace` package that counts the number of attributes that were dropped for surpassing the `AttributePerLinkCountLimit` configured in the Span's `SpanLimits`.
  This new type replaces the equal-named API `Link` type found in the `otel/trace` package for most usages within the SDK.
  For example, instances of this type are now returned by the `Links()` function of `ReadOnlySpan`s provided in places like the `OnEnd` function of `SpanProcessor` implementations. (#2118)
- Added the `SpanRecorder` type to the `go.opentelemetry.io/otel/skd/trace/tracetest` package.
  This type can be used with the default SDK as a `SpanProcessor` during testing. (#2132)

### Changed

- The `SpanModels` function is now exported from the `go.opentelemetry.io/otel/exporters/zipkin` package to convert OpenTelemetry spans into Zipkin model spans. (#2027)
- Rename the `"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc".RetrySettings` to `RetryConfig`. (#2095)

### Deprecated

- The `TextMapCarrier` and `TextMapPropagator` from the `go.opentelemetry.io/otel/oteltest` package and their associated creation functions (`TextMapCarrier`, `NewTextMapPropagator`) are deprecated. (#2114)
- The `Harness` type from the `go.opentelemetry.io/otel/oteltest` package and its associated creation function, `NewHarness` are deprecated and will be removed in the next release. (#2123)
- The `TraceStateFromKeyValues` function from the `go.opentelemetry.io/otel/oteltest` package is deprecated.
  Use the `trace.ParseTraceState` function instead. (#2122)

### Removed

- Removed the deprecated package `go.opentelemetry.io/otel/exporters/trace/jaeger`. (#2020)
- Removed the deprecated package `go.opentelemetry.io/otel/exporters/trace/zipkin`. (#2020)
- Removed the `"go.opentelemetry.io/otel/sdk/resource".WithBuiltinDetectors` function.
  The explicit `With*` options for every built-in detector should be used instead. (#2026 #2097)
- Removed the `WithMaxAttempts` and `WithBackoff` options from the `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp` package.
  The retry logic of the package has been updated to match the `otlptracegrpc` package and accordingly a `WithRetry` option is added that should be used instead. (#2095)
- Removed `DroppedAttributeCount` field from `otel/trace.Link` struct. (#2118)

### Fixed

- When using WithNewRoot, don't use the parent context for making sampling decisions. (#2032)
- `oteltest.Tracer` now creates a valid `SpanContext` when using `WithNewRoot`. (#2073)
- OS type detector now sets the correct `dragonflybsd` value for DragonFly BSD. (#2092)
- The OTel span status is correctly transformed into the OTLP status in the `go.opentelemetry.io/otel/exporters/otlp/otlptrace` package.
  This fix will by default set the status to `Unset` if it is not explicitly set to `Ok` or `Error`. (#2099 #2102)
- The `Inject` method for the `"go.opentelemetry.io/otel/propagation".TraceContext` type no longer injects empty `tracestate` values. (#2108)
- Use `6831` as default Jaeger agent port instead of `6832`. (#2131)

## [Experimental Metrics v0.22.0] - 2021-07-19

### Added

- Adds HTTP support for OTLP metrics exporter. (#2022)

### Removed

- Removed the deprecated package `go.opentelemetry.io/otel/exporters/metric/prometheus`. (#2020)

## [1.0.0-RC1] / 0.21.0 - 2021-06-18

With this release we are introducing a split in module versions.  The tracing API and SDK are entering the `v1.0.0` Release Candidate phase with `v1.0.0-RC1`
while the experimental metrics API and SDK continue with `v0.x` releases at `v0.21.0`.  Modules at major version 1 or greater will not depend on modules
with major version 0.

### Added

- Adds `otlpgrpc.WithRetry`option for configuring the retry policy for transient errors on the otlp/gRPC exporter. (#1832)
  - The following status codes are defined as transient errors:
      | gRPC Status Code | Description |
      | ---------------- | ----------- |
      | 1  | Cancelled |
      | 4  | Deadline Exceeded |
      | 8  | Resource Exhausted |
      | 10 | Aborted |
      | 10 | Out of Range |
      | 14 | Unavailable |
      | 15 | Data Loss |
- Added `Status` type to the `go.opentelemetry.io/otel/sdk/trace` package to represent the status of a span. (#1874)
- Added `SpanStub` type and its associated functions to the `go.opentelemetry.io/otel/sdk/trace/tracetest` package.
  This type can be used as a testing replacement for the `SpanSnapshot` that was removed from the `go.opentelemetry.io/otel/sdk/trace` package. (#1873)
- Adds support for scheme in `OTEL_EXPORTER_OTLP_ENDPOINT` according to the spec. (#1886)
- Adds `trace.WithSchemaURL` option for configuring the tracer with a Schema URL. (#1889)
- Added an example of using OpenTelemetry Go as a trace context forwarder. (#1912)
- `ParseTraceState` is added to the `go.opentelemetry.io/otel/trace` package.
  It can be used to decode a `TraceState` from a `tracestate` header string value. (#1937)
- Added `Len` method to the `TraceState` type in the `go.opentelemetry.io/otel/trace` package.
  This method returns the number of list-members the `TraceState` holds. (#1937)
- Creates package `go.opentelemetry.io/otel/exporters/otlp/otlptrace` that defines a trace exporter that uses a `otlptrace.Client` to send data.
  Creates package `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc` implementing a gRPC `otlptrace.Client` and offers convenience functions, `NewExportPipeline` and `InstallNewPipeline`, to setup and install a `otlptrace.Exporter` in tracing .(#1922)
- Added `Baggage`, `Member`, and `Property` types to the `go.opentelemetry.io/otel/baggage` package along with their related functions. (#1967)
- Added `ContextWithBaggage`, `ContextWithoutBaggage`, and `FromContext` functions to the `go.opentelemetry.io/otel/baggage` package.
  These functions replace the `Set`, `Value`, `ContextWithValue`, `ContextWithoutValue`, and `ContextWithEmpty` functions from that package and directly work with the new `Baggage` type. (#1967)
- The `OTEL_SERVICE_NAME` environment variable is the preferred source for `service.name`, used by the environment resource detector if a service name is present both there and in `OTEL_RESOURCE_ATTRIBUTES`. (#1969)
- Creates package `go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp` implementing an HTTP `otlptrace.Client` and offers convenience functions, `NewExportPipeline` and `InstallNewPipeline`, to setup and install a `otlptrace.Exporter` in tracing. (#1963)
- Changes `go.opentelemetry.io/otel/sdk/resource.NewWithAttributes` to require a schema URL. The old function is still available as `resource.NewSchemaless`. This is a breaking change. (#1938)
- Several builtin resource detectors now correctly populate the schema URL. (#1938)
- Creates package `go.opentelemetry.io/otel/exporters/otlp/otlpmetric` that defines a metrics exporter that uses a `otlpmetric.Client` to send data.
- Creates package `go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc` implementing a gRPC `otlpmetric.Client` and offers convenience functions, `New` and `NewUnstarted`, to create an `otlpmetric.Exporter`.(#1991)
- Added `go.opentelemetry.io/otel/exporters/stdout/stdouttrace` exporter. (#2005)
- Added `go.opentelemetry.io/otel/exporters/stdout/stdoutmetric` exporter. (#2005)
- Added a `TracerProvider()` method to the `"go.opentelemetry.io/otel/trace".Span` interface. This can be used to obtain a `TracerProvider` from a given span that utilizes the same trace processing pipeline.  (#2009)

### Changed

- Make `NewSplitDriver` from `go.opentelemetry.io/otel/exporters/otlp` take variadic arguments instead of a `SplitConfig` item.
  `NewSplitDriver` now automatically implements an internal `noopDriver` for `SplitConfig` fields that are not initialized. (#1798)
- `resource.New()` now creates a Resource without builtin detectors. Previous behavior is now achieved by using `WithBuiltinDetectors` Option. (#1810)
- Move the `Event` type from the `go.opentelemetry.io/otel` package to the `go.opentelemetry.io/otel/sdk/trace` package. (#1846)
- CI builds validate against last two versions of Go, dropping 1.14 and adding 1.16. (#1865)
- BatchSpanProcessor now report export failures when calling `ForceFlush()` method. (#1860)
- `Set.Encoded(Encoder)` no longer caches the result of an encoding. (#1855)
- Renamed `CloudZoneKey` to `CloudAvailabilityZoneKey` in Resource semantic conventions according to spec. (#1871)
- The `StatusCode` and `StatusMessage` methods of the `ReadOnlySpan` interface and the `Span` produced by the `go.opentelemetry.io/otel/sdk/trace` package have been replaced with a single `Status` method.
  This method returns the status of a span using the new `Status` type. (#1874)
- Updated `ExportSpans` method of the`SpanExporter` interface type to accept `ReadOnlySpan`s instead of the removed `SpanSnapshot`.
  This brings the export interface into compliance with the specification in that it now accepts an explicitly immutable type instead of just an implied one. (#1873)
- Unembed `SpanContext` in `Link`. (#1877)
- Generate Semantic conventions from the specification YAML. (#1891)
- Spans created by the global `Tracer` obtained from `go.opentelemetry.io/otel`, prior to a functioning `TracerProvider` being set, now propagate the span context from their parent if one exists. (#1901)
- The `"go.opentelemetry.io/otel".Tracer` function now accepts tracer options. (#1902)
- Move the `go.opentelemetry.io/otel/unit` package to `go.opentelemetry.io/otel/metric/unit`. (#1903)
- Changed `go.opentelemetry.io/otel/trace.TracerConfig` to conform to the [Contributing guidelines](CONTRIBUTING.md#config.) (#1921)
- Changed `go.opentelemetry.io/otel/trace.SpanConfig` to conform to the [Contributing guidelines](CONTRIBUTING.md#config). (#1921)
- Changed `span.End()` now only accepts Options that are allowed at `End()`. (#1921)
- Changed `go.opentelemetry.io/otel/metric.InstrumentConfig` to conform to the [Contributing guidelines](CONTRIBUTING.md#config). (#1921)
- Changed `go.opentelemetry.io/otel/metric.MeterConfig` to conform to the [Contributing guidelines](CONTRIBUTING.md#config). (#1921)
- Refactored option types according to the contribution style guide. (#1882)
- Move the `go.opentelemetry.io/otel/trace.TraceStateFromKeyValues` function to the `go.opentelemetry.io/otel/oteltest` package.
  This function is preserved for testing purposes where it may be useful to create a `TraceState` from `attribute.KeyValue`s, but it is not intended for production use.
  The new `ParseTraceState` function should be used to create a `TraceState`. (#1931)
- Updated `MarshalJSON` method of the `go.opentelemetry.io/otel/trace.TraceState` type to marshal the type into the string representation of the `TraceState`. (#1931)
- The `TraceState.Delete` method from the `go.opentelemetry.io/otel/trace` package no longer returns an error in addition to a `TraceState`. (#1931)
- Updated `Get` method of the `TraceState` type from the `go.opentelemetry.io/otel/trace` package to accept a `string` instead of an `attribute.Key` type. (#1931)
- Updated `Insert` method of the `TraceState` type from the `go.opentelemetry.io/otel/trace` package to accept a pair of `string`s instead of an `attribute.KeyValue` type. (#1931)
- Updated `Delete` method of the `TraceState` type from the `go.opentelemetry.io/otel/trace` package to accept a `string` instead of an `attribute.Key` type. (#1931)
- Renamed `NewExporter` to `New` in the `go.opentelemetry.io/otel/exporters/stdout` package. (#1985)
- Renamed `NewExporter` to `New` in the `go.opentelemetry.io/otel/exporters/metric/prometheus` package. (#1985)
- Renamed `NewExporter` to `New` in the `go.opentelemetry.io/otel/exporters/trace/jaeger` package. (#1985)
- Renamed `NewExporter` to `New` in the `go.opentelemetry.io/otel/exporters/trace/zipkin` package. (#1985)
- Renamed `NewExporter` to `New` in the `go.opentelemetry.io/otel/exporters/otlp` package. (#1985)
- Renamed `NewUnstartedExporter` to `NewUnstarted` in the `go.opentelemetry.io/otel/exporters/otlp` package. (#1985)
- The `go.opentelemetry.io/otel/semconv` package has been moved to `go.opentelemetry.io/otel/semconv/v1.4.0` to allow for multiple [telemetry schema](https://github.com/open-telemetry/oteps/blob/main/text/0152-telemetry-schemas.md) versions to be used concurrently. (#1987)
- Metrics test helpers in `go.opentelemetry.io/otel/oteltest` have been moved to `go.opentelemetry.io/otel/metric/metrictest`. (#1988)

### Deprecated

- The `go.opentelemetry.io/otel/exporters/metric/prometheus` is deprecated, use `go.opentelemetry.io/otel/exporters/prometheus` instead. (#1993)
- The `go.opentelemetry.io/otel/exporters/trace/jaeger` is deprecated, use `go.opentelemetry.io/otel/exporters/jaeger` instead. (#1993)
- The `go.opentelemetry.io/otel/exporters/trace/zipkin` is deprecated, use `go.opentelemetry.io/otel/exporters/zipkin` instead. (#1993)

### Removed

- Removed `resource.WithoutBuiltin()`. Use `resource.New()`. (#1810)
- Unexported types `resource.FromEnv`, `resource.Host`, and `resource.TelemetrySDK`, Use the corresponding `With*()` to use individually. (#1810)
- Removed the `Tracer` and `IsRecording` method from the `ReadOnlySpan` in the `go.opentelemetry.io/otel/sdk/trace`.
  The `Tracer` method is not a required to be included in this interface and given the mutable nature of the tracer that is associated with a span, this method is not appropriate.
  The `IsRecording` method returns if the span is recording or not.
  A read-only span value does not need to know if updates to it will be recorded or not.
  By definition, it cannot be updated so there is no point in communicating if an update is recorded. (#1873)
- Removed the `SpanSnapshot` type from the `go.opentelemetry.io/otel/sdk/trace` package.
  The use of this type has been replaced with the use of the explicitly immutable `ReadOnlySpan` type.
  When a concrete representation of a read-only span is needed for testing, the newly added `SpanStub` in the `go.opentelemetry.io/otel/sdk/trace/tracetest` package should be used. (#1873)
- Removed the `Tracer` method from the `Span` interface in the `go.opentelemetry.io/otel/trace` package.
  Using the same tracer that created a span introduces the error where an instrumentation library's `Tracer` is used by other code instead of their own.
  The `"go.opentelemetry.io/otel".Tracer` function or a `TracerProvider` should be used to acquire a library specific `Tracer` instead. (#1900)
  - The `TracerProvider()` method on the `Span` interface may also be used to obtain a `TracerProvider` using the same trace processing pipeline. (#2009)
- The `http.url` attribute generated by `HTTPClientAttributesFromHTTPRequest` will no longer include username or password information. (#1919)
- Removed `IsEmpty` method of the `TraceState` type in the `go.opentelemetry.io/otel/trace` package in favor of using the added `TraceState.Len` method. (#1931)
- Removed `Set`, `Value`, `ContextWithValue`, `ContextWithoutValue`, and `ContextWithEmpty` functions in the `go.opentelemetry.io/otel/baggage` package.
  Handling of baggage is now done using the added `Baggage` type and related context functions (`ContextWithBaggage`, `ContextWithoutBaggage`, and `FromContext`) in that package. (#1967)
- The `InstallNewPipeline` and `NewExportPipeline` creation functions in all the exporters (prometheus, otlp, stdout, jaeger, and zipkin) have been removed.
  These functions were deemed premature attempts to provide convenience that did not achieve this aim. (#1985)
- The `go.opentelemetry.io/otel/exporters/otlp` exporter has been removed.  Use `go.opentelemetry.io/otel/exporters/otlp/otlptrace` instead. (#1990)
- The `go.opentelemetry.io/otel/exporters/stdout` exporter has been removed.  Use `go.opentelemetry.io/otel/exporters/stdout/stdouttrace` or `go.opentelemetry.io/otel/exporters/stdout/stdoutmetric` instead. (#2005)

### Fixed

- Only report errors from the `"go.opentelemetry.io/otel/sdk/resource".Environment` function when they are not `nil`. (#1850, #1851)
- The `Shutdown` method of the simple `SpanProcessor` in the `go.opentelemetry.io/otel/sdk/trace` package now honors the context deadline or cancellation. (#1616, #1856)
- BatchSpanProcessor now drops span batches that failed to be exported. (#1860)
- Use `http://localhost:14268/api/traces` as default Jaeger collector endpoint instead of `http://localhost:14250`. (#1898)
- Allow trailing and leading whitespace in the parsing of a `tracestate` header. (#1931)
- Add logic to determine if the channel is closed to fix Jaeger exporter test panic with close closed channel. (#1870, #1973)
- Avoid transport security when OTLP endpoint is a Unix socket. (#2001)

### Security

## [0.20.0] - 2021-04-23

### Added

- The OTLP exporter now has two new convenience functions, `NewExportPipeline` and `InstallNewPipeline`, setup and install the exporter in tracing and metrics pipelines. (#1373)
- Adds semantic conventions for exceptions. (#1492)
- Added Jaeger Environment variables: `OTEL_EXPORTER_JAEGER_AGENT_HOST`, `OTEL_EXPORTER_JAEGER_AGENT_PORT`
  These environment variables can be used to override Jaeger agent hostname and port (#1752)
- Option `ExportTimeout` was added to batch span processor. (#1755)
- `trace.TraceFlags` is now a defined type over `byte` and `WithSampled(bool) TraceFlags` and `IsSampled() bool` methods have been added to it. (#1770)
- The `Event` and `Link` struct types from the `go.opentelemetry.io/otel` package now include a `DroppedAttributeCount` field to record the number of attributes that were not recorded due to configured limits being reached. (#1771)
- The Jaeger exporter now reports dropped attributes for a Span event in the exported log. (#1771)
- Adds test to check BatchSpanProcessor ignores `OnEnd` and `ForceFlush` post `Shutdown`. (#1772)
- Extract resource attributes from the `OTEL_RESOURCE_ATTRIBUTES` environment variable and merge them with the `resource.Default` resource as well as resources provided to the `TracerProvider` and metric `Controller`. (#1785)
- Added `WithOSType` resource configuration option to set OS (Operating System) type resource attribute (`os.type`). (#1788)
- Added `WithProcess*` resource configuration options to set Process resource attributes. (#1788)
  - `process.pid`
  - `process.executable.name`
  - `process.executable.path`
  - `process.command_args`
  - `process.owner`
  - `process.runtime.name`
  - `process.runtime.version`
  - `process.runtime.description`
- Adds `k8s.node.name` and `k8s.node.uid` attribute keys to the `semconv` package. (#1789)
- Added support for configuring OTLP/HTTP and OTLP/gRPC Endpoints, TLS Certificates, Headers, Compression and Timeout via Environment Variables. (#1758, #1769 and #1811)
  - `OTEL_EXPORTER_OTLP_ENDPOINT`
  - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
  - `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`
  - `OTEL_EXPORTER_OTLP_HEADERS`
  - `OTEL_EXPORTER_OTLP_TRACES_HEADERS`
  - `OTEL_EXPORTER_OTLP_METRICS_HEADERS`
  - `OTEL_EXPORTER_OTLP_COMPRESSION`
  - `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION`
  - `OTEL_EXPORTER_OTLP_METRICS_COMPRESSION`
  - `OTEL_EXPORTER_OTLP_TIMEOUT`
  - `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT`
  - `OTEL_EXPORTER_OTLP_METRICS_TIMEOUT`
  - `OTEL_EXPORTER_OTLP_CERTIFICATE`
  - `OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE`
  - `OTEL_EXPORTER_OTLP_METRICS_CERTIFICATE`
- Adds `otlpgrpc.WithTimeout` option for configuring timeout to the otlp/gRPC exporter. (#1821)
- Adds `jaeger.WithMaxPacketSize` option for configuring maximum UDP packet size used when connecting to the Jaeger agent. (#1853)

### Fixed

- The `Span.IsRecording` implementation from `go.opentelemetry.io/otel/sdk/trace` always returns false when not being sampled. (#1750)
- The Jaeger exporter now correctly sets tags for the Span status code and message.
  This means it uses the correct tag keys (`"otel.status_code"`, `"otel.status_description"`) and does not set the status message as a tag unless it is set on the span. (#1761)
- The Jaeger exporter now correctly records Span event's names using the `"event"` key for a tag.
  Additionally, this tag is overridden, as specified in the OTel specification, if the event contains an attribute with that key. (#1768)
- Zipkin Exporter: Ensure mapping between OTel and Zipkin span data complies with the specification. (#1688)
- Fixed typo for default service name in Jaeger Exporter. (#1797)
- Fix flaky OTLP for the reconnnection of the client connection. (#1527, #1814)
- Fix Jaeger exporter dropping of span batches that exceed the UDP packet size limit.
  Instead, the exporter now splits the batch into smaller sendable batches. (#1828)

### Changed

- Span `RecordError` now records an `exception` event to comply with the semantic convention specification. (#1492)
- Jaeger exporter was updated to use thrift v0.14.1. (#1712)
- Migrate from using internally built and maintained version of the OTLP to the one hosted at `go.opentelemetry.io/proto/otlp`. (#1713)
- Migrate from using `github.com/gogo/protobuf` to `google.golang.org/protobuf` to match `go.opentelemetry.io/proto/otlp`. (#1713)
- The storage of a local or remote Span in a `context.Context` using its SpanContext is unified to store just the current Span.
  The Span's SpanContext can now self-identify as being remote or not.
  This means that `"go.opentelemetry.io/otel/trace".ContextWithRemoteSpanContext` will now overwrite any existing current Span, not just existing remote Spans, and make it the current Span in a `context.Context`. (#1731)
- Improve OTLP/gRPC exporter connection errors. (#1737)
- Information about a parent span context in a `"go.opentelemetry.io/otel/export/trace".SpanSnapshot` is unified in a new `Parent` field.
  The existing `ParentSpanID` and `HasRemoteParent` fields are removed in favor of this. (#1748)
- The `ParentContext` field of the `"go.opentelemetry.io/otel/sdk/trace".SamplingParameters` is updated to hold a `context.Context` containing the parent span.
  This changes it to make `SamplingParameters` conform with the OpenTelemetry specification. (#1749)
- Updated Jaeger Environment Variables: `JAEGER_ENDPOINT`, `JAEGER_USER`, `JAEGER_PASSWORD`
  to `OTEL_EXPORTER_JAEGER_ENDPOINT`, `OTEL_EXPORTER_JAEGER_USER`, `OTEL_EXPORTER_JAEGER_PASSWORD` in compliance with OTel specification. (#1752)
- Modify `BatchSpanProcessor.ForceFlush` to abort after timeout/cancellation. (#1757)
- The `DroppedAttributeCount` field of the `Span` in the `go.opentelemetry.io/otel` package now only represents the number of attributes dropped for the span itself.
  It no longer is a conglomerate of itself, events, and link attributes that have been dropped. (#1771)
- Make `ExportSpans` in Jaeger Exporter honor context deadline. (#1773)
- Modify Zipkin Exporter default service name, use default resource's serviceName instead of empty. (#1777)
- The `go.opentelemetry.io/otel/sdk/export/trace` package is merged into the `go.opentelemetry.io/otel/sdk/trace` package. (#1778)
- The prometheus.InstallNewPipeline example is moved from comment to example test (#1796)
- The convenience functions for the stdout exporter have been updated to return the `TracerProvider` implementation and enable the shutdown of the exporter. (#1800)
- Replace the flush function returned from the Jaeger exporter's convenience creation functions (`InstallNewPipeline` and `NewExportPipeline`) with the `TracerProvider` implementation they create.
  This enables the caller to shutdown and flush using the related `TracerProvider` methods. (#1822)
- Updated the Jaeger exporter to have a default endpoint, `http://localhost:14250`, for the collector. (#1824)
- Changed the function `WithCollectorEndpoint` in the Jaeger exporter to no longer accept an endpoint as an argument.
  The endpoint can be passed with the `CollectorEndpointOption` using the `WithEndpoint` function or by setting the `OTEL_EXPORTER_JAEGER_ENDPOINT` environment variable value appropriately. (#1824)
- The Jaeger exporter no longer batches exported spans itself, instead it relies on the SDK's `BatchSpanProcessor` for this functionality. (#1830)
- The Jaeger exporter creation functions (`NewRawExporter`, `NewExportPipeline`, and `InstallNewPipeline`) no longer accept the removed `Option` type as a variadic argument. (#1830)

### Removed

- Removed Jaeger Environment variables: `JAEGER_SERVICE_NAME`, `JAEGER_DISABLED`, `JAEGER_TAGS`
  These environment variables will no longer be used to override values of the Jaeger exporter (#1752)
- No longer set the links for a `Span` in `go.opentelemetry.io/otel/sdk/trace` that is configured to be a new root.
  This is unspecified behavior that the OpenTelemetry community plans to standardize in the future.
  To prevent backwards incompatible changes when it is specified, these links are removed. (#1726)
- Setting error status while recording error with Span from oteltest package. (#1729)
- The concept of a remote and local Span stored in a context is unified to just the current Span.
  Because of this `"go.opentelemetry.io/otel/trace".RemoteSpanContextFromContext` is removed as it is no longer needed.
  Instead, `"go.opentelemetry.io/otel/trace".SpanContextFromContex` can be used to return the current Span.
  If needed, that Span's `SpanContext.IsRemote()` can then be used to determine if it is remote or not. (#1731)
- The `HasRemoteParent` field of the `"go.opentelemetry.io/otel/sdk/trace".SamplingParameters` is removed.
  This field is redundant to the information returned from the `Remote` method of the `SpanContext` held in the `ParentContext` field. (#1749)
- The `trace.FlagsDebug` and `trace.FlagsDeferred` constants have been removed and will be localized to the B3 propagator. (#1770)
- Remove `Process` configuration, `WithProcessFromEnv` and `ProcessFromEnv`, and type from the Jaeger exporter package.
  The information that could be configured in the `Process` struct should be configured in a `Resource` instead. (#1776, #1804)
- Remove the `WithDisabled` option from the Jaeger exporter.
  To disable the exporter unregister it from the `TracerProvider` or use a no-operation `TracerProvider`. (#1806)
- Removed the functions `CollectorEndpointFromEnv` and `WithCollectorEndpointOptionFromEnv` from the Jaeger exporter.
  These functions for retrieving specific environment variable values are redundant of other internal functions and
  are not intended for end user use. (#1824)
- Removed the Jaeger exporter `WithSDKOptions` `Option`.
  This option was used to set SDK options for the exporter creation convenience functions.
  These functions are provided as a way to easily setup or install the exporter with what are deemed reasonable SDK settings for common use cases.
  If the SDK needs to be configured differently, the `NewRawExporter` function and direct setup of the SDK with the desired settings should be used. (#1825)
- The `WithBufferMaxCount` and `WithBatchMaxCount` `Option`s from the Jaeger exporter are removed.
  The exporter no longer batches exports, instead relying on the SDK's `BatchSpanProcessor` for this functionality. (#1830)
- The Jaeger exporter `Option` type is removed.
  The type is no longer used by the exporter to configure anything.
  All the previous configurations these options provided were duplicates of SDK configuration.
  They have been removed in favor of using the SDK configuration and focuses the exporter configuration to be only about the endpoints it will send telemetry to. (#1830)

## [0.19.0] - 2021-03-18

### Added

- Added `Marshaler` config option to `otlphttp` to enable otlp over json or protobufs. (#1586)
- A `ForceFlush` method to the `"go.opentelemetry.io/otel/sdk/trace".TracerProvider` to flush all registered `SpanProcessor`s. (#1608)
- Added `WithSampler` and `WithSpanLimits` to tracer provider. (#1633, #1702)
- `"go.opentelemetry.io/otel/trace".SpanContext` now has a `remote` property, and `IsRemote()` predicate, that is true when the `SpanContext` has been extracted from remote context data. (#1701)
- A `Valid` method to the `"go.opentelemetry.io/otel/attribute".KeyValue` type. (#1703)

### Changed

- `trace.SpanContext` is now immutable and has no exported fields. (#1573)
  - `trace.NewSpanContext()` can be used in conjunction with the `trace.SpanContextConfig` struct to initialize a new `SpanContext` where all values are known.
- Update the `ForceFlush` method signature to the `"go.opentelemetry.io/otel/sdk/trace".SpanProcessor` to accept a `context.Context` and return an error. (#1608)
- Update the `Shutdown` method to the `"go.opentelemetry.io/otel/sdk/trace".TracerProvider` return an error on shutdown failure. (#1608)
- The SimpleSpanProcessor will now shut down the enclosed `SpanExporter` and gracefully ignore subsequent calls to `OnEnd` after `Shutdown` is called. (#1612)
- `"go.opentelemetry.io/sdk/metric/controller.basic".WithPusher` is replaced with `WithExporter` to provide consistent naming across project. (#1656)
- Added non-empty string check for trace `Attribute` keys. (#1659)
- Add `description` to SpanStatus only when `StatusCode` is set to error. (#1662)
- Jaeger exporter falls back to `resource.Default`'s `service.name` if the exported Span does not have one. (#1673)
- Jaeger exporter populates Jaeger's Span Process from Resource. (#1673)
- Renamed the `LabelSet` method of `"go.opentelemetry.io/otel/sdk/resource".Resource` to `Set`. (#1692)
- Changed `WithSDK` to `WithSDKOptions` to accept variadic arguments of `TracerProviderOption` type in `go.opentelemetry.io/otel/exporters/trace/jaeger` package. (#1693)
- Changed `WithSDK` to `WithSDKOptions` to accept variadic arguments of `TracerProviderOption` type in `go.opentelemetry.io/otel/exporters/trace/zipkin` package. (#1693)

### Removed

- Removed `serviceName` parameter from Zipkin exporter and uses resource instead. (#1549)
- Removed `WithConfig` from tracer provider to avoid overriding configuration. (#1633)
- Removed the exported `SimpleSpanProcessor` and `BatchSpanProcessor` structs.
   These are now returned as a SpanProcessor interface from their respective constructors. (#1638)
- Removed `WithRecord()` from `trace.SpanOption` when creating a span. (#1660)
- Removed setting status to `Error` while recording an error as a span event in `RecordError`. (#1663)
- Removed `jaeger.WithProcess` configuration option. (#1673)
- Removed `ApplyConfig` method from `"go.opentelemetry.io/otel/sdk/trace".TracerProvider` and the now unneeded `Config` struct. (#1693)

### Fixed

- Jaeger Exporter: Ensure mapping between OTEL and Jaeger span data complies with the specification. (#1626)
- `SamplingResult.TraceState` is correctly propagated to a newly created span's `SpanContext`. (#1655)
- The `otel-collector` example now correctly flushes metric events prior to shutting down the exporter. (#1678)
- Do not set span status message in `SpanStatusFromHTTPStatusCode` if it can be inferred from `http.status_code`. (#1681)
- Synchronization issues in global trace delegate implementation. (#1686)
- Reduced excess memory usage by global `TracerProvider`. (#1687)

## [0.18.0] - 2021-03-03

### Added

- Added `resource.Default()` for use with meter and tracer providers. (#1507)
- `AttributePerEventCountLimit` and `AttributePerLinkCountLimit` for `SpanLimits`. (#1535)
- Added `Keys()` method to `propagation.TextMapCarrier` and `propagation.HeaderCarrier` to adapt `http.Header` to this interface. (#1544)
- Added `code` attributes to `go.opentelemetry.io/otel/semconv` package. (#1558)
- Compatibility testing suite in the CI system for the following systems. (#1567)
   | OS      | Go Version | Architecture |
   | ------- | ---------- | ------------ |
   | Ubuntu  | 1.15       | amd64        |
   | Ubuntu  | 1.14       | amd64        |
   | Ubuntu  | 1.15       | 386          |
   | Ubuntu  | 1.14       | 386          |
   | MacOS   | 1.15       | amd64        |
   | MacOS   | 1.14       | amd64        |
   | Windows | 1.15       | amd64        |
   | Windows | 1.14       | amd64        |
   | Windows | 1.15       | 386          |
   | Windows | 1.14       | 386          |

### Changed

- Replaced interface `oteltest.SpanRecorder` with its existing implementation
  `StandardSpanRecorder`. (#1542)
- Default span limit values to 128. (#1535)
- Rename `MaxEventsPerSpan`, `MaxAttributesPerSpan` and `MaxLinksPerSpan` to `EventCountLimit`, `AttributeCountLimit` and `LinkCountLimit`, and move these fields into `SpanLimits`. (#1535)
- Renamed the `otel/label` package to `otel/attribute`. (#1541)
- Vendor the Jaeger exporter's dependency on Apache Thrift. (#1551)
- Parallelize the CI linting and testing. (#1567)
- Stagger timestamps in exact aggregator tests. (#1569)
- Changed all examples to use `WithBatchTimeout(5 * time.Second)` rather than `WithBatchTimeout(5)`. (#1621)
- Prevent end-users from implementing some interfaces (#1575)

  ```
      "otel/exporters/otlp/otlphttp".Option
      "otel/exporters/stdout".Option
      "otel/oteltest".Option
      "otel/trace".TracerOption
      "otel/trace".SpanOption
      "otel/trace".EventOption
      "otel/trace".LifeCycleOption
      "otel/trace".InstrumentationOption
      "otel/sdk/resource".Option
      "otel/sdk/trace".ParentBasedSamplerOption
      "otel/sdk/trace".ReadOnlySpan
      "otel/sdk/trace".ReadWriteSpan
  ```

### Removed

- Removed attempt to resample spans upon changing the span name with `span.SetName()`. (#1545)
- The `test-benchmark` is no longer a dependency of the `precommit` make target. (#1567)
- Removed the `test-386` make target.
   This was replaced with a full compatibility testing suite (i.e. multi OS/arch) in the CI system. (#1567)

### Fixed

- The sequential timing check of timestamps in the stdout exporter are now setup explicitly to be sequential (#1571). (#1572)
- Windows build of Jaeger tests now compiles with OS specific functions (#1576). (#1577)
- The sequential timing check of timestamps of go.opentelemetry.io/otel/sdk/metric/aggregator/lastvalue are now setup explicitly to be sequential (#1578). (#1579)
- Validate tracestate header keys with vendors according to the W3C TraceContext specification (#1475). (#1581)
- The OTLP exporter includes related labels for translations of a GaugeArray (#1563). (#1570)

## [0.17.0] - 2021-02-12

### Changed

- Rename project default branch from `master` to `main`. (#1505)
- Reverse order in which `Resource` attributes are merged, per change in spec. (#1501)
- Add tooling to maintain "replace" directives in go.mod files automatically. (#1528)
- Create new modules: otel/metric, otel/trace, otel/oteltest, otel/sdk/export/metric, otel/sdk/metric (#1528)
- Move metric-related public global APIs from otel to otel/metric/global. (#1528)

## Fixed

- Fixed otlpgrpc reconnection issue.
- The example code in the README.md of `go.opentelemetry.io/otel/exporters/otlp` is moved to a compiled example test and used the new `WithAddress` instead of `WithEndpoint`. (#1513)
- The otel-collector example now uses the default OTLP receiver port of the collector.

## [0.16.0] - 2021-01-13

### Added

- Add the `ReadOnlySpan` and `ReadWriteSpan` interfaces to provide better control for accessing span data. (#1360)
- `NewGRPCDriver` function returns a `ProtocolDriver` that maintains a single gRPC connection to the collector. (#1369)
- Added documentation about the project's versioning policy. (#1388)
- Added `NewSplitDriver` for OTLP exporter that allows sending traces and metrics to different endpoints. (#1418)
- Added codeql workflow to GitHub Actions (#1428)
- Added Gosec workflow to GitHub Actions (#1429)
- Add new HTTP driver for OTLP exporter in `exporters/otlp/otlphttp`. Currently it only supports the binary protobuf payloads. (#1420)
- Add an OpenCensus exporter bridge. (#1444)

### Changed

- Rename `internal/testing` to `internal/internaltest`. (#1449)
- Rename `export.SpanData` to `export.SpanSnapshot` and use it only for exporting spans. (#1360)
- Store the parent's full `SpanContext` rather than just its span ID in the `span` struct. (#1360)
- Improve span duration accuracy. (#1360)
- Migrated CI/CD from CircleCI to GitHub Actions (#1382)
- Remove duplicate checkout from GitHub Actions workflow (#1407)
- Metric `array` aggregator renamed `exact` to match its `aggregation.Kind` (#1412)
- Metric `exact` aggregator includes per-point timestamps (#1412)
- Metric stdout exporter uses MinMaxSumCount aggregator for ValueRecorder instruments (#1412)
- `NewExporter` from `exporters/otlp` now takes a `ProtocolDriver` as a parameter. (#1369)
- Many OTLP Exporter options became gRPC ProtocolDriver options. (#1369)
- Unify endpoint API that related to OTel exporter. (#1401)
- Optimize metric histogram aggregator to re-use its slice of buckets. (#1435)
- Metric aggregator Count() and histogram Bucket.Counts are consistently `uint64`. (1430)
- Histogram aggregator accepts functional options, uses default boundaries if none given. (#1434)
- `SamplingResult` now passed a `Tracestate` from the parent `SpanContext` (#1432)
- Moved gRPC driver for OTLP exporter to `exporters/otlp/otlpgrpc`. (#1420)
- The `TraceContext` propagator now correctly propagates `TraceState` through the `SpanContext`. (#1447)
- Metric Push and Pull Controller components are combined into a single "basic" Controller:
  - `WithExporter()` and `Start()` to configure Push behavior
  - `Start()` is optional; use `Collect()` and `ForEach()` for Pull behavior
  - `Start()` and `Stop()` accept Context. (#1378)
- The `Event` type is moved from the `otel/sdk/export/trace` package to the `otel/trace` API package. (#1452)

### Removed

- Remove `errUninitializedSpan` as its only usage is now obsolete. (#1360)
- Remove Metric export functionality related to quantiles and summary data points: this is not specified (#1412)
- Remove DDSketch metric aggregator; our intention is to re-introduce this as an option of the histogram aggregator after [new OTLP histogram data types](https://github.com/open-telemetry/opentelemetry-proto/pull/226) are released (#1412)

### Fixed

- `BatchSpanProcessor.Shutdown()` will now shutdown underlying `export.SpanExporter`. (#1443)

## [0.15.0] - 2020-12-10

### Added

- The `WithIDGenerator` `TracerProviderOption` is added to the `go.opentelemetry.io/otel/trace` package to configure an `IDGenerator` for the `TracerProvider`. (#1363)

### Changed

- The Zipkin exporter now uses the Span status code to determine. (#1328)
- `NewExporter` and `Start` functions in `go.opentelemetry.io/otel/exporters/otlp` now receive `context.Context` as a first parameter. (#1357)
- Move the OpenCensus example into `example` directory. (#1359)
- Moved the SDK's `internal.IDGenerator` interface in to the `sdk/trace` package to enable support for externally-defined ID generators. (#1363)
- Bump `github.com/google/go-cmp` from 0.5.3 to 0.5.4 (#1374)
- Bump `github.com/golangci/golangci-lint` in `/internal/tools` (#1375)

### Fixed

- Metric SDK `SumObserver` and `UpDownSumObserver` instruments correctness fixes. (#1381)

## [0.14.0] - 2020-11-19

### Added

- An `EventOption` and the related `NewEventConfig` function are added to the `go.opentelemetry.io/otel` package to configure Span events. (#1254)
- A `TextMapPropagator` and associated `TextMapCarrier` are added to the `go.opentelemetry.io/otel/oteltest` package to test `TextMap` type propagators and their use. (#1259)
- `SpanContextFromContext` returns `SpanContext` from context. (#1255)
- `TraceState` has been added to `SpanContext`. (#1340)
- `DeploymentEnvironmentKey` added to `go.opentelemetry.io/otel/semconv` package. (#1323)
- Add an OpenCensus to OpenTelemetry tracing bridge. (#1305)
- Add a parent context argument to `SpanProcessor.OnStart` to follow the specification. (#1333)
- Add missing tests for `sdk/trace/attributes_map.go`. (#1337)

### Changed

- Move the `go.opentelemetry.io/otel/api/trace` package into `go.opentelemetry.io/otel/trace` with the following changes. (#1229) (#1307)
  - `ID` has been renamed to `TraceID`.
  - `IDFromHex` has been renamed to `TraceIDFromHex`.
  - `EmptySpanContext` is removed.
- Move the `go.opentelemetry.io/otel/api/trace/tracetest` package into `go.opentelemetry.io/otel/oteltest`. (#1229)
- OTLP Exporter updates:
  - supports OTLP v0.6.0 (#1230, #1354)
  - supports configurable aggregation temporality (default: Cumulative, optional: Stateless). (#1296)
- The Sampler is now called on local child spans. (#1233)
- The `Kind` type from the `go.opentelemetry.io/otel/api/metric` package was renamed to `InstrumentKind` to more specifically describe what it is and avoid semantic ambiguity. (#1240)
- The `MetricKind` method of the `Descriptor` type in the `go.opentelemetry.io/otel/api/metric` package was renamed to `Descriptor.InstrumentKind`.
   This matches the returned type and fixes misuse of the term metric. (#1240)
- Move test harness from the `go.opentelemetry.io/otel/api/apitest` package into `go.opentelemetry.io/otel/oteltest`. (#1241)
- Move the `go.opentelemetry.io/otel/api/metric/metrictest` package into `go.opentelemetry.io/oteltest` as part of #964. (#1252)
- Move the `go.opentelemetry.io/otel/api/metric` package into `go.opentelemetry.io/otel/metric` as part of #1303. (#1321)
- Move the `go.opentelemetry.io/otel/api/metric/registry` package into `go.opentelemetry.io/otel/metric/registry` as a part of #1303. (#1316)
- Move the `Number` type (together with related functions) from `go.opentelemetry.io/otel/api/metric` package into `go.opentelemetry.io/otel/metric/number` as a part of #1303. (#1316)
- The function signature of the Span `AddEvent` method in `go.opentelemetry.io/otel` is updated to no longer take an unused context and instead take a required name and a variable number of `EventOption`s. (#1254)
- The function signature of the Span `RecordError` method in `go.opentelemetry.io/otel` is updated to no longer take an unused context and instead take a required error value and a variable number of `EventOption`s. (#1254)
- Move the `go.opentelemetry.io/otel/api/global` package to `go.opentelemetry.io/otel`. (#1262) (#1330)
- Move the `Version` function from `go.opentelemetry.io/otel/sdk` to `go.opentelemetry.io/otel`. (#1330)
- Rename correlation context header from `"otcorrelations"` to `"baggage"` to match the OpenTelemetry specification. (#1267)
- Fix `Code.UnmarshalJSON` to work with valid JSON only. (#1276)
- The `resource.New()` method changes signature to support builtin attributes and functional options, including `telemetry.sdk.*` and
  `host.name` semantic conventions; the former method is renamed `resource.NewWithAttributes`. (#1235)
- The Prometheus exporter now exports non-monotonic counters (i.e. `UpDownCounter`s) as gauges. (#1210)
- Correct the `Span.End` method documentation in the `otel` API to state updates are not allowed on a span after it has ended. (#1310)
- Updated span collection limits for attribute, event and link counts to 1000 (#1318)
- Renamed `semconv.HTTPUrlKey` to `semconv.HTTPURLKey`. (#1338)

### Removed

- The `ErrInvalidHexID`, `ErrInvalidTraceIDLength`, `ErrInvalidSpanIDLength`, `ErrInvalidSpanIDLength`, or `ErrNilSpanID` from the `go.opentelemetry.io/otel` package are unexported now. (#1243)
- The `AddEventWithTimestamp` method on the `Span` interface in `go.opentelemetry.io/otel` is removed due to its redundancy.
   It is replaced by using the `AddEvent` method with a `WithTimestamp` option. (#1254)
- The `MockSpan` and `MockTracer` types are removed from `go.opentelemetry.io/otel/oteltest`.
   `Tracer` and `Span` from the same module should be used in their place instead. (#1306)
- `WorkerCount` option is removed from `go.opentelemetry.io/otel/exporters/otlp`. (#1350)
- Remove the following labels types: INT32, UINT32, UINT64 and FLOAT32. (#1314)

### Fixed

- Rename `MergeItererator` to `MergeIterator` in the `go.opentelemetry.io/otel/label` package. (#1244)
- The `go.opentelemetry.io/otel/api/global` packages global TextMapPropagator now delegates functionality to a globally set delegate for all previously returned propagators. (#1258)
- Fix condition in `label.Any`. (#1299)
- Fix global `TracerProvider` to pass options to its configured provider. (#1329)
- Fix missing handler for `ExactKind` aggregator in OTLP metrics transformer (#1309)

## [0.13.0] - 2020-10-08

### Added

- OTLP Metric exporter supports Histogram aggregation. (#1209)
- The `Code` struct from the `go.opentelemetry.io/otel/codes` package now supports JSON marshaling and unmarshaling as well as implements the `Stringer` interface. (#1214)
- A Baggage API to implement the OpenTelemetry specification. (#1217)
- Add Shutdown method to sdk/trace/provider, shutdown processors in the order they were registered. (#1227)

### Changed

- Set default propagator to no-op propagator. (#1184)
- The `HTTPSupplier`, `HTTPExtractor`, `HTTPInjector`, and `HTTPPropagator` from the `go.opentelemetry.io/otel/api/propagation` package were replaced with unified `TextMapCarrier` and `TextMapPropagator` in the `go.opentelemetry.io/otel/propagation` package. (#1212) (#1325)
- The `New` function from the `go.opentelemetry.io/otel/api/propagation` package was replaced with `NewCompositeTextMapPropagator` in the `go.opentelemetry.io/otel` package. (#1212)
- The status codes of the `go.opentelemetry.io/otel/codes` package have been updated to match the latest OpenTelemetry specification.
   They now are `Unset`, `Error`, and `Ok`.
   They no longer track the gRPC codes. (#1214)
- The `StatusCode` field of the `SpanData` struct in the `go.opentelemetry.io/otel/sdk/export/trace` package now uses the codes package from this package instead of the gRPC project. (#1214)
- Move the `go.opentelemetry.io/otel/api/baggage` package into `go.opentelemetry.io/otel/baggage`. (#1217) (#1325)
- A `Shutdown` method of `SpanProcessor` and all its implementations receives a context and returns an error. (#1264)

### Fixed

- Copies of data from arrays and slices passed to `go.opentelemetry.io/otel/label.ArrayValue()` are now used in the returned `Value` instead of using the mutable data itself. (#1226)

### Removed

- The `ExtractHTTP` and `InjectHTTP` functions from the `go.opentelemetry.io/otel/api/propagation` package were removed. (#1212)
- The `Propagators` interface from the `go.opentelemetry.io/otel/api/propagation` package was removed to conform to the OpenTelemetry specification.
   The explicit `TextMapPropagator` type can be used in its place as this is the `Propagator` type the specification defines. (#1212)
- The `SetAttribute` method of the `Span` from the `go.opentelemetry.io/otel/api/trace` package was removed given its redundancy with the `SetAttributes` method. (#1216)
- The internal implementation of Baggage storage is removed in favor of using the new Baggage API functionality. (#1217)
- Remove duplicate hostname key `HostHostNameKey` in Resource semantic conventions. (#1219)
- Nested array/slice support has been removed. (#1226)

## [0.12.0] - 2020-09-24

### Added

- A `SpanConfigure` function in `go.opentelemetry.io/otel/api/trace` to create a new `SpanConfig` from `SpanOption`s. (#1108)
- In the `go.opentelemetry.io/otel/api/trace` package, `NewTracerConfig` was added to construct new `TracerConfig`s.
   This addition was made to conform with our project option conventions. (#1155)
- Instrumentation library information was added to the Zipkin exporter. (#1119)
- The `SpanProcessor` interface now has a `ForceFlush()` method. (#1166)
- More semantic conventions for k8s as resource attributes. (#1167)

### Changed

- Add reconnecting udp connection type to Jaeger exporter.
   This change adds a new optional implementation of the udp conn interface used to detect changes to an agent's host dns record.
   It then adopts the new destination address to ensure the exporter doesn't get stuck. This change was ported from jaegertracing/jaeger-client-go#520. (#1063)
- Replace `StartOption` and `EndOption` in `go.opentelemetry.io/otel/api/trace` with `SpanOption`.
   This change is matched by replacing the `StartConfig` and `EndConfig` with a unified `SpanConfig`. (#1108)
- Replace the `LinkedTo` span option in `go.opentelemetry.io/otel/api/trace` with `WithLinks`.
   This is be more consistent with our other option patterns, i.e. passing the item to be configured directly instead of its component parts, and provides a cleaner function signature. (#1108)
- The `go.opentelemetry.io/otel/api/trace` `TracerOption` was changed to an interface to conform to project option conventions. (#1109)
- Move the `B3` and `TraceContext` from within the `go.opentelemetry.io/otel/api/trace` package to their own `go.opentelemetry.io/otel/propagators` package.
    This removal of the propagators is reflective of the OpenTelemetry specification for these propagators as well as cleans up the `go.opentelemetry.io/otel/api/trace` API. (#1118)
- Rename Jaeger tags used for instrumentation library information to reflect changes in OpenTelemetry specification. (#1119)
- Rename `ProbabilitySampler` to `TraceIDRatioBased` and change semantics to ignore parent span sampling status. (#1115)
- Move `tools` package under `internal`. (#1141)
- Move `go.opentelemetry.io/otel/api/correlation` package to `go.opentelemetry.io/otel/api/baggage`. (#1142)
   The `correlation.CorrelationContext` propagator has been renamed `baggage.Baggage`.  Other exported functions and types are unchanged.
- Rename `ParentOrElse` sampler to `ParentBased` and allow setting samplers depending on parent span. (#1153)
- In the `go.opentelemetry.io/otel/api/trace` package, `SpanConfigure` was renamed to `NewSpanConfig`. (#1155)
- Change `dependabot.yml` to add a `Skip Changelog` label to dependabot-sourced PRs. (#1161)
- The [configuration style guide](https://github.com/open-telemetry/opentelemetry-go/blob/master/CONTRIBUTING.md#config) has been updated to
   recommend the use of `newConfig()` instead of `configure()`. (#1163)
- The `otlp.Config` type has been unexported and changed to `otlp.config`, along with its initializer. (#1163)
- Ensure exported interface types include parameter names and update the
   Style Guide to reflect this styling rule. (#1172)
- Don't consider unset environment variable for resource detection to be an error. (#1170)
- Rename `go.opentelemetry.io/otel/api/metric.ConfigureInstrument` to `NewInstrumentConfig` and
  `go.opentelemetry.io/otel/api/metric.ConfigureMeter` to `NewMeterConfig`.
- ValueObserver instruments use LastValue aggregator by default. (#1165)
- OTLP Metric exporter supports LastValue aggregation. (#1165)
- Move the `go.opentelemetry.io/otel/api/unit` package to `go.opentelemetry.io/otel/unit`. (#1185)
- Rename `Provider` to `MeterProvider` in the `go.opentelemetry.io/otel/api/metric` package. (#1190)
- Rename `NoopProvider` to `NoopMeterProvider` in the `go.opentelemetry.io/otel/api/metric` package. (#1190)
- Rename `NewProvider` to `NewMeterProvider` in the `go.opentelemetry.io/otel/api/metric/metrictest` package. (#1190)
- Rename `Provider` to `MeterProvider` in the `go.opentelemetry.io/otel/api/metric/registry` package. (#1190)
- Rename `NewProvider` to `NewMeterProvider` in the `go.opentelemetry.io/otel/api/metri/registryc` package. (#1190)
- Rename `Provider` to `TracerProvider` in the `go.opentelemetry.io/otel/api/trace` package. (#1190)
- Rename `NoopProvider` to `NoopTracerProvider` in the `go.opentelemetry.io/otel/api/trace` package. (#1190)
- Rename `Provider` to `TracerProvider` in the `go.opentelemetry.io/otel/api/trace/tracetest` package. (#1190)
- Rename `NewProvider` to `NewTracerProvider` in the `go.opentelemetry.io/otel/api/trace/tracetest` package. (#1190)
- Rename `WrapperProvider` to `WrapperTracerProvider` in the `go.opentelemetry.io/otel/bridge/opentracing` package. (#1190)
- Rename `NewWrapperProvider` to `NewWrapperTracerProvider` in the `go.opentelemetry.io/otel/bridge/opentracing` package. (#1190)
- Rename `Provider` method of the pull controller to `MeterProvider` in the `go.opentelemetry.io/otel/sdk/metric/controller/pull` package. (#1190)
- Rename `Provider` method of the push controller to `MeterProvider` in the `go.opentelemetry.io/otel/sdk/metric/controller/push` package. (#1190)
- Rename `ProviderOptions` to `TracerProviderConfig` in the `go.opentelemetry.io/otel/sdk/trace` package. (#1190)
- Rename `ProviderOption` to `TracerProviderOption` in the `go.opentelemetry.io/otel/sdk/trace` package. (#1190)
- Rename `Provider` to `TracerProvider` in the `go.opentelemetry.io/otel/sdk/trace` package. (#1190)
- Rename `NewProvider` to `NewTracerProvider` in the `go.opentelemetry.io/otel/sdk/trace` package. (#1190)
- Renamed `SamplingDecision` values to comply with OpenTelemetry specification change. (#1192)
- Renamed Zipkin attribute names from `ot.status_code & ot.status_description` to `otel.status_code & otel.status_description`. (#1201)
- The default SDK now invokes registered `SpanProcessor`s in the order they were registered with the `TracerProvider`. (#1195)
- Add test of spans being processed by the `SpanProcessor`s in the order they were registered. (#1203)

### Removed

- Remove the B3 propagator from `go.opentelemetry.io/otel/propagators`. It is now located in the
   `go.opentelemetry.io/contrib/propagators/` module. (#1191)
- Remove the semantic convention for HTTP status text, `HTTPStatusTextKey` from package `go.opentelemetry.io/otel/semconv`. (#1194)

### Fixed

- Zipkin example no longer mentions `ParentSampler`, corrected to `ParentBased`. (#1171)
- Fix missing shutdown processor in otel-collector example. (#1186)
- Fix missing shutdown processor in basic and namedtracer examples. (#1197)

## [0.11.0] - 2020-08-24

### Added

- Support for exporting array-valued attributes via OTLP. (#992)
- `Noop` and `InMemory` `SpanBatcher` implementations to help with testing integrations. (#994)
- Support for filtering metric label sets. (#1047)
- A dimensionality-reducing metric Processor. (#1057)
- Integration tests for more OTel Collector Attribute types. (#1062)
- A new `WithSpanProcessor` `ProviderOption` is added to the `go.opentelemetry.io/otel/sdk/trace` package to create a `Provider` and automatically register the `SpanProcessor`. (#1078)

### Changed

- Rename `sdk/metric/processor/test` to `sdk/metric/processor/processortest`. (#1049)
- Rename `sdk/metric/controller/test` to `sdk/metric/controller/controllertest`. (#1049)
- Rename `api/testharness` to `api/apitest`. (#1049)
- Rename `api/trace/testtrace` to `api/trace/tracetest`. (#1049)
- Change Metric Processor to merge multiple observations. (#1024)
- The `go.opentelemetry.io/otel/bridge/opentracing` bridge package has been made into its own module.
   This removes the package dependencies of this bridge from the rest of the OpenTelemetry based project. (#1038)
- Renamed `go.opentelemetry.io/otel/api/standard` package to `go.opentelemetry.io/otel/semconv` to avoid the ambiguous and generic name `standard` and better describe the package as containing OpenTelemetry semantic conventions. (#1016)
- The environment variable used for resource detection has been changed from `OTEL_RESOURCE_LABELS` to `OTEL_RESOURCE_ATTRIBUTES` (#1042)
- Replace `WithSyncer` with `WithBatcher` in examples. (#1044)
- Replace the `google.golang.org/grpc/codes` dependency in the API with an equivalent `go.opentelemetry.io/otel/codes` package. (#1046)
- Merge the `go.opentelemetry.io/otel/api/label` and `go.opentelemetry.io/otel/api/kv` into the new `go.opentelemetry.io/otel/label` package. (#1060)
- Unify Callback Function Naming.
   Rename `*Callback` with `*Func`. (#1061)
- CI builds validate against last two versions of Go, dropping 1.13 and adding 1.15. (#1064)
- The `go.opentelemetry.io/otel/sdk/export/trace` interfaces `SpanSyncer` and `SpanBatcher` have been replaced with a specification compliant `Exporter` interface.
   This interface still supports the export of `SpanData`, but only as a slice.
   Implementation are also required now to return any error from `ExportSpans` if one occurs as well as implement a `Shutdown` method for exporter clean-up. (#1078)
- The `go.opentelemetry.io/otel/sdk/trace` `NewBatchSpanProcessor` function no longer returns an error.
   If a `nil` exporter is passed as an argument to this function, instead of it returning an error, it now returns a `BatchSpanProcessor` that handles the export of `SpanData` by not taking any action. (#1078)
- The `go.opentelemetry.io/otel/sdk/trace` `NewProvider` function to create a `Provider` no longer returns an error, instead only a `*Provider`.
   This change is related to `NewBatchSpanProcessor` not returning an error which was the only error this function would return. (#1078)

### Removed

- Duplicate, unused API sampler interface. (#999)
   Use the [`Sampler` interface](https://github.com/open-telemetry/opentelemetry-go/blob/v0.11.0/sdk/trace/sampling.go) provided by the SDK instead.
- The `grpctrace` instrumentation was moved to the `go.opentelemetry.io/contrib` repository and out of this repository.
   This move includes moving the `grpc` example to the `go.opentelemetry.io/contrib` as well. (#1027)
- The `WithSpan` method of the `Tracer` interface.
   The functionality this method provided was limited compared to what a user can provide themselves.
   It was removed with the understanding that if there is sufficient user need it can be added back based on actual user usage. (#1043)
- The `RegisterSpanProcessor` and `UnregisterSpanProcessor` functions.
   These were holdovers from an approach prior to the TracerProvider design. They were not used anymore. (#1077)
- The `oterror` package. (#1026)
- The `othttp` and `httptrace` instrumentations were moved to `go.opentelemetry.io/contrib`. (#1032)

### Fixed

- The `semconv.HTTPServerMetricAttributesFromHTTPRequest()` function no longer generates the high-cardinality `http.request.content.length` label. (#1031)
- Correct instrumentation version tag in Jaeger exporter. (#1037)
- The SDK span will now set an error event if the `End` method is called during a panic (i.e. it was deferred). (#1043)
- Move internally generated protobuf code from the `go.opentelemetry.io/otel` to the OTLP exporter to reduce dependency overhead. (#1050)
- The `otel-collector` example referenced outdated collector processors. (#1006)

## [0.10.0] - 2020-07-29

This release migrates the default OpenTelemetry SDK into its own Go module, decoupling the SDK from the API and reducing dependencies for instrumentation packages.

### Added

- The Zipkin exporter now has `NewExportPipeline` and `InstallNewPipeline` constructor functions to match the common pattern.
    These function build a new exporter with default SDK options and register the exporter with the `global` package respectively. (#944)
- Add propagator option for gRPC instrumentation. (#986)
- The `testtrace` package now tracks the `trace.SpanKind` for each span. (#987)

### Changed

- Replace the `RegisterGlobal` `Option` in the Jaeger exporter with an `InstallNewPipeline` constructor function.
   This matches the other exporter constructor patterns and will register a new exporter after building it with default configuration. (#944)
- The trace (`go.opentelemetry.io/otel/exporters/trace/stdout`) and metric (`go.opentelemetry.io/otel/exporters/metric/stdout`) `stdout` exporters are now merged into a single exporter at `go.opentelemetry.io/otel/exporters/stdout`.
   This new exporter was made into its own Go module to follow the pattern of all exporters and decouple it from the `go.opentelemetry.io/otel` module. (#956, #963)
- Move the `go.opentelemetry.io/otel/exporters/test` test package to `go.opentelemetry.io/otel/sdk/export/metric/metrictest`. (#962)
- The `go.opentelemetry.io/otel/api/kv/value` package was merged into the parent `go.opentelemetry.io/otel/api/kv` package. (#968)
  - `value.Bool` was replaced with `kv.BoolValue`.
  - `value.Int64` was replaced with `kv.Int64Value`.
  - `value.Uint64` was replaced with `kv.Uint64Value`.
  - `value.Float64` was replaced with `kv.Float64Value`.
  - `value.Int32` was replaced with `kv.Int32Value`.
  - `value.Uint32` was replaced with `kv.Uint32Value`.
  - `value.Float32` was replaced with `kv.Float32Value`.
  - `value.String` was replaced with `kv.StringValue`.
  - `value.Int` was replaced with `kv.IntValue`.
  - `value.Uint` was replaced with `kv.UintValue`.
  - `value.Array` was replaced with `kv.ArrayValue`.
- Rename `Infer` to `Any` in the `go.opentelemetry.io/otel/api/kv` package. (#972)
- Change `othttp` to use the `httpsnoop` package to wrap the `ResponseWriter` so that optional interfaces (`http.Hijacker`, `http.Flusher`, etc.) that are implemented by the original `ResponseWriter`are also implemented by the wrapped `ResponseWriter`. (#979)
- Rename `go.opentelemetry.io/otel/sdk/metric/aggregator/test` package to `go.opentelemetry.io/otel/sdk/metric/aggregator/aggregatortest`. (#980)
- Make the SDK into its own Go module called `go.opentelemetry.io/otel/sdk`. (#985)
- Changed the default trace `Sampler` from `AlwaysOn` to `ParentOrElse(AlwaysOn)`. (#989)

### Removed

- The `IndexedAttribute` function from the `go.opentelemetry.io/otel/api/label` package was removed in favor of `IndexedLabel` which it was synonymous with. (#970)

### Fixed

- Bump github.com/golangci/golangci-lint from 1.28.3 to 1.29.0 in /tools. (#953)
- Bump github.com/google/go-cmp from 0.5.0 to 0.5.1. (#957)
- Use `global.Handle` for span export errors in the OTLP exporter. (#946)
- Correct Go language formatting in the README documentation. (#961)
- Remove default SDK dependencies from the `go.opentelemetry.io/otel/api` package. (#977)
- Remove default SDK dependencies from the `go.opentelemetry.io/otel/instrumentation` package. (#983)
- Move documented examples for `go.opentelemetry.io/otel/instrumentation/grpctrace` interceptors into Go example tests. (#984)

## [0.9.0] - 2020-07-20

### Added

- A new Resource Detector interface is included to allow resources to be automatically detected and included. (#939)
- A Detector to automatically detect resources from an environment variable. (#939)
- Github action to generate protobuf Go bindings locally in `internal/opentelemetry-proto-gen`. (#938)
- OTLP .proto files from `open-telemetry/opentelemetry-proto` imported as a git submodule under `internal/opentelemetry-proto`.
   References to `github.com/open-telemetry/opentelemetry-proto` changed to `go.opentelemetry.io/otel/internal/opentelemetry-proto-gen`. (#942)

### Changed

- Non-nil value `struct`s for key-value pairs will be marshalled using JSON rather than `Sprintf`. (#948)

### Removed

- Removed dependency on `github.com/open-telemetry/opentelemetry-collector`. (#943)

## [0.8.0] - 2020-07-09

### Added

- The `B3Encoding` type to represent the B3 encoding(s) the B3 propagator can inject.
   A value for HTTP supported encodings (Multiple Header: `MultipleHeader`, Single Header: `SingleHeader`) are included. (#882)
- The `FlagsDeferred` trace flag to indicate if the trace sampling decision has been deferred. (#882)
- The `FlagsDebug` trace flag to indicate if the trace is a debug trace. (#882)
- Add `peer.service` semantic attribute. (#898)
- Add database-specific semantic attributes. (#899)
- Add semantic convention for `faas.coldstart` and `container.id`. (#909)
- Add http content size semantic conventions. (#905)
- Include `http.request_content_length` in HTTP request basic attributes. (#905)
- Add semantic conventions for operating system process resource attribute keys. (#919)
- The Jaeger exporter now has a `WithBatchMaxCount` option to specify the maximum number of spans sent in a batch. (#931)

### Changed

- Update `CONTRIBUTING.md` to ask for updates to `CHANGELOG.md` with each pull request. (#879)
- Use lowercase header names for B3 Multiple Headers. (#881)
- The B3 propagator `SingleHeader` field has been replaced with `InjectEncoding`.
   This new field can be set to combinations of the `B3Encoding` bitmasks and will inject trace information in these encodings.
   If no encoding is set, the propagator will default to `MultipleHeader` encoding. (#882)
- The B3 propagator now extracts from either HTTP encoding of B3 (Single Header or Multiple Header) based on what is contained in the header.
   Preference is given to Single Header encoding with Multiple Header being the fallback if Single Header is not found or is invalid.
   This behavior change is made to dynamically support all correctly encoded traces received instead of having to guess the expected encoding prior to receiving. (#882)
- Extend semantic conventions for RPC. (#900)
- To match constant naming conventions in the `api/standard` package, the `FaaS*` key names are appended with a suffix of `Key`. (#920)
  - `"api/standard".FaaSName` -> `FaaSNameKey`
  - `"api/standard".FaaSID` -> `FaaSIDKey`
  - `"api/standard".FaaSVersion` -> `FaaSVersionKey`
  - `"api/standard".FaaSInstance` -> `FaaSInstanceKey`

### Removed

- The `FlagsUnused` trace flag is removed.
   The purpose of this flag was to act as the inverse of `FlagsSampled`, the inverse of `FlagsSampled` is used instead. (#882)
- The B3 header constants (`B3SingleHeader`, `B3DebugFlagHeader`, `B3TraceIDHeader`, `B3SpanIDHeader`, `B3SampledHeader`, `B3ParentSpanIDHeader`) are removed.
   If B3 header keys are needed [the authoritative OpenZipkin package constants](https://pkg.go.dev/github.com/openzipkin/zipkin-go@v0.2.2/propagation/b3?tab=doc#pkg-constants) should be used instead. (#882)

### Fixed

- The B3 Single Header name is now correctly `b3` instead of the previous `X-B3`. (#881)
- The B3 propagator now correctly supports sampling only values (`b3: 0`, `b3: 1`, or `b3: d`) for a Single B3 Header. (#882)
- The B3 propagator now propagates the debug flag.
   This removes the behavior of changing the debug flag into a set sampling bit.
   Instead, this now follow the B3 specification and omits the `X-B3-Sampling` header. (#882)
- The B3 propagator now tracks "unset" sampling state (meaning "defer the decision") and does not set the `X-B3-Sampling` header when injecting. (#882)
- Bump github.com/itchyny/gojq from 0.10.3 to 0.10.4 in /tools. (#883)
- Bump github.com/opentracing/opentracing-go from v1.1.1-0.20190913142402-a7454ce5950e to v1.2.0. (#885)
- The tracing time conversion for OTLP spans is now correctly set to `UnixNano`. (#896)
- Ensure span status is not set to `Unknown` when no HTTP status code is provided as it is assumed to be `200 OK`. (#908)
- Ensure `httptrace.clientTracer` closes `http.headers` span. (#912)
- Prometheus exporter will not apply stale updates or forget inactive metrics. (#903)
- Add test for api.standard `HTTPClientAttributesFromHTTPRequest`. (#905)
- Bump github.com/golangci/golangci-lint from 1.27.0 to 1.28.1 in /tools. (#901, #913)
- Update otel-colector example to use the v0.5.0 collector. (#915)
- The `grpctrace` instrumentation uses a span name conforming to the OpenTelemetry semantic conventions (does not contain a leading slash (`/`)). (#922)
- The `grpctrace` instrumentation includes an `rpc.method` attribute now set to the gRPC method name. (#900, #922)
- The `grpctrace` instrumentation `rpc.service` attribute now contains the package name if one exists.
   This is in accordance with OpenTelemetry semantic conventions. (#922)
- Correlation Context extractor will no longer insert an empty map into the returned context when no valid values are extracted. (#923)
- Bump google.golang.org/api from 0.28.0 to 0.29.0 in /exporters/trace/jaeger. (#925)
- Bump github.com/itchyny/gojq from 0.10.4 to 0.11.0 in /tools. (#926)
- Bump github.com/golangci/golangci-lint from 1.28.1 to 1.28.2 in /tools. (#930)

## [0.7.0] - 2020-06-26

This release implements the v0.5.0 version of the OpenTelemetry specification.

### Added

- The othttp instrumentation now includes default metrics. (#861)
- This CHANGELOG file to track all changes in the project going forward.
- Support for array type attributes. (#798)
- Apply transitive dependabot go.mod dependency updates as part of a new automatic Github workflow. (#844)
- Timestamps are now passed to exporters for each export. (#835)
- Add new `Accumulation` type to metric SDK to transport telemetry from `Accumulator`s to `Processor`s.
   This replaces the prior `Record` `struct` use for this purpose. (#835)
- New dependabot integration to automate package upgrades. (#814)
- `Meter` and `Tracer` implementations accept instrumentation version version as an optional argument.
   This instrumentation version is passed on to exporters. (#811) (#805) (#802)
- The OTLP exporter includes the instrumentation version in telemetry it exports. (#811)
- Environment variables for Jaeger exporter are supported. (#796)
- New `aggregation.Kind` in the export metric API. (#808)
- New example that uses OTLP and the collector. (#790)
- Handle errors in the span `SetName` during span initialization. (#791)
- Default service config to enable retries for retry-able failed requests in the OTLP exporter and an option to override this default. (#777)
- New `go.opentelemetry.io/otel/api/oterror` package to uniformly support error handling and definitions for the project. (#778)
- New `global` default implementation of the `go.opentelemetry.io/otel/api/oterror.Handler` interface to be used to handle errors prior to an user defined `Handler`.
   There is also functionality for the user to register their `Handler` as well as a convenience function `Handle` to handle an error with this global `Handler`(#778)
- Options to specify propagators for httptrace and grpctrace instrumentation. (#784)
- The required `application/json` header for the Zipkin exporter is included in all exports. (#774)
- Integrate HTTP semantics helpers from the contrib repository into the `api/standard` package. #769

### Changed

- Rename `Integrator` to `Processor` in the metric SDK. (#863)
- Rename `AggregationSelector` to `AggregatorSelector`. (#859)
- Rename `SynchronizedCopy` to `SynchronizedMove`. (#858)
- Rename `simple` integrator to `basic` integrator. (#857)
- Merge otlp collector examples. (#841)
- Change the metric SDK to support cumulative, delta, and pass-through exporters directly.
   With these changes, cumulative and delta specific exporters are able to request the correct kind of aggregation from the SDK. (#840)
- The `Aggregator.Checkpoint` API is renamed to `SynchronizedCopy` and adds an argument, a different `Aggregator` into which the copy is stored. (#812)
- The `export.Aggregator` contract is that `Update()` and `SynchronizedCopy()` are synchronized with each other.
   All the aggregation interfaces (`Sum`, `LastValue`, ...) are not meant to be synchronized, as the caller is expected to synchronize aggregators at a higher level after the `Accumulator`.
   Some of the `Aggregators` used unnecessary locking and that has been cleaned up. (#812)
- Use of `metric.Number` was replaced by `int64` now that we use `sync.Mutex` in the `MinMaxSumCount` and `Histogram` `Aggregators`. (#812)
- Replace `AlwaysParentSample` with `ParentSample(fallback)` to match the OpenTelemetry v0.5.0 specification. (#810)
- Rename `sdk/export/metric/aggregator` to `sdk/export/metric/aggregation`. #808
- Send configured headers with every request in the OTLP exporter, instead of just on connection creation. (#806)
- Update error handling for any one off error handlers, replacing, instead, with the `global.Handle` function. (#791)
- Rename `plugin` directory to `instrumentation` to match the OpenTelemetry specification. (#779)
- Makes the argument order to Histogram and DDSketch `New()` consistent. (#781)

### Removed

- `Uint64NumberKind` and related functions from the API. (#864)
- Context arguments from `Aggregator.Checkpoint` and `Integrator.Process` as they were unused. (#803)
- `SpanID` is no longer included in parameters for sampling decision to match the OpenTelemetry specification. (#775)

### Fixed

- Upgrade OTLP exporter to opentelemetry-proto matching the opentelemetry-collector v0.4.0 release. (#866)
- Allow changes to `go.sum` and `go.mod` when running dependabot tidy-up. (#871)
- Bump github.com/stretchr/testify from 1.4.0 to 1.6.1. (#824)
- Bump github.com/prometheus/client_golang from 1.7.0 to 1.7.1 in /exporters/metric/prometheus. (#867)
- Bump google.golang.org/grpc from 1.29.1 to 1.30.0 in /exporters/trace/jaeger. (#853)
- Bump google.golang.org/grpc from 1.29.1 to 1.30.0 in /exporters/trace/zipkin. (#854)
- Bumps github.com/golang/protobuf from 1.3.2 to 1.4.2 (#848)
- Bump github.com/stretchr/testify from 1.4.0 to 1.6.1 in /exporters/otlp (#817)
- Bump github.com/golangci/golangci-lint from 1.25.1 to 1.27.0 in /tools (#828)
- Bump github.com/prometheus/client_golang from 1.5.0 to 1.7.0 in /exporters/metric/prometheus (#838)
- Bump github.com/stretchr/testify from 1.4.0 to 1.6.1 in /exporters/trace/jaeger (#829)
- Bump github.com/benbjohnson/clock from 1.0.0 to 1.0.3 (#815)
- Bump github.com/stretchr/testify from 1.4.0 to 1.6.1 in /exporters/trace/zipkin (#823)
- Bump github.com/itchyny/gojq from 0.10.1 to 0.10.3 in /tools (#830)
- Bump github.com/stretchr/testify from 1.4.0 to 1.6.1 in /exporters/metric/prometheus (#822)
- Bump google.golang.org/grpc from 1.27.1 to 1.29.1 in /exporters/trace/zipkin (#820)
- Bump google.golang.org/grpc from 1.27.1 to 1.29.1 in /exporters/trace/jaeger (#831)
- Bump github.com/google/go-cmp from 0.4.0 to 0.5.0 (#836)
- Bump github.com/google/go-cmp from 0.4.0 to 0.5.0 in /exporters/trace/jaeger (#837)
- Bump github.com/google/go-cmp from 0.4.0 to 0.5.0 in /exporters/otlp (#839)
- Bump google.golang.org/api from 0.20.0 to 0.28.0 in /exporters/trace/jaeger (#843)
- Set span status from HTTP status code in the othttp instrumentation. (#832)
- Fixed typo in push controller comment. (#834)
- The `Aggregator` testing has been updated and cleaned. (#812)
- `metric.Number(0)` expressions are replaced by `0` where possible. (#812)
- Fixed `global` `handler_test.go` test failure. #804
- Fixed `BatchSpanProcessor.Shutdown` to wait until all spans are processed. (#766)
- Fixed OTLP example's accidental early close of exporter. (#807)
- Ensure zipkin exporter reads and closes response body. (#788)
- Update instrumentation to use `api/standard` keys instead of custom keys. (#782)
- Clean up tools and RELEASING documentation. (#762)

## [0.6.0] - 2020-05-21

### Added

- Support for `Resource`s in the prometheus exporter. (#757)
- New pull controller. (#751)
- New `UpDownSumObserver` instrument. (#750)
- OpenTelemetry collector demo. (#711)
- New `SumObserver` instrument. (#747)
- New `UpDownCounter` instrument. (#745)
- New timeout `Option` and configuration function `WithTimeout` to the push controller. (#742)
- New `api/standards` package to implement semantic conventions and standard key-value generation. (#731)

### Changed

- Rename `Register*` functions in the metric API to `New*` for all `Observer` instruments. (#761)
- Use `[]float64` for histogram boundaries, not `[]metric.Number`. (#758)
- Change OTLP example to use exporter as a trace `Syncer` instead of as an unneeded `Batcher`. (#756)
- Replace `WithResourceAttributes()` with `WithResource()` in the trace SDK. (#754)
- The prometheus exporter now uses the new pull controller. (#751)
- Rename `ScheduleDelayMillis` to `BatchTimeout` in the trace `BatchSpanProcessor`.(#752)
- Support use of synchronous instruments in asynchronous callbacks (#725)
- Move `Resource` from the `Export` method parameter into the metric export `Record`. (#739)
- Rename `Observer` instrument to `ValueObserver`. (#734)
- The push controller now has a method (`Provider()`) to return a `metric.Provider` instead of the old `Meter` method that acted as a `metric.Provider`. (#738)
- Replace `Measure` instrument by `ValueRecorder` instrument. (#732)
- Rename correlation context header from `"Correlation-Context"` to `"otcorrelations"` to match the OpenTelemetry specification. (#727)

### Fixed

- Ensure gRPC `ClientStream` override methods do not panic in grpctrace package. (#755)
- Disable parts of `BatchSpanProcessor` test until a fix is found. (#743)
- Fix `string` case in `kv` `Infer` function. (#746)
- Fix panic in grpctrace client interceptors. (#740)
- Refactor the `api/metrics` push controller and add `CheckpointSet` synchronization. (#737)
- Rewrite span batch process queue batching logic. (#719)
- Remove the push controller named Meter map. (#738)
- Fix Histogram aggregator initial state (fix #735). (#736)
- Ensure golang alpine image is running `golang-1.14` for examples. (#733)
- Added test for grpctrace `UnaryInterceptorClient`. (#695)
- Rearrange `api/metric` code layout. (#724)

## [0.5.0] - 2020-05-13

### Added

- Batch `Observer` callback support. (#717)
- Alias `api` types to root package of project. (#696)
- Create basic `othttp.Transport` for simple client instrumentation. (#678)
- `SetAttribute(string, interface{})` to the trace API. (#674)
- Jaeger exporter option that allows user to specify custom http client. (#671)
- `Stringer` and `Infer` methods to `key`s. (#662)

### Changed

- Rename `NewKey` in the `kv` package to just `Key`. (#721)
- Move `core` and `key` to `kv` package. (#720)
- Make the metric API `Meter` a `struct` so the abstract `MeterImpl` can be passed and simplify implementation. (#709)
- Rename SDK `Batcher` to `Integrator` to match draft OpenTelemetry SDK specification. (#710)
- Rename SDK `Ungrouped` integrator to `simple.Integrator` to match draft OpenTelemetry SDK specification. (#710)
- Rename SDK `SDK` `struct` to `Accumulator` to match draft OpenTelemetry SDK specification. (#710)
- Move `Number` from `core` to `api/metric` package. (#706)
- Move `SpanContext` from `core` to `trace` package. (#692)
- Change traceparent header from `Traceparent` to `traceparent` to implement the W3C specification. (#681)

### Fixed

- Update tooling to run generators in all submodules. (#705)
- gRPC interceptor regexp to match methods without a service name. (#683)
- Use a `const` for padding 64-bit B3 trace IDs. (#701)
- Update `mockZipkin` listen address from `:0` to `127.0.0.1:0`. (#700)
- Left-pad 64-bit B3 trace IDs with zero. (#698)
- Propagate at least the first W3C tracestate header. (#694)
- Remove internal `StateLocker` implementation. (#688)
- Increase instance size CI system uses. (#690)
- Add a `key` benchmark and use reflection in `key.Infer()`. (#679)
- Fix internal `global` test by using `global.Meter` with `RecordBatch()`. (#680)
- Reimplement histogram using mutex instead of `StateLocker`. (#669)
- Switch `MinMaxSumCount` to a mutex lock implementation instead of `StateLocker`. (#667)
- Update documentation to not include any references to `WithKeys`. (#672)
- Correct misspelling. (#668)
- Fix clobbering of the span context if extraction fails. (#656)
- Bump `golangci-lint` and work around the corrupting bug. (#666) (#670)

## [0.4.3] - 2020-04-24

### Added

- `Dockerfile` and `docker-compose.yml` to run example code. (#635)
- New `grpctrace` package that provides gRPC client and server interceptors for both unary and stream connections. (#621)
- New `api/label` package, providing common label set implementation. (#651)
- Support for JSON marshaling of `Resources`. (#654)
- `TraceID` and `SpanID` implementations for `Stringer` interface. (#642)
- `RemoteAddrKey` in the othttp plugin to include the HTTP client address in top-level spans. (#627)
- `WithSpanFormatter` option to the othttp plugin. (#617)
- Updated README to include section for compatible libraries and include reference to the contrib repository. (#612)
- The prometheus exporter now supports exporting histograms. (#601)
- A `String` method to the `Resource` to return a hashable identifier for a now unique resource. (#613)
- An `Iter` method to the `Resource` to return an array `AttributeIterator`. (#613)
- An `Equal` method to the `Resource` test the equivalence of resources. (#613)
- An iterable structure (`AttributeIterator`) for `Resource` attributes.

### Changed

- zipkin export's `NewExporter` now requires a `serviceName` argument to ensure this needed values is provided. (#644)
- Pass `Resources` through the metrics export pipeline. (#659)

### Removed

- `WithKeys` option from the metric API. (#639)

### Fixed

- Use the `label.Set.Equivalent` value instead of an encoding in the batcher. (#658)
- Correct typo `trace.Exporter` to `trace.SpanSyncer` in comments. (#653)
- Use type names for return values in jaeger exporter. (#648)
- Increase the visibility of the `api/key` package by updating comments and fixing usages locally. (#650)
- `Checkpoint` only after `Update`; Keep records in the `sync.Map` longer. (#647)
- Do not cache `reflect.ValueOf()` in metric Labels. (#649)
- Batch metrics exported from the OTLP exporter based on `Resource` and labels. (#626)
- Add error wrapping to the prometheus exporter. (#631)
- Update the OTLP exporter batching of traces to use a unique `string` representation of an associated `Resource` as the batching key. (#623)
- Update OTLP `SpanData` transform to only include the `ParentSpanID` if one exists. (#614)
- Update `Resource` internal representation to uniquely and reliably identify resources. (#613)
- Check return value from `CheckpointSet.ForEach` in prometheus exporter. (#622)
- Ensure spans created by httptrace client tracer reflect operation structure. (#618)
- Create a new recorder rather than reuse when multiple observations in same epoch for asynchronous instruments. #610
- The default port the OTLP exporter uses to connect to the OpenTelemetry collector is updated to match the one the collector listens on by default. (#611)

## [0.4.2] - 2020-03-31

### Fixed

- Fix `pre_release.sh` to update version in `sdk/opentelemetry.go`. (#607)
- Fix time conversion from internal to OTLP in OTLP exporter. (#606)

## [0.4.1] - 2020-03-31

### Fixed

- Update `tag.sh` to create signed tags. (#604)

## [0.4.0] - 2020-03-30

### Added

- New API package `api/metric/registry` that exposes a `MeterImpl` wrapper for use by SDKs to generate unique instruments. (#580)
- Script to verify examples after a new release. (#579)

### Removed

- The dogstatsd exporter due to lack of support.
   This additionally removes support for statsd. (#591)
- `LabelSet` from the metric API.
   This is replaced by a `[]core.KeyValue` slice. (#595)
- `Labels` from the metric API's `Meter` interface. (#595)

### Changed

- The metric `export.Labels` became an interface which the SDK implements and the `export` package provides a simple, immutable implementation of this interface intended for testing purposes. (#574)
- Renamed `internal/metric.Meter` to `MeterImpl`. (#580)
- Renamed `api/global/internal.obsImpl` to `asyncImpl`. (#580)

### Fixed

- Corrected missing return in mock span. (#582)
- Update License header for all source files to match CNCF guidelines and include a test to ensure it is present. (#586) (#596)
- Update to v0.3.0 of the OTLP in the OTLP exporter. (#588)
- Update pre-release script to be compatible between GNU and BSD based systems. (#592)
- Add a `RecordBatch` benchmark. (#594)
- Moved span transforms of the OTLP exporter to the internal package. (#593)
- Build both go-1.13 and go-1.14 in circleci to test for all supported versions of Go. (#569)
- Removed unneeded allocation on empty labels in OLTP exporter. (#597)
- Update `BatchedSpanProcessor` to process the queue until no data but respect max batch size. (#599)
- Update project documentation godoc.org links to pkg.go.dev. (#602)

## [0.3.0] - 2020-03-21

This is a first official beta release, which provides almost fully complete metrics, tracing, and context propagation functionality.
There is still a possibility of breaking changes.

### Added

- Add `Observer` metric instrument. (#474)
- Add global `Propagators` functionality to enable deferred initialization for propagators registered before the first Meter SDK is installed. (#494)
- Simplified export setup pipeline for the jaeger exporter to match other exporters. (#459)
- The zipkin trace exporter. (#495)
- The OTLP exporter to export metric and trace telemetry to the OpenTelemetry collector. (#497) (#544) (#545)
- Add `StatusMessage` field to the trace `Span`. (#524)
- Context propagation in OpenTracing bridge in terms of OpenTelemetry context propagation. (#525)
- The `Resource` type was added to the SDK. (#528)
- The global API now supports a `Tracer` and `Meter` function as shortcuts to getting a global `*Provider` and calling these methods directly. (#538)
- The metric API now defines a generic `MeterImpl` interface to support general purpose `Meter` construction.
   Additionally, `SyncImpl` and `AsyncImpl` are added to support general purpose instrument construction. (#560)
- A metric `Kind` is added to represent the `MeasureKind`, `ObserverKind`, and `CounterKind`. (#560)
- Scripts to better automate the release process. (#576)

### Changed

- Default to to use `AlwaysSampler` instead of `ProbabilitySampler` to match OpenTelemetry specification. (#506)
- Renamed `AlwaysSampleSampler` to `AlwaysOnSampler` in the trace API. (#511)
- Renamed `NeverSampleSampler` to `AlwaysOffSampler` in the trace API. (#511)
- The `Status` field of the `Span` was changed to `StatusCode` to disambiguate with the added `StatusMessage`. (#524)
- Updated the trace `Sampler` interface conform to the OpenTelemetry specification. (#531)
- Rename metric API `Options` to `Config`. (#541)
- Rename metric `Counter` aggregator to be `Sum`. (#541)
- Unify metric options into `Option` from instrument specific options. (#541)
- The trace API's `TraceProvider` now support `Resource`s. (#545)
- Correct error in zipkin module name. (#548)
- The jaeger trace exporter now supports `Resource`s. (#551)
- Metric SDK now supports `Resource`s.
   The `WithResource` option was added to configure a `Resource` on creation and the `Resource` method was added to the metric `Descriptor` to return the associated `Resource`. (#552)
- Replace `ErrNoLastValue` and `ErrEmptyDataSet` by `ErrNoData` in the metric SDK. (#557)
- The stdout trace exporter now supports `Resource`s. (#558)
- The metric `Descriptor` is now included at the API instead of the SDK. (#560)
- Replace `Ordered` with an iterator in `export.Labels`. (#567)

### Removed

- The vendor specific Stackdriver. It is now hosted on 3rd party vendor infrastructure. (#452)
- The `Unregister` method for metric observers as it is not in the OpenTelemetry specification. (#560)
- `GetDescriptor` from the metric SDK. (#575)
- The `Gauge` instrument from the metric API. (#537)

### Fixed

- Make histogram aggregator checkpoint consistent. (#438)
- Update README with import instructions and how to build and test. (#505)
- The default label encoding was updated to be unique. (#508)
- Use `NewRoot` in the othttp plugin for public endpoints. (#513)
- Fix data race in `BatchedSpanProcessor`. (#518)
- Skip test-386 for Mac OS 10.15.x (Catalina and upwards). #521
- Use a variable-size array to represent ordered labels in maps. (#523)
- Update the OTLP protobuf and update changed import path. (#532)
- Use `StateLocker` implementation in `MinMaxSumCount`. (#546)
- Eliminate goroutine leak in histogram stress test. (#547)
- Update OTLP exporter with latest protobuf. (#550)
- Add filters to the othttp plugin. (#556)
- Provide an implementation of the `Header*` filters that do not depend on Go 1.14. (#565)
- Encode labels once during checkpoint.
   The checkpoint function is executed in a single thread so we can do the encoding lazily before passing the encoded version of labels to the exporter.
   This is a cheap and quick way to avoid encoding the labels on every collection interval. (#572)
- Run coverage over all packages in `COVERAGE_MOD_DIR`. (#573)

## [0.2.3] - 2020-03-04

### Added

- `RecordError` method on `Span`s in the trace API to Simplify adding error events to spans. (#473)
- Configurable push frequency for exporters setup pipeline. (#504)

### Changed

- Rename the `exporter` directory to `exporters`.
   The `go.opentelemetry.io/otel/exporter/trace/jaeger` package was mistakenly released with a `v1.0.0` tag instead of `v0.1.0`.
   This resulted in all subsequent releases not becoming the default latest.
   A consequence of this was that all `go get`s pulled in the incompatible `v0.1.0` release of that package when pulling in more recent packages from other otel packages.
   Renaming the `exporter` directory to `exporters` fixes this issue by renaming the package and therefore clearing any existing dependency tags.
   Consequentially, this action also renames *all* exporter packages. (#502)

### Removed

- The `CorrelationContextHeader` constant in the `correlation` package is no longer exported. (#503)

## [0.2.2] - 2020-02-27

### Added

- `HTTPSupplier` interface in the propagation API to specify methods to retrieve and store a single value for a key to be associated with a carrier. (#467)
- `HTTPExtractor` interface in the propagation API to extract information from an `HTTPSupplier` into a context. (#467)
- `HTTPInjector` interface in the propagation API to inject information into an `HTTPSupplier.` (#467)
- `Config` and configuring `Option` to the propagator API. (#467)
- `Propagators` interface in the propagation API to contain the set of injectors and extractors for all supported carrier formats. (#467)
- `HTTPPropagator` interface in the propagation API to inject and extract from an `HTTPSupplier.` (#467)
- `WithInjectors` and `WithExtractors` functions to the propagator API to configure injectors and extractors to use. (#467)
- `ExtractHTTP` and `InjectHTTP` functions to apply configured HTTP extractors and injectors to a passed context. (#467)
- Histogram aggregator. (#433)
- `DefaultPropagator` function and have it return `trace.TraceContext` as the default context propagator. (#456)
- `AlwaysParentSample` sampler to the trace API. (#455)
- `WithNewRoot` option function to the trace API to specify the created span should be considered a root span. (#451)

### Changed

- Renamed `WithMap` to `ContextWithMap` in the correlation package. (#481)
- Renamed `FromContext` to `MapFromContext` in the correlation package. (#481)
- Move correlation context propagation to correlation package. (#479)
- Do not default to putting remote span context into links. (#480)
- `Tracer.WithSpan` updated to accept `StartOptions`. (#472)
- Renamed `MetricKind` to `Kind` to not stutter in the type usage. (#432)
- Renamed the `export` package to `metric` to match directory structure. (#432)
- Rename the `api/distributedcontext` package to `api/correlation`. (#444)
- Rename the `api/propagators` package to `api/propagation`. (#444)
- Move the propagators from the `propagators` package into the `trace` API package. (#444)
- Update `Float64Gauge`, `Int64Gauge`, `Float64Counter`, `Int64Counter`, `Float64Measure`, and `Int64Measure` metric methods to use value receivers instead of pointers. (#462)
- Moved all dependencies of tools package to a tools directory. (#466)

### Removed

- Binary propagators. (#467)
- NOOP propagator. (#467)

### Fixed

- Upgraded `github.com/golangci/golangci-lint` from `v1.21.0` to `v1.23.6` in `tools/`. (#492)
- Fix a possible nil-dereference crash (#478)
- Correct comments for `InstallNewPipeline` in the stdout exporter. (#483)
- Correct comments for `InstallNewPipeline` in the dogstatsd exporter. (#484)
- Correct comments for `InstallNewPipeline` in the prometheus exporter. (#482)
- Initialize `onError` based on `Config` in prometheus exporter. (#486)
- Correct module name in prometheus exporter README. (#475)
- Removed tracer name prefix from span names. (#430)
- Fix `aggregator_test.go` import package comment. (#431)
- Improved detail in stdout exporter. (#436)
- Fix a dependency issue (generate target should depend on stringer, not lint target) in Makefile. (#442)
- Reorders the Makefile targets within `precommit` target so we generate files and build the code before doing linting, so we can get much nicer errors about syntax errors from the compiler. (#442)
- Reword function documentation in gRPC plugin. (#446)
- Send the `span.kind` tag to Jaeger from the jaeger exporter. (#441)
- Fix `metadataSupplier` in the jaeger exporter to overwrite the header if existing instead of appending to it. (#441)
- Upgraded to Go 1.13 in CI. (#465)
- Correct opentelemetry.io URL in trace SDK documentation. (#464)
- Refactored reference counting logic in SDK determination of stale records. (#468)
- Add call to `runtime.Gosched` in instrument `acquireHandle` logic to not block the collector. (#469)

## [0.2.1.1] - 2020-01-13

### Fixed

- Use stateful batcher on Prometheus exporter fixing regression introduced in #395. (#428)

## [0.2.1] - 2020-01-08

### Added

- Global meter forwarding implementation.
   This enables deferred initialization for metric instruments registered before the first Meter SDK is installed. (#392)
- Global trace forwarding implementation.
   This enables deferred initialization for tracers registered before the first Trace SDK is installed. (#406)
- Standardize export pipeline creation in all exporters. (#395)
- A testing, organization, and comments for 64-bit field alignment. (#418)
- Script to tag all modules in the project. (#414)

### Changed

- Renamed `propagation` package to `propagators`. (#362)
- Renamed `B3Propagator` propagator to `B3`. (#362)
- Renamed `TextFormatPropagator` propagator to `TextFormat`. (#362)
- Renamed `BinaryPropagator` propagator to `Binary`. (#362)
- Renamed `BinaryFormatPropagator` propagator to `BinaryFormat`. (#362)
- Renamed `NoopTextFormatPropagator` propagator to `NoopTextFormat`. (#362)
- Renamed `TraceContextPropagator` propagator to `TraceContext`. (#362)
- Renamed `SpanOption` to `StartOption` in the trace API. (#369)
- Renamed `StartOptions` to `StartConfig` in the trace API. (#369)
- Renamed `EndOptions` to `EndConfig` in the trace API. (#369)
- `Number` now has a pointer receiver for its methods. (#375)
- Renamed `CurrentSpan` to `SpanFromContext` in the trace API. (#379)
- Renamed `SetCurrentSpan` to `ContextWithSpan` in the trace API. (#379)
- Renamed `Message` in Event to `Name` in the trace API. (#389)
- Prometheus exporter no longer aggregates metrics, instead it only exports them. (#385)
- Renamed `HandleImpl` to `BoundInstrumentImpl` in the metric API. (#400)
- Renamed `Float64CounterHandle` to `Float64CounterBoundInstrument` in the metric API. (#400)
- Renamed `Int64CounterHandle` to `Int64CounterBoundInstrument` in the metric API. (#400)
- Renamed `Float64GaugeHandle` to `Float64GaugeBoundInstrument` in the metric API. (#400)
- Renamed `Int64GaugeHandle` to `Int64GaugeBoundInstrument` in the metric API. (#400)
- Renamed `Float64MeasureHandle` to `Float64MeasureBoundInstrument` in the metric API. (#400)
- Renamed `Int64MeasureHandle` to `Int64MeasureBoundInstrument` in the metric API. (#400)
- Renamed `Release` method for bound instruments in the metric API to `Unbind`. (#400)
- Renamed `AcquireHandle` method for bound instruments in the metric API to `Bind`. (#400)
- Renamed the `File` option in the stdout exporter to `Writer`. (#404)
- Renamed all `Options` to `Config` for all metric exports where this wasn't already the case.

### Fixed

- Aggregator import path corrected. (#421)
- Correct links in README. (#368)
- The README was updated to match latest code changes in its examples. (#374)
- Don't capitalize error statements. (#375)
- Fix ignored errors. (#375)
- Fix ambiguous variable naming. (#375)
- Removed unnecessary type casting. (#375)
- Use named parameters. (#375)
- Updated release schedule. (#378)
- Correct http-stackdriver example module name. (#394)
- Removed the `http.request` span in `httptrace` package. (#397)
- Add comments in the metrics SDK (#399)
- Initialize checkpoint when creating ddsketch aggregator to prevent panic when merging into a empty one. (#402) (#403)
- Add documentation of compatible exporters in the README. (#405)
- Typo fix. (#408)
- Simplify span check logic in SDK tracer implementation. (#419)

## [0.2.0] - 2019-12-03

### Added

- Unary gRPC tracing example. (#351)
- Prometheus exporter. (#334)
- Dogstatsd metrics exporter. (#326)

### Changed

- Rename `MaxSumCount` aggregation to `MinMaxSumCount` and add the `Min` interface for this aggregation. (#352)
- Rename `GetMeter` to `Meter`. (#357)
- Rename `HTTPTraceContextPropagator` to `TraceContextPropagator`. (#355)
- Rename `HTTPB3Propagator` to `B3Propagator`. (#355)
- Rename `HTTPTraceContextPropagator` to `TraceContextPropagator`. (#355)
- Move `/global` package to `/api/global`. (#356)
- Rename `GetTracer` to `Tracer`. (#347)

### Removed

- `SetAttribute` from the `Span` interface in the trace API. (#361)
- `AddLink` from the `Span` interface in the trace API. (#349)
- `Link` from the `Span` interface in the trace API. (#349)

### Fixed

- Exclude example directories from coverage report. (#365)
- Lint make target now implements automatic fixes with `golangci-lint` before a second run to report the remaining issues. (#360)
- Drop `GO111MODULE` environment variable in Makefile as Go 1.13 is the project specified minimum version and this is environment variable is not needed for that version of Go. (#359)
- Run the race checker for all test. (#354)
- Redundant commands in the Makefile are removed. (#354)
- Split the `generate` and `lint` targets of the Makefile. (#354)
- Renames `circle-ci` target to more generic `ci` in Makefile. (#354)
- Add example Prometheus binary to gitignore. (#358)
- Support negative numbers with the `MaxSumCount`. (#335)
- Resolve race conditions in `push_test.go` identified in #339. (#340)
- Use `/usr/bin/env bash` as a shebang in scripts rather than `/bin/bash`. (#336)
- Trace benchmark now tests both `AlwaysSample` and `NeverSample`.
   Previously it was testing `AlwaysSample` twice. (#325)
- Trace benchmark now uses a `[]byte` for `TraceID` to fix failing test. (#325)
- Added a trace benchmark to test variadic functions in `setAttribute` vs `setAttributes` (#325)
- The `defaultkeys` batcher was only using the encoded label set as its map key while building a checkpoint.
   This allowed distinct label sets through, but any metrics sharing a label set could be overwritten or merged incorrectly.
   This was corrected. (#333)

## [0.1.2] - 2019-11-18

### Fixed

- Optimized the `simplelru` map for attributes to reduce the number of allocations. (#328)
- Removed unnecessary unslicing of parameters that are already a slice. (#324)

## [0.1.1] - 2019-11-18

This release contains a Metrics SDK with stdout exporter and supports basic aggregations such as counter, gauges, array, maxsumcount, and ddsketch.

### Added

- Metrics stdout export pipeline. (#265)
- Array aggregation for raw measure metrics. (#282)
- The core.Value now have a `MarshalJSON` method. (#281)

### Removed

- `WithService`, `WithResources`, and `WithComponent` methods of tracers. (#314)
- Prefix slash in `Tracer.Start()` for the Jaeger example. (#292)

### Changed

- Allocation in LabelSet construction to reduce GC overhead. (#318)
- `trace.WithAttributes` to append values instead of replacing (#315)
- Use a formula for tolerance in sampling tests. (#298)
- Move export types into trace and metric-specific sub-directories. (#289)
- `SpanKind` back to being based on an `int` type. (#288)

### Fixed

- URL to OpenTelemetry website in README. (#323)
- Name of othttp default tracer. (#321)
- `ExportSpans` for the stackdriver exporter now handles `nil` context. (#294)
- CI modules cache to correctly restore/save from/to the cache. (#316)
- Fix metric SDK race condition between `LoadOrStore` and the assignment `rec.recorder = i.meter.exporter.AggregatorFor(rec)`. (#293)
- README now reflects the new code structure introduced with these changes. (#291)
- Make the basic example work. (#279)

## [0.1.0] - 2019-11-04

This is the first release of open-telemetry go library.
It contains api and sdk for trace and meter.

### Added

- Initial OpenTelemetry trace and metric API prototypes.
- Initial OpenTelemetry trace, metric, and export SDK packages.
- A wireframe bridge to support compatibility with OpenTracing.
- Example code for a basic, http-stackdriver, http, jaeger, and named tracer setup.
- Exporters for Jaeger, Stackdriver, and stdout.
- Propagators for binary, B3, and trace-context protocols.
- Project information and guidelines in the form of a README and CONTRIBUTING.
- Tools to build the project and a Makefile to automate the process.
- Apache-2.0 license.
- CircleCI build CI manifest files.
- CODEOWNERS file to track owners of this project.

[Unreleased]: https://github.com/open-telemetry/opentelemetry-go/compare/v1.15.1...HEAD
[1.15.1/0.38.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.15.1
[1.15.0/0.38.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.15.0
[1.15.0-rc.2/0.38.0-rc.2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.15.0-rc.2
[1.15.0-rc.1/0.38.0-rc.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.15.0-rc.1
[1.14.0/0.37.0/0.0.4]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.14.0
[1.13.0/0.36.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.13.0
[1.12.0/0.35.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.12.0
[1.11.2/0.34.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.11.2
[1.11.1/0.33.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.11.1
[1.11.0/0.32.3]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.11.0
[0.32.2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/sdk/metric/v0.32.2
[0.32.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/sdk/metric/v0.32.1
[0.32.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/sdk/metric/v0.32.0
[1.10.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.10.0
[1.9.0/0.0.3]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.9.0
[1.8.0/0.31.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.8.0
[1.7.0/0.30.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.7.0
[0.29.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/metric/v0.29.0
[1.6.3]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.6.3
[1.6.2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.6.2
[1.6.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.6.1
[1.6.0/0.28.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.6.0
[1.5.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.5.0
[1.4.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.4.1
[1.4.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.4.0
[1.3.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.3.0
[1.2.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.2.0
[1.1.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.1.0
[1.0.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.0.1
[Metrics 0.24.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/metric/v0.24.0
[1.0.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.0.0
[1.0.0-RC3]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.0.0-RC3
[1.0.0-RC2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.0.0-RC2
[Experimental Metrics v0.22.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/metric/v0.22.0
[1.0.0-RC1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v1.0.0-RC1
[0.20.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.20.0
[0.19.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.19.0
[0.18.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.18.0
[0.17.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.17.0
[0.16.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.16.0
[0.15.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.15.0
[0.14.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.14.0
[0.13.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.13.0
[0.12.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.12.0
[0.11.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.11.0
[0.10.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.10.0
[0.9.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.9.0
[0.8.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.8.0
[0.7.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.7.0
[0.6.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.6.0
[0.5.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.5.0
[0.4.3]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.4.3
[0.4.2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.4.2
[0.4.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.4.1
[0.4.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.4.0
[0.3.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.3.0
[0.2.3]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.2.3
[0.2.2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.2.2
[0.2.1.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.2.1.1
[0.2.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.2.1
[0.2.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.2.0
[0.1.2]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.1.2
[0.1.1]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.1.1
[0.1.0]: https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.1.0

[Go 1.20]: https://go.dev/doc/go1.20
[Go 1.19]: https://go.dev/doc/go1.19
[Go 1.18]: https://go.dev/doc/go1.18
