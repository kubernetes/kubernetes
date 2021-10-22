## 1.7.1 / 2020-06-23

* [BUGFIX] API client: Actually propagate start/end parameters of `LabelNames` and `LabelValues`. #771

## 1.7.0 / 2020-06-17

* [CHANGE] API client: Add start/end parameters to `LabelNames` and `LabelValues`. #767
* [FEATURE] testutil: Add `GatherAndCount` and enable filtering in `CollectAndCount` #753
* [FEATURE] API client: Add support for `status` and `runtimeinfo` endpoints. #755
* [ENHANCEMENT] Wrapping `nil` with a `WrapRegistererWith...` function creates a no-op `Registerer`.  #764
* [ENHANCEMENT] promlint: Allow Kelvin as a base unit for cases like color temperature. #761
* [BUGFIX] push: Properly handle empty job and label values. #752

## 1.6.0 / 2020-04-28

* [FEATURE] testutil: Add lint checks for metrics, including a sub-package `promlint` to expose the linter engine for external usage. #739 #743
* [ENHANCEMENT] API client: Improve error messages. #731
* [BUGFIX] process collector: Fix `process_resident_memory_bytes` on 32bit MS Windows. #734

## 1.5.1 / 2020-03-14

* [BUGFIX] promhttp: Remove another superfluous `WriteHeader` call. #726

## 1.5.0 / 2020-03-03

* [FEATURE] promauto: Add a factory to allow automatic registration with a local registry. #713
* [FEATURE] promauto: Add `NewUntypedFunc`. #713
* [FEATURE] API client: Support new metadata endpoint. #718

## 1.4.1 / 2020-02-07

* [BUGFIX] Fix timestamp of exemplars in `CounterVec`. #710

## 1.4.0 / 2020-01-27

* [CHANGE] Go collector: Improve doc string for `go_gc_duration_seconds`. #702
* [FEATURE] Support a subset of OpenMetrics, including exemplars. Needs opt-in via `promhttp.HandlerOpts`. **EXPERIMENTAL** #706
* [FEATURE] Add `testutil.CollectAndCount`. #703

## 1.3.0 / 2019-12-21

* [FEATURE] Support tags in Graphite bridge. #668
* [BUGFIX] API client: Actually return Prometheus warnings. #699

## 1.2.1 / 2019-10-17

* [BUGFIX] Fix regression in the implementation of `Registerer.Unregister`. #663

## 1.2.0 / 2019-10-15

* [FEATURE] Support pushing to Pushgateway v0.10+. #652
* [ENHANCEMENT] Improve hashing to make a spurious `AlreadyRegisteredError` less likely to occur. #657
* [ENHANCEMENT] API client: Add godoc examples. #630
* [BUGFIX] promhttp: Correctly call WriteHeader in HTTP middleware. #634

## 1.1.0 / 2019-08-01

* [CHANGE] API client: Format time as UTC rather than RFC3339Nano. #617
* [CHANGE] API client: Add warnings to `LabelValues` and `LabelNames` calls. #609
* [FEATURE] Push: Support base64 encoding in grouping key. #624
* [FEATURE] Push: Add Delete method to Pusher. #613

## 1.0.0 / 2019-06-15

_This release removes all previously deprecated features, resulting in the breaking changes listed below. As this is v1.0.0, semantic versioning applies from now on, with the exception of the API client and parts marked explicitly as experimental._

* [CHANGE] Remove objectives from the default `Summary`. (Objectives have to be set explicitly in the `SummaryOpts`.) #600
* [CHANGE] Remove all HTTP related feature in the `prometheus` package. (Use the `promhttp` package instead.)  #600
* [CHANGE] Remove `push.FromGatherer`, `push.AddFromGatherer`, `push.Collectors`. (Use `push.New` instead.) #600
* [CHANGE] API client: Pass warnings through on non-error responses. #599
* [CHANGE] API client: Add warnings to `Series` call. #603
* [FEATURE] Make process collector work on Microsoft Windows. **EXPERIMENTAL** #596
* [FEATURE] API client: Add `/labels` call. #604
* [BUGFIX] Make `AlreadyRegisteredError` usable for wrapped registries. #607

## 0.9.4 / 2019-06-07
* [CHANGE] API client: Switch to alert values as strings. #585
* [FEATURE] Add a collector for Go module build information. #595
* [FEATURE] promhttp: Add an counter for internal errors during HTTP exposition. #594
* [FEATURE] API client: Support target metadata API. #590
* [FEATURE] API client: Support storage warnings. #562
* [ENHANCEMENT] API client: Improve performance handling JSON. #570
* [BUGFIX] Reduce test flakiness. #573

## 0.9.3 / 2019-05-16
* [CHANGE] Required Go version is now 1.9+. #561
* [FEATURE] API client: Add POST with get fallback for Query/QueryRange. #557
* [FEATURE] API client: Add alerts endpoint. #552
* [FEATURE] API client: Add rules endpoint. #508
* [FEATURE] push: Add option to pick metrics format. #540
* [ENHANCEMENT] Limit time the Go collector may take to collect memstats,
  returning results from the previous collection in case of a timeout. #568
* [ENHANCEMENT] Pusher now requires only a thin interface instead of a full
  `http.Client`, facilitating mocking and custom HTTP client implementation.
  #559
* [ENHANCEMENT] Memory usage improvement for histograms and summaries without
  objectives. #536
* [ENHANCEMENT] Summaries without objectives are now lock-free. #521
* [BUGFIX] promhttp: `InstrumentRoundTripperTrace` now takes into account a pre-set context. #582
* [BUGFIX] `TestCounterAddLarge` now works on all platforms. #567
* [BUGFIX] Fix `promhttp` examples. #535 #544
* [BUGFIX] API client: Wait for done before writing to shared response
  body. #532
* [BUGFIX] API client: Deal with discovered labels properly. #529

## 0.9.2 / 2018-12-06
* [FEATURE] Support for Go modules. #501
* [FEATURE] `Timer.ObserveDuration` returns observed duration. #509
* [ENHANCEMENT] Improved doc comments and error messages. #504 
* [BUGFIX] Fix race condition during metrics gathering. #512
* [BUGFIX] Fix testutil metric comparison for Histograms and empty labels. #494
  #498

## 0.9.1 / 2018-11-03
* [FEATURE] Add `WriteToTextfile` function to facilitate the creation of
  *.prom files for the textfile collector of the node exporter. #489
* [ENHANCEMENT] More descriptive error messages for inconsistent label
  cardinality. #487
* [ENHANCEMENT] Exposition: Use a GZIP encoder pool to avoid allocations in
  high-frequency scrape scenarios. #366
* [ENHANCEMENT] Exposition: Streaming serving of metrics data while encoding.
  #482
* [ENHANCEMENT] API client: Add a way to return the body of a 5xx response.
  #479

## 0.9.0 / 2018-10-15
* [CHANGE] Go1.6 is no longer supported.
* [CHANGE] More refinements of the `Registry` consistency checks: Duplicated
  labels are now detected, but inconsistent label dimensions are now allowed.
  Collisions with the “magic” metric and label names in Summaries and
  Histograms are detected now. #108 #417 #471
* [CHANGE] Changed `ProcessCollector` constructor. #219
* [CHANGE] Changed Go counter `go_memstats_heap_released_bytes_total` to gauge
  `go_memstats_heap_released_bytes`. #229
* [CHANGE] Unexported `LabelPairSorter`. #453
* [CHANGE] Removed the `Untyped` metric from direct instrumentation. #340
* [CHANGE] Unexported `MetricVec`. #319
* [CHANGE] Removed deprecated `Set` method from `Counter` #247
* [CHANGE] Removed deprecated `RegisterOrGet` and `MustRegisterOrGet`. #247
* [CHANGE] API client: Introduced versioned packages.
* [FEATURE] A `Registerer` can be wrapped with prefixes and labels. #357
* [FEATURE] “Describe by collect” helper function. #239
* [FEATURE] Added package `testutil`. #58
* [FEATURE] Timestamp can be explicitly set for const metrics. #187
* [FEATURE] “Unchecked” collectors are possible now without cheating. #47
* [FEATURE] Pushing to the Pushgateway reworked in package `push` to support
  many new features. (The old functions are still usable but deprecated.) #372
  #341
* [FEATURE] Configurable connection limit for scrapes. #179
* [FEATURE] New HTTP middlewares to instrument `http.Handler` and
  `http.RoundTripper`. The old middlewares and the pre-instrumented `/metrics`
  handler are (strongly) deprecated. #316 #57 #101 #224
* [FEATURE] “Currying” for metric vectors. #320
* [FEATURE] A `Summary` can be created without quantiles. #118
* [FEATURE] Added a `Timer` helper type. #231
* [FEATURE] Added a Graphite bridge. #197
* [FEATURE] Help strings are now optional. #460
* [FEATURE] Added `process_virtual_memory_max_bytes` metric. #438 #440
* [FEATURE] Added `go_gc_cpu_fraction` and `go_threads` metrics. #281 #277
* [FEATURE] Added `promauto` package with auto-registering metrics. #385 #393
* [FEATURE] Add `SetToCurrentTime` method to `Gauge`. #259
* [FEATURE] API client: Add AlertManager, Status, and Target methods. #402
* [FEATURE] API client: Add admin methods. #398
* [FEATURE] API client: Support series API. #361
* [FEATURE] API client: Support querying label values.
* [ENHANCEMENT] Smarter creation of goroutines during scraping. Solves memory
  usage spikes in certain situations. #369
* [ENHANCEMENT] Counters are now faster if dealing with integers only. #367
* [ENHANCEMENT] Improved label validation. #274 #335
* [BUGFIX] Creating a const metric with an invalid `Desc` returns an error. #460
* [BUGFIX] Histogram observations don't race any longer with exposition. #275
* [BUGFIX] Fixed goroutine leaks. #236 #472
* [BUGFIX] Fixed an error message for exponential histogram buckets. #467
* [BUGFIX] Fixed data race writing to the metric map. #401
* [BUGFIX] API client: Decode JSON on a 4xx respons but do not on 204
  responses. #476 #414

## 0.8.0 / 2016-08-17
* [CHANGE] Registry is doing more consistency checks. This might break
  existing setups that used to export inconsistent metrics.
* [CHANGE] Pushing to Pushgateway moved to package `push` and changed to allow
  arbitrary grouping.
* [CHANGE] Removed `SelfCollector`.
* [CHANGE] Removed `PanicOnCollectError` and `EnableCollectChecks` methods.
* [CHANGE] Moved packages to the prometheus/common repo: `text`, `model`,
  `extraction`.
* [CHANGE] Deprecated a number of functions.
* [FEATURE] Allow custom registries. Added `Registerer` and `Gatherer`
  interfaces.
* [FEATURE] Separated HTTP exposition, allowing custom HTTP handlers (package
  `promhttp`) and enabling the creation of other exposition mechanisms.
* [FEATURE] `MustRegister` is variadic now, allowing registration of many
  collectors in one call.
* [FEATURE] Added HTTP API v1 package.
* [ENHANCEMENT] Numerous documentation improvements.
* [ENHANCEMENT] Improved metric sorting.
* [ENHANCEMENT] Inlined fnv64a hashing for improved performance.
* [ENHANCEMENT] Several test improvements.
* [BUGFIX] Handle collisions in MetricVec.

## 0.7.0 / 2015-07-27
* [CHANGE] Rename ExporterLabelPrefix to ExportedLabelPrefix.
* [BUGFIX] Closed gaps in metric consistency check.
* [BUGFIX] Validate LabelName/LabelSet on JSON unmarshaling.
* [ENHANCEMENT] Document the possibility to create "empty" metrics in
  a metric vector.
* [ENHANCEMENT] Fix and clarify various doc comments and the README.md.
* [ENHANCEMENT] (Kind of) solve "The Proxy Problem" of http.InstrumentHandler.
* [ENHANCEMENT] Change responseWriterDelegator.written to int64.

## 0.6.0 / 2015-06-01
* [CHANGE] Rename process_goroutines to go_goroutines.
* [ENHANCEMENT] Validate label names during YAML decoding.
* [ENHANCEMENT] Add LabelName regular expression.
* [BUGFIX] Ensure alignment of struct members for 32-bit systems.

## 0.5.0 / 2015-05-06
* [BUGFIX] Removed a weakness in the fingerprinting aka signature code.
  This makes fingerprinting slower and more allocation-heavy, but the
  weakness was too severe to be tolerated.
* [CHANGE] As a result of the above, Metric.Fingerprint is now returning
  a different fingerprint. To keep the same fingerprint, the new method
  Metric.FastFingerprint was introduced, which will be used by the
  Prometheus server for storage purposes (implying that a collision
  detection has to be added, too).
* [ENHANCEMENT] The Metric.Equal and Metric.Before do not depend on
  fingerprinting anymore, removing the possibility of an undetected
  fingerprint collision.
* [FEATURE] The Go collector in the exposition library includes garbage
  collection stats.
* [FEATURE] The exposition library allows to create constant "throw-away"
  summaries and histograms.
* [CHANGE] A number of new reserved labels and prefixes.

## 0.4.0 / 2015-04-08
* [CHANGE] Return NaN when Summaries have no observations yet.
* [BUGFIX] Properly handle Summary decay upon Write().
* [BUGFIX] Fix the documentation link to the consumption library.
* [FEATURE] Allow the metric family injection hook to merge with existing
  metric families.
* [ENHANCEMENT] Removed cgo dependency and conditional compilation of procfs.
* [MAINTENANCE] Adjusted to changes in matttproud/golang_protobuf_extensions.

## 0.3.2 / 2015-03-11
* [BUGFIX] Fixed the receiver type of COWMetric.Set(). This method is
  only used by the Prometheus server internally.
* [CLEANUP] Added licenses of vendored code left out by godep.

## 0.3.1 / 2015-03-04
* [ENHANCEMENT] Switched fingerprinting functions from own free list to
  sync.Pool.
* [CHANGE] Makefile uses Go 1.4.2 now (only relevant for examples and tests).

## 0.3.0 / 2015-03-03
* [CHANGE] Changed the fingerprinting for metrics. THIS WILL INVALIDATE ALL
  PERSISTED FINGERPRINTS. IF YOU COMPILE THE PROMETHEUS SERVER WITH THIS
  VERSION, YOU HAVE TO WIPE THE PREVIOUSLY CREATED STORAGE.
* [CHANGE] LabelValuesToSignature removed. (Nobody had used it, and it was
  arguably broken.)
* [CHANGE] Vendored dependencies. Those are only used by the Makefile. If
  client_golang is used as a library, the vendoring will stay out of your way.
* [BUGFIX] Remove a weakness in the fingerprinting for metrics. (This made
  the fingerprinting change above necessary.)
* [FEATURE] Added new fingerprinting functions SignatureForLabels and
  SignatureWithoutLabels to be used by the Prometheus server. These functions
  require fewer allocations than the ones currently used by the server.

## 0.2.0 / 2015-02-23
* [FEATURE] Introduce new Histagram metric type.
* [CHANGE] Ignore process collector errors for now (better error handling
  pending).
* [CHANGE] Use clear error interface for process pidFn.
* [BUGFIX] Fix Go download links for several archs and OSes.
* [ENHANCEMENT] Massively improve Gauge and Counter performance.
* [ENHANCEMENT] Catch illegal label names for summaries in histograms.
* [ENHANCEMENT] Reduce allocations during fingerprinting.
* [ENHANCEMENT] Remove cgo dependency. procfs package will only be included if
  both cgo is available and the build is for an OS with procfs.
* [CLEANUP] Clean up code style issues.
* [CLEANUP] Mark slow test as such and exclude them from travis.
* [CLEANUP] Update protobuf library package name.
* [CLEANUP] Updated vendoring of beorn7/perks.

## 0.1.0 / 2015-02-02
* [CLEANUP] Introduced semantic versioning and changelog. From now on,
  changes will be reported in this file.
