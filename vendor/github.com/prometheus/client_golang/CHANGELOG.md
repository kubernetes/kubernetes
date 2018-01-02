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
