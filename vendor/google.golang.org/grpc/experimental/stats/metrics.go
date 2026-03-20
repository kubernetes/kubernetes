/*
 *
 * Copyright 2024 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package stats contains experimental metrics/stats API's.
package stats

import (
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/stats"
)

// MetricsRecorder records on metrics derived from metric registry.
// Implementors must embed UnimplementedMetricsRecorder.
type MetricsRecorder interface {
	// RecordInt64Count records the measurement alongside labels on the int
	// count associated with the provided handle.
	RecordInt64Count(handle *Int64CountHandle, incr int64, labels ...string)
	// RecordFloat64Count records the measurement alongside labels on the float
	// count associated with the provided handle.
	RecordFloat64Count(handle *Float64CountHandle, incr float64, labels ...string)
	// RecordInt64Histo records the measurement alongside labels on the int
	// histo associated with the provided handle.
	RecordInt64Histo(handle *Int64HistoHandle, incr int64, labels ...string)
	// RecordFloat64Histo records the measurement alongside labels on the float
	// histo associated with the provided handle.
	RecordFloat64Histo(handle *Float64HistoHandle, incr float64, labels ...string)
	// RecordInt64Gauge records the measurement alongside labels on the int
	// gauge associated with the provided handle.
	RecordInt64Gauge(handle *Int64GaugeHandle, incr int64, labels ...string)
	// RecordInt64UpDownCounter records the measurement alongside labels on the int
	// count associated with the provided handle.
	RecordInt64UpDownCount(handle *Int64UpDownCountHandle, incr int64, labels ...string)
	// RegisterAsyncReporter registers a reporter to produce metric values for
	// only the listed descriptors. The returned function must be called when
	// the metrics are no longer needed, which will remove the reporter. The
	// returned method needs to be idempotent and concurrent safe.
	RegisterAsyncReporter(reporter AsyncMetricReporter, descriptors ...AsyncMetric) func()

	// EnforceMetricsRecorderEmbedding is included to force implementers to embed
	// another implementation of this interface, allowing gRPC to add methods
	// without breaking users.
	internal.EnforceMetricsRecorderEmbedding
}

// AsyncMetricReporter is an interface for types that record metrics asynchronously
// for the set of descriptors they are registered with. The AsyncMetricsRecorder
// parameter is used to record values for these metrics.
//
// Implementations must make unique recordings across all registered
// AsyncMetricReporters. Meaning, they should not report values for a metric with
// the same attributes as another AsyncMetricReporter will report.
//
// Implementations must be concurrent-safe.
type AsyncMetricReporter interface {
	// Report records metric values using the provided recorder.
	Report(AsyncMetricsRecorder) error
}

// AsyncMetricReporterFunc is an adapter to allow the use of ordinary functions as
// AsyncMetricReporters.
type AsyncMetricReporterFunc func(AsyncMetricsRecorder) error

// Report calls f(r).
func (f AsyncMetricReporterFunc) Report(r AsyncMetricsRecorder) error {
	return f(r)
}

// AsyncMetricsRecorder records on asynchronous metrics derived from metric registry.
type AsyncMetricsRecorder interface {
	// RecordInt64AsyncGauge records the measurement alongside labels on the int
	// count associated with the provided handle asynchronously
	RecordInt64AsyncGauge(handle *Int64AsyncGaugeHandle, incr int64, labels ...string)
}

// Metrics is an experimental legacy alias of the now-stable stats.MetricSet.
// Metrics will be deleted in a future release.
type Metrics = stats.MetricSet

// Metric was replaced by direct usage of strings.
type Metric = string

// NewMetrics is an experimental legacy alias of the now-stable
// stats.NewMetricSet.  NewMetrics will be deleted in a future release.
func NewMetrics(metrics ...Metric) *Metrics {
	return stats.NewMetricSet(metrics...)
}

// UnimplementedMetricsRecorder must be embedded to have forward compatible implementations.
type UnimplementedMetricsRecorder struct {
	internal.EnforceMetricsRecorderEmbedding
}

// RecordInt64Count provides a no-op implementation.
func (UnimplementedMetricsRecorder) RecordInt64Count(*Int64CountHandle, int64, ...string) {}

// RecordFloat64Count provides a no-op implementation.
func (UnimplementedMetricsRecorder) RecordFloat64Count(*Float64CountHandle, float64, ...string) {}

// RecordInt64Histo provides a no-op implementation.
func (UnimplementedMetricsRecorder) RecordInt64Histo(*Int64HistoHandle, int64, ...string) {}

// RecordFloat64Histo provides a no-op implementation.
func (UnimplementedMetricsRecorder) RecordFloat64Histo(*Float64HistoHandle, float64, ...string) {}

// RecordInt64Gauge provides a no-op implementation.
func (UnimplementedMetricsRecorder) RecordInt64Gauge(*Int64GaugeHandle, int64, ...string) {}

// RecordInt64UpDownCount provides a no-op implementation.
func (UnimplementedMetricsRecorder) RecordInt64UpDownCount(*Int64UpDownCountHandle, int64, ...string) {
}

// RegisterAsyncReporter provides a no-op implementation.
func (UnimplementedMetricsRecorder) RegisterAsyncReporter(AsyncMetricReporter, ...AsyncMetric) func() {
	// No-op: Return an empty function to ensure caller doesn't panic on nil function call
	return func() {}
}
