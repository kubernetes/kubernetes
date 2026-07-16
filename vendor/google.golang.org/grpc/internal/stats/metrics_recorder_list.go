/*
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
 */

package stats

import (
	"fmt"

	estats "google.golang.org/grpc/experimental/stats"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/stats"
)

// MetricsRecorderList forwards Record calls to all of its metricsRecorders.
//
// It eats any record calls where the label values provided do not match the
// number of label keys.
type MetricsRecorderList struct {
	internal.EnforceMetricsRecorderEmbedding
	// metricsRecorders are the metrics recorders this list will forward to.
	metricsRecorders []estats.MetricsRecorder
}

// NewMetricsRecorderList creates a new metric recorder list with all the stats
// handlers provided which implement the MetricsRecorder interface.
// If no stats handlers provided implement the MetricsRecorder interface,
// the MetricsRecorder list returned is a no-op.
func NewMetricsRecorderList(shs []stats.Handler) *MetricsRecorderList {
	var mrs []estats.MetricsRecorder
	for _, sh := range shs {
		if mr, ok := sh.(estats.MetricsRecorder); ok {
			mrs = append(mrs, mr)
		}
	}
	return &MetricsRecorderList{
		metricsRecorders: mrs,
	}
}

func verifyLabels(desc *estats.MetricDescriptor, labelsRecv ...string) {
	if got, want := len(labelsRecv), len(desc.Labels)+len(desc.OptionalLabels); got != want {
		panic(fmt.Sprintf("Received %d labels in call to record metric %q, but expected %d.", got, desc.Name, want))
	}
}

// RecordInt64Count records the measurement alongside labels on the int
// count associated with the provided handle.
func (l *MetricsRecorderList) RecordInt64Count(handle *estats.Int64CountHandle, incr int64, labels ...string) {
	verifyLabels(handle.Descriptor(), labels...)

	for _, metricRecorder := range l.metricsRecorders {
		metricRecorder.RecordInt64Count(handle, incr, labels...)
	}
}

// RecordInt64UpDownCount records the measurement alongside labels on the int
// count associated with the provided handle.
func (l *MetricsRecorderList) RecordInt64UpDownCount(handle *estats.Int64UpDownCountHandle, incr int64, labels ...string) {
	verifyLabels(handle.Descriptor(), labels...)

	for _, metricRecorder := range l.metricsRecorders {
		metricRecorder.RecordInt64UpDownCount(handle, incr, labels...)
	}
}

// RecordFloat64Count records the measurement alongside labels on the float
// count associated with the provided handle.
func (l *MetricsRecorderList) RecordFloat64Count(handle *estats.Float64CountHandle, incr float64, labels ...string) {
	verifyLabels(handle.Descriptor(), labels...)

	for _, metricRecorder := range l.metricsRecorders {
		metricRecorder.RecordFloat64Count(handle, incr, labels...)
	}
}

// RecordInt64Histo records the measurement alongside labels on the int
// histo associated with the provided handle.
func (l *MetricsRecorderList) RecordInt64Histo(handle *estats.Int64HistoHandle, incr int64, labels ...string) {
	verifyLabels(handle.Descriptor(), labels...)

	for _, metricRecorder := range l.metricsRecorders {
		metricRecorder.RecordInt64Histo(handle, incr, labels...)
	}
}

// RecordFloat64Histo records the measurement alongside labels on the float
// histo associated with the provided handle.
func (l *MetricsRecorderList) RecordFloat64Histo(handle *estats.Float64HistoHandle, incr float64, labels ...string) {
	verifyLabels(handle.Descriptor(), labels...)

	for _, metricRecorder := range l.metricsRecorders {
		metricRecorder.RecordFloat64Histo(handle, incr, labels...)
	}
}

// RecordInt64Gauge records the measurement alongside labels on the int
// gauge associated with the provided handle.
func (l *MetricsRecorderList) RecordInt64Gauge(handle *estats.Int64GaugeHandle, incr int64, labels ...string) {
	verifyLabels(handle.Descriptor(), labels...)

	for _, metricRecorder := range l.metricsRecorders {
		metricRecorder.RecordInt64Gauge(handle, incr, labels...)
	}
}

// RegisterAsyncReporter forwards the registration to all underlying metrics
// recorders.
//
// It returns a cleanup function that, when called, invokes the cleanup function
// returned by each underlying recorder, ensuring the reporter is unregistered
// from all of them.
func (l *MetricsRecorderList) RegisterAsyncReporter(reporter estats.AsyncMetricReporter, metrics ...estats.AsyncMetric) func() {
	descriptorsMap := make(map[*estats.MetricDescriptor]bool, len(metrics))
	for _, m := range metrics {
		descriptorsMap[m.Descriptor()] = true
	}
	unregisterFns := make([]func(), 0, len(l.metricsRecorders))
	for _, mr := range l.metricsRecorders {
		// Wrap the AsyncMetricsRecorder to intercept calls to RecordInt64Gauge
		// and validate the labels.
		wrappedCallback := func(recorder estats.AsyncMetricsRecorder) error {
			wrappedRecorder := &asyncRecorderWrapper{
				delegate:    recorder,
				descriptors: descriptorsMap,
			}
			return reporter.Report(wrappedRecorder)
		}
		unregisterFns = append(unregisterFns, mr.RegisterAsyncReporter(estats.AsyncMetricReporterFunc(wrappedCallback), metrics...))
	}

	// Wrap the cleanup function using the internal delegate.
	// In production, this returns realCleanup as-is.
	// In tests, the leak checker can swap this to track the registration lifetime.
	return internal.AsyncReporterCleanupDelegate(defaultCleanUp(unregisterFns))
}

func defaultCleanUp(unregisterFns []func()) func() {
	return func() {
		for _, unregister := range unregisterFns {
			unregister()
		}
	}
}

type asyncRecorderWrapper struct {
	delegate    estats.AsyncMetricsRecorder
	descriptors map[*estats.MetricDescriptor]bool
}

// RecordIntAsync64Gauge records the measurement alongside labels on the int
// gauge associated with the provided handle.
func (w *asyncRecorderWrapper) RecordInt64AsyncGauge(handle *estats.Int64AsyncGaugeHandle, value int64, labels ...string) {
	// Ensure only metrics for descriptors passed during callback registration
	// are emitted.
	d := handle.Descriptor()
	if _, ok := w.descriptors[d]; !ok {
		return
	}
	// Validate labels and delegate.
	verifyLabels(d, labels...)
	w.delegate.RecordInt64AsyncGauge(handle, value, labels...)
}
