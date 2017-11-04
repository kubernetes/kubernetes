// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus

// Untyped is a Metric that represents a single numerical value that can
// arbitrarily go up and down.
//
// An Untyped metric works the same as a Gauge. The only difference is that to
// no type information is implied.
//
// To create Untyped instances, use NewUntyped.
//
// Deprecated: The Untyped type is deprecated because it doesn't make sense in
// direct instrumentation. If you need to mirror an external metric of unknown
// type (usually while writing exporters), Use MustNewConstMetric to create an
// untyped metric instance on the fly.
type Untyped interface {
	Metric
	Collector

	// Set sets the Untyped metric to an arbitrary value.
	Set(float64)
	// Inc increments the Untyped metric by 1.
	Inc()
	// Dec decrements the Untyped metric by 1.
	Dec()
	// Add adds the given value to the Untyped metric. (The value can be
	// negative, resulting in a decrease.)
	Add(float64)
	// Sub subtracts the given value from the Untyped metric. (The value can
	// be negative, resulting in an increase.)
	Sub(float64)
}

// UntypedOpts is an alias for Opts. See there for doc comments.
type UntypedOpts Opts

// NewUntyped creates a new Untyped metric from the provided UntypedOpts.
func NewUntyped(opts UntypedOpts) Untyped {
	return newValue(NewDesc(
		BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		nil,
		opts.ConstLabels,
	), UntypedValue, 0)
}

// UntypedVec is a Collector that bundles a set of Untyped metrics that all
// share the same Desc, but have different values for their variable
// labels. This is used if you want to count the same thing partitioned by
// various dimensions. Create instances with NewUntypedVec.
type UntypedVec struct {
	*MetricVec
}

// NewUntypedVec creates a new UntypedVec based on the provided UntypedOpts and
// partitioned by the given label names. At least one label name must be
// provided.
func NewUntypedVec(opts UntypedOpts, labelNames []string) *UntypedVec {
	desc := NewDesc(
		BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		labelNames,
		opts.ConstLabels,
	)
	return &UntypedVec{
		MetricVec: newMetricVec(desc, func(lvs ...string) Metric {
			return newValue(desc, UntypedValue, 0, lvs...)
		}),
	}
}

// GetMetricWithLabelValues replaces the method of the same name in
// MetricVec. The difference is that this method returns an Untyped and not a
// Metric so that no type conversion is required.
func (m *UntypedVec) GetMetricWithLabelValues(lvs ...string) (Untyped, error) {
	metric, err := m.MetricVec.GetMetricWithLabelValues(lvs...)
	if metric != nil {
		return metric.(Untyped), err
	}
	return nil, err
}

// GetMetricWith replaces the method of the same name in MetricVec. The
// difference is that this method returns an Untyped and not a Metric so that no
// type conversion is required.
func (m *UntypedVec) GetMetricWith(labels Labels) (Untyped, error) {
	metric, err := m.MetricVec.GetMetricWith(labels)
	if metric != nil {
		return metric.(Untyped), err
	}
	return nil, err
}

// WithLabelValues works as GetMetricWithLabelValues, but panics where
// GetMetricWithLabelValues would have returned an error. By not returning an
// error, WithLabelValues allows shortcuts like
//     myVec.WithLabelValues("404", "GET").Add(42)
func (m *UntypedVec) WithLabelValues(lvs ...string) Untyped {
	return m.MetricVec.WithLabelValues(lvs...).(Untyped)
}

// With works as GetMetricWith, but panics where GetMetricWithLabels would have
// returned an error. By not returning an error, With allows shortcuts like
//     myVec.With(Labels{"code": "404", "method": "GET"}).Add(42)
func (m *UntypedVec) With(labels Labels) Untyped {
	return m.MetricVec.With(labels).(Untyped)
}

// UntypedFunc is an Untyped whose value is determined at collect time by
// calling a provided function.
//
// To create UntypedFunc instances, use NewUntypedFunc.
type UntypedFunc interface {
	Metric
	Collector
}

// NewUntypedFunc creates a new UntypedFunc based on the provided
// UntypedOpts. The value reported is determined by calling the given function
// from within the Write method. Take into account that metric collection may
// happen concurrently. If that results in concurrent calls to Write, like in
// the case where an UntypedFunc is directly registered with Prometheus, the
// provided function must be concurrency-safe.
func NewUntypedFunc(opts UntypedOpts, function func() float64) UntypedFunc {
	return newValueFunc(NewDesc(
		BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		nil,
		opts.ConstLabels,
	), UntypedValue, function)
}
