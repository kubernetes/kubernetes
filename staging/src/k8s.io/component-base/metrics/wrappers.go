/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"context"
	"errors"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// This file contains a series of interfaces which we explicitly define for
// integrating with prometheus. We redefine the interfaces explicitly here
// so that we can prevent breakage if methods are ever added to prometheus
// variants of them.

// Collector defines a subset of prometheus.Collector interface methods
type Collector interface {
	Describe(chan<- *prometheus.Desc)
	Collect(chan<- prometheus.Metric)
}

// Metric defines a subset of prometheus.Metric interface methods
type Metric interface {
	Desc() *prometheus.Desc
	Write(*dto.Metric) error
}

// CounterMetric is a Metric that represents a single numerical value that only ever
// goes up. That implies that it cannot be used to count items whose number can
// also go down, e.g. the number of currently running goroutines. Those
// "counters" are represented by Gauges.

// CounterMetric is an interface which defines a subset of the interface provided by prometheus.Counter
type CounterMetric interface {
	Inc()
	Add(float64)
}

// CounterVecMetric is an interface which prometheus.CounterVec satisfies.
type CounterVecMetric interface {
	WithLabelValues(...string) CounterMetric
	With(prometheus.Labels) CounterMetric
}

// GaugeMetric is an interface which defines a subset of the interface provided by prometheus.Gauge
type GaugeMetric interface {
	Set(float64)
	Inc()
	Dec()
	Add(float64)
	Write(out *dto.Metric) error
	SetToCurrentTime()
}

// GaugeMetricVec is a collection of Gauges that differ only in label values.
// This is really just one Metric.
// It might be better called GaugeVecMetric, but that pattern of name is already
// taken by the other pattern --- which is treacherous.  The treachery is that
// WithLabelValues can return an object that is permanently broken (i.e., a noop).
type GaugeMetricVec interface {
	Set(value float64, labelValues ...string)
	Inc(labelValues ...string)
	Dec(labelValues ...string)
	Add(delta float64, labelValues ...string)
	SetToCurrentTime(labelValues ...string)

	SetForLabels(value float64, labels map[string]string)
	IncForLabels(labels map[string]string)
	DecForLabels(labels map[string]string)
	AddForLabels(delta float64, labels map[string]string)
	SetToCurrentTimeForLabels(labels map[string]string)

	// WithLabelValues, if called after this vector has been
	// registered in at least one registry and this vector is not
	// hidden, will return a GaugeMetric that is NOT a noop along
	// with nil error.  If called on a hidden vector then it will
	// return a noop and a nil error.  Otherwise it returns a noop
	// and an error that passes ErrIsNotRegistered.
	WithLabelValues(labelValues ...string) (GaugeMetric, error)

	// With, if called after this vector has been
	// registered in at least one registry and this vector is not
	// hidden, will return a GaugeMetric that is NOT a noop along
	// with nil error.  If called on a hidden vector then it will
	// return a noop and a nil error.  Otherwise it returns a noop
	// and an error that passes ErrIsNotRegistered.
	With(labels map[string]string) (GaugeMetric, error)

	// Delete asserts that the vec should have no member for the given label set.
	// The returned bool indicates whether there was a change.
	// The return will certainly be `false` if the given label set has the wrong
	// set of label names.
	Delete(map[string]string) bool

	// Reset removes all the members
	Reset()
}

// PreContextGaugeMetricVec is something that can construct a GaugeMetricVec
// that uses a given Context.
type PreContextGaugeMetricVec interface {
	// WithContext creates a GaugeMetricVec that uses the given Context
	WithContext(ctx context.Context) GaugeMetricVec
}

// RegisterableGaugeMetricVec is the intersection of Registerable and GaugeMetricVec
type RegisterableGaugeMetricVec interface {
	Registerable
	GaugeMetricVec
}

// PreContextAndRegisterableGaugeMetricVec is the intersection of
// PreContextGaugeMetricVec and RegisterableGaugeMetricVec
type PreContextAndRegisterableGaugeMetricVec interface {
	PreContextGaugeMetricVec
	RegisterableGaugeMetricVec
}

// ObserverMetric captures individual observations.
type ObserverMetric interface {
	Observe(float64)
}

// PromRegistry is an interface which implements a subset of prometheus.Registerer and
// prometheus.Gatherer interfaces
type PromRegistry interface {
	Register(prometheus.Collector) error
	MustRegister(...prometheus.Collector)
	Unregister(prometheus.Collector) bool
	Gather() ([]*dto.MetricFamily, error)
}

// Gatherer is the interface for the part of a registry in charge of gathering
// the collected metrics into a number of MetricFamilies.
type Gatherer interface {
	prometheus.Gatherer
}

// GaugeFunc is a Gauge whose value is determined at collect time by calling a
// provided function.
//
// To create GaugeFunc instances, use NewGaugeFunc.
type GaugeFunc interface {
	Metric
	Collector
}

func ErrIsNotRegistered(err error) bool {
	return err == errNotRegistered
}

var errNotRegistered = errors.New("metric vec is not registered yet")
