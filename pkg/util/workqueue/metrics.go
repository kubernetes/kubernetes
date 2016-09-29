/*
Copyright 2016 The Kubernetes Authors.

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

package workqueue

import (
	"sync"
	"time"
)

// You need to import the prometheus package to create and register prometheus
// metrics, otherwise all the operations of the metrics are noop.

type queueMetrics interface {
	add(item t)
	get(item t)
	done(item t)
}

// GaugeMetric that represents a single numerical value that can
// arbitrarily go up and down.
// Its methods are a subset of prometheus.Gauge.
type GaugeMetric interface {
	Inc()
	Dec()
}

type noopGaugeMetric struct{}

func (noopGaugeMetric) Inc() {}
func (noopGaugeMetric) Dec() {}

// CounterMetric represents a single numerical value that only ever
// goes up.
// Its methods are a subset of prometheus.Counter.
type CounterMetric interface {
	Inc()
}

type noopCounterMetric struct{}

func (noopCounterMetric) Inc() {}

// SummaryMetric captures individual observations.
// Its methods are a subset of prometheus.Observe.
type SummaryMetric interface {
	Observe(float64)
}

type noopSummaryMetric struct{}

func (noopSummaryMetric) Observe(float64) {}

type defaultQueueMetrics struct {
	// current depth of a workqueue
	depth GaugeMetric
	// total number of adds handled by a workqueue
	adds CounterMetric
	// how long an item stays in a workqueue
	latency SummaryMetric
	// how long processing an item from a workqueue takes
	workDuration         SummaryMetric
	addTimes             map[t]time.Time
	processingStartTimes map[t]time.Time
}

// DepthMetricProvider creates DepthMetric.
type DepthMetricProvider interface {
	NewDepthMetric(name string) GaugeMetric
}

type noopDepthMetricProvider struct{}

func (_ noopDepthMetricProvider) NewDepthMetric(name string) GaugeMetric {
	return noopGaugeMetric{}
}

// AddsMetricProvider creates AddsMetric.
type AddsMetricProvider interface {
	NewAddsMetric(name string) CounterMetric
}

type noopAddsMetricProvider struct{}

func (_ noopAddsMetricProvider) NewAddsMetric(name string) CounterMetric {
	return noopCounterMetric{}
}

// LatencyMetricProvider creates LatencyMetric.
type LatencyMetricProvider interface {
	NewLatencyMetric(name string) SummaryMetric
}

type noopLatencyMetricProvider struct{}

func (_ noopLatencyMetricProvider) NewLatencyMetric(name string) SummaryMetric {
	return noopSummaryMetric{}
}

// WorkDurationMetricProvider creates WorkDurationMetric.
type WorkDurationMetricProvider interface {
	NewWorkDurationMetric(name string) SummaryMetric
}

type noopWorkDurationMetricProvider struct{}

func (_ noopWorkDurationMetricProvider) NewWorkDurationMetric(name string) SummaryMetric {
	return noopSummaryMetric{}
}

func (m *defaultQueueMetrics) add(item t) {
	if m == nil {
		return
	}

	m.adds.Inc()
	m.depth.Inc()
	if _, exists := m.addTimes[item]; !exists {
		m.addTimes[item] = time.Now()
	}
}

func (m *defaultQueueMetrics) get(item t) {
	if m == nil {
		return
	}

	m.depth.Dec()
	m.processingStartTimes[item] = time.Now()
	if startTime, exists := m.addTimes[item]; exists {
		m.latency.Observe(sinceInMicroseconds(startTime))
		delete(m.addTimes, item)
	}
}

func (m *defaultQueueMetrics) done(item t) {
	if m == nil {
		return
	}

	if startTime, exists := m.processingStartTimes[item]; exists {
		m.workDuration.Observe(sinceInMicroseconds(startTime))
		delete(m.processingStartTimes, item)
	}
}

// Gets the time since the specified start in microseconds.
func sinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

type retryMetrics interface {
	retry()
}

type defaultRetryMetrics struct {
	retries CounterMetric
}

// RetriesMetricProvider creates RetriesMetric.
type RetriesMetricProvider interface {
	NewRetriesMetric(name string) CounterMetric
}

type noopRetriesMetricProvider struct{}

func (_ noopRetriesMetricProvider) NewRetriesMetric(name string) CounterMetric {
	return noopCounterMetric{}
}

func (m *defaultRetryMetrics) retry() {
	if m == nil {
		return
	}

	m.retries.Inc()
}

type metricsFactory struct {
	depthMetricProvider        DepthMetricProvider
	addsMetricProvider         AddsMetricProvider
	latencyMetricProvider      LatencyMetricProvider
	workDurationMetricProvider WorkDurationMetricProvider
	retriesMetricProvider      RetriesMetricProvider
	setProviders               sync.Once
}

func (f metricsFactory) newQueueMetrics(name string) queueMetrics {
	var ret *defaultQueueMetrics
	if len(name) == 0 {
		return ret
	}
	return &defaultQueueMetrics{
		depth:                f.depthMetricProvider.NewDepthMetric(name),
		adds:                 f.addsMetricProvider.NewAddsMetric(name),
		latency:              f.latencyMetricProvider.NewLatencyMetric(name),
		workDuration:         f.workDurationMetricProvider.NewWorkDurationMetric(name),
		addTimes:             map[t]time.Time{},
		processingStartTimes: map[t]time.Time{},
	}
}

func (f metricsFactory) newRetryMetrics(name string) retryMetrics {
	var ret *defaultRetryMetrics
	if len(name) == 0 {
		return ret
	}
	return &defaultRetryMetrics{
		retries: f.retriesMetricProvider.NewRetriesMetric(name),
	}
}

// SetProviders sets the metrics providers of the metricsFactory.
func (f metricsFactory) SetProviders(
	depthMetricProvider DepthMetricProvider,
	addsMetricProvider AddsMetricProvider,
	latencyMetricProvider LatencyMetricProvider,
	workDurationMetricProvider WorkDurationMetricProvider,
	retriesMetricProvider RetriesMetricProvider,
) {
	f.setProviders.Do(func() {
		f.depthMetricProvider = depthMetricProvider
		f.addsMetricProvider = addsMetricProvider
		f.latencyMetricProvider = latencyMetricProvider
		f.workDurationMetricProvider = workDurationMetricProvider
		f.retriesMetricProvider = retriesMetricProvider
	})
}

// DefaultMetricsFactory is a factory that produces queueMetrics and retryMetrics.
var DefaultMetricsFactory = metricsFactory{
	depthMetricProvider:        noopDepthMetricProvider{},
	addsMetricProvider:         noopAddsMetricProvider{},
	latencyMetricProvider:      noopLatencyMetricProvider{},
	workDurationMetricProvider: noopWorkDurationMetricProvider{},
	retriesMetricProvider:      noopRetriesMetricProvider{},
}
