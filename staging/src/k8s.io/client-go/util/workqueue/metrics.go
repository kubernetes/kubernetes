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

// This file provides abstractions for setting the provider (e.g., prometheus)
// of metrics.

type queueMetrics interface {
	add(item t)
	get(item t)
	done(item t)
	updateUnfinishedWork()
}

// GaugeMetric represents a single numerical value that can arbitrarily go up
// and down.
type GaugeMetric interface {
	Inc()
	Dec()
}

// SettableGaugeMetric represents a single numerical value that can arbitrarily go up
// and down. (Separate from GaugeMetric to preserve backwards compatibility.)
type SettableGaugeMetric interface {
	Set(float64)
}

// CounterMetric represents a single numerical value that only ever
// goes up.
type CounterMetric interface {
	Inc()
}

// SummaryMetric captures individual observations.
type SummaryMetric interface {
	Observe(float64)
}

type noopMetric struct{}

func (noopMetric) Inc()            {}
func (noopMetric) Dec()            {}
func (noopMetric) Set(float64)     {}
func (noopMetric) Observe(float64) {}

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

	// how long have current threads been working?
	unfinishedWorkMicroseconds SettableGaugeMetric
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

func (m *defaultQueueMetrics) updateUnfinishedWork() {
	var total float64
	if m.processingStartTimes != nil {
		for _, t := range m.processingStartTimes {
			total += sinceInMicroseconds(t)
		}
	}
	m.unfinishedWorkMicroseconds.Set(total)
}

type noMetrics struct{}

func (noMetrics) add(item t)            {}
func (noMetrics) get(item t)            {}
func (noMetrics) done(item t)           {}
func (noMetrics) updateUnfinishedWork() {}

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

func (m *defaultRetryMetrics) retry() {
	if m == nil {
		return
	}

	m.retries.Inc()
}

// MetricsProvider generates various metrics used by the queue.
type MetricsProvider interface {
	NewDepthMetric(name string) GaugeMetric
	NewAddsMetric(name string) CounterMetric
	NewLatencyMetric(name string) SummaryMetric
	NewWorkDurationMetric(name string) SummaryMetric
	NewUnfinishedWorkMicrosecondsMetric(name string) SettableGaugeMetric
	NewRetriesMetric(name string) CounterMetric
}

type noopMetricsProvider struct{}

func (_ noopMetricsProvider) NewDepthMetric(name string) GaugeMetric {
	return noopMetric{}
}

func (_ noopMetricsProvider) NewAddsMetric(name string) CounterMetric {
	return noopMetric{}
}

func (_ noopMetricsProvider) NewLatencyMetric(name string) SummaryMetric {
	return noopMetric{}
}

func (_ noopMetricsProvider) NewWorkDurationMetric(name string) SummaryMetric {
	return noopMetric{}
}

func (_ noopMetricsProvider) NewUnfinishedWorkMicrosecondsMetric(name string) SettableGaugeMetric {
	return noopMetric{}
}

func (_ noopMetricsProvider) NewRetriesMetric(name string) CounterMetric {
	return noopMetric{}
}

var metricsFactory = struct {
	metricsProvider MetricsProvider
	setProviders    sync.Once
}{
	metricsProvider: noopMetricsProvider{},
}

func newQueueMetrics(name string) queueMetrics {
	mp := metricsFactory.metricsProvider
	if len(name) == 0 || mp == (noopMetricsProvider{}) {
		return noMetrics{}
	}
	return &defaultQueueMetrics{
		depth:                      mp.NewDepthMetric(name),
		adds:                       mp.NewAddsMetric(name),
		latency:                    mp.NewLatencyMetric(name),
		workDuration:               mp.NewWorkDurationMetric(name),
		unfinishedWorkMicroseconds: mp.NewUnfinishedWorkMicrosecondsMetric(name),
		addTimes:                   map[t]time.Time{},
		processingStartTimes:       map[t]time.Time{},
	}
}

func newRetryMetrics(name string) retryMetrics {
	var ret *defaultRetryMetrics
	if len(name) == 0 {
		return ret
	}
	return &defaultRetryMetrics{
		retries: metricsFactory.metricsProvider.NewRetriesMetric(name),
	}
}

// SetProvider sets the metrics provider of the metricsFactory.
func SetProvider(metricsProvider MetricsProvider) {
	metricsFactory.setProviders.Do(func() {
		metricsFactory.metricsProvider = metricsProvider
	})
}
