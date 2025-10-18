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

// This file provides abstractions for setting the provider (e.g., prometheus)
// of metrics.

package cache

import (
	"sync"
)

// GaugeMetric represents a single numerical value that can arbitrarily go up
// and down.
type GaugeMetric interface {
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

// HistogramMetric represents a metric that captures observations in buckets
type HistogramMetric interface {
	Observe(float64)
}

type noopMetric struct{}

func (noopMetric) Inc()            {}
func (noopMetric) Dec()            {}
func (noopMetric) Observe(float64) {}
func (noopMetric) Set(float64)     {}

// reflectorMetrics is a struct that holds the metrics for the reflector.
type reflectorMetrics struct {
	// listDuration measures the time taken to complete list operations
	listDuration HistogramMetric
}

// MetricsProvider generates various metrics used by the reflector.
type MetricsProvider interface {
	NewListsMetric(name string, group string, resource string) CounterMetric
	NewListDurationMetric(name string, group string, resource string) HistogramMetric
	NewItemsInListMetric(name string, group string, resource string) GaugeMetric

	NewWatchesMetric(name string, group string, resource string) CounterMetric
	NewShortWatchesMetric(name string, group string, resource string) CounterMetric
	NewWatchDurationMetric(name string, group string, resource string) HistogramMetric
	NewItemsInWatchMetric(name string, group string, resource string) GaugeMetric

	NewLastResourceVersionMetric(name string, group string, resource string) GaugeMetric
}

type noopMetricsProvider struct{}

func (noopMetricsProvider) NewListDurationMetric(name string, group string, resource string) HistogramMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewListsMetric(name string, group string, resource string) CounterMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewItemsInListMetric(name string, group string, resource string) GaugeMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewLastResourceVersionMetric(name string, group string, resource string) GaugeMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewWatchesMetric(name string, group string, resource string) CounterMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewShortWatchesMetric(name string, group string, resource string) CounterMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewWatchDurationMetric(name string, group string, resource string) HistogramMetric {
	return noopMetric{}
}

func (noopMetricsProvider) NewItemsInWatchMetric(name string, group string, resource string) GaugeMetric {
	return noopMetric{}
}

var metricsFactory = struct {
	metricsProvider MetricsProvider
	setProviders    sync.Once
}{
	metricsProvider: noopMetricsProvider{},
}

// newReflectorMetrics creates a new reflectorMetrics object with the given name, group and resource.
// It uses the provided metricsProvider to create individual metrics, or falls back to the global
// provider if none is specified. Returns nil if name is empty.
func newReflectorMetrics(name string, group string, resource string, metricsProvider MetricsProvider) *reflectorMetrics {
	var ret *reflectorMetrics
	if name == "" {
		return ret
	}

	if metricsProvider == nil {
		metricsProvider = metricsFactory.metricsProvider
	}

	return &reflectorMetrics{
		listDuration: metricsProvider.NewListDurationMetric(name, group, resource),
	}
}

// SetReflectorMetricsProvider sets the metrics provider
func SetReflectorMetricsProvider(metricsProvider MetricsProvider) {
	metricsFactory.setProviders.Do(func() {
		metricsFactory.metricsProvider = metricsProvider
	})
}
