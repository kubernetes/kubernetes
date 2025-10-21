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

	"k8s.io/utils/clock"
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
	clock clock.Clock

	// numberOfLists tracks the cumulative count of list operations performed
	numberOfLists CounterMetric
	// listDuration measures the time taken to complete list operations
	listDuration HistogramMetric
	// numberOfItemsInList tracks the current number of items in the most recent list
	numberOfItemsInList GaugeMetric
	// lastResourceVersion tracks the most recent resource version seen from list/watch operations
	lastResourceVersion GaugeMetric

	// numberOfWatches tracks the cumulative count of watch operations performed
	numberOfWatches CounterMetric
	// numberOfShortWatches tracks the cumulative count of short watch operations performed
	numberOfShortWatches CounterMetric
	// watchDuration measures the time taken to complete watch operations
	watchDuration HistogramMetric
	// numberOfItemsInWatch tracks the current number of items in the most recent watch
	numberOfItemsInWatch GaugeMetric
}

// ReflectorMetricsProvider defines an interface for creating metrics that track reflector operations.
type ReflectorMetricsProvider interface {
	NewListsMetric(name string, resourceType string) CounterMetric
	NewListDurationMetric(name string, resourceType string) HistogramMetric
	NewItemsInListMetric(name string, resourceType string) GaugeMetric
	NewLastResourceVersionMetric(name string, resourceType string) GaugeMetric

	NewWatchesMetric(name string, resourceType string) CounterMetric
	NewShortWatchesMetric(name string, resourceType string) CounterMetric
	NewWatchDurationMetric(name string, resourceType string) HistogramMetric
	NewItemsInWatchMetric(name string, resourceType string) GaugeMetric
}

type noopMetricsProvider struct{}

func (noopMetricsProvider) NewListsMetric(name string, resourceType string) CounterMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewListDurationMetric(name string, resourceType string) HistogramMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewItemsInListMetric(name string, resourceType string) GaugeMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewWatchesMetric(name string, resourceType string) CounterMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewShortWatchesMetric(name string, resourceType string) CounterMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewWatchDurationMetric(name string, resourceType string) HistogramMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewItemsInWatchMetric(name string, resourceType string) GaugeMetric {
	return noopMetric{}
}
func (noopMetricsProvider) NewLastResourceVersionMetric(name string, resourceType string) GaugeMetric {
	return noopMetric{}
}

var globalReflectorMetricsProvider ReflectorMetricsProvider = noopMetricsProvider{}
var setGlobalReflectorMetricsProvider sync.Once

// newReflectorMetrics creates a new reflectorMetrics object with the given name and resource type.
// It uses the provided metricsProvider to create individual metrics, or falls back to the global
// provider if none is specified. Returns nil if either name or resourceType is empty.
func newReflectorMetrics(name string, resourceType string, metricsProvider ReflectorMetricsProvider) *reflectorMetrics {
	var ret *reflectorMetrics
	if name == "" || resourceType == "" {
		return ret
	}

	if metricsProvider == nil {
		metricsProvider = globalReflectorMetricsProvider
	}

	return &reflectorMetrics{
		clock:                &clock.RealClock{},
		numberOfLists:        metricsProvider.NewListsMetric(name, resourceType),
		listDuration:         metricsProvider.NewListDurationMetric(name, resourceType),
		numberOfItemsInList:  metricsProvider.NewItemsInListMetric(name, resourceType),
		numberOfWatches:      metricsProvider.NewWatchesMetric(name, resourceType),
		numberOfShortWatches: metricsProvider.NewShortWatchesMetric(name, resourceType),
		watchDuration:        metricsProvider.NewWatchDurationMetric(name, resourceType),
		numberOfItemsInWatch: metricsProvider.NewItemsInWatchMetric(name, resourceType),
		lastResourceVersion:  metricsProvider.NewLastResourceVersionMetric(name, resourceType),
	}
}

// SetReflectorMetricsProvider sets the metrics provider.
func SetReflectorMetricsProvider(metricsProvider ReflectorMetricsProvider) {
	setGlobalReflectorMetricsProvider.Do(func() {
		globalReflectorMetricsProvider = metricsProvider
	})
}
