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

type noopMetric struct{}

func (noopMetric) Inc()            {}
func (noopMetric) Dec()            {}
func (noopMetric) Observe(float64) {}
func (noopMetric) Set(float64)     {}

type reflectorMetrics struct {
	numberOfLists       CounterMetric
	listDuration        SummaryMetric
	numberOfItemsInList SummaryMetric

	numberOfWatches      CounterMetric
	numberOfShortWatches CounterMetric
	watchDuration        SummaryMetric
	numberOfItemsInWatch SummaryMetric

	lastResourceVersion GaugeMetric
}

// MetricsProvider generates various metrics used by the reflector.
type MetricsProvider interface {
	NewListsMetric(name string) CounterMetric
	NewListDurationMetric(name string) SummaryMetric
	NewItemsInListMetric(name string) SummaryMetric

	NewWatchesMetric(name string) CounterMetric
	NewShortWatchesMetric(name string) CounterMetric
	NewWatchDurationMetric(name string) SummaryMetric
	NewItemsInWatchMetric(name string) SummaryMetric

	NewLastResourceVersionMetric(name string) GaugeMetric
}

type noopMetricsProvider struct{}

func (noopMetricsProvider) NewListsMetric(name string) CounterMetric         { return noopMetric{} }
func (noopMetricsProvider) NewListDurationMetric(name string) SummaryMetric  { return noopMetric{} }
func (noopMetricsProvider) NewItemsInListMetric(name string) SummaryMetric   { return noopMetric{} }
func (noopMetricsProvider) NewWatchesMetric(name string) CounterMetric       { return noopMetric{} }
func (noopMetricsProvider) NewShortWatchesMetric(name string) CounterMetric  { return noopMetric{} }
func (noopMetricsProvider) NewWatchDurationMetric(name string) SummaryMetric { return noopMetric{} }
func (noopMetricsProvider) NewItemsInWatchMetric(name string) SummaryMetric  { return noopMetric{} }
func (noopMetricsProvider) NewLastResourceVersionMetric(name string) GaugeMetric {
	return noopMetric{}
}

var metricsFactory = struct {
	metricsProvider MetricsProvider
	setProviders    sync.Once
}{
	metricsProvider: noopMetricsProvider{},
}

// SetReflectorMetricsProvider sets the metrics provider
func SetReflectorMetricsProvider(metricsProvider MetricsProvider) {
	metricsFactory.setProviders.Do(func() {
		metricsFactory.metricsProvider = metricsProvider
	})
}
