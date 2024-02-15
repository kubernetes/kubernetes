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

// HistogramMetric counts individual observations.
type HistogramMetric interface {
	Observe(float64)
}

type noopMetric struct{}

func (noopMetric) Inc()            {}
func (noopMetric) Dec()            {}
func (noopMetric) Observe(float64) {}
func (noopMetric) Set(float64)     {}

var noop = noopMetric{}

type reflectorMetrics struct {
	name                 string
	numberOfLists        CounterMetric
	listDuration         HistogramMetric
	numberOfItemsInList  HistogramMetric
	numberOfWatchLists   CounterMetric
	numberOfWatches      CounterMetric
	numberOfShortWatches CounterMetric
	watchDuration        HistogramMetric
	numberOfItemsInWatch HistogramMetric
	lastResourceVersion  GaugeMetric
}

func (r *reflectorMetrics) numberOfListsInc() {
	if r == nil {
		return
	}
	r.numberOfLists.Inc()
}

func (r *reflectorMetrics) listDurationObserve(value float64) {
	if r == nil {
		return
	}
	r.listDuration.Observe(value)
}

func (r *reflectorMetrics) numberOfItemsInListObserve(value float64) {
	if r == nil {
		return
	}
	r.numberOfItemsInList.Observe(value)
}

func (r *reflectorMetrics) numberOfWatchListsInc() {
	if r == nil {
		return
	}
	r.numberOfWatchLists.Inc()
}

func (r *reflectorMetrics) numberOfWatchesInc() {
	if r == nil {
		return
	}
	r.numberOfWatches.Inc()
}

func (r *reflectorMetrics) numberOfShortWatchesInc() {
	if r == nil {
		return
	}
	r.numberOfShortWatches.Inc()
}

func (r *reflectorMetrics) watchDurationObserve(value float64) {
	if r == nil {
		return
	}
	r.watchDuration.Observe(value)
}

func (r *reflectorMetrics) numberOfItemsInWatchObserve(value float64) {
	if r == nil {
		return
	}
	r.numberOfItemsInWatch.Observe(value)
}

func (r *reflectorMetrics) setLastResourceVersion(value float64) {
	if r == nil {
		return
	}
	r.lastResourceVersion.Set(value)
}

// loadMetrics init reflector metrics
func (r *reflectorMetrics) loadMetrics() {
	if r == nil {
		return
	}
	metricsProvider := metricsFactory.provider()
	r.numberOfLists = metricsProvider.NewListsMetric(r.name)
	r.listDuration = metricsProvider.NewListDurationMetric(r.name)
	r.numberOfItemsInList = metricsProvider.NewItemsInListMetric(r.name)
	r.numberOfWatchLists = metricsProvider.NewWatchListsMetrics(r.name)
	r.numberOfWatches = metricsProvider.NewWatchesMetric(r.name)
	r.numberOfShortWatches = metricsProvider.NewShortWatchesMetric(r.name)
	r.watchDuration = metricsProvider.NewWatchDurationMetric(r.name)
	r.numberOfItemsInWatch = metricsProvider.NewItemsInWatchMetric(r.name)
	r.lastResourceVersion = metricsProvider.NewLastResourceVersionMetric(r.name)
}

// cleanUpMetrics delete reflector metrics.
// To free up memory and avoid nil, reset to noop.
func (r *reflectorMetrics) cleanUpMetrics() {
	if r == nil {
		return
	}
	metricsProvider := metricsFactory.provider()

	// free up memory used by metrics
	r.numberOfLists = noop
	metricsProvider.DeleteListsMetric(r.name)

	r.listDuration = noop
	metricsProvider.DeleteListDurationMetric(r.name)

	r.numberOfItemsInList = noop
	metricsProvider.DeleteItemsInListMetric(r.name)

	r.numberOfWatchLists = noop
	metricsProvider.DeleteWatchListsMetrics(r.name)

	r.numberOfWatches = noop
	metricsProvider.DeleteWatchesMetric(r.name)

	r.numberOfShortWatches = noop
	metricsProvider.DeleteShortWatchesMetric(r.name)

	r.watchDuration = noop
	metricsProvider.DeleteWatchDurationMetric(r.name)

	r.numberOfItemsInWatch = noop
	metricsProvider.DeleteItemsInWatchMetric(r.name)

	r.lastResourceVersion = noop
	metricsProvider.DeleteLastResourceVersionMetric(r.name)
}

// MetricsProvider generates various metrics used by the reflector.
type MetricsProvider interface {
	NewListsMetric(name string) CounterMetric
	DeleteListsMetric(name string)

	NewListDurationMetric(name string) HistogramMetric
	DeleteListDurationMetric(name string)

	NewItemsInListMetric(name string) HistogramMetric
	DeleteItemsInListMetric(name string)

	NewWatchListsMetrics(name string) CounterMetric
	DeleteWatchListsMetrics(name string)

	NewWatchesMetric(name string) CounterMetric
	DeleteWatchesMetric(name string)

	NewShortWatchesMetric(name string) CounterMetric
	DeleteShortWatchesMetric(name string)

	NewWatchDurationMetric(name string) HistogramMetric
	DeleteWatchDurationMetric(name string)

	NewItemsInWatchMetric(name string) HistogramMetric
	DeleteItemsInWatchMetric(name string)

	NewLastResourceVersionMetric(name string) GaugeMetric
	DeleteLastResourceVersionMetric(name string)
}

type noopMetricsProvider struct{}

func (noopMetricsProvider) NewListsMetric(name string) CounterMetric             { return noopMetric{} }
func (noopMetricsProvider) DeleteListsMetric(name string)                        {}
func (noopMetricsProvider) NewListDurationMetric(name string) HistogramMetric    { return noopMetric{} }
func (noopMetricsProvider) DeleteListDurationMetric(name string)                 {}
func (noopMetricsProvider) NewItemsInListMetric(name string) HistogramMetric     { return noopMetric{} }
func (noopMetricsProvider) DeleteItemsInListMetric(name string)                  {}
func (noopMetricsProvider) NewWatchListsMetrics(name string) CounterMetric       { return noopMetric{} }
func (noopMetricsProvider) DeleteWatchListsMetrics(name string)                  {}
func (noopMetricsProvider) NewWatchesMetric(name string) CounterMetric           { return noopMetric{} }
func (noopMetricsProvider) DeleteWatchesMetric(name string)                      {}
func (noopMetricsProvider) NewShortWatchesMetric(name string) CounterMetric      { return noopMetric{} }
func (noopMetricsProvider) DeleteShortWatchesMetric(name string)                 {}
func (noopMetricsProvider) NewWatchDurationMetric(name string) HistogramMetric   { return noopMetric{} }
func (noopMetricsProvider) DeleteWatchDurationMetric(name string)                {}
func (noopMetricsProvider) NewItemsInWatchMetric(name string) HistogramMetric    { return noopMetric{} }
func (noopMetricsProvider) DeleteItemsInWatchMetric(name string)                 {}
func (noopMetricsProvider) NewLastResourceVersionMetric(name string) GaugeMetric { return noopMetric{} }
func (noopMetricsProvider) DeleteLastResourceVersionMetric(name string)          {}

type reflectorMetricsFactoryInterface interface {
	setProvider(metricsProvider MetricsProvider)
	provider() MetricsProvider
}

type reflectorMetricsFactory struct {
	metricsProvider MetricsProvider
	onlyOnce        sync.Once
}

func (metricsFactory *reflectorMetricsFactory) setProvider(metricsProvider MetricsProvider) {
	metricsFactory.onlyOnce.Do(func() {
		metricsFactory.metricsProvider = metricsProvider
	})
}

func (metricsFactory *reflectorMetricsFactory) provider() MetricsProvider {
	return metricsFactory.metricsProvider
}

var metricsFactory reflectorMetricsFactoryInterface = &reflectorMetricsFactory{
	metricsProvider: noopMetricsProvider{},
}

func newReflectorMetrics(name string) *reflectorMetrics {
	var ret *reflectorMetrics
	if len(name) == 0 {
		return ret
	}
	r := &reflectorMetrics{
		name: name,
	}
	r.loadMetrics()
	return r
}

// SetReflectorMetricsProvider sets the metrics provider
func SetReflectorMetricsProvider(metricsProvider MetricsProvider) {
	metricsFactory.setProvider(metricsProvider)
}

// SwapReflectorMetricsFactory set the metrics factory to the specified value, and returns the original value.
// it is for testing purposes only and should not be used in production.
func SwapReflectorMetricsFactory(factory reflectorMetricsFactoryInterface) reflectorMetricsFactoryInterface {
	tmp := metricsFactory
	metricsFactory = factory
	return tmp
}
