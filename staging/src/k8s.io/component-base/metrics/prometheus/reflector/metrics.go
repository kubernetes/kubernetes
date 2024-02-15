/*
Copyright 2024 The Kubernetes Authors.

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

package reflector

import (
	"k8s.io/client-go/features"
	"k8s.io/client-go/tools/cache"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	ReflectorSubsystem     = "reflector"
	ListsTotalKey          = "lists_total"
	ListsDurationKey       = "list_duration_seconds"
	ItemsPerListKey        = "items_per_list"
	WatchListsTotalKey     = "watchLists_total"
	WatchesTotalKey        = "watches_total"
	ShortWatchesTotalKey   = "short_watches_total"
	WatchDurationKey       = "watch_duration_seconds"
	ItemsPerWatchKey       = "items_per_watch"
	LastResourceVersionKey = "last_resource_version"
)

var (
	listsTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem: ReflectorSubsystem,
		Name:      ListsTotalKey,
		Help:      "Total number of API lists done by the reflectors",
	}, []string{"name"})
	listsDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem: ReflectorSubsystem,
		Name:      ListsDurationKey,
		Help:      "How long an API list takes to return and decode for the reflectors",
	}, []string{"name"})
	itemsPerList = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem: ReflectorSubsystem,
		Name:      ItemsPerListKey,
		Help:      "How many items an API list returns to the reflectors",
	}, []string{"name"})
	watchListsTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem: ReflectorSubsystem,
		Name:      WatchListsTotalKey,
		Help:      "Total number of API initial watchLists done by the reflectors",
	}, []string{"name"})
	watchesTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem: ReflectorSubsystem,
		Name:      WatchesTotalKey,
		Help:      "Total number of API watches done by the reflectors",
	}, []string{"name"})
	shortWatchesTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem: ReflectorSubsystem,
		Name:      ShortWatchesTotalKey,
		Help:      "Total number of short API watches done by the reflectors",
	}, []string{"name"})
	watchDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem: ReflectorSubsystem,
		Name:      WatchDurationKey,
		Help:      "How long an API watch takes to return and decode for the reflectors",
	}, []string{"name"})
	itemsPerWatch = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem: ReflectorSubsystem,
		Name:      ItemsPerWatchKey,
		Help:      "How many items an API watch returns to the reflectors",
	}, []string{"name"})
	lastResourceVersion = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem: ReflectorSubsystem,
		Name:      LastResourceVersionKey,
		Help:      "Last resource version seen for the reflectors",
	}, []string{"name"})
	metrics = []k8smetrics.Registerable{
		listsTotal, watchListsTotal, listsDuration, itemsPerList, watchesTotal, shortWatchesTotal, watchDuration, itemsPerWatch, lastResourceVersion,
	}
)

func init() {
	for _, m := range metrics {
		legacyregistry.MustRegister(m)
	}
	// check InformerMetrics is enabled in envVarFeatureGatesEnabled.
	if features.FeatureGates().Enabled(features.InformerMetrics) {
		LoadReflectorMetrics()
	}
}

// LoadReflectorMetrics is called when InformerMetrics is enabled
// in kube's feature gate(i.e. command line flag). It should be called
// before new Reflector.
func LoadReflectorMetrics() {
	cache.SetReflectorMetricsProvider(reflectorMetricsProvider{})
}

type reflectorMetricsProvider struct{}

func (reflectorMetricsProvider) NewListsMetric(name string) cache.CounterMetric {
	return listsTotal.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteListsMetric(name string) {
	listsTotal.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewListDurationMetric(name string) cache.HistogramMetric {
	return listsDuration.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteListDurationMetric(name string) {
	listsDuration.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewItemsInListMetric(name string) cache.HistogramMetric {
	return itemsPerList.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteItemsInListMetric(name string) {
	itemsPerList.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewWatchListsMetrics(name string) cache.CounterMetric {
	return watchListsTotal.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteWatchListsMetrics(name string) {
	watchListsTotal.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewWatchesMetric(name string) cache.CounterMetric {
	return watchesTotal.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteWatchesMetric(name string) {
	watchesTotal.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewShortWatchesMetric(name string) cache.CounterMetric {
	return shortWatchesTotal.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteShortWatchesMetric(name string) {
	shortWatchesTotal.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewWatchDurationMetric(name string) cache.HistogramMetric {
	return watchDuration.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteWatchDurationMetric(name string) {
	watchDuration.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewItemsInWatchMetric(name string) cache.HistogramMetric {
	return itemsPerWatch.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteItemsInWatchMetric(name string) {
	itemsPerWatch.Delete(map[string]string{"name": name})
}
func (reflectorMetricsProvider) NewLastResourceVersionMetric(name string) cache.GaugeMetric {
	return lastResourceVersion.WithLabelValues(name)
}
func (reflectorMetricsProvider) DeleteLastResourceVersionMetric(name string) {
	lastResourceVersion.Delete(map[string]string{"name": name})
}
