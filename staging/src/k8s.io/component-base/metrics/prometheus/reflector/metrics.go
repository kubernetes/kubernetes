/*
Copyright 2025 The Kubernetes Authors.

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
	"k8s.io/client-go/tools/cache"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	ReflectorSubsystem     = "reflector"
	ListsTotalKey          = "lists_total"
	ListsDurationKey       = "list_duration_seconds"
	ListedItemsKey         = "listed_items"
	WatchesTotalKey        = "watches_total"
	ShortWatchesTotalKey   = "short_watches_total"
	WatchDurationKey       = "watch_duration_seconds"
	WatchedObjectsKey      = "watched_objects"
	LastResourceVersionKey = "last_resource_version"
)

var (
	listsTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ListsTotalKey,
		Help:           "Total number of API lists done by the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "group", "resource"})

	listsDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ListsDurationKey,
		Help:           "How long an API list takes to return and decode for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
		Buckets: []float64{0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
			4, 5, 6, 8, 10, 15, 20, 30, 45, 60},
	}, []string{"name", "group", "resource"})

	itemsPerList = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ListedItemsKey,
		Help:           "Number of items returned by an API list for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "group", "resource"})

	watchesTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           WatchesTotalKey,
		Help:           "Total number of API watches done by the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "group", "resource"})

	shortWatchesTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ShortWatchesTotalKey,
		Help:           "Total number of short API watches done by the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "group", "resource"})

	watchDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           WatchDurationKey,
		Help:           "How long an API watch takes to return and decode for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
		Buckets: []float64{0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
			4, 5, 6, 8, 10, 15, 20, 30, 45, 60},
	}, []string{"name", "group", "resource"})

	itemsPerWatch = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           WatchedObjectsKey,
		Help:           "Number of items returned by an API watch for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "group", "resource"})

	lastResourceVersion = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           LastResourceVersionKey,
		Help:           "Last resource version seen by the reflector",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "group", "resource"})

	metrics = []k8smetrics.Registerable{
		listsTotal, listsDuration, itemsPerList, watchesTotal, shortWatchesTotal, watchDuration, itemsPerWatch, lastResourceVersion,
	}
)

type reflectorMetricsProvider struct{}

// Register registers reflector metrics to the provided registry.
func Register() {
	for _, m := range metrics {
		legacyregistry.MustRegister(m)
	}
	cache.SetReflectorMetricsProvider(reflectorMetricsProvider{})
}

func (reflectorMetricsProvider) NewListsMetric(name string, group string, resource string) cache.CounterMetric {
	return listsTotal.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewListDurationMetric(name string, group string, resource string) cache.HistogramMetric {
	return listsDuration.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewItemsInListMetric(name string, group string, resource string) cache.GaugeMetric {
	return itemsPerList.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewWatchesMetric(name string, group string, resource string) cache.CounterMetric {
	return watchesTotal.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewShortWatchesMetric(name string, group string, resource string) cache.CounterMetric {
	return shortWatchesTotal.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewWatchDurationMetric(name string, group string, resource string) cache.HistogramMetric {
	return watchDuration.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewItemsInWatchMetric(name string, group string, resource string) cache.GaugeMetric {
	return itemsPerWatch.WithLabelValues(name, group, resource)
}

func (reflectorMetricsProvider) NewLastResourceVersionMetric(name string, group string, resource string) cache.GaugeMetric {
	return lastResourceVersion.WithLabelValues(name, group, resource)
}
