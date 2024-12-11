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
	}, []string{"name", "resource_type"})

	listsDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ListsDurationKey,
		Help:           "How long an API list takes to return and decode for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
		Buckets: []float64{0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
			4, 5, 6, 8, 10, 15, 20, 30, 45, 60},
	}, []string{"name", "resource_type"})

	itemsPerList = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ListedItemsKey,
		Help:           "Number of items returned by an API list for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "resource_type"})

	watchesTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           WatchesTotalKey,
		Help:           "Total number of API watches done by the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "resource_type"})

	shortWatchesTotal = k8smetrics.NewCounterVec(&k8smetrics.CounterOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           ShortWatchesTotalKey,
		Help:           "Total number of short API watches done by the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "resource_type"})

	watchDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           WatchDurationKey,
		Help:           "How long an API watch takes to return and decode for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
		Buckets:        []float64{0.05, 0.1, 0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 300, 600},
	}, []string{"name", "resource_type"})

	itemsPerWatch = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           WatchedObjectsKey,
		Help:           "Number of items returned by an API watch for the reflectors",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "resource_type"})

	lastResourceVersion = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      ReflectorSubsystem,
		Name:           LastResourceVersionKey,
		Help:           "Last resource version seen by the reflector",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "resource_type"})

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

func (reflectorMetricsProvider) NewListsMetric(name string, resourceType string) cache.CounterMetric {
	return listsTotal.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewListDurationMetric(name string, resourceType string) cache.HistogramMetric {
	return listsDuration.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewItemsInListMetric(name string, resourceType string) cache.GaugeMetric {
	return itemsPerList.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewWatchesMetric(name string, resourceType string) cache.CounterMetric {
	return watchesTotal.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewShortWatchesMetric(name string, resourceType string) cache.CounterMetric {
	return shortWatchesTotal.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewWatchDurationMetric(name string, resourceType string) cache.HistogramMetric {
	return watchDuration.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewItemsInWatchMetric(name string, resourceType string) cache.GaugeMetric {
	return itemsPerWatch.WithLabelValues(name, resourceType)
}

func (reflectorMetricsProvider) NewLastResourceVersionMetric(name string, resourceType string) cache.GaugeMetric {
	return lastResourceVersion.WithLabelValues(name, resourceType)
}
