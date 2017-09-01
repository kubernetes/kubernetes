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

// Package prometheus sets the cache DefaultMetricsFactory to produce
// prometheus metrics. To use this package, you just have to import it.
package prometheus

import (
	"k8s.io/client-go/tools/cache"

	"github.com/prometheus/client_golang/prometheus"
)

const reflectorSubsystem = "reflector"

var (
	listsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: reflectorSubsystem,
		Name:      "lists_total",
		Help:      "Total number of API lists done by the reflectors",
	}, []string{"name"})

	listsDuration = prometheus.NewSummaryVec(prometheus.SummaryOpts{
		Subsystem: reflectorSubsystem,
		Name:      "list_duration_seconds",
		Help:      "How long an API list takes to return and decode for the reflectors",
	}, []string{"name"})

	itemsPerList = prometheus.NewSummaryVec(prometheus.SummaryOpts{
		Subsystem: reflectorSubsystem,
		Name:      "items_per_list",
		Help:      "How many items an API list returns to the reflectors",
	}, []string{"name"})

	watchesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: reflectorSubsystem,
		Name:      "watches_total",
		Help:      "Total number of API watches done by the reflectors",
	}, []string{"name"})

	shortWatchesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: reflectorSubsystem,
		Name:      "short_watches_total",
		Help:      "Total number of short API watches done by the reflectors",
	}, []string{"name"})

	watchDuration = prometheus.NewSummaryVec(prometheus.SummaryOpts{
		Subsystem: reflectorSubsystem,
		Name:      "watch_duration_seconds",
		Help:      "How long an API watch takes to return and decode for the reflectors",
	}, []string{"name"})

	itemsPerWatch = prometheus.NewSummaryVec(prometheus.SummaryOpts{
		Subsystem: reflectorSubsystem,
		Name:      "items_per_watch",
		Help:      "How many items an API watch returns to the reflectors",
	}, []string{"name"})
)

func init() {
	prometheus.MustRegister(listsTotal)
	prometheus.MustRegister(listsDuration)
	prometheus.MustRegister(itemsPerList)
	prometheus.MustRegister(watchesTotal)
	prometheus.MustRegister(shortWatchesTotal)
	prometheus.MustRegister(watchDuration)
	prometheus.MustRegister(itemsPerWatch)

	cache.SetReflectorMetricsProvider(prometheusMetricsProvider{})
}

type prometheusMetricsProvider struct{}

func (prometheusMetricsProvider) NewListsMetric(name string) cache.CounterMetric {
	return listsTotal.WithLabelValues(name)
}

// use summary to get averages and percentiles
func (prometheusMetricsProvider) NewListDurationMetric(name string) cache.SummaryMetric {
	return listsDuration.WithLabelValues(name)
}

// use summary to get averages and percentiles
func (prometheusMetricsProvider) NewItemsInListMetric(name string) cache.SummaryMetric {
	return itemsPerList.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewWatchesMetric(name string) cache.CounterMetric {
	return watchesTotal.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewShortWatchesMetric(name string) cache.CounterMetric {
	return shortWatchesTotal.WithLabelValues(name)
}

// use summary to get averages and percentiles
func (prometheusMetricsProvider) NewWatchDurationMetric(name string) cache.SummaryMetric {
	return watchDuration.WithLabelValues(name)
}

// use summary to get averages and percentiles
func (prometheusMetricsProvider) NewItemsInWatchMetric(name string) cache.SummaryMetric {
	return itemsPerWatch.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewLastResourceVersionMetric(name string) cache.GaugeMetric {
	rv := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "last_resource_version",
		Help:      "last resource version seen for the reflectors",
	})
	prometheus.MustRegister(rv)
	return rv
}
