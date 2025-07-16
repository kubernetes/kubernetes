/*
Copyright 2021 The Kubernetes Authors.

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

package metrics

import (
	"sync"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "watch_cache"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	listCacheCount = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "cache_list_total",
			Help:           "Number of LIST requests served from watch cache",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource_prefix", "index"},
	)
	listCacheNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "cache_list_fetched_objects_total",
			Help:           "Number of objects read from watch cache in the course of serving a LIST request",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource_prefix", "index"},
	)
	listCacheNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "cache_list_returned_objects_total",
			Help:           "Number of objects returned for a LIST request from watch cache",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource_prefix"},
	)
	InitCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "init_events_total",
			Help:           "Counter of init events processed in watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	EventsReceivedCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "events_received_total",
			Help:           "Counter of events received in watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	EventsCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "events_dispatched_total",
			Help:           "Counter of events dispatched in watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	TerminatedWatchersCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "terminated_watchers_total",
			Help:           "Counter of watchers closed due to unresponsiveness broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheResourceVersion = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "resource_version",
			Help:           "Current resource version of watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheCapacityIncreaseTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "capacity_increase_total",
			Help:           "Total number of watch cache capacity increase events broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheCapacityDecreaseTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "capacity_decrease_total",
			Help:           "Total number of watch cache capacity decrease events broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	WatchCacheCapacity = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "capacity",
			Help:           "Total capacity of watch cache broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	WatchCacheInitializations = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "initializations_total",
			Help:           "Counter of watch cache initializations broken by resource type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)

	WatchCacheReadWait = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "read_wait_seconds",
			Help:           "Histogram of time spent waiting for a watch cache to become fresh.",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3},
		}, []string{"resource"})

	ConsistentReadTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "consistent_read_total",
			Help:           "Counter for consistent reads from cache.",
			StabilityLevel: compbasemetrics.ALPHA,
		}, []string{"resource", "success", "fallback"})

	StorageConsistencyCheckTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace:      namespace,
			Name:           "storage_consistency_checks_total",
			Help:           "Counter for status of consistency checks between etcd and watch cache",
			StabilityLevel: compbasemetrics.INTERNAL,
		}, []string{"resource", "status"})
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(listCacheCount)
		legacyregistry.MustRegister(listCacheNumFetched)
		legacyregistry.MustRegister(listCacheNumReturned)
		legacyregistry.MustRegister(InitCounter)
		legacyregistry.MustRegister(EventsReceivedCounter)
		legacyregistry.MustRegister(EventsCounter)
		legacyregistry.MustRegister(TerminatedWatchersCounter)
		legacyregistry.MustRegister(watchCacheResourceVersion)
		legacyregistry.MustRegister(watchCacheCapacityIncreaseTotal)
		legacyregistry.MustRegister(watchCacheCapacityDecreaseTotal)
		legacyregistry.MustRegister(WatchCacheCapacity)
		legacyregistry.MustRegister(WatchCacheInitializations)
		legacyregistry.MustRegister(WatchCacheReadWait)
		legacyregistry.MustRegister(ConsistentReadTotal)
		legacyregistry.MustRegister(StorageConsistencyCheckTotal)
	})
}

// RecordListCacheMetrics notes various metrics of the cost to serve a LIST request
func RecordListCacheMetrics(resourcePrefix, indexName string, numFetched, numReturned int) {
	listCacheCount.WithLabelValues(resourcePrefix, indexName).Inc()
	listCacheNumFetched.WithLabelValues(resourcePrefix, indexName).Add(float64(numFetched))
	listCacheNumReturned.WithLabelValues(resourcePrefix).Add(float64(numReturned))
}

// RecordResourceVersion sets the current resource version for a given resource type.
func RecordResourceVersion(resourcePrefix string, resourceVersion uint64) {
	watchCacheResourceVersion.WithLabelValues(resourcePrefix).Set(float64(resourceVersion))
}

// RecordsWatchCacheCapacityChange record watchCache capacity resize(increase or decrease) operations.
func RecordsWatchCacheCapacityChange(objType string, old, new int) {
	WatchCacheCapacity.WithLabelValues(objType).Set(float64(new))
	if old < new {
		watchCacheCapacityIncreaseTotal.WithLabelValues(objType).Inc()
		return
	}
	watchCacheCapacityDecreaseTotal.WithLabelValues(objType).Inc()
}
