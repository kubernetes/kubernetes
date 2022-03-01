/*
Copyright 2020 The Kubernetes Authors.

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

package cacher

import (
	"k8s.io/component-base/metrics"
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
	initCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "apiserver_init_events_total",
			Help:           "Counter of init events processed in watch cache broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	eventsCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "events_dispatched_total",
			Help:           "Counter of events dispatched in watch cache broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	terminatedWatchersCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "apiserver_terminated_watchers_total",
			Help:           "Counter of watchers closed due to unresponsiveness broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheCapacityIncreaseTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "watch_cache_capacity_increase_total",
			Help:           "Total number of watch cache capacity increase events broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheCapacityDecreaseTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "watch_cache_capacity_decrease_total",
			Help:           "Total number of watch cache capacity decrease events broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheCapacity = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name:           "watch_cache_capacity",
			Help:           "Total capacity of watch cache broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	watchCacheInitializations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "watch_cache_initializations_total",
			Help:           "Counter of watch cache initializations broken by resource type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)
)

func init() {
	legacyregistry.MustRegister(initCounter)
	legacyregistry.MustRegister(eventsCounter)
	legacyregistry.MustRegister(terminatedWatchersCounter)
	legacyregistry.MustRegister(watchCacheCapacityIncreaseTotal)
	legacyregistry.MustRegister(watchCacheCapacityDecreaseTotal)
	legacyregistry.MustRegister(watchCacheCapacity)
	legacyregistry.MustRegister(watchCacheInitializations)
}

// recordsWatchCacheCapacityChange record watchCache capacity resize(increase or decrease) operations.
func recordsWatchCacheCapacityChange(objType string, old, new int) {
	watchCacheCapacity.WithLabelValues(objType).Set(float64(new))
	if old < new {
		watchCacheCapacityIncreaseTotal.WithLabelValues(objType).Inc()
		return
	}
	watchCacheCapacityDecreaseTotal.WithLabelValues(objType).Inc()
}
