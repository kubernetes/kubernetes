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

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/20190404-kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	initCounter = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "apiserver_init_events_total",
			Help:           "Counter of init events processed in watchcache broken by resource type.",
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
)

func init() {
	legacyregistry.MustRegister(initCounter)
	legacyregistry.MustRegister(watchCacheCapacityIncreaseTotal)
	legacyregistry.MustRegister(watchCacheCapacityDecreaseTotal)
}

// recordsWatchCacheCapacityChange record watchCache capacity resize(increase or decrease) operations.
func recordsWatchCacheCapacityChange(objType string, old, new int) {
	if old < new {
		watchCacheCapacityIncreaseTotal.WithLabelValues(objType).Inc()
		return
	}
	watchCacheCapacityDecreaseTotal.WithLabelValues(objType).Inc()
}
