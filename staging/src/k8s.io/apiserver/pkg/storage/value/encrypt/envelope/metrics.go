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

package envelope

import (
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace        = "apiserver"
	subsystem        = "envelope_encryption"
	fromStorageLabel = "from_storage"
	toStorageLabel   = "to_storage"
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
	lockLastFromStorage sync.Mutex
	lockLastToStorage   sync.Mutex

	lastFromStorage time.Time
	lastToStorage   time.Time

	dekCacheFillPercent = metrics.NewGauge(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dek_cache_fill_percent",
			Help:           "Percent of the cache slots currently occupied by cached DEKs.",
			StabilityLevel: metrics.ALPHA,
		},
	)

	dekCacheInterArrivals = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dek_cache_inter_arrival_time_seconds",
			Help:           "Time (in seconds) of inter arrival of transformation requests.",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(60, 2, 10),
		},
		[]string{"transformation_type"},
	)
)

var registerMetricsFunc sync.Once

func registerMetrics() {
	registerMetricsFunc.Do(func() {
		legacyregistry.MustRegister(dekCacheFillPercent)
		legacyregistry.MustRegister(dekCacheInterArrivals)
	})
}

func recordArrival(transformationType string, start time.Time) {
	switch transformationType {
	case fromStorageLabel:
		lockLastFromStorage.Lock()
		defer lockLastFromStorage.Unlock()

		if lastFromStorage.IsZero() {
			lastFromStorage = start
		}
		dekCacheInterArrivals.WithLabelValues(transformationType).Observe(start.Sub(lastFromStorage).Seconds())
		lastFromStorage = start
	case toStorageLabel:
		lockLastToStorage.Lock()
		defer lockLastToStorage.Unlock()

		if lastToStorage.IsZero() {
			lastToStorage = start
		}
		dekCacheInterArrivals.WithLabelValues(transformationType).Observe(start.Sub(lastToStorage).Seconds())
		lastToStorage = start
	}
}
