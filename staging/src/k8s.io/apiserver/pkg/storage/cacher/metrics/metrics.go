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
			Name:           "apiserver_cache_list_total",
			Help:           "Number of LIST requests served from watch cache",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource_prefix", "index"},
	)
	listCacheNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_cache_list_fetched_objects_total",
			Help:           "Number of objects read from watch cache in the course of serving a LIST request",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource_prefix", "index"},
	)
	listCacheNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_cache_list_returned_objects_total",
			Help:           "Number of objects returned for a LIST request from watch cache",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource_prefix"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(listCacheCount)
		legacyregistry.MustRegister(listCacheNumFetched)
		legacyregistry.MustRegister(listCacheNumReturned)
	})
}

// RecordListCacheMetrics notes various metrics of the cost to serve a LIST request
func RecordListCacheMetrics(resourcePrefix, indexName string, numFetched, numReturned int) {
	listCacheCount.WithLabelValues(resourcePrefix, indexName).Inc()
	listCacheNumFetched.WithLabelValues(resourcePrefix, indexName).Add(float64(numFetched))
	listCacheNumReturned.WithLabelValues(resourcePrefix).Add(float64(numReturned))
}
