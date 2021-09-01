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
	"strconv"
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
			Name:           "apiserver_list_cache_count",
			Help:           "Number of LIST requests served from watch cache, split by path_prefix, index_used, and predicate_complexity",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"path_prefix", "index_used", "predicate_complexity"},
	)
	listCacheNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_list_cache_num_fetched",
			Help:           "Number of objects read from watch cache in the course of serving a LIST request, split by path_prefix and index_used",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"path_prefix", "index_used"},
	)
	listCacheNumEvals = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_list_cache_num_selector_evals",
			Help:           "Number of objects tested in the course of serving a LIST request from watch cache, split by path_prefix and predicate_complexity",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"path_prefix", "predicate_complexity"},
	)
	listCacheNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_list_cache_num_returned",
			Help:           "Number of objects returned for a LIST request from watch cache, split by path_prefix",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"path_prefix"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(listCacheCount)
		legacyregistry.MustRegister(listCacheNumFetched)
		legacyregistry.MustRegister(listCacheNumEvals)
		legacyregistry.MustRegister(listCacheNumReturned)
	})
}

// RecordListCacheMetrics notes various metrics of the cost to serve a LIST request
func RecordListCacheMetrics(pathPrefix, indexUsed string, numFetched, predicateComplexity, numEvals, numReturned int) {
	pcStr := strconv.FormatInt(int64(predicateComplexity), 10)
	listCacheCount.WithLabelValues(pathPrefix, indexUsed, pcStr).Inc()
	listCacheNumFetched.WithLabelValues(pathPrefix, indexUsed).Add(float64(numFetched))
	listCacheNumEvals.WithLabelValues(pathPrefix, pcStr).Add(float64(numEvals))
	listCacheNumReturned.WithLabelValues(pathPrefix).Add(float64(numReturned))
}
