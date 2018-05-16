/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apiserver/pkg/metrics"
)

var (
	cacheHitCounter = metrics.NewResettableCounter(prometheus.CounterOpts{
		Name: "etcd_helper_cache_hit_count",
		Help: "Counter of etcd helper cache hits.",
	})
	cacheMissCounter = metrics.NewResettableCounter(prometheus.CounterOpts{
		Name: "etcd_helper_cache_miss_count",
		Help: "Counter of etcd helper cache miss.",
	})
	cacheEntryCounter = metrics.NewResettableCounter(prometheus.CounterOpts{
		Name: "etcd_helper_cache_entry_count",
		Help: "Counter of etcd helper cache entries. This can be different from etcd_helper_cache_miss_count " +
			"because two concurrent threads can miss the cache and generate the same entry twice.",
	})
	cacheGetLatency = metrics.NewResettableSummary(
		prometheus.SummaryOpts{
			Name: "etcd_request_cache_get_latencies_summary",
			Help: "Latency in microseconds of getting an object from etcd cache",
		},
	)
	cacheAddLatency = metrics.NewResettableSummary(
		prometheus.SummaryOpts{
			Name: "etcd_request_cache_add_latencies_summary",
			Help: "Latency in microseconds of adding an object to etcd cache",
		},
	)
	etcdRequestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "etcd_request_latencies_summary",
			Help: "Etcd request latency summary in microseconds for each operation and object type.",
		},
		[]string{"operation", "type"},
	)
	objectCounts = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "etcd_object_counts",
			Help: "Number of stored objects at the time of last check split by kind.",
		},
		[]string{"resource"},
	)
)

// Metrics returns etcd storage metrics.
func Metrics() metrics.Group {
	return metrics.NewGroup(
		cacheHitCounter,
		cacheMissCounter,
		cacheEntryCounter,
		cacheAddLatency,
		cacheGetLatency,
		etcdRequestLatenciesSummary,
		objectCounts,
	)
}

func UpdateObjectCount(resourcePrefix string, count int64) {
	objectCounts.WithLabelValues(resourcePrefix).Set(float64(count))
}

func RecordEtcdRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatenciesSummary.WithLabelValues(verb, resource).Observe(float64(time.Since(startTime) / time.Microsecond))
}

func ObserveGetCache(startTime time.Time) {
	cacheGetLatency.Observe(float64(time.Since(startTime) / time.Microsecond))
}

func ObserveAddCache(startTime time.Time) {
	cacheAddLatency.Observe(float64(time.Since(startTime) / time.Microsecond))
}

func ObserveCacheHit() {
	cacheHitCounter.Inc()
}

func ObserveCacheMiss() {
	cacheMissCounter.Inc()
}

func ObserveNewEntry() {
	cacheEntryCounter.Inc()
}
