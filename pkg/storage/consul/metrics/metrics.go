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
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	cacheHitCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "consul_helper_cache_hit_count",
			Help: "Counter of consul helper cache hits.",
		},
	)
	cacheMissCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "consul_helper_cache_miss_count",
			Help: "Counter of consul helper cache miss.",
		},
	)
	cacheEntryCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "consul_helper_cache_entry_count",
			Help: "Counter of consul helper cache entries. This can be different from consul_helper_cache_miss_count " +
				"because two concurrent threads can miss the cache and generate the same entry twice.",
		},
	)
	cacheGetLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Name: "consul_request_cache_get_latencies_summary",
			Help: "Latency in microseconds of getting an object from consul cache",
		},
	)
	cacheAddLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Name: "consul_request_cache_add_latencies_summary",
			Help: "Latency in microseconds of adding an object to consul cache",
		},
	)
	consulRequestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "consul_request_latencies_summary",
			Help: "Consul request latency summary in microseconds for each operation and object type.",
		},
		[]string{"operation", "type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(cacheHitCounter)
		prometheus.MustRegister(cacheMissCounter)
		prometheus.MustRegister(cacheEntryCounter)
		prometheus.MustRegister(cacheAddLatency)
		prometheus.MustRegister(cacheGetLatency)
		prometheus.MustRegister(consulRequestLatenciesSummary)
	})
}

func RecordConsulRequestLatency(verb, resource string, startTime time.Time) {
	consulRequestLatenciesSummary.WithLabelValues(verb, resource).Observe(float64(time.Since(startTime) / time.Microsecond))
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

func Reset() {
	cacheHitCounter.Set(0)
	cacheMissCounter.Set(0)
	cacheEntryCounter.Set(0)
	// TODO: Reset cacheAddLatency.
	// TODO: Reset cacheGetLatency.
	consulRequestLatenciesSummary.Reset()
}
