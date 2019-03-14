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
	cacheHitCounterOpts = prometheus.CounterOpts{
		Name: "etcd_helper_cache_hit_total",
		Help: "Counter of etcd helper cache hits.",
	}
	cacheHitCounter      = prometheus.NewCounter(cacheHitCounterOpts)
	cacheMissCounterOpts = prometheus.CounterOpts{
		Name: "etcd_helper_cache_miss_total",
		Help: "Counter of etcd helper cache miss.",
	}
	cacheMissCounter      = prometheus.NewCounter(cacheMissCounterOpts)
	cacheEntryCounterOpts = prometheus.CounterOpts{
		Name: "etcd_helper_cache_entry_total",
		Help: "Counter of etcd helper cache entries. This can be different from etcd_helper_cache_miss_count " +
			"because two concurrent threads can miss the cache and generate the same entry twice.",
	}
	cacheEntryCounter = prometheus.NewCounter(cacheEntryCounterOpts)
	cacheGetLatency   = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name: "etcd_request_cache_get_duration_seconds",
			Help: "Latency in seconds of getting an object from etcd cache",
		},
	)
	cacheAddLatency = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name: "etcd_request_cache_add_duration_seconds",
			Help: "Latency in seconds of adding an object to etcd cache",
		},
	)
	etcdRequestLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "etcd_request_duration_seconds",
			Help: "Etcd request latency in seconds for each operation and object type.",
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

	deprecatedCacheHitCounterOpts = prometheus.CounterOpts{
		Name: "etcd_helper_cache_hit_count",
		Help: "(Deprecated) Counter of etcd helper cache hits.",
	}
	deprecatedCacheHitCounter      = prometheus.NewCounter(deprecatedCacheHitCounterOpts)
	deprecatedCacheMissCounterOpts = prometheus.CounterOpts{
		Name: "etcd_helper_cache_miss_count",
		Help: "(Deprecated) Counter of etcd helper cache miss.",
	}
	deprecatedCacheMissCounter      = prometheus.NewCounter(deprecatedCacheMissCounterOpts)
	deprecatedCacheEntryCounterOpts = prometheus.CounterOpts{
		Name: "etcd_helper_cache_entry_count",
		Help: "(Deprecated) Counter of etcd helper cache entries. This can be different from etcd_helper_cache_miss_count " +
			"because two concurrent threads can miss the cache and generate the same entry twice.",
	}
	deprecatedCacheEntryCounter = prometheus.NewCounter(deprecatedCacheEntryCounterOpts)
	deprecatedCacheGetLatency   = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Name: "etcd_request_cache_get_latencies_summary",
			Help: "(Deprecated) Latency in microseconds of getting an object from etcd cache",
		},
	)
	deprecatedCacheAddLatency = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Name: "etcd_request_cache_add_latencies_summary",
			Help: "(Deprecated) Latency in microseconds of adding an object to etcd cache",
		},
	)
	deprecatedEtcdRequestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "etcd_request_latencies_summary",
			Help: "(Deprecated) Etcd request latency summary in microseconds for each operation and object type.",
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
		prometheus.MustRegister(etcdRequestLatency)
		prometheus.MustRegister(objectCounts)

		// TODO(danielqsj): Remove the following metrics, they are deprecated
		prometheus.MustRegister(deprecatedCacheHitCounter)
		prometheus.MustRegister(deprecatedCacheMissCounter)
		prometheus.MustRegister(deprecatedCacheEntryCounter)
		prometheus.MustRegister(deprecatedCacheAddLatency)
		prometheus.MustRegister(deprecatedCacheGetLatency)
		prometheus.MustRegister(deprecatedEtcdRequestLatenciesSummary)
	})
}

func UpdateObjectCount(resourcePrefix string, count int64) {
	objectCounts.WithLabelValues(resourcePrefix).Set(float64(count))
}

func RecordEtcdRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatency.WithLabelValues(verb, resource).Observe(sinceInSeconds(startTime))
	deprecatedEtcdRequestLatenciesSummary.WithLabelValues(verb, resource).Observe(sinceInMicroseconds(startTime))
}

func ObserveGetCache(startTime time.Time) {
	cacheGetLatency.Observe(sinceInSeconds(startTime))
	deprecatedCacheGetLatency.Observe(sinceInMicroseconds(startTime))
}

func ObserveAddCache(startTime time.Time) {
	cacheAddLatency.Observe(sinceInSeconds(startTime))
	deprecatedCacheAddLatency.Observe(sinceInMicroseconds(startTime))
}

func ObserveCacheHit() {
	cacheHitCounter.Inc()
	deprecatedCacheHitCounter.Inc()
}

func ObserveCacheMiss() {
	cacheMissCounter.Inc()
	deprecatedCacheMissCounter.Inc()
}

func ObserveNewEntry() {
	cacheEntryCounter.Inc()
	deprecatedCacheEntryCounter.Inc()
}

func Reset() {
	cacheHitCounter = prometheus.NewCounter(cacheHitCounterOpts)
	cacheMissCounter = prometheus.NewCounter(cacheMissCounterOpts)
	cacheEntryCounter = prometheus.NewCounter(cacheEntryCounterOpts)
	// TODO: Reset cacheAddLatency.
	// TODO: Reset cacheGetLatency.
	etcdRequestLatency.Reset()

	deprecatedCacheHitCounter = prometheus.NewCounter(deprecatedCacheHitCounterOpts)
	deprecatedCacheMissCounter = prometheus.NewCounter(deprecatedCacheMissCounterOpts)
	deprecatedCacheEntryCounter = prometheus.NewCounter(deprecatedCacheEntryCounterOpts)
	deprecatedEtcdRequestLatenciesSummary.Reset()
}

// sinceInMicroseconds gets the time since the specified start in microseconds.
func sinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

// sinceInSeconds gets the time since the specified start in seconds.
func sinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
