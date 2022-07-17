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
	etcdRequestLatency = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name: "etcd_request_duration_seconds",
			Help: "Etcd request latency in seconds for each operation and object type.",
			// Etcd request latency in seconds for each operation and object type.
			// This metric is used for verifying etcd api call latencies SLO
			// keep consistent with apiserver metric 'requestLatencies' in
			// staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go
			Buckets: []float64{0.005, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3,
				4, 5, 6, 8, 10, 15, 20, 30, 45, 60},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "type"},
	)
	objectCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "apiserver_storage_objects",
			Help:           "Number of stored objects at the time of last check split by kind.",
			StabilityLevel: compbasemetrics.STABLE,
		},
		[]string{"resource"},
	)
	dbTotalSize = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "etcd_db_total_size_in_bytes",
			Help:           "Total size of the etcd database file physically allocated in bytes.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"endpoint"},
	)
	etcdBookmarkCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "etcd_bookmark_counts",
			Help:           "Number of etcd bookmarks (progress notify events) split by kind.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)
	etcdLeaseObjectCounts = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "etcd_lease_object_counts",
			Help:           "Number of objects attached to a single etcd lease.",
			Buckets:        []float64{10, 50, 100, 500, 1000, 2500, 5000},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{},
	)
	listStorageCount = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_total",
			Help:           "Number of LIST requests served from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)
	listStorageNumFetched = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_fetched_objects_total",
			Help:           "Number of objects read from storage in the course of serving a LIST request",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)
	listStorageNumSelectorEvals = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_evaluated_objects_total",
			Help:           "Number of objects tested in the course of serving a LIST request from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)
	listStorageNumReturned = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_storage_list_returned_objects_total",
			Help:           "Number of objects returned for a LIST request from storage",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"resource"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(etcdRequestLatency)
		legacyregistry.MustRegister(objectCounts)
		legacyregistry.MustRegister(dbTotalSize)
		legacyregistry.MustRegister(etcdBookmarkCounts)
		legacyregistry.MustRegister(etcdLeaseObjectCounts)
		legacyregistry.MustRegister(listStorageCount)
		legacyregistry.MustRegister(listStorageNumFetched)
		legacyregistry.MustRegister(listStorageNumSelectorEvals)
		legacyregistry.MustRegister(listStorageNumReturned)
	})
}

// UpdateObjectCount sets the apiserver_storage_object_counts metric.
func UpdateObjectCount(resourcePrefix string, count int64) {
	objectCounts.WithLabelValues(resourcePrefix).Set(float64(count))
}

// RecordEtcdRequestLatency sets the etcd_request_duration_seconds metrics.
func RecordEtcdRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatency.WithLabelValues(verb, resource).Observe(sinceInSeconds(startTime))
}

// RecordEtcdBookmark updates the etcd_bookmark_counts metric.
func RecordEtcdBookmark(resource string) {
	etcdBookmarkCounts.WithLabelValues(resource).Inc()
}

// Reset resets the etcd_request_duration_seconds metric.
func Reset() {
	etcdRequestLatency.Reset()
}

// sinceInSeconds gets the time since the specified start in seconds.
func sinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

// UpdateEtcdDbSize sets the etcd_db_total_size_in_bytes metric.
func UpdateEtcdDbSize(ep string, size int64) {
	dbTotalSize.WithLabelValues(ep).Set(float64(size))
}

// UpdateLeaseObjectCount sets the etcd_lease_object_counts metric.
func UpdateLeaseObjectCount(count int64) {
	// Currently we only store one previous lease, since all the events have the same ttl.
	// See pkg/storage/etcd3/lease_manager.go
	etcdLeaseObjectCounts.WithLabelValues().Observe(float64(count))
}

// RecordListEtcd3Metrics notes various metrics of the cost to serve a LIST request
func RecordStorageListMetrics(resource string, numFetched, numEvald, numReturned int) {
	listStorageCount.WithLabelValues(resource).Inc()
	listStorageNumFetched.WithLabelValues(resource).Add(float64(numFetched))
	listStorageNumSelectorEvals.WithLabelValues(resource).Add(float64(numEvald))
	listStorageNumReturned.WithLabelValues(resource).Add(float64(numReturned))
}
