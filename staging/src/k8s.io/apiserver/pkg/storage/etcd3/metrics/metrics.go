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
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/20190404-kubernetes-control-plane-metrics-stability.md#stability-classes)
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
			// Keeping it similar to the buckets used by the apiserver_request_duration_seconds metric so that
			// api latency and etcd latency can be more comparable side by side.
			Buckets: []float64{.005, .01, .025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7,
				0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "type"},
	)
	objectCounts = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "etcd_object_counts",
			Help:           "Number of stored objects at the time of last check split by kind.",
			StabilityLevel: compbasemetrics.ALPHA,
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
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(etcdRequestLatency)
		legacyregistry.MustRegister(objectCounts)
		legacyregistry.MustRegister(dbTotalSize)
	})
}

// UpdateObjectCount sets the etcd_object_counts metric.
func UpdateObjectCount(resourcePrefix string, count int64) {
	objectCounts.WithLabelValues(resourcePrefix).Set(float64(count))
}

// RecordEtcdRequestLatency sets the etcd_request_duration_seconds metrics.
func RecordEtcdRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatency.WithLabelValues(verb, resource).Observe(sinceInSeconds(startTime))
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
