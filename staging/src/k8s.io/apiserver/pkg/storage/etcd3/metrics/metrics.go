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
			Name:           "etcd_request_duration_seconds",
			Help:           "Etcd request latency in seconds for each operation and object type.",
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
	requestSize = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "etcd_request_size_bytes",
			Help:           "Etcd request size in bytes for each operation and object type.",
			Buckets:        []float64{0, 64, 128, 256, 512, 1028, 2048, 4096, 8192, 16384, 32768, 65536, 1048576},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "type"},
	)
	responseSize = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "etcd_response_size_bytes",
			Help:           "Etcd response size in bytes for each operation and object type.",
			Buckets:        []float64{0, 64, 128, 256, 512, 1028, 2048, 4096, 8192, 16384, 32768, 65536, 1048576},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "type"},
	)

	deprecatedEtcdRequestLatenciesSummary = compbasemetrics.NewSummaryVec(
		&compbasemetrics.SummaryOpts{
			Name:           "etcd_request_latencies_summary",
			Help:           "(Deprecated) Etcd request latency summary in microseconds for each operation and object type.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"operation", "type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(etcdRequestLatency)
		legacyregistry.MustRegister(objectCounts)
		legacyregistry.MustRegister(requestSize)
		legacyregistry.MustRegister(responseSize)

		// TODO(danielqsj): Remove the following metrics, they are deprecated
		legacyregistry.MustRegister(deprecatedEtcdRequestLatenciesSummary)
	})
}

// UpdateObjectCount sets the etcd_object_counts metric.
func UpdateObjectCount(resourcePrefix string, count int64) {
	objectCounts.WithLabelValues(resourcePrefix).Set(float64(count))
}

// RecordEtcdRequestLatency sets the etcd_request_duration_seconds metrics.
func RecordEtcdRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatency.WithLabelValues(verb, resource).Observe(sinceInSeconds(startTime))
	deprecatedEtcdRequestLatenciesSummary.WithLabelValues(verb, resource).Observe(sinceInMicroseconds(startTime))
}

func RecordEtcdRequestSize(verb, resource string, sizeInBytes uint64) {
	requestSize.WithLabelValues(verb, resource).Observe(float64(sizeInBytes))
}

func RecordEtcdResponseSize(verb, resource string, sizeInBytes uint64) {
	responseSize.WithLabelValues(verb, resource).Observe(float64(sizeInBytes))
}

// Reset resets the etcd_request_duration_seconds metric.
func Reset() {
	etcdRequestLatency.Reset()

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
