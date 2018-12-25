/*
Copyright 2019 The Kubernetes Authors.

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

package etcdmetrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	etcdRequestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "etcd_request_latency_seconds",
			Help:    "etcd request latency in seconds for each operation and object type.",
			Buckets: prometheus.ExponentialBuckets(0.001, 5, 7),
		},
		[]string{"operation", "type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(etcdRequestLatencies)
	})
}

// RecordRequestLatency records the latency of the etcd request.
func RecordRequestLatency(verb, resource string, startTime time.Time) {
	etcdRequestLatencies.WithLabelValues(verb, resource).Observe(time.Since(startTime).Seconds())
}

// Reset resets the etcdRequestLatencies.
func Reset() {
	etcdRequestLatencies.Reset()
}
