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

package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	kubeletSubsystem = "kubelet"
)

var (
	// HTTPRequests tracks the number of the http requests received since the server started.
	HTTPRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: kubeletSubsystem,
			Name:      "http_requests_total",
			Help:      "Number of the http requests received since the server started",
		},
		// server_type aims to differentiate the readonly server and the readwrite server.
		// long_running marks whether the request is long-running or not.
		// Currently, long-running requests include exec/attach/portforward/debug.
		[]string{"method", "path", "server_type", "long_running"},
	)
	// HTTPRequestsDuration tracks the duration in seconds to serve http requests.
	HTTPRequestsDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: kubeletSubsystem,
			Name:      "http_requests_duration_seconds",
			Help:      "Duration in seconds to serve http requests",
			// Use DefBuckets for now, will customize the buckets if necessary.
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "path", "server_type", "long_running"},
	)
	// HTTPInflightRequests tracks the number of the inflight http requests.
	HTTPInflightRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: kubeletSubsystem,
			Name:      "http_inflight_requests",
			Help:      "Number of the inflight http requests",
		},
		[]string{"method", "path", "server_type", "long_running"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(HTTPRequests)
		prometheus.MustRegister(HTTPRequestsDuration)
		prometheus.MustRegister(HTTPInflightRequests)
	})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
