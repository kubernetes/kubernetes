// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus_test

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	// apiRequestDuration tracks the duration separate for each HTTP status
	// class (1xx, 2xx, ...). This creates a fair amount of time series on
	// the Prometheus server. Usually, you would track the duration of
	// serving HTTP request without partitioning by outcome. Do something
	// like this only if needed. Also note how only status classes are
	// tracked, not every single status code. The latter would create an
	// even larger amount of time series. Request counters partitioned by
	// status code are usually OK as each counter only creates one time
	// series. Histograms are way more expensive, so partition with care and
	// only where you really need separate latency tracking. Partitioning by
	// status class is only an example. In concrete cases, other partitions
	// might make more sense.
	apiRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "api_request_duration_seconds",
			Help:    "Histogram for the request duration of the public API, partitioned by status class.",
			Buckets: prometheus.ExponentialBuckets(0.1, 1.5, 5),
		},
		[]string{"status_class"},
	)
)

func handler(w http.ResponseWriter, r *http.Request) {
	status := http.StatusOK
	// The ObserverFunc gets called by the deferred ObserveDuration and
	// decides which Histogram's Observe method is called.
	timer := prometheus.NewTimer(prometheus.ObserverFunc(func(v float64) {
		switch {
		case status >= 500: // Server error.
			apiRequestDuration.WithLabelValues("5xx").Observe(v)
		case status >= 400: // Client error.
			apiRequestDuration.WithLabelValues("4xx").Observe(v)
		case status >= 300: // Redirection.
			apiRequestDuration.WithLabelValues("3xx").Observe(v)
		case status >= 200: // Success.
			apiRequestDuration.WithLabelValues("2xx").Observe(v)
		default: // Informational.
			apiRequestDuration.WithLabelValues("1xx").Observe(v)
		}
	}))
	defer timer.ObserveDuration()

	// Handle the request. Set status accordingly.
	// ...
}

func ExampleTimer_complex() {
	http.HandleFunc("/api", handler)
}
