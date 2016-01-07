/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	// TODO(a-robinson): Add unit tests for the handling of these metrics once
	// the upstream library supports it.
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "apiserver_request_count",
			Help: "Counter of apiserver requests broken out for each verb, API resource, client, and HTTP response code.",
		},
		[]string{"verb", "resource", "client", "code"},
	)
	requestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "apiserver_request_latencies",
			Help: "Response latency distribution in microseconds for each verb, resource and client.",
			// Use buckets ranging from 125 ms to 8 seconds.
			Buckets: prometheus.ExponentialBuckets(125000, 2.0, 7),
		},
		[]string{"verb", "resource"},
	)
	requestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "apiserver_request_latencies_summary",
			Help: "Response latency summary in microseconds for each verb and resource.",
			// Make the sliding window of 1h.
			MaxAge: time.Hour,
		},
		[]string{"verb", "resource"},
	)
)

// Register all metrics.
func Register() {
	prometheus.MustRegister(requestCounter)
	prometheus.MustRegister(requestLatencies)
	prometheus.MustRegister(requestLatenciesSummary)
}

func Monitor(verb, resource *string, client string, httpCode *int, reqStart time.Time) {
	requestCounter.WithLabelValues(*verb, *resource, client, strconv.Itoa(*httpCode)).Inc()
	requestLatencies.WithLabelValues(*verb, *resource).Observe(float64((time.Since(reqStart)) / time.Microsecond))
	requestLatenciesSummary.WithLabelValues(*verb, *resource).Observe(float64((time.Since(reqStart)) / time.Microsecond))
}

func Reset(w http.ResponseWriter, req *http.Request) {
	requestCounter.Reset()
	requestLatencies.Reset()
	requestLatenciesSummary.Reset()
	io.WriteString(w, "metrics reset\n")
}
