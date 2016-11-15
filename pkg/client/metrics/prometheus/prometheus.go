/*
Copyright 2016 The Kubernetes Authors.

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

// Package prometheus creates and registers prometheus metrics with
// rest clients. To use this package, you just have to import it.
package prometheus

import (
	"net/url"
	"time"

	"k8s.io/kubernetes/pkg/client/metrics"

	"github.com/prometheus/client_golang/prometheus"
)

const restClientSubsystem = "rest_client"

var (
	// requestLatency is a Prometheus Summary metric type partitioned by
	// "verb" and "url" labels. It is used for the rest client latency metrics.
	requestLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: restClientSubsystem,
			Name:      "request_latency_microseconds",
			Help:      "Request latency in microseconds. Broken down by verb and URL",
			MaxAge:    time.Hour,
		},
		[]string{"verb", "url"},
	)

	requestResult = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: restClientSubsystem,
			Name:      "request_status_codes",
			Help:      "Number of http requests, partitioned by metadata",
		},
		[]string{"code", "method", "host"},
	)
)

func init() {
	prometheus.MustRegister(requestLatency)
	prometheus.MustRegister(requestResult)
	metrics.Register(&latencyAdapter{requestLatency}, &resultAdapter{requestResult})
}

type latencyAdapter struct {
	m *prometheus.SummaryVec
}

func (l *latencyAdapter) Observe(verb string, u url.URL, latency time.Duration) {
	microseconds := float64(latency) / float64(time.Microsecond)
	l.m.WithLabelValues(verb, u.String()).Observe(microseconds)
}

type resultAdapter struct {
	m *prometheus.CounterVec
}

func (r *resultAdapter) Increment(code, method, host string) {
	r.m.WithLabelValues(code, method, host).Inc()
}
