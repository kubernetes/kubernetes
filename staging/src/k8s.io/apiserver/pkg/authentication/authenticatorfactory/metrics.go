/*
Copyright 2021 The Kubernetes Authors.

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

package authenticatorfactory

import (
	"context"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

type registerables []compbasemetrics.Registerable

// init registers all metrics.
func init() {
	for _, metric := range metrics {
		legacyregistry.MustRegister(metric)
	}
}

var (
	requestTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_delegated_authn_request_total",
			Help:           "Number of HTTP requests partitioned by status code.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"code"},
	)

	requestLatency = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "apiserver_delegated_authn_request_duration_seconds",
			Help:           "Request latency in seconds. Broken down by status code.",
			Buckets:        []float64{0.25, 0.5, 0.7, 1, 1.5, 3, 5, 10},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"code"},
	)

	metrics = registerables{
		requestTotal,
		requestLatency,
	}
)

// RecordRequestTotal increments the total number of requests for the delegated authentication.
func RecordRequestTotal(ctx context.Context, code string) {
	requestTotal.WithContext(ctx).WithLabelValues(code).Inc()
}

// RecordRequestLatency measures request latency in seconds for the delegated authentication. Broken down by status code.
func RecordRequestLatency(ctx context.Context, code string, latency float64) {
	requestLatency.WithContext(ctx).WithLabelValues(code).Observe(latency)
}
