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

// Package metrics provides utilities for registering client metrics to Prometheus.
package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

const restClientSubsystem = "rest_client"

var (
	// RequestLatency is a Prometheus Summary metric type partitioned by
	// "verb" and "url" labels. It is used for the rest client latency metrics.
	RequestLatency = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Subsystem: restClientSubsystem,
			Name:      "request_latency_microseconds",
			Help:      "Request latency in microseconds. Broken down by verb and URL",
			MaxAge:    time.Hour,
		},
		[]string{"verb", "url"},
	)

	RequestResult = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: restClientSubsystem,
			Name:      "request_status_codes",
			Help:      "Number of http requests, partitioned by metadata",
		},
		[]string{"code", "method", "host"},
	)
)

var registerMetrics sync.Once

// Register registers all metrics to Prometheus with
// respect to the RequestLatency.
func Register() {
	// Register the metrics.
	registerMetrics.Do(func() {
		prometheus.MustRegister(RequestLatency)
		prometheus.MustRegister(RequestResult)
	})
}

// Calculates the time since the specified start in microseconds.
func SinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}
