/*
Copyright The Kubernetes Authors.

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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	// Subsystem is the subsystem name used for metrics.
	Subsystem = "resourcepoolstatusrequest_controller"
)

var (
	// RequestsProcessed tracks the total number of ResourcePoolStatusRequests processed
	// (both successes and failures).
	RequestsProcessed = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      Subsystem,
			Name:           "requests_processed_total",
			Help:           "Total number of ResourcePoolStatusRequests processed",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"driver_name"},
	)

	// RequestProcessingErrors tracks the number of errors during request processing.
	RequestProcessingErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      Subsystem,
			Name:           "request_processing_errors_total",
			Help:           "Total number of errors encountered while processing ResourcePoolStatusRequests",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"driver_name"},
	)

	// IncompleteRequests tracks requests whose status was written while at
	// least one pool was still incomplete, after retries were exhausted.
	IncompleteRequests = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      Subsystem,
			Name:           "incomplete_requests_total",
			Help:           "Total number of ResourcePoolStatusRequests completed with at least one incomplete pool after retries were exhausted",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"driver_name"},
	)

	// RequestProcessingDuration tracks the time from the first sync attempt of
	// a request until its status is written, including requeues. The 30s and
	// 60s buckets make the 30s completion SLO measurable.
	RequestProcessingDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      Subsystem,
			Name:           "request_processing_duration_seconds",
			Help:           "Time from the first processing attempt of a ResourcePoolStatusRequest until its status is written, including retries",
			Buckets:        []float64{0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 16.384, 30, 60},
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"driver_name"},
	)
)

var registerOnce sync.Once

// Register registers all metrics with the legacy registry.
func Register() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(RequestsProcessed)
		legacyregistry.MustRegister(RequestProcessingErrors)
		legacyregistry.MustRegister(IncompleteRequests)
		legacyregistry.MustRegister(RequestProcessingDuration)
	})
}
