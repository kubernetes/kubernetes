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
	// RequestsProcessed tracks the total number of ResourcePoolStatusRequests processed.
	RequestsProcessed = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      Subsystem,
			Name:           "requests_processed_total",
			Help:           "Total number of ResourcePoolStatusRequests successfully processed",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// RequestProcessingErrors tracks the number of errors during request processing.
	RequestProcessingErrors = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      Subsystem,
			Name:           "request_processing_errors_total",
			Help:           "Total number of errors encountered while processing ResourcePoolStatusRequests",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// RequestProcessingDuration tracks the time taken to process requests.
	RequestProcessingDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      Subsystem,
			Name:           "request_processing_duration_seconds",
			Help:           "Time taken to process a ResourcePoolStatusRequest",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// PoolsCounted tracks the number of pools counted in status calculations.
	PoolsCounted = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      Subsystem,
			Name:           "pools_counted",
			Help:           "Number of pools counted per ResourcePoolStatusRequest",
			Buckets:        metrics.ExponentialBuckets(1, 2, 10),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// CacheRebuildDuration tracks the time taken to rebuild caches.
	CacheRebuildDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      Subsystem,
			Name:           "cache_rebuild_duration_seconds",
			Help:           "Time taken to rebuild internal caches",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"cache_type"},
	)
)

var registerOnce sync.Once

// Register registers all metrics with the legacy registry.
func Register() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(RequestsProcessed)
		legacyregistry.MustRegister(RequestProcessingErrors)
		legacyregistry.MustRegister(RequestProcessingDuration)
		legacyregistry.MustRegister(PoolsCounted)
		legacyregistry.MustRegister(CacheRebuildDuration)
	})
}
