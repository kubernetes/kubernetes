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

package compression

import (
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "storage"

	toStorage   = "to_storage"
	fromStorage = "from_storage"

	outcomeDeflate            = "deflate"
	outcomeStored             = "stored"
	outcomeUnframedBelowMin   = "unframed_below_threshold"
	outcomeUnframedAboveMax   = "unframed_above_max"
	outcomeUnframedLegacyRead = "unframed_legacy"

	directionToFramed   = "to_framed"
	directionToUnframed = "to_unframed"
)

// There's deliberately no ratio or byte-count metric. On a resource with few
// objects, either one comes close to revealing each object's compressibility.
// That's a finer side channel than the stored sizes that compression already
// shows etcd readers, and /metrics has far more readers than etcd does.
var (
	operationsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compression_operations_total",
			Help:           "Number of storage compression operations by resource, operation and outcome. The outcome tells compressed, stored verbatim and unframed values apart.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource", "operation", "outcome"},
	)

	durationSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compression_duration_seconds",
			Help:           "Time spent compressing and decompressing storage values, not counting any encryption at rest or the wait for a decompression slot.",
			Buckets:        metrics.ExponentialBuckets(5e-6, 2, 20),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation"},
	)

	// formatErrorsTotal has no reason label: it would tell someone tampering with
	// etcd which check their bytes failed. The reason is logged at V(4) instead.
	formatErrorsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compression_format_errors_total",
			Help:           "Number of storage compression frames that failed their checksum or did not parse, by resource.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	unsupportedAlgorithmTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compression_unsupported_algorithm_total",
			Help:           "Number of storage values read whose compression algorithm this apiserver does not implement, by resource. A newer apiserver wrote them.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource"},
	)

	// rewritesNeededTotal is counted on reads, so it goes quiet once a migration
	// has converged.
	rewritesNeededTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compression_rewrites_needed_total",
			Help:           "Number of reads that reported a value stale because its compression framing disagreed with the policy, by resource and direction.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource", "direction"},
	)

	// inflationWaitSeconds has no labels because the semaphore is process-wide.
	// It's a vec so that no series exists until something inflates.
	inflationWaitSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "compression_inflation_wait_seconds",
			Help:           "Time spent waiting for a decompression slot.",
			Buckets:        metrics.ExponentialBuckets(1e-5, 4, 10),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(operationsTotal)
		legacyregistry.MustRegister(durationSeconds)
		legacyregistry.MustRegister(formatErrorsTotal)
		legacyregistry.MustRegister(unsupportedAlgorithmTotal)
		legacyregistry.MustRegister(rewritesNeededTotal)
		legacyregistry.MustRegister(inflationWaitSeconds)
	})
}

func recordOperation(resource, operation, outcome string) {
	operationsTotal.WithLabelValues(resource, operation, outcome).Inc()
}

func recordDuration(operation string, elapsed time.Duration) {
	durationSeconds.WithLabelValues(operation).Observe(elapsed.Seconds())
}

func recordFormatError(resource string) {
	formatErrorsTotal.WithLabelValues(resource).Inc()
}

func recordUnsupportedAlgorithm(resource string) {
	unsupportedAlgorithmTotal.WithLabelValues(resource).Inc()
}

func recordRewriteNeeded(resource, direction string) {
	rewritesNeededTotal.WithLabelValues(resource, direction).Inc()
}

func recordInflationWait(elapsed time.Duration) {
	inflationWaitSeconds.WithLabelValues().Observe(elapsed.Seconds())
}
