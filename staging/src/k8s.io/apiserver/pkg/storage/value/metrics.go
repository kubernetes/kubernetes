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

package value

import (
	"errors"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "storage"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	transformerLatencies = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "transformation_duration_seconds",
			Help:      "Latencies in seconds of value transformation operations.",
			// In-process transformations (ex. AES CBC) complete on the order of 20 microseconds. However, when
			// external KMS is involved latencies may climb into hundreds of milliseconds.
			Buckets:        metrics.ExponentialBuckets(5e-6, 2, 25),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"transformation_type", "transformer_prefix"},
	)

	transformerOperationsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "transformation_operations_total",
			Help:           "Total number of transformations. Successful transformation will have a status 'OK' and a varied status string when the transformation fails. This status and transformation_type fields may be used for alerting on encryption/decryption failure using transformation_type from_storage for decryption and to_storage for encryption",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"transformation_type", "transformer_prefix", "status"},
	)

	envelopeTransformationCacheMissTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "envelope_transformation_cache_misses_total",
			Help:           "Total number of cache misses while accessing key decryption key(KEK).",
			StabilityLevel: metrics.ALPHA,
		},
	)

	dataKeyGenerationLatencies = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "data_key_generation_duration_seconds",
			Help:           "Latencies in seconds of data encryption key(DEK) generation operations.",
			Buckets:        metrics.ExponentialBuckets(5e-6, 2, 14),
			StabilityLevel: metrics.ALPHA,
		},
	)

	dataKeyGenerationFailuresTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "data_key_generation_failures_total",
			Help:           "Total number of failed data encryption key(DEK) generation operations.",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(transformerLatencies)
		legacyregistry.MustRegister(transformerOperationsTotal)
		legacyregistry.MustRegister(envelopeTransformationCacheMissTotal)
		legacyregistry.MustRegister(dataKeyGenerationLatencies)
		legacyregistry.MustRegister(dataKeyGenerationFailuresTotal)
	})
}

// RecordTransformation records latencies and count of TransformFromStorage and TransformToStorage operations.
// Note that transformation_failures_total metric is deprecated, use transformation_operations_total instead.
func RecordTransformation(transformationType, transformerPrefix string, elapsed time.Duration, err error) {
	transformerOperationsTotal.WithLabelValues(transformationType, transformerPrefix, getErrorCode(err)).Inc()

	if err == nil {
		transformerLatencies.WithLabelValues(transformationType, transformerPrefix).Observe(elapsed.Seconds())
	}
}

// RecordCacheMiss records a miss on Key Encryption Key(KEK) - call to KMS was required to decrypt KEK.
func RecordCacheMiss() {
	envelopeTransformationCacheMissTotal.Inc()
}

// RecordDataKeyGeneration records latencies and count of Data Encryption Key generation operations.
func RecordDataKeyGeneration(start time.Time, err error) {
	if err != nil {
		dataKeyGenerationFailuresTotal.Inc()
		return
	}

	dataKeyGenerationLatencies.Observe(sinceInSeconds(start))
}

// sinceInSeconds gets the time since the specified start in seconds.
func sinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}

type gRPCError interface {
	GRPCStatus() *status.Status
}

func getErrorCode(err error) string {
	if err == nil {
		return codes.OK.String()
	}

	// handle errors wrapped with fmt.Errorf and similar
	var s gRPCError
	if errors.As(err, &s) {
		return s.GRPCStatus().Code().String()
	}

	// This is not gRPC error. The operation must have failed before gRPC
	// method was called, otherwise we would get gRPC error.
	return "unknown-non-grpc"
}
