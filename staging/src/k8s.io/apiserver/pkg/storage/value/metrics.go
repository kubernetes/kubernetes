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
	"sync"
	"time"

	"google.golang.org/grpc/status"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	namespace = "apiserver"
	subsystem = "storage"
)

var (
	transformerLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "transformation_duration_seconds",
			Help:      "Latencies in seconds of value transformation operations.",
			// In-process transformations (ex. AES CBC) complete on the order of 20 microseconds. However, when
			// external KMS is involved latencies may climb into milliseconds.
			Buckets: prometheus.ExponentialBuckets(5e-6, 2, 14),
		},
		[]string{"transformation_type"},
	)
	deprecatedTransformerLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "transformation_latencies_microseconds",
			Help:      "(Deprecated) Latencies in microseconds of value transformation operations.",
			// In-process transformations (ex. AES CBC) complete on the order of 20 microseconds. However, when
			// external KMS is involved latencies may climb into milliseconds.
			Buckets: prometheus.ExponentialBuckets(5, 2, 14),
		},
		[]string{"transformation_type"},
	)

	transformerOperationsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "transformation_operations_total",
			Help:      "Total number of transformations.",
		},
		[]string{"transformation_type", "transformer_prefix", "status"},
	)

	deprecatedTransformerFailuresTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "transformation_failures_total",
			Help:      "(Deprecated) Total number of failed transformation operations.",
		},
		[]string{"transformation_type"},
	)

	envelopeTransformationCacheMissTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "envelope_transformation_cache_misses_total",
			Help:      "Total number of cache misses while accessing key decryption key(KEK).",
		},
	)

	dataKeyGenerationLatencies = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "data_key_generation_duration_seconds",
			Help:      "Latencies in seconds of data encryption key(DEK) generation operations.",
			Buckets:   prometheus.ExponentialBuckets(5e-6, 2, 14),
		},
	)
	deprecatedDataKeyGenerationLatencies = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "data_key_generation_latencies_microseconds",
			Help:      "(Deprecated) Latencies in microseconds of data encryption key(DEK) generation operations.",
			Buckets:   prometheus.ExponentialBuckets(5, 2, 14),
		},
	)
	dataKeyGenerationFailuresTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "data_key_generation_failures_total",
			Help:      "Total number of failed data encryption key(DEK) generation operations.",
		},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(transformerLatencies)
		prometheus.MustRegister(deprecatedTransformerLatencies)
		prometheus.MustRegister(transformerOperationsTotal)
		prometheus.MustRegister(deprecatedTransformerFailuresTotal)
		prometheus.MustRegister(envelopeTransformationCacheMissTotal)
		prometheus.MustRegister(dataKeyGenerationLatencies)
		prometheus.MustRegister(deprecatedDataKeyGenerationLatencies)
		prometheus.MustRegister(dataKeyGenerationFailuresTotal)
	})
}

// RecordTransformation records latencies and count of TransformFromStorage and TransformToStorage operations.
// Note that transformation_failures_total metric is deprecated, use transformation_operations_total instead.
func RecordTransformation(transformationType, transformerPrefix string, start time.Time, err error) {
	transformerOperationsTotal.WithLabelValues(transformationType, transformerPrefix, status.Code(err).String()).Inc()

	switch {
	case err == nil:
		transformerLatencies.WithLabelValues(transformationType).Observe(sinceInSeconds(start))
		deprecatedTransformerLatencies.WithLabelValues(transformationType).Observe(sinceInMicroseconds(start))
	default:
		deprecatedTransformerFailuresTotal.WithLabelValues(transformationType).Inc()
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
	deprecatedDataKeyGenerationLatencies.Observe(sinceInMicroseconds(start))
}

// sinceInMicroseconds gets the time since the specified start in microseconds.
func sinceInMicroseconds(start time.Time) float64 {
	return float64(time.Since(start).Nanoseconds() / time.Microsecond.Nanoseconds())
}

// sinceInSeconds gets the time since the specified start in seconds.
func sinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
