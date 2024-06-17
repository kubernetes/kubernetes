/*
Copyright 2020 The Kubernetes Authors.

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
	"crypto/sha256"
	"errors"
	"fmt"
	"hash"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
	"k8s.io/utils/lru"
)

const (
	namespace        = "apiserver"
	subsystem        = "envelope_encryption"
	FromStorageLabel = "from_storage"
	ToStorageLabel   = "to_storage"
)

type metricLabels struct {
	transformationType string
	providerName       string
	keyIDHash          string
	apiServerIDHash    string
}

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	lockLastFromStorage   sync.Mutex
	lockLastToStorage     sync.Mutex
	lockRecordKeyID       sync.Mutex
	lockRecordKeyIDStatus sync.Mutex

	lastFromStorage                                 time.Time
	lastToStorage                                   time.Time
	keyIDHashTotalMetricLabels                      *lru.Cache
	keyIDHashStatusLastTimestampSecondsMetricLabels *lru.Cache
	cacheSize                                       = 100

	// This metric is only used for KMS v1 API.
	dekCacheFillPercent = metrics.NewGauge(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dek_cache_fill_percent",
			Help:           "Percent of the cache slots currently occupied by cached DEKs.",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// This metric is only used for KMS v1 API.
	dekCacheInterArrivals = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dek_cache_inter_arrival_time_seconds",
			Help:           "Time (in seconds) of inter arrival of transformation requests.",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(60, 2, 10),
		},
		[]string{"transformation_type"},
	)

	// These metrics are made public to be used by unit tests.
	KMSOperationsLatencyMetric = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "kms_operations_latency_seconds",
			Help:           "KMS operation duration with gRPC error code status total.",
			StabilityLevel: metrics.ALPHA,
			// Use custom buckets to avoid the default buckets which are too small for KMS operations.
			// Start 0.1ms with the last bucket being [~52s, +Inf)
			Buckets: metrics.ExponentialBuckets(0.0001, 2, 20),
		},
		[]string{"provider_name", "method_name", "grpc_status_code"},
	)

	// keyIDHashTotal is the number of times a keyID is used
	// e.g. apiserver_envelope_encryption_key_id_hash_total counter
	// apiserver_envelope_encryption_key_id_hash_total{apiserver_id_hash="sha256",key_id_hash="sha256",
	// provider_name="providerName",transformation_type="from_storage"} 1
	KeyIDHashTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "key_id_hash_total",
			Help:           "Number of times a keyID is used split by transformation type, provider, and apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"transformation_type", "provider_name", "key_id_hash", "apiserver_id_hash"},
	)

	// keyIDHashLastTimestampSeconds is the last time in seconds when a keyID was used
	// e.g. apiserver_envelope_encryption_key_id_hash_last_timestamp_seconds{apiserver_id_hash="sha256",key_id_hash="sha256", provider_name="providerName",transformation_type="from_storage"} 1.674865558833728e+09
	KeyIDHashLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "key_id_hash_last_timestamp_seconds",
			Help:           "The last time in seconds when a keyID was used.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"transformation_type", "provider_name", "key_id_hash", "apiserver_id_hash"},
	)

	// keyIDHashStatusLastTimestampSeconds is the last time in seconds when a keyID was returned by the Status RPC call.
	// e.g. apiserver_envelope_encryption_key_id_hash_status_last_timestamp_seconds{apiserver_id_hash="sha256",key_id_hash="sha256", provider_name="providerName"} 1.674865558833728e+09
	KeyIDHashStatusLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "key_id_hash_status_last_timestamp_seconds",
			Help:           "The last time in seconds when a keyID was returned by the Status RPC call.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"provider_name", "key_id_hash", "apiserver_id_hash"},
	)

	InvalidKeyIDFromStatusTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "invalid_key_id_from_status_total",
			Help:           "Number of times an invalid keyID is returned by the Status RPC call split by error.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"provider_name", "error"},
	)

	DekSourceCacheSize = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dek_source_cache_size",
			Help:           "Number of records in data encryption key (DEK) source cache. On a restart, this value is an approximation of the number of decrypt RPC calls the server will make to the KMS plugin.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"provider_name"},
	)
)

var registerMetricsFunc sync.Once
var hashPool *sync.Pool

func registerLRUMetrics() {
	if keyIDHashTotalMetricLabels != nil {
		keyIDHashTotalMetricLabels.Clear()
	}
	if keyIDHashStatusLastTimestampSecondsMetricLabels != nil {
		keyIDHashStatusLastTimestampSecondsMetricLabels.Clear()
	}

	keyIDHashTotalMetricLabels = lru.NewWithEvictionFunc(cacheSize, func(key lru.Key, _ interface{}) {
		item := key.(metricLabels)
		if deleted := KeyIDHashTotal.DeleteLabelValues(item.transformationType, item.providerName, item.keyIDHash, item.apiServerIDHash); deleted {
			klog.InfoS("Deleted keyIDHashTotalMetricLabels", "transformationType", item.transformationType,
				"providerName", item.providerName, "keyIDHash", item.keyIDHash, "apiServerIDHash", item.apiServerIDHash)
		}
		if deleted := KeyIDHashLastTimestampSeconds.DeleteLabelValues(item.transformationType, item.providerName, item.keyIDHash, item.apiServerIDHash); deleted {
			klog.InfoS("Deleted keyIDHashLastTimestampSecondsMetricLabels", "transformationType", item.transformationType,
				"providerName", item.providerName, "keyIDHash", item.keyIDHash, "apiServerIDHash", item.apiServerIDHash)
		}
	})
	keyIDHashStatusLastTimestampSecondsMetricLabels = lru.NewWithEvictionFunc(cacheSize, func(key lru.Key, _ interface{}) {
		item := key.(metricLabels)
		if deleted := KeyIDHashStatusLastTimestampSeconds.DeleteLabelValues(item.providerName, item.keyIDHash, item.apiServerIDHash); deleted {
			klog.InfoS("Deleted keyIDHashStatusLastTimestampSecondsMetricLabels", "providerName", item.providerName, "keyIDHash", item.keyIDHash, "apiServerIDHash", item.apiServerIDHash)
		}
	})
}
func RegisterMetrics() {
	registerMetricsFunc.Do(func() {
		registerLRUMetrics()
		hashPool = &sync.Pool{
			New: func() interface{} {
				return sha256.New()
			},
		}
		legacyregistry.MustRegister(dekCacheFillPercent)
		legacyregistry.MustRegister(dekCacheInterArrivals)
		legacyregistry.MustRegister(DekSourceCacheSize)
		legacyregistry.MustRegister(KeyIDHashTotal)
		legacyregistry.MustRegister(KeyIDHashLastTimestampSeconds)
		legacyregistry.MustRegister(KeyIDHashStatusLastTimestampSeconds)
		legacyregistry.MustRegister(InvalidKeyIDFromStatusTotal)
		legacyregistry.MustRegister(KMSOperationsLatencyMetric)
	})
}

// RecordKeyID records total count and last time in seconds when a KeyID was used for TransformFromStorage and TransformToStorage operations
func RecordKeyID(transformationType, providerName, keyID, apiServerID string) {
	lockRecordKeyID.Lock()
	defer lockRecordKeyID.Unlock()

	keyIDHash, apiServerIDHash := addLabelToCache(keyIDHashTotalMetricLabels, transformationType, providerName, keyID, apiServerID)
	KeyIDHashTotal.WithLabelValues(transformationType, providerName, keyIDHash, apiServerIDHash).Inc()
	KeyIDHashLastTimestampSeconds.WithLabelValues(transformationType, providerName, keyIDHash, apiServerIDHash).SetToCurrentTime()
}

// RecordKeyIDFromStatus records last time in seconds when a KeyID was returned by the Status RPC call.
func RecordKeyIDFromStatus(providerName, keyID, apiServerID string) {
	lockRecordKeyIDStatus.Lock()
	defer lockRecordKeyIDStatus.Unlock()

	keyIDHash, apiServerIDHash := addLabelToCache(keyIDHashStatusLastTimestampSecondsMetricLabels, "", providerName, keyID, apiServerID)
	KeyIDHashStatusLastTimestampSeconds.WithLabelValues(providerName, keyIDHash, apiServerIDHash).SetToCurrentTime()
}

func RecordInvalidKeyIDFromStatus(providerName, errCode string) {
	InvalidKeyIDFromStatusTotal.WithLabelValues(providerName, errCode).Inc()
}

func RecordArrival(transformationType string, start time.Time) {
	switch transformationType {
	case FromStorageLabel:
		lockLastFromStorage.Lock()
		defer lockLastFromStorage.Unlock()

		if lastFromStorage.IsZero() {
			lastFromStorage = start
		}
		dekCacheInterArrivals.WithLabelValues(transformationType).Observe(start.Sub(lastFromStorage).Seconds())
		lastFromStorage = start
	case ToStorageLabel:
		lockLastToStorage.Lock()
		defer lockLastToStorage.Unlock()

		if lastToStorage.IsZero() {
			lastToStorage = start
		}
		dekCacheInterArrivals.WithLabelValues(transformationType).Observe(start.Sub(lastToStorage).Seconds())
		lastToStorage = start
	}
}

func RecordDekCacheFillPercent(percent float64) {
	dekCacheFillPercent.Set(percent)
}

func RecordDekSourceCacheSize(providerName string, size int) {
	DekSourceCacheSize.WithLabelValues(providerName).Set(float64(size))
}

// RecordKMSOperationLatency records the latency of KMS operation.
func RecordKMSOperationLatency(providerName, methodName string, duration time.Duration, err error) {
	KMSOperationsLatencyMetric.WithLabelValues(providerName, methodName, getErrorCode(err)).Observe(duration.Seconds())
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

func getHash(data string) string {
	if len(data) == 0 {
		return ""
	}
	h := hashPool.Get().(hash.Hash)
	h.Reset()
	h.Write([]byte(data))
	dataHash := fmt.Sprintf("sha256:%x", h.Sum(nil))
	hashPool.Put(h)
	return dataHash
}

func addLabelToCache(c *lru.Cache, transformationType, providerName, keyID, apiServerID string) (string, string) {
	keyIDHash := getHash(keyID)
	apiServerIDHash := getHash(apiServerID)
	c.Add(metricLabels{
		transformationType: transformationType,
		providerName:       providerName,
		keyIDHash:          keyIDHash,
		apiServerIDHash:    apiServerIDHash,
	}, nil) // value is irrelevant, this is a set and not a map
	return keyIDHash, apiServerIDHash
}
