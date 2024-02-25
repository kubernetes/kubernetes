/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"hash"
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "encryption_config_controller"
)

var (
	encryptionConfigAutomaticReloadsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reloads_total",
			Help:           "Total number of reload successes and failures of encryption configuration split by apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "apiserver_id_hash"},
	)

	// deprecatedEncryptionConfigAutomaticReloadFailureTotal has been deprecated in 1.30.0
	// use encryptionConfigAutomaticReloadsTotal instead
	deprecatedEncryptionConfigAutomaticReloadFailureTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:         namespace,
			Subsystem:         subsystem,
			Name:              "automatic_reload_failures_total",
			Help:              "Total number of failed automatic reloads of encryption configuration split by apiserver identity.",
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.30.0",
		},
		[]string{"apiserver_id_hash"},
	)

	// deprecatedEncryptionConfigAutomaticReloadSuccessTotal has been deprecated in 1.30.0
	// use encryptionConfigAutomaticReloadsTotal instead
	deprecatedEncryptionConfigAutomaticReloadSuccessTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:         namespace,
			Subsystem:         subsystem,
			Name:              "automatic_reload_success_total",
			Help:              "Total number of successful automatic reloads of encryption configuration split by apiserver identity.",
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.30.0",
		},
		[]string{"apiserver_id_hash"},
	)

	encryptionConfigAutomaticReloadLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_last_timestamp_seconds",
			Help:           "Timestamp of the last successful or failed automatic reload of encryption configuration split by apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "apiserver_id_hash"},
	)
)

var registerMetrics sync.Once
var hashPool *sync.Pool

func RegisterMetrics() {
	registerMetrics.Do(func() {
		hashPool = &sync.Pool{
			New: func() interface{} {
				return sha256.New()
			},
		}
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadsTotal)
		legacyregistry.MustRegister(deprecatedEncryptionConfigAutomaticReloadFailureTotal)
		legacyregistry.MustRegister(deprecatedEncryptionConfigAutomaticReloadSuccessTotal)
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadLastTimestampSeconds)
	})
}

func RecordEncryptionConfigAutomaticReloadFailure(apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	encryptionConfigAutomaticReloadsTotal.WithLabelValues("failure", apiServerIDHash).Inc()
	deprecatedEncryptionConfigAutomaticReloadFailureTotal.WithLabelValues(apiServerIDHash).Inc()
	recordEncryptionConfigAutomaticReloadTimestamp("failure", apiServerIDHash)
}

func RecordEncryptionConfigAutomaticReloadSuccess(apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	encryptionConfigAutomaticReloadsTotal.WithLabelValues("success", apiServerIDHash).Inc()
	deprecatedEncryptionConfigAutomaticReloadSuccessTotal.WithLabelValues(apiServerIDHash).Inc()
	recordEncryptionConfigAutomaticReloadTimestamp("success", apiServerIDHash)
}

func recordEncryptionConfigAutomaticReloadTimestamp(result, apiServerIDHash string) {
	encryptionConfigAutomaticReloadLastTimestampSeconds.WithLabelValues(result, apiServerIDHash).SetToCurrentTime()
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
