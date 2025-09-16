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

	"k8s.io/apiserver/pkg/util/configmetrics"
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

	encryptionConfigLastConfigInfo = metrics.NewDesc(
		metrics.BuildFQName(namespace, subsystem, "last_config_info"),
		"Information about the last applied encryption configuration with hash as label, split by apiserver identity.",
		[]string{"apiserver_id_hash", "hash"},
		nil,
		metrics.ALPHA,
		"",
	)
)

var registerMetrics sync.Once
var hashPool *sync.Pool
var configHashProvider = configmetrics.NewAtomicHashProvider()

func RegisterMetrics() {
	registerMetrics.Do(func() {
		hashPool = &sync.Pool{
			New: func() interface{} {
				return sha256.New()
			},
		}
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadsTotal)
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadLastTimestampSeconds)
		legacyregistry.CustomMustRegister(configmetrics.NewConfigInfoCustomCollector(encryptionConfigLastConfigInfo, configHashProvider))
	})
}

func ResetMetricsForTest() {
	encryptionConfigAutomaticReloadsTotal.Reset()
	encryptionConfigAutomaticReloadLastTimestampSeconds.Reset()
}

func RecordEncryptionConfigAutomaticReloadFailure(apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	encryptionConfigAutomaticReloadsTotal.WithLabelValues("failure", apiServerIDHash).Inc()
	recordEncryptionConfigAutomaticReloadTimestamp("failure", apiServerIDHash)
}

func RecordEncryptionConfigAutomaticReloadSuccess(apiServerID, encryptionConfigDataHash string) {
	apiServerIDHash := getHash(apiServerID)
	encryptionConfigAutomaticReloadsTotal.WithLabelValues("success", apiServerIDHash).Inc()
	recordEncryptionConfigAutomaticReloadTimestamp("success", apiServerIDHash)

	RecordEncryptionConfigLastConfigInfo(apiServerID, encryptionConfigDataHash)
}

func recordEncryptionConfigAutomaticReloadTimestamp(result, apiServerIDHash string) {
	encryptionConfigAutomaticReloadLastTimestampSeconds.WithLabelValues(result, apiServerIDHash).SetToCurrentTime()
}

func RecordEncryptionConfigLastConfigInfo(apiServerID, encryptionConfigDataHash string) {
	configHashProvider.SetHashes(getHash(apiServerID), encryptionConfigDataHash)
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
