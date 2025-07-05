/*
Copyright 2024 The Kubernetes Authors.

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
	"encoding/binary"
	"fmt"
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "authentication_config_controller"
)

var (
	authenticationConfigAutomaticReloadsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reloads_total",
			Help:           "Total number of automatic reloads of authentication configuration split by status and apiserver identity.",
			StabilityLevel: metrics.BETA,
		},
		[]string{"status", "apiserver_id_hash"},
	)

	authenticationConfigAutomaticReloadLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_last_timestamp_seconds",
			Help:           "Timestamp of the last automatic reload of authentication configuration split by status and apiserver identity.",
			StabilityLevel: metrics.BETA,
		},
		[]string{"status", "apiserver_id_hash"},
	)

	authenticationConfigAutomaticReloadLastConfigHash = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_last_config_hash",
			Help:           "Hash of the last applied authentication configuration split by apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"apiserver_id_hash"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(authenticationConfigAutomaticReloadsTotal)
		legacyregistry.MustRegister(authenticationConfigAutomaticReloadLastTimestampSeconds)
		legacyregistry.MustRegister(authenticationConfigAutomaticReloadLastConfigHash)
	})
}

func ResetMetricsForTest() {
	authenticationConfigAutomaticReloadsTotal.Reset()
	authenticationConfigAutomaticReloadLastTimestampSeconds.Reset()
	authenticationConfigAutomaticReloadLastConfigHash.Reset()
}

func RecordAuthenticationConfigAutomaticReloadFailure(apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	authenticationConfigAutomaticReloadsTotal.WithLabelValues("failure", apiServerIDHash).Inc()
	authenticationConfigAutomaticReloadLastTimestampSeconds.WithLabelValues("failure", apiServerIDHash).SetToCurrentTime()
}

func RecordAuthenticationConfigAutomaticReloadSuccess(apiServerID, authConfigData string) {
	apiServerIDHash := getHash(apiServerID)
	authenticationConfigAutomaticReloadsTotal.WithLabelValues("success", apiServerIDHash).Inc()
	authenticationConfigAutomaticReloadLastTimestampSeconds.WithLabelValues("success", apiServerIDHash).SetToCurrentTime()

	hash := sha256.Sum256([]byte(authConfigData))
	truncated := binary.BigEndian.Uint64(hash[0:8]) // Truncate to 8 bytes for the gauge

	authenticationConfigAutomaticReloadLastConfigHash.WithLabelValues(apiServerIDHash).Set(float64(truncated))
}

func getHash(data string) string {
	if len(data) == 0 {
		return ""
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(data)))
}
