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
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "encryption_config_controller"
)

var (
	encryptionConfigAutomaticReloadFailureTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_failures_total",
			Help:           "Total number of failed automatic reloads of encryption configuration.",
			StabilityLevel: metrics.ALPHA,
		},
	)

	encryptionConfigAutomaticReloadSuccessTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_success_total",
			Help:           "Total number of successful automatic reloads of encryption configuration.",
			StabilityLevel: metrics.ALPHA,
		},
	)

	encryptionConfigAutomaticReloadLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_last_timestamp_seconds",
			Help:           "Timestamp of the last successful or failed automatic reload of encryption configuration.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status"},
	)
)

var registerMetrics sync.Once

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadFailureTotal)
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadSuccessTotal)
		legacyregistry.MustRegister(encryptionConfigAutomaticReloadLastTimestampSeconds)
	})
}

func RecordEncryptionConfigAutomaticReloadFailure() {
	encryptionConfigAutomaticReloadFailureTotal.Inc()
	recordEncryptionConfigAutomaticReloadTimestamp("failure")
}

func RecordEncryptionConfigAutomaticReloadSuccess() {
	encryptionConfigAutomaticReloadSuccessTotal.Inc()
	recordEncryptionConfigAutomaticReloadTimestamp("success")
}

func recordEncryptionConfigAutomaticReloadTimestamp(result string) {
	encryptionConfigAutomaticReloadLastTimestampSeconds.WithLabelValues(result).SetToCurrentTime()
}
