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

// Package metrics provides metrics for manifest-based admission configuration.
package metrics

import (
	"crypto/sha256"
	"fmt"
	"sync"

	"k8s.io/apiserver/pkg/util/configmetrics"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "manifest_admission_config_controller"
)

var (
	admissionManifestAutomaticReloadsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reloads_total",
			Help:           "Total number of automatic reloads of admission manifest configuration split by status, plugin, and apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "plugin", "apiserver_id_hash"},
	)

	admissionManifestAutomaticReloadLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "automatic_reload_last_timestamp_seconds",
			Help:           "Timestamp of the last automatic reload of admission manifest configuration split by status, plugin, and apiserver identity.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "plugin", "apiserver_id_hash"},
	)

	admissionManifestLastConfigInfo = metrics.NewDesc(
		metrics.BuildFQName(namespace, subsystem, "last_config_info"),
		"Information about the last applied admission manifest configuration with hash as label, split by plugin and apiserver identity.",
		[]string{"plugin", "apiserver_id_hash", "hash"},
		nil,
		metrics.ALPHA,
		"",
	)
)

// ManifestType represents the admission plugin name for which manifests are being tracked.
type ManifestType string

const (
	// ValidatingWebhookManifestType represents validating webhook configurations.
	ValidatingWebhookManifestType ManifestType = "ValidatingAdmissionWebhook"
	// MutatingWebhookManifestType represents mutating webhook configurations.
	MutatingWebhookManifestType ManifestType = "MutatingAdmissionWebhook"
	// VAPManifestType represents validating admission policy configurations.
	VAPManifestType ManifestType = "ValidatingAdmissionPolicy"
	// MAPManifestType represents mutating admission policy configurations.
	MAPManifestType ManifestType = "MutatingAdmissionPolicy"
)

var registerMetrics sync.Once

// configHashProviders stores providers per plugin for config info metrics.
// Each plugin has its own provider that stores [apiserver_id_hash, config_hash].
var configHashProviders = map[ManifestType]*configmetrics.AtomicHashProvider{
	ValidatingWebhookManifestType: configmetrics.NewAtomicHashProvider(),
	MutatingWebhookManifestType:   configmetrics.NewAtomicHashProvider(),
	VAPManifestType:               configmetrics.NewAtomicHashProvider(),
	MAPManifestType:               configmetrics.NewAtomicHashProvider(),
}

// multiTypeConfigInfoCollector emits config info metrics for multiple manifest types.
type multiTypeConfigInfoCollector struct {
	metrics.BaseStableCollector
	desc *metrics.Desc
}

var _ metrics.StableCollector = &multiTypeConfigInfoCollector{}

func (c *multiTypeConfigInfoCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- c.desc
}

func (c *multiTypeConfigInfoCollector) CollectWithStability(ch chan<- metrics.Metric) {
	// Emit a metric for each plugin that has data
	for manifestType, provider := range configHashProviders {
		hashes := provider.GetCurrentHashes()
		if len(hashes) >= 2 && len(hashes[0]) > 0 {
			// hashes contains [apiserver_id_hash, config_hash]
			// Prepend the plugin to match label order: ["plugin", "apiserver_id_hash", "hash"]
			labelValues := append([]string{string(manifestType)}, hashes...)
			ch <- metrics.NewLazyConstMetric(c.desc, metrics.GaugeValue, 1, labelValues...)
		}
	}
}

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(admissionManifestAutomaticReloadsTotal)
		legacyregistry.MustRegister(admissionManifestAutomaticReloadLastTimestampSeconds)
		// Use a custom collector that emits metrics for all manifest types
		legacyregistry.CustomMustRegister(&multiTypeConfigInfoCollector{desc: admissionManifestLastConfigInfo})
	})
}

func ResetMetricsForTest() {
	admissionManifestAutomaticReloadsTotal.Reset()
	admissionManifestAutomaticReloadLastTimestampSeconds.Reset()
}

// RecordAutomaticReloadFailure records a failed reload attempt for the given manifest type.
func RecordAutomaticReloadFailure(manifestType ManifestType, apiServerID string) {
	apiServerIDHash := getHash(apiServerID)
	admissionManifestAutomaticReloadsTotal.WithLabelValues("failure", string(manifestType), apiServerIDHash).Inc()
	admissionManifestAutomaticReloadLastTimestampSeconds.WithLabelValues("failure", string(manifestType), apiServerIDHash).SetToCurrentTime()
}

// RecordAutomaticReloadSuccess records a successful reload for the given manifest type.
func RecordAutomaticReloadSuccess(manifestType ManifestType, apiServerID, configHash string) {
	apiServerIDHash := getHash(apiServerID)
	admissionManifestAutomaticReloadsTotal.WithLabelValues("success", string(manifestType), apiServerIDHash).Inc()
	admissionManifestAutomaticReloadLastTimestampSeconds.WithLabelValues("success", string(manifestType), apiServerIDHash).SetToCurrentTime()

	RecordLastConfigInfo(manifestType, apiServerID, configHash)
}

// RecordLastConfigInfo records the hash of the last successfully loaded configuration.
func RecordLastConfigInfo(manifestType ManifestType, apiServerID, configHash string) {
	if provider, ok := configHashProviders[manifestType]; ok {
		// Store [apiserver_id_hash, config_hash] - the multiTypeConfigInfoCollector prepends the type
		provider.SetHashes(getHash(apiServerID), configHash)
	}
}

func getHash(data string) string {
	if len(data) == 0 {
		return ""
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(data)))
}
