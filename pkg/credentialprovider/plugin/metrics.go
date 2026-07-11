/*
Copyright 2021 The Kubernetes Authors.

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

package plugin

import (
	"sync"

	"k8s.io/apiserver/pkg/util/configmetrics"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	KubeletSubsystem = "kubelet"
)

var (
	registerOnce sync.Once

	kubeletCredentialProviderPluginErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "credential_provider_plugin_errors_total",
			Help:           "Number of errors from credential provider plugin",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"plugin_name"},
	)

	kubeletCredentialProviderPluginDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      KubeletSubsystem,
			Name:           "credential_provider_plugin_duration",
			Help:           "Duration of execution in seconds for credential provider plugin",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"plugin_name"},
	)

	// kubeletCredentialProviderConfigInfo provides information about the credential provider configuration.
	// The hash label is typically constant for the lifetime of the kubelet process, changing only when
	// the configuration is updated and the kubelet is restarted.
	kubeletCredentialProviderConfigInfo = metrics.NewDesc(
		metrics.BuildFQName("", KubeletSubsystem, "credential_provider_config_info"),
		"Information about the last applied credential provider configuration with hash as label",
		[]string{"hash"},
		nil,
		metrics.ALPHA,
		"",
	)
)

var configHashProvider = configmetrics.NewAtomicHashProvider()

// registerMetrics registers credential provider metrics.
func registerMetrics() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(kubeletCredentialProviderPluginErrors)
		legacyregistry.MustRegister(kubeletCredentialProviderPluginDuration)
		legacyregistry.CustomMustRegister(configmetrics.NewConfigInfoCustomCollector(kubeletCredentialProviderConfigInfo, configHashProvider))
	})
}

// recordCredentialProviderConfigHash records the hash of the credential provider configuration
func recordCredentialProviderConfigHash(configHash string) {
	configHashProvider.SetHashes(configHash)
}
