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
			Name:           "credential_provider_plugin_errors",
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
)

// registerMetrics registers credential provider metrics.
func registerMetrics() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(kubeletCredentialProviderPluginErrors)
		legacyregistry.MustRegister(kubeletCredentialProviderPluginDuration)
	})
}
