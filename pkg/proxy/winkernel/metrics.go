/*
Copyright 2017 The Kubernetes Authors.

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

package winkernel

import (
	"sync"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/proxy/metrics"
)

var registerMetricsOnce sync.Once

// RegisterMetrics registers kube-proxy metrics for Windows modes.
func RegisterMetrics() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(metrics.SyncProxyRulesLatency)
		legacyregistry.MustRegister(metrics.SyncProxyRulesLastTimestamp)
		legacyregistry.MustRegister(metrics.EndpointChangesPending)
		legacyregistry.MustRegister(metrics.EndpointChangesTotal)
		legacyregistry.MustRegister(metrics.ServiceChangesPending)
		legacyregistry.MustRegister(metrics.ServiceChangesTotal)
		legacyregistry.MustRegister(metrics.SyncProxyRulesLastQueuedTimestamp)
	})
}
