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

package rest

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "kube_apiserver"
	subsystem = "pod_logs"

	usageEnforce     = "enforce_tls"
	usageSkipAllowed = "skip_tls_allowed"
)

var (
	// podLogsUsage counts and categorizes how the insecure backend skip TLS option is used and allowed.
	podLogsUsage = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "insecure_backend_total",
			Help:           "Total number of requests for pods/logs sliced by usage type: enforce_tls, skip_tls_allowed, skip_tls_denied",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"usage"},
	)

	// deprecatedPodLogsUsage counts and categorizes how the insecure backend skip TLS option is used and allowed.
	deprecatedPodLogsUsage = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:         namespace,
			Subsystem:         subsystem,
			Name:              "pods_logs_insecure_backend_total",
			Help:              "Total number of requests for pods/logs sliced by usage type: enforce_tls, skip_tls_allowed, skip_tls_denied",
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.27.0",
		},
		[]string{"usage"},
	)

	// podLogsTLSFailure counts how many attempts to get pod logs fail on tls verification
	podLogsTLSFailure = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "backend_tls_failure_total",
			Help:           "Total number of requests for pods/logs that failed due to kubelet server TLS verification",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// deprecatedPodLogsTLSFailure counts how many attempts to get pod logs fail on tls verification
	deprecatedPodLogsTLSFailure = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:         namespace,
			Subsystem:         subsystem,
			Name:              "pods_logs_backend_tls_failure_total",
			Help:              "Total number of requests for pods/logs that failed due to kubelet server TLS verification",
			StabilityLevel:    metrics.ALPHA,
			DeprecatedVersion: "1.27.0",
		},
	)
)

var registerMetricsOnce sync.Once

func registerMetrics() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(podLogsUsage)
		legacyregistry.MustRegister(podLogsTLSFailure)
		legacyregistry.MustRegister(deprecatedPodLogsUsage)
		legacyregistry.MustRegister(deprecatedPodLogsTLSFailure)
	})
}
