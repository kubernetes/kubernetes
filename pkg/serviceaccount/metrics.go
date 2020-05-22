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

package serviceaccount

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const kubeServiceAccountSubsystem = "serviceaccount"

var (
	// LegacyTokensTotal is the number of legacy tokens used against apiserver.
	legacyTokensTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeServiceAccountSubsystem,
			Name:           "legacy_tokens_total",
			Help:           "Cumulative legacy service account tokens used",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// StaleTokensTotal is the number of stale projected tokens not refreshed on
	// client side.
	staleTokensTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeServiceAccountSubsystem,
			Name:           "stale_tokens_total",
			Help:           "Cumulative stale projected service account tokens used",
			StabilityLevel: metrics.ALPHA,
		},
	)

	// ValidTokensTotal is the number of valid projected tokens used.
	validTokensTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      kubeServiceAccountSubsystem,
			Name:           "valid_tokens_total",
			Help:           "Cumulative valid projected service account tokens used",
			StabilityLevel: metrics.ALPHA,
		},
	)
)

var registerMetricsOnce sync.Once

func RegisterMetrics() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(legacyTokensTotal)
		legacyregistry.MustRegister(staleTokensTotal)
		legacyregistry.MustRegister(validTokensTotal)
	})
}
