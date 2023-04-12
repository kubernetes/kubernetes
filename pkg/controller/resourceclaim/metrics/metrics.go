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

package metrics

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// ResourceClaimSubsystem - subsystem name used for ResourceClaim creation
const ResourceClaimSubsystem = "resourceclaim_controller"

var (
	// ResourceClaimCreateAttempts tracks the number of
	// ResourceClaims().Create calls (both successful and unsuccessful)
	ResourceClaimCreateAttempts = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      ResourceClaimSubsystem,
			Name:           "create_attempts_total",
			Help:           "Number of ResourceClaims creation requests",
			StabilityLevel: metrics.ALPHA,
		})
	// ResourceClaimCreateFailures tracks the number of unsuccessful
	// ResourceClaims().Create calls
	ResourceClaimCreateFailures = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      ResourceClaimSubsystem,
			Name:           "create_failures_total",
			Help:           "Number of ResourceClaims creation request failures",
			StabilityLevel: metrics.ALPHA,
		})
)

var registerMetrics sync.Once

// RegisterMetrics registers ResourceClaim metrics.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ResourceClaimCreateAttempts)
		legacyregistry.MustRegister(ResourceClaimCreateFailures)
	})
}
