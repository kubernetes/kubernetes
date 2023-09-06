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

// DynamicResourceSubsystem - subsystem name used for Endpoint Slices.
const DynamicResourceSubsystem = "resource_controller"

var (
	// DynamicResourceCreateAttempts tracks the number of
	// ResourceClaim().Create calls (both successful and unsuccessful)
	DynamicResourceCreateAttempts = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      DynamicResourceSubsystem,
			Name:           "create_total",
			Help:           "Number of ResourceClaim creation requests",
			StabilityLevel: metrics.ALPHA,
		})
	// DynamicResourceCreateFailures tracks the number of unsuccessful
	// ResourceClaim().Create calls
	DynamicResourceCreateFailures = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      DynamicResourceSubsystem,
			Name:           "create_failures_total",
			Help:           "Number of ResourceClaim failed creation requests",
			StabilityLevel: metrics.ALPHA,
		})
)

var registerMetrics sync.Once

// RegisterMetrics registers DynamicResource metrics.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(DynamicResourceCreateAttempts)
		legacyregistry.MustRegister(DynamicResourceCreateFailures)
	})
}
