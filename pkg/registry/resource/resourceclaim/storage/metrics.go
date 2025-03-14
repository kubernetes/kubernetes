/*
Copyright 2025 The Kubernetes Authors.

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

package storage

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "kube_apiserver"
	subsystem = "resourceclaim"
)

var (
	// resourceClaimUpdateStatusDevicesAttempts tracks the number of
	// ResourceClaims().Update calls (both successful and unsuccessful)
	resourceClaimUpdateStatusDevicesAttempts = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "update_status_devices_attempts_total",
			Help:           "Number of ResourceClaims update requests",
			StabilityLevel: metrics.ALPHA,
		})
	// resourceClaimUpdateStatusDevicesFailures tracks the number of unsuccessful
	// ResourceClaims().Update calls
	resourceClaimUpdateStatusDevicesFailures = metrics.NewCounter(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "update_status_devices_failures_total",
			Help:           "Number of ResourceClaims update request failures",
			StabilityLevel: metrics.ALPHA,
		})
)

func init() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(resourceClaimUpdateStatusDevicesAttempts)
		legacyregistry.MustRegister(resourceClaimUpdateStatusDevicesFailures)
	})
}

var registerMetricsOnce sync.Once
