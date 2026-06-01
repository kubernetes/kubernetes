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

package metrics

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// subsystem is intentionally generic because these metrics are exposed in kube-controller-manager and kube-scheduler.
const subsystem = "dynamic_resource_allocation"

var (
	// ResourceClaimCreate tracks the total number of
	// ResourceClaims creation requests
	// categorized by their creation status and admin access.
	// Used by kube-controller-manager and kube-scheduler, so
	// the component where this metric gets collected is another dimension.
	ResourceClaimCreate = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "resourceclaim_creates_total",
			Help:           "Number of ResourceClaims creation requests, categorized by creation status and admin access",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"status", "admin_access"},
	)
)

var registerMetrics sync.Once

// RegisterMetrics registers ResourceClaim metrics.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ResourceClaimCreate)
	})
}
