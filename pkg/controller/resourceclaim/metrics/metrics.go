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

// subsystem is intentionally generic because similar metrics exist also elsewhere.
const subsystem = "dynamic_resource_allocation"

type NumResourceClaimLabels struct {
	Allocated   string
	AdminAccess string
	Source      string
}

var (
	// NumResourceClaimsDesc tracks the number of ResourceClaims,
	// categorized by their allocation status, admin access, and source.
	// Source can be 'resource_claim_template' (created from a template),
	// 'extended_resource' (extended resources), or empty (manually created by a user).
	NumResourceClaimsDesc = metrics.NewDesc(
		metrics.BuildFQName("", subsystem, "resource_claims"),
		"Number of ResourceClaims, categorized by allocation status, admin access, and source. "+
			"Source can be 'resource_claim_template' (created from a template), "+
			"'extended_resource' (extended resources), or empty (manually created by a user).",
		[]string{"allocated", "admin_access", "source"}, nil,
		metrics.ALPHA, "")
)

var registerMetrics sync.Once

// testMode indicates whether we're running in test mode
// In test mode, we don't register the custom collector in the global registry
var testMode bool

// SetTestMode enables or disables test mode
func SetTestMode(enabled bool) {
	testMode = enabled
}

// RegisterMetrics registers ResourceClaim metrics.
func RegisterMetrics(collector metrics.StableCollector) {
	registerMetrics.Do(func() {
		if !testMode && collector != nil {
			// Only register custom collector in non-test mode
			legacyregistry.CustomMustRegister(collector)
		}
	})
}
