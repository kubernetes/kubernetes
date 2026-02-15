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

package metrics

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

// Mock collector for testing NumResourceClaimsDesc
type mockResourceClaimCollector struct {
	metrics.BaseStableCollector
	allocated   string
	adminAccess string
	source      string
	value       float64
}

func (m *mockResourceClaimCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- NumResourceClaimsDesc
}

func (m *mockResourceClaimCollector) CollectWithStability(ch chan<- metrics.Metric) {
	ch <- metrics.NewLazyConstMetric(
		NumResourceClaimsDesc,
		metrics.GaugeValue,
		m.value,
		m.allocated,
		m.adminAccess,
		m.source,
	)
}

func TestResourceClaimMetrics(t *testing.T) {
	// Reset metrics to ensure clean state
	ResourceClaimCreate.Reset()

	// Register metrics
	SetTestMode(false)
	collector := &mockResourceClaimCollector{
		allocated:   "true",
		adminAccess: "true",
		source:      "resource_claim_template",
		value:       5.0,
	}
	RegisterMetrics(collector)

	// Test ResourceClaimCreate
	ResourceClaimCreate.WithLabelValues("success", "true").Inc()
	ResourceClaimCreate.WithLabelValues("error", "false").Inc()

	wantCreate := `
		# HELP resourceclaim_controller_creates_total [ALPHA] Number of ResourceClaims creation requests, categorized by creation status and admin access
		# TYPE resourceclaim_controller_creates_total counter
		resourceclaim_controller_creates_total{admin_access="false",status="error"} 1
		resourceclaim_controller_creates_total{admin_access="true",status="success"} 1
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantCreate), "resourceclaim_controller_creates_total"); err != nil {
		t.Fatal(err)
	}

	// Test NumResourceClaimsDesc (resource_claims)
	wantResourceClaims := `
		# HELP resourceclaim_controller_resource_claims [BETA] Number of ResourceClaims, categorized by allocation status, admin access, and source. Source can be 'resource_claim_template' (created from a template), 'extended_resource' (extended resources), or empty (manually created by a user).
		# TYPE resourceclaim_controller_resource_claims gauge
		resourceclaim_controller_resource_claims{admin_access="true",allocated="true",source="resource_claim_template"} 5
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantResourceClaims), "resourceclaim_controller_resource_claims"); err != nil {
		t.Fatal(err)
	}
}
