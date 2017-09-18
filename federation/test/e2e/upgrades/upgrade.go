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

// Package upgrades provides a framework for testing Kubernetes federation
// features before, during, and after different types of upgrades.
package upgrades

import fedframework "k8s.io/kubernetes/federation/test/e2e/framework"

// FederationUpgradeType represents different types of federation upgrades.
type FederationUpgradeType int

const (
	// FCPUpgrade indicates that federation control plane is being upgraded.
	FCPUpgrade FederationUpgradeType = iota

	// FederatedClustersUpgrade indicates that federated clusters are being upgraded.
	FederatedClustersUpgrade

	// FCPUpgradeFollowedByFederatedClustersUpgrade indicates that federation control plane is upgraded
	// followed by federated clusters upgrade.
	FCPUpgradeFollowedByFederatedClustersUpgrade

	// FederatedClustersUpgradeFollowedByFCPUpgrade indicates that federated clusters are upgraded
	// followed by federation control plane upgrade.
	FederatedClustersUpgradeFollowedByFCPUpgrade
)

// Test is an interface for federation upgrade tests.
type Test interface {
	// Setup should create and verify whatever objects need to
	// exist before the upgrade disruption starts.
	Setup(f *fedframework.Framework)

	// Test will run during the upgrade. When the upgrade is
	// complete, done will be closed and final validation can
	// begin.
	Test(f *fedframework.Framework, done <-chan struct{}, upgrade FederationUpgradeType)

	// TearDown should clean up any objects that are created that
	// aren't already cleaned up by the framework.
	Teardown(f *fedframework.Framework)
}
