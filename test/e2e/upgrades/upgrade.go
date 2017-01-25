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

// Package upgrades provides a framework for testing Kubernetes
// features before, during, and after different types of upgrades.
package upgrades

import "k8s.io/kubernetes/test/e2e/framework"

// UpgradeType represents different types of upgrades.
type UpgradeType int

const (
	// MasterUpgrade indicates that only the master is being upgraded.
	MasterUpgrade UpgradeType = iota

	// NodeUpgrade indicates that only the nodes are being upgraded.
	NodeUpgrade

	// ClusterUpgrade indicates that both master and nodes are
	// being upgraded.
	ClusterUpgrade
)

// Test is an interface for upgrade tests.
type Test interface {
	// Setup should create and verify whatever objects need to
	// exist before the upgrade disruption starts.
	Setup(f *framework.Framework)

	// Test will run during the upgrade. When the upgrade is
	// complete, done will be closed and final validation can
	// begin.
	Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType)

	// TearDown should clean up any objects that are created that
	// aren't already cleaned up by the framework.
	Teardown(f *framework.Framework)
}
