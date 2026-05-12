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

package framework

import "time"

var defaultTimeouts = TimeoutContext{
	Poll:                      2 * time.Second, // from the former e2e/framework/pod poll interval
	PodStart:                  5 * time.Minute,
	PodStartShort:             2 * time.Minute,
	PodStartSlow:              15 * time.Minute,
	PodDelete:                 5 * time.Minute,
	ClaimProvision:            5 * time.Minute,
	ClaimProvisionShort:       1 * time.Minute,
	DataSourceProvision:       5 * time.Minute,
	ClaimBound:                3 * time.Minute,
	PVReclaim:                 3 * time.Minute,
	PVBound:                   3 * time.Minute,
	PVCreate:                  3 * time.Minute,
	PVDelete:                  5 * time.Minute,
	PVDeleteSlow:              20 * time.Minute,
	SnapshotCreate:            5 * time.Minute,
	SnapshotDelete:            5 * time.Minute,
	SnapshotControllerMetrics: 5 * time.Minute,
	SystemPodsStartup:         10 * time.Minute,
	NodeSchedulable:           30 * time.Minute,
	SystemDaemonsetStartup:    5 * time.Minute,
	NodeNotReady:              3 * time.Minute,
}

// TimeoutContext contains timeout settings for several actions.
type TimeoutContext struct {
	// Poll is how long to wait between API calls when waiting for some condition.
	Poll time.Duration

	// PodStart is how long to wait for the pod to be started.
	// This value is the default for gomega.Eventually.
	PodStart time.Duration

	// PodStartShort is same as `PodStart`, but shorter.
	// Use it in a case-by-case basis, mostly when you are sure pod start will not be delayed.
	// This value is the default for gomega.Consistently.
	PodStartShort time.Duration

	// PodStartSlow is same as `PodStart`, but longer.
	// Use it in a case-by-case basis, mostly when you are sure pod start will take longer than usual.
	PodStartSlow time.Duration

	// PodDelete is how long to wait for the pod to be deleted.
	PodDelete time.Duration

	// ClaimProvision is how long claims have to become dynamically provisioned.
	ClaimProvision time.Duration

	// DataSourceProvision is how long claims have to become dynamically provisioned from source claim.
	DataSourceProvision time.Duration

	// ClaimProvisionShort is the same as `ClaimProvision`, but shorter.
	ClaimProvisionShort time.Duration

	// ClaimBound is how long claims have to become bound.
	ClaimBound time.Duration

	// PVReclaim is how long PVs have to become reclaimed.
	PVReclaim time.Duration

	// PVBound is how long PVs have to become bound.
	PVBound time.Duration

	// PVCreate is how long PVs have to be created.
	PVCreate time.Duration

	// PVDelete is how long PVs have to become deleted.
	PVDelete time.Duration

	// PVDeleteSlow is the same as PVDelete, but slower.
	PVDeleteSlow time.Duration

	// SnapshotCreate is how long for snapshot to create snapshotContent.
	SnapshotCreate time.Duration

	// SnapshotDelete is how long for snapshot to delete snapshotContent.
	SnapshotDelete time.Duration

	// SnapshotControllerMetrics is how long to wait for snapshot controller metrics.
	SnapshotControllerMetrics time.Duration

	// SystemPodsStartup is how long to wait for system pods to be running.
	SystemPodsStartup time.Duration

	// NodeSchedulable is how long to wait for all nodes to be schedulable.
	NodeSchedulable time.Duration

	// SystemDaemonsetStartup is how long to wait for all system daemonsets to be ready.
	SystemDaemonsetStartup time.Duration

	// NodeNotReady is how long to wait for a node to be not ready.
	NodeNotReady time.Duration
}

// NewTimeoutContext returns a TimeoutContext with all values set either to
// hard-coded defaults or a value that was configured when running the E2E
// suite. Should be called after command line parsing.
func NewTimeoutContext() *TimeoutContext {
	// Make a copy, otherwise the caller would have the ability to modify
	// the original values.
	copy := TestContext.timeouts
	return &copy
}

// PollInterval defines how long to wait between API server queries while
// waiting for some condition.
//
// This value is the default for gomega.Eventually and gomega.Consistently.
func PollInterval() time.Duration {
	return TestContext.timeouts.Poll
}
