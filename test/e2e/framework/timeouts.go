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

const (
	// Default timeouts to be used in TimeoutContext
	podStartTimeout                  = 5 * time.Minute
	podStartShortTimeout             = 2 * time.Minute
	podStartSlowTimeout              = 15 * time.Minute
	podDeleteTimeout                 = 5 * time.Minute
	claimProvisionTimeout            = 5 * time.Minute
	claimProvisionShortTimeout       = 1 * time.Minute
	claimBoundTimeout                = 3 * time.Minute
	pvReclaimTimeout                 = 3 * time.Minute
	pvBoundTimeout                   = 3 * time.Minute
	pvCreateTimeout                  = 3 * time.Minute
	pvDeleteTimeout                  = 3 * time.Minute
	pvDeleteSlowTimeout              = 20 * time.Minute
	snapshotCreateTimeout            = 5 * time.Minute
	snapshotDeleteTimeout            = 5 * time.Minute
	snapshotControllerMetricsTimeout = 5 * time.Minute
)

// TimeoutContext contains timeout settings for several actions.
type TimeoutContext struct {
	// PodStart is how long to wait for the pod to be started.
	PodStart time.Duration

	// PodStartShort is same as `PodStart`, but shorter.
	// Use it in a case-by-case basis, mostly when you are sure pod start will not be delayed.
	PodStartShort time.Duration

	// PodStartSlow is same as `PodStart`, but longer.
	// Use it in a case-by-case basis, mostly when you are sure pod start will take longer than usual.
	PodStartSlow time.Duration

	// PodDelete is how long to wait for the pod to be deleted.
	PodDelete time.Duration

	// ClaimProvision is how long claims have to become dynamically provisioned.
	ClaimProvision time.Duration

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
}

// NewTimeoutContextWithDefaults returns a TimeoutContext with default values.
func NewTimeoutContextWithDefaults() *TimeoutContext {
	return &TimeoutContext{
		PodStart:                  podStartTimeout,
		PodStartShort:             podStartShortTimeout,
		PodStartSlow:              podStartSlowTimeout,
		PodDelete:                 podDeleteTimeout,
		ClaimProvision:            claimProvisionTimeout,
		ClaimProvisionShort:       claimProvisionShortTimeout,
		ClaimBound:                claimBoundTimeout,
		PVReclaim:                 pvReclaimTimeout,
		PVBound:                   pvBoundTimeout,
		PVCreate:                  pvCreateTimeout,
		PVDelete:                  pvDeleteTimeout,
		PVDeleteSlow:              pvDeleteSlowTimeout,
		SnapshotCreate:            snapshotCreateTimeout,
		SnapshotDelete:            snapshotDeleteTimeout,
		SnapshotControllerMetrics: snapshotControllerMetricsTimeout,
	}
}
