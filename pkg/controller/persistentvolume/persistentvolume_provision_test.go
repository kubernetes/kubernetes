/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package persistentvolume

import (
	"errors"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

// Test single call to syncVolume, expecting provisioning to happen.
// 1. Fill in the controller with initial data
// 2. Call the syncVolume *once*.
// 3. Compare resulting volumes with expected volumes.
func TestProvisionSync(t *testing.T) {
	tests := []controllerTest{
		{
			// Provision a volume
			"11-1 - successful provision",
			novolumes,
			newVolumeArray("pv-provisioned-for-uid11-1", "1Gi", "uid11-1", "claim11-1", api.VolumeBound, api.PersistentVolumeReclaimDelete, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim11-1", "uid11-1", "1Gi", "", api.ClaimPending, annClass),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-1", "uid11-1", "1Gi", "", api.ClaimPending, annClass),
			noevents, wrapTestWithControllerConfig(operationProvision, []error{nil}, testSyncClaim),
		},
		{
			// Provision failure - plugin not found
			"11-2 - plugin not found",
			novolumes,
			novolumes,
			newClaimArray("claim11-2", "uid11-2", "1Gi", "", api.ClaimPending, annClass),
			newClaimArray("claim11-2", "uid11-2", "1Gi", "", api.ClaimPending, annClass),
			[]string{"Warning ProvisioningFailed"},
			testSyncClaim,
		},
		{
			// Provision failure - newProvisioner returns error
			"11-3 - newProvisioner failure",
			novolumes,
			novolumes,
			newClaimArray("claim11-3", "uid11-3", "1Gi", "", api.ClaimPending, annClass),
			newClaimArray("claim11-3", "uid11-3", "1Gi", "", api.ClaimPending, annClass),
			[]string{"Warning ProvisioningFailed"},
			wrapTestWithControllerConfig(operationProvision, []error{}, testSyncClaim),
		},
		{
			// Provision failure - Provision returns error
			"11-4 - provision failure",
			novolumes,
			novolumes,
			newClaimArray("claim11-4", "uid11-4", "1Gi", "", api.ClaimPending, annClass),
			newClaimArray("claim11-4", "uid11-4", "1Gi", "", api.ClaimPending, annClass),
			[]string{"Warning ProvisioningFailed"},
			wrapTestWithControllerConfig(operationProvision, []error{errors.New("Moc provisioner error")}, testSyncClaim),
		},
		{
			// Provision success - there is already a volume available, still
			// we provision a new one when requested.
			"11-6 - provisioning when there is a volume available",
			newVolumeArray("volume11-6", "1Gi", "", "", api.VolumePending, api.PersistentVolumeReclaimRetain),
			[]*api.PersistentVolume{
				newVolume("volume11-6", "1Gi", "", "", api.VolumePending, api.PersistentVolumeReclaimRetain),
				newVolume("pv-provisioned-for-uid11-6", "1Gi", "uid11-6", "claim11-6", api.VolumeBound, api.PersistentVolumeReclaimDelete, annBoundByController, annDynamicallyProvisioned),
			},
			newClaimArray("claim11-6", "uid11-6", "1Gi", "", api.ClaimPending, annClass),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-6", "uid11-6", "1Gi", "", api.ClaimPending, annClass),
			noevents,
			// No provisioning plugin confingure - makes the test fail when
			// the controller errorneously tries to provision something
			wrapTestWithControllerConfig(operationProvision, []error{nil}, testSyncClaim),
		},
		/*		{
				// Provision success? - claim is bound before provisioner creates
				// a volume.
				"11-7 - claim is bound before provisioning",
				novolumes,
				novolumes,
				[]*api.PersistentVolumeClaim{
					newClaim("claim11-7", "uid11-7", "1Gi", "", api.ClaimPending, annClass),
				},
				[]*api.PersistentVolumeClaim{
					newClaim("claim11-7", "uid11-7", "1Gi", "volume11-7", api.ClaimBound, annClass, annBindCompleted),
				},
				[]string{"Warning ProvisioningFailed"},
				getSyncClaimWithOperation(operationProvision, []error{errors.New("Moc provisioner error")}),
			}, */
	}
	runSyncTests(t, tests)
}

// Test multiple calls to syncClaim/syncVolume and periodic sync of all
// volume/claims. The test follows this pattern:
// 0. Load the controller with initial data.
// 1. Call controllerTest.testCall() once as in TestSync()
// 2. For all volumes/claims changed by previous syncVolume/syncClaim calls,
//    call appropriate syncVolume/syncClaim (simulating "volume/claim changed"
//    events). Go to 2. if these calls change anything.
// 3. When all changes are processed and no new changes were made, call
//    syncVolume/syncClaim on all volumes/claims (simulating "periodic sync").
// 4. If some changes were done by step 3., go to 2. (simulation of
//    "volume/claim updated" events, eventually performing step 3. again)
// 5. When 3. does not do any changes, finish the tests and compare final set
//    of volumes/claims with expected claims/volumes and report differences.
// Some limit of calls in enforced to prevent endless loops.
func TestProvisionMultiSync(t *testing.T) {
	tests := []controllerTest{
		{
			// Provision a volume with binding
			"12-1 - successful provision",
			novolumes,
			newVolumeArray("pv-provisioned-for-uid12-1", "1Gi", "uid12-1", "claim12-1", api.VolumeBound, api.PersistentVolumeReclaimDelete, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim12-1", "uid12-1", "1Gi", "", api.ClaimPending, annClass),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim12-1", "uid12-1", "1Gi", "pv-provisioned-for-uid12-1", api.ClaimBound, annClass, annBoundByController, annBindCompleted),
			noevents, wrapTestWithControllerConfig(operationProvision, []error{nil}, testSyncClaim),
		},
	}

	runMultisyncTests(t, tests)
}
