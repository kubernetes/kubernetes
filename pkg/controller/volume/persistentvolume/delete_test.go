/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/storage"
)

// Test single call to syncVolume, expecting recycling to happen.
// 1. Fill in the controller with initial data
// 2. Call the syncVolume *once*.
// 3. Compare resulting volumes with expected volumes.
func TestDeleteSync(t *testing.T) {
	tests := []controllerTest{
		{
			// delete volume bound by controller
			"8-1 - successful delete",
			newVolumeArray("volume8-1", "1Gi", "uid8-1", "claim8-1", api.VolumeBound, api.PersistentVolumeReclaimDelete, annBoundByController),
			novolumes,
			noclaims,
			noclaims,
			noevents, noerrors,
			// Inject deleter into the controller and call syncVolume. The
			// deleter simulates one delete() call that succeeds.
			wrapTestWithReclaimCalls(operationDelete, []error{nil}, testSyncVolume),
		},
		{
			// delete volume bound by user
			"8-2 - successful delete with prebound volume",
			newVolumeArray("volume8-2", "1Gi", "uid8-2", "claim8-2", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			novolumes,
			noclaims,
			noclaims,
			noevents, noerrors,
			// Inject deleter into the controller and call syncVolume. The
			// deleter simulates one delete() call that succeeds.
			wrapTestWithReclaimCalls(operationDelete, []error{nil}, testSyncVolume),
		},
		{
			// delete failure - plugin not found
			"8-3 - plugin not found",
			newVolumeArray("volume8-3", "1Gi", "uid8-3", "claim8-3", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			withMessage("Error getting deleter volume plugin for volume \"volume8-3\": no volume plugin matched", newVolumeArray("volume8-3", "1Gi", "uid8-3", "claim8-3", api.VolumeFailed, api.PersistentVolumeReclaimDelete)),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedDelete"}, noerrors, testSyncVolume,
		},
		{
			// delete failure - newDeleter returns error
			"8-4 - newDeleter returns error",
			newVolumeArray("volume8-4", "1Gi", "uid8-4", "claim8-4", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			withMessage("Failed to create deleter for volume \"volume8-4\": Mock plugin error: no deleteCalls configured", newVolumeArray("volume8-4", "1Gi", "uid8-4", "claim8-4", api.VolumeFailed, api.PersistentVolumeReclaimDelete)),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedDelete"}, noerrors,
			wrapTestWithReclaimCalls(operationDelete, []error{}, testSyncVolume),
		},
		{
			// delete failure - delete() returns error
			"8-5 - delete returns error",
			newVolumeArray("volume8-5", "1Gi", "uid8-5", "claim8-5", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			withMessage("Delete of volume \"volume8-5\" failed: Mock delete error", newVolumeArray("volume8-5", "1Gi", "uid8-5", "claim8-5", api.VolumeFailed, api.PersistentVolumeReclaimDelete)),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedDelete"}, noerrors,
			wrapTestWithReclaimCalls(operationDelete, []error{errors.New("Mock delete error")}, testSyncVolume),
		},
		{
			// delete success(?) - volume is deleted before doDelete() starts
			"8-6 - volume is deleted before deleting",
			newVolumeArray("volume8-6", "1Gi", "uid8-6", "claim8-6", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			novolumes,
			noclaims,
			noclaims,
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithReclaimCalls(operationDelete, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *volumeReactor) {
				// Delete the volume before delete operation starts
				reactor.lock.Lock()
				delete(reactor.volumes, "volume8-6")
				reactor.lock.Unlock()
			}),
		},
		{
			// delete success(?) - volume is bound just at the time doDelete()
			// starts. This simulates "volume no longer needs recycling,
			// skipping".
			"8-7 - volume is bound before deleting",
			newVolumeArray("volume8-7", "1Gi", "uid8-7", "claim8-7", api.VolumeBound, api.PersistentVolumeReclaimDelete, annBoundByController),
			newVolumeArray("volume8-7", "1Gi", "uid8-7", "claim8-7", api.VolumeBound, api.PersistentVolumeReclaimDelete, annBoundByController),
			noclaims,
			newClaimArray("claim8-7", "uid8-7", "10Gi", "volume8-7", api.ClaimBound),
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithReclaimCalls(operationDelete, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *volumeReactor) {
				reactor.lock.Lock()
				defer reactor.lock.Unlock()
				// Bind the volume to resurrected claim (this should never
				// happen)
				claim := newClaim("claim8-7", "uid8-7", "10Gi", "volume8-7", api.ClaimBound)
				reactor.claims[claim.Name] = claim
				ctrl.claims.Add(claim)
				volume := reactor.volumes["volume8-7"]
				volume.Status.Phase = api.VolumeBound
			}),
		},
		{
			// delete success - volume bound by user is deleted, while a new
			// claim is created with another UID.
			"8-9 - prebound volume is deleted while the claim exists",
			newVolumeArray("volume8-9", "1Gi", "uid8-9", "claim8-9", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			novolumes,
			newClaimArray("claim8-9", "uid8-9-x", "10Gi", "", api.ClaimPending),
			newClaimArray("claim8-9", "uid8-9-x", "10Gi", "", api.ClaimPending),
			noevents, noerrors,
			// Inject deleter into the controller and call syncVolume. The
			// deleter simulates one delete() call that succeeds.
			wrapTestWithReclaimCalls(operationDelete, []error{nil}, testSyncVolume),
		},
	}
	runSyncTests(t, tests, []*storage.StorageClass{})
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
func TestDeleteMultiSync(t *testing.T) {
	tests := []controllerTest{
		{
			// delete failure - delete returns error. The controller should
			// try again.
			"9-1 - delete returns error",
			newVolumeArray("volume9-1", "1Gi", "uid9-1", "claim9-1", api.VolumeBound, api.PersistentVolumeReclaimDelete),
			novolumes,
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedDelete"}, noerrors,
			wrapTestWithReclaimCalls(operationDelete, []error{errors.New("Mock delete error"), nil}, testSyncVolume),
		},
	}

	runMultisyncTests(t, tests, []*storage.StorageClass{}, "")
}
