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

	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
)

// Test single call to syncVolume, expecting recycling to happen.
// 1. Fill in the controller with initial data
// 2. Call the syncVolume *once*.
// 3. Compare resulting volumes with expected volumes.
func TestRecycleSync(t *testing.T) {
	tests := []controllerTest{
		{
			// recycle volume bound by controller
			"6-1 - successful recycle",
			newVolumeArray("volume6-1", "1Gi", "uid6-1", "claim6-1", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, annBoundByController),
			newVolumeArray("volume6-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle),
			noclaims,
			noclaims,
			noevents, noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// recycle volume bound by user
			"6-2 - successful recycle with prebound volume",
			newVolumeArray("volume6-2", "1Gi", "uid6-2", "claim6-2", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			newVolumeArray("volume6-2", "1Gi", "", "claim6-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle),
			noclaims,
			noclaims,
			noevents, noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// recycle failure - plugin not found
			"6-3 - plugin not found",
			newVolumeArray("volume6-3", "1Gi", "uid6-3", "claim6-3", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			withMessage("No recycler plugin found for the volume!", newVolumeArray("volume6-3", "1Gi", "uid6-3", "claim6-3", v1.VolumeFailed, v1.PersistentVolumeReclaimRecycle)),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedRecycle"}, noerrors, testSyncVolume,
		},
		{
			// recycle failure - newRecycler returns error
			"6-4 - newRecycler returns error",
			newVolumeArray("volume6-4", "1Gi", "uid6-4", "claim6-4", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			withMessage("Failed to create recycler: Mock plugin error: no recycleCalls configured", newVolumeArray("volume6-4", "1Gi", "uid6-4", "claim6-4", v1.VolumeFailed, v1.PersistentVolumeReclaimRecycle)),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedRecycle"}, noerrors,
			wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume),
		},
		{
			// recycle failure - recycle returns error
			"6-5 - recycle returns error",
			newVolumeArray("volume6-5", "1Gi", "uid6-5", "claim6-5", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			withMessage("Recycler failed: Mock recycle error", newVolumeArray("volume6-5", "1Gi", "uid6-5", "claim6-5", v1.VolumeFailed, v1.PersistentVolumeReclaimRecycle)),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedRecycle"}, noerrors,
			wrapTestWithReclaimCalls(operationRecycle, []error{errors.New("Mock recycle error")}, testSyncVolume),
		},
		{
			// recycle success(?) - volume is deleted before doRecycle() starts
			"6-6 - volume is deleted before recycling",
			newVolumeArray("volume6-6", "1Gi", "uid6-6", "claim6-6", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			novolumes,
			noclaims,
			noclaims,
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *volumeReactor) {
				// Delete the volume before recycle operation starts
				reactor.lock.Lock()
				delete(reactor.volumes, "volume6-6")
				reactor.lock.Unlock()
			}),
		},
		{
			// recycle success(?) - volume is recycled by previous recycler just
			// at the time new doRecycle() starts. This simulates "volume no
			// longer needs recycling, skipping".
			"6-7 - volume is deleted before recycling",
			newVolumeArray("volume6-7", "1Gi", "uid6-7", "claim6-7", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, annBoundByController),
			newVolumeArray("volume6-7", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle),
			noclaims,
			noclaims,
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *volumeReactor) {
				// Mark the volume as Available before the recycler starts
				reactor.lock.Lock()
				volume := reactor.volumes["volume6-7"]
				volume.Spec.ClaimRef = nil
				volume.Status.Phase = v1.VolumeAvailable
				volume.Annotations = nil
				reactor.lock.Unlock()
			}),
		},
		{
			// recycle success(?) - volume bound by user is recycled by previous
			// recycler just at the time new doRecycle() starts. This simulates
			// "volume no longer needs recycling, skipping" with volume bound by
			// user.
			"6-8 - prebound volume is deleted before recycling",
			newVolumeArray("volume6-8", "1Gi", "uid6-8", "claim6-8", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			newVolumeArray("volume6-8", "1Gi", "", "claim6-8", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle),
			noclaims,
			noclaims,
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *volumeReactor) {
				// Mark the volume as Available before the recycler starts
				reactor.lock.Lock()
				volume := reactor.volumes["volume6-8"]
				volume.Spec.ClaimRef.UID = ""
				volume.Status.Phase = v1.VolumeAvailable
				reactor.lock.Unlock()
			}),
		},
		{
			// recycle success - volume bound by user is recycled, while a new
			// claim is created with another UID.
			"6-9 - prebound volume is recycled while the claim exists",
			newVolumeArray("volume6-9", "1Gi", "uid6-9", "claim6-9", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			newVolumeArray("volume6-9", "1Gi", "", "claim6-9", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle),
			newClaimArray("claim6-9", "uid6-9-x", "10Gi", "", v1.ClaimPending),
			newClaimArray("claim6-9", "uid6-9-x", "10Gi", "", v1.ClaimPending),
			noevents, noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// volume has unknown reclaim policy - failure expected
			"6-10 - unknown reclaim policy",
			newVolumeArray("volume6-10", "1Gi", "uid6-10", "claim6-10", v1.VolumeBound, "Unknown"),
			withMessage("Volume has unrecognized PersistentVolumeReclaimPolicy", newVolumeArray("volume6-10", "1Gi", "uid6-10", "claim6-10", v1.VolumeFailed, "Unknown")),
			noclaims,
			noclaims,
			[]string{"Warning VolumeUnknownReclaimPolicy"}, noerrors, testSyncVolume,
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
func TestRecycleMultiSync(t *testing.T) {
	tests := []controllerTest{
		{
			// recycle failure - recycle returns error. The controller should
			// try again.
			"7-1 - recycle returns error",
			newVolumeArray("volume7-1", "1Gi", "uid7-1", "claim7-1", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle),
			newVolumeArray("volume7-1", "1Gi", "", "claim7-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedRecycle"}, noerrors,
			wrapTestWithReclaimCalls(operationRecycle, []error{errors.New("Mock recycle error"), nil}, testSyncVolume),
		},
	}

	runMultisyncTests(t, tests, []*storage.StorageClass{}, "")
}
