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

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2/ktesting"
	pvtesting "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/testing"
)

// Test single call to syncVolume, expecting recycling to happen.
// 1. Fill in the controller with initial data
// 2. Call the syncVolume *once*.
// 3. Compare resulting volumes with expected volumes.
func TestRecycleSync(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	runningPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "runningPod",
			Namespace: testNamespace,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "vol1",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "runningClaim",
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}

	pendingPod := runningPod.DeepCopy()
	pendingPod.Name = "pendingPod"
	pendingPod.Status.Phase = v1.PodPending
	pendingPod.Spec.Volumes[0].PersistentVolumeClaim.ClaimName = "pendingClaim"

	completedPod := runningPod.DeepCopy()
	completedPod.Name = "completedPod"
	completedPod.Status.Phase = v1.PodSucceeded
	completedPod.Spec.Volumes[0].PersistentVolumeClaim.ClaimName = "completedClaim"

	pods := []*v1.Pod{
		runningPod,
		pendingPod,
		completedPod,
	}

	tests := []controllerTest{
		{
			// recycle volume bound by controller
			name:            "6-1 - successful recycle",
			initialVolumes:  newVolumeArray("volume6-1", "1Gi", "uid6-1", "claim6-1", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume6-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			test: wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// recycle volume bound by user
			name:            "6-2 - successful recycle with prebound volume",
			initialVolumes:  newVolumeArray("volume6-2", "1Gi", "uid6-2", "claim6-2", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: newVolumeArray("volume6-2", "1Gi", "", "claim6-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			test: wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// recycle failure - plugin not found
			name:            "6-3 - plugin not found",
			initialVolumes:  newVolumeArray("volume6-3", "1Gi", "uid6-3", "claim6-3", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: withMessage("No recycler plugin found for the volume!", newVolumeArray("volume6-3", "1Gi", "uid6-3", "claim6-3", v1.VolumeFailed, v1.PersistentVolumeReclaimRecycle, classEmpty)),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  []string{"Warning VolumeFailedRecycle"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// recycle failure - Recycle returns error
			name:            "6-4 - newRecycler returns error",
			initialVolumes:  newVolumeArray("volume6-4", "1Gi", "uid6-4", "claim6-4", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: withMessage("Recycle failed: Mock plugin error: no recycleCalls configured", newVolumeArray("volume6-4", "1Gi", "uid6-4", "claim6-4", v1.VolumeFailed, v1.PersistentVolumeReclaimRecycle, classEmpty)),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  []string{"Warning VolumeFailedRecycle"},
			errors:          noerrors,
			test:            wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume),
		},
		{
			// recycle failure - recycle returns error
			name:            "6-5 - recycle returns error",
			initialVolumes:  newVolumeArray("volume6-5", "1Gi", "uid6-5", "claim6-5", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: withMessage("Recycle failed: Mock recycle error", newVolumeArray("volume6-5", "1Gi", "uid6-5", "claim6-5", v1.VolumeFailed, v1.PersistentVolumeReclaimRecycle, classEmpty)),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  []string{"Warning VolumeFailedRecycle"},
			errors:          noerrors,
			test:            wrapTestWithReclaimCalls(operationRecycle, []error{errors.New("Mock recycle error")}, testSyncVolume),
		},
		{
			// recycle success(?) - volume is deleted before doRecycle() starts
			name:            "6-6 - volume is deleted before recycling",
			initialVolumes:  newVolumeArray("volume6-6", "1Gi", "uid6-6", "claim6-6", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: novolumes,
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			test: wrapTestWithInjectedOperation(ctx, wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
				// Delete the volume before recycle operation starts
				reactor.DeleteVolume("volume6-6")
			}),
		},
		{
			// recycle success(?) - volume is recycled by previous recycler just
			// at the time new doRecycle() starts. This simulates "volume no
			// longer needs recycling, skipping".
			name:            "6-7 - volume is deleted before recycling",
			initialVolumes:  newVolumeArray("volume6-7", "1Gi", "uid6-7", "claim6-7", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume6-7", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			test: wrapTestWithInjectedOperation(ctx, wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
				// Mark the volume as Available before the recycler starts
				reactor.MarkVolumeAvailable("volume6-7")
			}),
		},
		{
			// recycle success(?) - volume bound by user is recycled by previous
			// recycler just at the time new doRecycle() starts. This simulates
			// "volume no longer needs recycling, skipping" with volume bound by
			// user.
			name:            "6-8 - prebound volume is deleted before recycling",
			initialVolumes:  newVolumeArray("volume6-8", "1Gi", "uid6-8", "claim6-8", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: newVolumeArray("volume6-8", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			test: wrapTestWithInjectedOperation(ctx, wrapTestWithReclaimCalls(operationRecycle, []error{}, testSyncVolume), func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
				// Mark the volume as Available before the recycler starts
				reactor.MarkVolumeAvailable("volume6-8")
			}),
		},
		{
			// recycle success - volume bound by user is recycled, while a new
			// claim is created with another UID.
			name:            "6-9 - prebound volume is recycled while the claim exists",
			initialVolumes:  newVolumeArray("volume6-9", "1Gi", "uid6-9", "claim6-9", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			expectedVolumes: newVolumeArray("volume6-9", "1Gi", "", "claim6-9", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   newClaimArray("claim6-9", "uid6-9-x", "10Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim6-9", "uid6-9-x", "10Gi", "", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			test: wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// volume has unknown reclaim policy - failure expected
			name:            "6-10 - unknown reclaim policy",
			initialVolumes:  newVolumeArray("volume6-10", "1Gi", "uid6-10", "claim6-10", v1.VolumeBound, "Unknown", classEmpty),
			expectedVolumes: withMessage("Volume has unrecognized PersistentVolumeReclaimPolicy", newVolumeArray("volume6-10", "1Gi", "uid6-10", "claim6-10", v1.VolumeFailed, "Unknown", classEmpty)),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  []string{"Warning VolumeUnknownReclaimPolicy"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// volume is used by a running pod - failure expected
			name:            "6-11 - used by running pod",
			initialVolumes:  newVolumeArray("volume6-11", "1Gi", "uid6-11", "runningClaim", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume6-11", "1Gi", "uid6-11", "runningClaim", v1.VolumeReleased, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  []string{"Normal VolumeFailedRecycle"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// volume is used by a pending pod - failure expected
			name:            "6-12 - used by pending pod",
			initialVolumes:  newVolumeArray("volume6-12", "1Gi", "uid6-12", "pendingClaim", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume6-12", "1Gi", "uid6-12", "pendingClaim", v1.VolumeReleased, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  []string{"Normal VolumeFailedRecycle"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// volume is used by a completed pod - recycle succeeds
			name:            "6-13 - used by completed pod",
			initialVolumes:  newVolumeArray("volume6-13", "1Gi", "uid6-13", "completedClaim", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume6-13", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			// Inject recycler into the controller and call syncVolume. The
			// recycler simulates one recycle() call that succeeds.
			test: wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
		{
			// volume is used by a completed pod, pod using claim with the same name bound to different pv is running, should recycle
			name:            "6-14 - seemingly used by running pod",
			initialVolumes:  newVolumeArray("volume6-14", "1Gi", "uid6-14", "completedClaim", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume6-14", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			initialClaims:   newClaimArray("completedClaim", "uid6-14-x", "10Gi", "", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("completedClaim", "uid6-14-x", "10Gi", "", v1.ClaimBound, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            wrapTestWithReclaimCalls(operationRecycle, []error{nil}, testSyncVolume),
		},
	}
	runSyncTests(t, ctx, tests, []*storage.StorageClass{}, pods)
}

// Test multiple calls to syncClaim/syncVolume and periodic sync of all
// volume/claims. The test follows this pattern:
//  0. Load the controller with initial data.
//  1. Call controllerTest.testCall() once as in TestSync()
//  2. For all volumes/claims changed by previous syncVolume/syncClaim calls,
//     call appropriate syncVolume/syncClaim (simulating "volume/claim changed"
//     events). Go to 2. if these calls change anything.
//  3. When all changes are processed and no new changes were made, call
//     syncVolume/syncClaim on all volumes/claims (simulating "periodic sync").
//  4. If some changes were done by step 3., go to 2. (simulation of
//     "volume/claim updated" events, eventually performing step 3. again)
//  5. When 3. does not do any changes, finish the tests and compare final set
//     of volumes/claims with expected claims/volumes and report differences.
//
// Some limit of calls in enforced to prevent endless loops.
func TestRecycleMultiSync(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tests := []controllerTest{
		{
			// recycle failure - recycle returns error. The controller should
			// try again.
			"7-1 - recycle returns error",
			newVolumeArray("volume7-1", "1Gi", "uid7-1", "claim7-1", v1.VolumeBound, v1.PersistentVolumeReclaimRecycle, classEmpty),
			newVolumeArray("volume7-1", "1Gi", "", "claim7-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRecycle, classEmpty),
			noclaims,
			noclaims,
			[]string{"Warning VolumeFailedRecycle"}, noerrors,
			wrapTestWithReclaimCalls(operationRecycle, []error{errors.New("Mock recycle error"), nil}, testSyncVolume),
		},
	}

	runMultisyncTests(t, ctx, tests, []*storage.StorageClass{}, "")
}
