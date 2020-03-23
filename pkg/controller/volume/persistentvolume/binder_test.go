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
	"testing"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
)

// Test single call to syncClaim and syncVolume methods.
// 1. Fill in the controller with initial data
// 2. Call the tested function (syncClaim/syncVolume) via
//    controllerTest.testCall *once*.
// 3. Compare resulting volumes and claims with expected volumes and claims.
func TestSync(t *testing.T) {
	labels := map[string]string{
		"foo": "true",
		"bar": "false",
	}

	tests := []controllerTest{
		// [Unit test set 1] User did not care which PV they get.
		// Test the matching with no claim.Spec.VolumeName and with various
		// volumes.
		{
			// syncClaim binds to a matching unbound volume.
			"1-1 - successful bind",
			newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume1-1", "1Gi", "uid1-1", "claim1-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "volume1-1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim does not do anything when there is no matching volume.
			"1-2 - noop",
			newVolumeArray("volume1-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume1-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim1-2", "uid1-2", "10Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim1-2", "uid1-2", "10Gi", "", v1.ClaimPending, nil),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// syncClaim resets claim.Status to Pending when there is no
			// matching volume.
			"1-3 - reset to Pending",
			newVolumeArray("volume1-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume1-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim1-3", "uid1-3", "10Gi", "", v1.ClaimBound, nil),
			newClaimArray("claim1-3", "uid1-3", "10Gi", "", v1.ClaimPending, nil),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// syncClaim binds claims to the smallest matching volume
			"1-4 - smallest volume",
			[]*v1.PersistentVolume{
				newVolume("volume1-4_1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-4_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolume{
				newVolume("volume1-4_1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-4_2", "1Gi", "uid1-4", "claim1-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			},
			newClaimArray("claim1-4", "uid1-4", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim1-4", "uid1-4", "1Gi", "volume1-4_2", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim binds a claim only to volume that points to it (by
			// name), even though a smaller one is available.
			"1-5 - prebound volume by name - success",
			[]*v1.PersistentVolume{
				newVolume("volume1-5_1", "10Gi", "", "claim1-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-5_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolume{
				newVolume("volume1-5_1", "10Gi", "uid1-5", "claim1-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-5_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			newClaimArray("claim1-5", "uid1-5", "1Gi", "", v1.ClaimPending, nil),
			withExpectedCapacity("10Gi", newClaimArray("claim1-5", "uid1-5", "1Gi", "volume1-5_1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim binds a claim only to volume that points to it (by
			// UID), even though a smaller one is available.
			"1-6 - prebound volume by UID - success",
			[]*v1.PersistentVolume{
				newVolume("volume1-6_1", "10Gi", "uid1-6", "claim1-6", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-6_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolume{
				newVolume("volume1-6_1", "10Gi", "uid1-6", "claim1-6", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-6_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			newClaimArray("claim1-6", "uid1-6", "1Gi", "", v1.ClaimPending, nil),
			withExpectedCapacity("10Gi", newClaimArray("claim1-6", "uid1-6", "1Gi", "volume1-6_1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim does not bind claim to a volume prebound to a claim with
			// same name and different UID
			"1-7 - prebound volume to different claim",
			newVolumeArray("volume1-7", "10Gi", "uid1-777", "claim1-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume1-7", "10Gi", "uid1-777", "claim1-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim1-7", "uid1-7", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim1-7", "uid1-7", "1Gi", "", v1.ClaimPending, nil),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// syncClaim completes binding - simulates controller crash after
			// PV.ClaimRef is saved
			"1-8 - complete bind after crash - PV bound",
			newVolumeArray("volume1-8", "1Gi", "uid1-8", "claim1-8", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newVolumeArray("volume1-8", "1Gi", "uid1-8", "claim1-8", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim1-8", "uid1-8", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim1-8", "uid1-8", "1Gi", "volume1-8", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim completes binding - simulates controller crash after
			// PV.Status is saved
			"1-9 - complete bind after crash - PV status saved",
			newVolumeArray("volume1-9", "1Gi", "uid1-9", "claim1-9", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newVolumeArray("volume1-9", "1Gi", "uid1-9", "claim1-9", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim1-9", "uid1-9", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim1-9", "uid1-9", "1Gi", "volume1-9", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim completes binding - simulates controller crash after
			// PVC.VolumeName is saved
			"1-10 - complete bind after crash - PVC bound",
			newVolumeArray("volume1-10", "1Gi", "uid1-10", "claim1-10", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newVolumeArray("volume1-10", "1Gi", "uid1-10", "claim1-10", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim1-10", "uid1-10", "1Gi", "volume1-10", v1.ClaimPending, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim1-10", "uid1-10", "1Gi", "volume1-10", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim binds a claim only when the label selector matches the volume
			"1-11 - bind when selector matches",
			withLabels(labels, newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withLabels(labels, newVolumeArray("volume1-1", "1Gi", "uid1-1", "claim1-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "volume1-1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim does not bind a claim when the label selector doesn't match
			"1-12 - do not bind when selector does not match",
			newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// syncClaim does not do anything when binding is delayed
			"1-13 - delayed binding",
			newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			[]string{"Normal WaitForFirstConsumer"},
			noerrors, testSyncClaim,
		},
		{
			// syncClaim binds when binding is delayed but PV is prebound to PVC
			"1-14 - successful prebound PV",
			newVolumeArray("volume1-1", "1Gi", "", "claim1-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			newVolumeArray("volume1-1", "1Gi", "uid1-1", "claim1-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classWait),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "volume1-1", v1.ClaimBound, &classWait, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim binds pre-bound PVC only to the volume it points to,
			// even if there is smaller volume available
			"1-15 - successful prebound PVC",
			[]*v1.PersistentVolume{
				newVolume("volume1-15_1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-15_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolume{
				newVolume("volume1-15_1", "10Gi", "uid1-15", "claim1-15", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
				newVolume("volume1-15_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			newClaimArray("claim1-15", "uid1-15", "1Gi", "volume1-15_1", v1.ClaimPending, nil),
			withExpectedCapacity("10Gi", newClaimArray("claim1-15", "uid1-15", "1Gi", "volume1-15_1", v1.ClaimBound, nil, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim does not bind pre-bound PVC to PV with different AccessMode
			"1-16 - successful prebound PVC",
			// PV has ReadWriteOnce
			newVolumeArray("volume1-16", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume1-16", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			claimWithAccessMode([]v1.PersistentVolumeAccessMode{v1.ReadWriteMany}, newClaimArray("claim1-16", "uid1-16", "1Gi", "volume1-16", v1.ClaimPending, nil)),
			claimWithAccessMode([]v1.PersistentVolumeAccessMode{v1.ReadWriteMany}, newClaimArray("claim1-16", "uid1-16", "1Gi", "volume1-16", v1.ClaimPending, nil)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim does not bind PVC to non-available PV if it's not pre-bind
			"1-17 - skip non-available PV if it's not pre-bind",
			[]*v1.PersistentVolume{
				newVolume("volume1-17-pending", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-failed", "1Gi", "", "", v1.VolumeFailed, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-released", "1Gi", "", "", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-empty", "1Gi", "", "", "", v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolume{
				newVolume("volume1-17-pending", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-failed", "1Gi", "", "", v1.VolumeFailed, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-released", "1Gi", "", "", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-empty", "1Gi", "", "", "", v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolumeClaim{
				newClaim("claim1-17", "uid1-17", "1Gi", "", v1.ClaimPending, nil),
			},
			[]*v1.PersistentVolumeClaim{
				newClaim("claim1-17", "uid1-17", "1Gi", "", v1.ClaimPending, nil),
			},
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim that scheduled to a selected node
			"1-18 - successful pre-bound PV to PVC provisioning",
			newVolumeArray("volume1-18", "1Gi", "uid1-18", "claim1-18", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			newVolumeArray("volume1-18", "1Gi", "uid1-18", "claim1-18", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classWait),
			claimWithAnnotation(pvutil.AnnSelectedNode, "node1",
				newClaimArray("claim1-18", "uid1-18", "1Gi", "", v1.ClaimPending, &classWait)),
			claimWithAnnotation(pvutil.AnnSelectedNode, "node1",
				newClaimArray("claim1-18", "uid1-18", "1Gi", "volume1-18", v1.ClaimBound, &classWait, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},

		// [Unit test set 2] User asked for a specific PV.
		// Test the binding when pv.ClaimRef is already set by controller or
		// by user.
		{
			// syncClaim with claim pre-bound to a PV that does not exist
			"2-1 - claim prebound to non-existing volume - noop",
			novolumes,
			novolumes,
			newClaimArray("claim2-1", "uid2-1", "10Gi", "volume2-1", v1.ClaimPending, nil),
			newClaimArray("claim2-1", "uid2-1", "10Gi", "volume2-1", v1.ClaimPending, nil),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that does not exist.
			// Check that the claim status is reset to Pending
			"2-2 - claim prebound to non-existing volume - reset status",
			novolumes,
			novolumes,
			newClaimArray("claim2-2", "uid2-2", "10Gi", "volume2-2", v1.ClaimBound, nil),
			newClaimArray("claim2-2", "uid2-2", "10Gi", "volume2-2", v1.ClaimPending, nil),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound. Check it gets bound and no pvutil.AnnBoundByController is set.
			"2-3 - claim prebound to unbound volume",
			newVolumeArray("volume2-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-3", "1Gi", "uid2-3", "claim2-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim2-3", "uid2-3", "1Gi", "volume2-3", v1.ClaimPending, nil),
			newClaimArray("claim2-3", "uid2-3", "1Gi", "volume2-3", v1.ClaimBound, nil, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// claim with claim pre-bound to a PV that is pre-bound to the claim
			// by name. Check it gets bound and no pvutil.AnnBoundByController is set.
			"2-4 - claim prebound to prebound volume by name",
			newVolumeArray("volume2-4", "1Gi", "", "claim2-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-4", "1Gi", "uid2-4", "claim2-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim2-4", "uid2-4", "1Gi", "volume2-4", v1.ClaimPending, nil),
			newClaimArray("claim2-4", "uid2-4", "1Gi", "volume2-4", v1.ClaimBound, nil, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that is pre-bound to the
			// claim by UID. Check it gets bound and no pvutil.AnnBoundByController is
			// set.
			"2-5 - claim prebound to prebound volume by UID",
			newVolumeArray("volume2-5", "1Gi", "uid2-5", "claim2-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-5", "1Gi", "uid2-5", "claim2-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim2-5", "uid2-5", "1Gi", "volume2-5", v1.ClaimPending, nil),
			newClaimArray("claim2-5", "uid2-5", "1Gi", "volume2-5", v1.ClaimBound, nil, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that is bound to different
			// claim. Check it's reset to Pending.
			"2-6 - claim prebound to already bound volume",
			newVolumeArray("volume2-6", "1Gi", "uid2-6_1", "claim2-6_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-6", "1Gi", "uid2-6_1", "claim2-6_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim2-6", "uid2-6", "1Gi", "volume2-6", v1.ClaimBound, nil),
			newClaimArray("claim2-6", "uid2-6", "1Gi", "volume2-6", v1.ClaimPending, nil),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound by controller to a PV that is bound to
			// different claim. Check it throws an error.
			"2-7 - claim bound by controller to already bound volume",
			newVolumeArray("volume2-7", "1Gi", "uid2-7_1", "claim2-7_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-7", "1Gi", "uid2-7_1", "claim2-7_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim2-7", "uid2-7", "1Gi", "volume2-7", v1.ClaimBound, nil, pvutil.AnnBoundByController),
			newClaimArray("claim2-7", "uid2-7", "1Gi", "volume2-7", v1.ClaimBound, nil, pvutil.AnnBoundByController),
			noevents, noerrors, testSyncClaimError,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, but does not match the selector. Check it gets bound
			// and no pvutil.AnnBoundByController is set.
			"2-8 - claim prebound to unbound volume that does not match the selector",
			newVolumeArray("volume2-8", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-8", "1Gi", "uid2-8", "claim2-8", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			withLabelSelector(labels, newClaimArray("claim2-8", "uid2-8", "1Gi", "volume2-8", v1.ClaimPending, nil)),
			withLabelSelector(labels, newClaimArray("claim2-8", "uid2-8", "1Gi", "volume2-8", v1.ClaimBound, nil, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, but its size is smaller than requested.
			//Check that the claim status is reset to Pending
			"2-9 - claim prebound to unbound volume that size is smaller than requested",
			newVolumeArray("volume2-9", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume2-9", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim2-9", "uid2-9", "2Gi", "volume2-9", v1.ClaimBound, nil),
			newClaimArray("claim2-9", "uid2-9", "2Gi", "volume2-9", v1.ClaimPending, nil),
			[]string{"Warning VolumeMismatch"}, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, but its class does not match. Check that the claim status is reset to Pending
			"2-10 - claim prebound to unbound volume that class is different",
			newVolumeArray("volume2-10", "1Gi", "1", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			newVolumeArray("volume2-10", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			newClaimArray("claim2-10", "uid2-10", "1Gi", "volume2-10", v1.ClaimBound, nil),
			newClaimArray("claim2-10", "uid2-10", "1Gi", "volume2-10", v1.ClaimPending, nil),
			[]string{"Warning VolumeMismatch"}, noerrors, testSyncClaim,
		},

		// [Unit test set 3] Syncing bound claim
		{
			// syncClaim with claim  bound and its claim.Spec.VolumeName is
			// removed. Check it's marked as Lost.
			"3-1 - bound claim with missing VolumeName",
			novolumes,
			novolumes,
			newClaimArray("claim3-1", "uid3-1", "10Gi", "", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim3-1", "uid3-1", "10Gi", "", v1.ClaimLost, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			[]string{"Warning ClaimLost"}, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound to non-existing volume. Check it's
			// marked as Lost.
			"3-2 - bound claim with missing volume",
			novolumes,
			novolumes,
			newClaimArray("claim3-2", "uid3-2", "10Gi", "volume3-2", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim3-2", "uid3-2", "10Gi", "volume3-2", v1.ClaimLost, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			[]string{"Warning ClaimLost"}, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound to unbound volume. Check it's bound.
			// Also check that Pending phase is set to Bound
			"3-3 - bound claim with unbound volume",
			newVolumeArray("volume3-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume3-3", "10Gi", "uid3-3", "claim3-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimPending, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound to volume with missing (or different)
			// volume.Spec.ClaimRef.UID. Check that the claim is marked as lost.
			"3-4 - bound claim with prebound volume",
			newVolumeArray("volume3-4", "10Gi", "claim3-4-x", "claim3-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume3-4", "10Gi", "claim3-4-x", "claim3-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim3-4", "uid3-4", "10Gi", "volume3-4", v1.ClaimPending, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim3-4", "uid3-4", "10Gi", "volume3-4", v1.ClaimLost, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			[]string{"Warning ClaimMisbound"}, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound to bound volume. Check that the
			// controller does not do anything. Also check that Pending phase is
			// set to Bound
			"3-5 - bound claim with bound volume",
			newVolumeArray("volume3-5", "10Gi", "uid3-5", "claim3-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume3-5", "10Gi", "uid3-5", "claim3-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim3-5", "uid3-5", "10Gi", "volume3-5", v1.ClaimPending, nil, pvutil.AnnBindCompleted),
			newClaimArray("claim3-5", "uid3-5", "10Gi", "volume3-5", v1.ClaimBound, nil, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound to a volume that is bound to different
			// claim. Check that the claim is marked as lost.
			// TODO: test that an event is emitted
			"3-6 - bound claim with bound volume",
			newVolumeArray("volume3-6", "10Gi", "uid3-6-x", "claim3-6-x", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume3-6", "10Gi", "uid3-6-x", "claim3-6-x", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim3-6", "uid3-6", "10Gi", "volume3-6", v1.ClaimPending, nil, pvutil.AnnBindCompleted),
			newClaimArray("claim3-6", "uid3-6", "10Gi", "volume3-6", v1.ClaimLost, nil, pvutil.AnnBindCompleted),
			[]string{"Warning ClaimMisbound"}, noerrors, testSyncClaim,
		},
		{
			// syncClaim with claim bound to unbound volume. Check it's bound
			// even if the claim's selector doesn't match the volume. Also
			// check that Pending phase is set to Bound
			"3-7 - bound claim with unbound volume where selector doesn't match",
			newVolumeArray("volume3-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume3-3", "10Gi", "uid3-3", "claim3-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			withLabelSelector(labels, newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimPending, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			withLabelSelector(labels, newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		// [Unit test set 4] All syncVolume tests.
		{
			// syncVolume with pending volume. Check it's marked as Available.
			"4-1 - pending volume",
			newVolumeArray("volume4-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			noclaims,
			noclaims,
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with prebound pending volume. Check it's marked as
			// Available.
			"4-2 - pending prebound volume",
			newVolumeArray("volume4-2", "10Gi", "", "claim4-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-2", "10Gi", "", "claim4-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			noclaims,
			noclaims,
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound to missing claim.
			// Check the volume gets Released
			"4-3 - bound volume with missing claim",
			newVolumeArray("volume4-3", "10Gi", "uid4-3", "claim4-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-3", "10Gi", "uid4-3", "claim4-3", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
			noclaims,
			noclaims,
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound to claim with different UID.
			// Check the volume gets Released.
			"4-4 - volume bound to claim with different UID",
			newVolumeArray("volume4-4", "10Gi", "uid4-4", "claim4-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-4", "10Gi", "uid4-4", "claim4-4", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim4-4", "uid4-4-x", "10Gi", "volume4-4", v1.ClaimBound, nil, pvutil.AnnBindCompleted),
			newClaimArray("claim4-4", "uid4-4-x", "10Gi", "volume4-4", v1.ClaimBound, nil, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound by controller to unbound claim.
			// Check syncVolume does not do anything.
			"4-5 - volume bound by controller to unbound claim",
			newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim4-5", "uid4-5", "10Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim4-5", "uid4-5", "10Gi", "", v1.ClaimPending, nil),
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound by user to unbound claim.
			// Check syncVolume does not do anything.
			"4-5 - volume bound by user to bound claim",
			newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim4-5", "uid4-5", "10Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim4-5", "uid4-5", "10Gi", "", v1.ClaimPending, nil),
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound to bound claim.
			// Check that the volume is marked as Bound.
			"4-6 - volume bound by to bound claim",
			newVolumeArray("volume4-6", "10Gi", "uid4-6", "claim4-6", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-6", "10Gi", "uid4-6", "claim4-6", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim4-6", "uid4-6", "10Gi", "volume4-6", v1.ClaimBound, nil),
			newClaimArray("claim4-6", "uid4-6", "10Gi", "volume4-6", v1.ClaimBound, nil),
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound by controller to claim bound to
			// another volume. Check that the volume is rolled back.
			"4-7 - volume bound by controller to claim bound somewhere else",
			newVolumeArray("volume4-7", "10Gi", "uid4-7", "claim4-7", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newVolumeArray("volume4-7", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim4-7", "uid4-7", "10Gi", "volume4-7-x", v1.ClaimBound, nil),
			newClaimArray("claim4-7", "uid4-7", "10Gi", "volume4-7-x", v1.ClaimBound, nil),
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume with volume bound by user to claim bound to
			// another volume. Check that the volume is marked as Available
			// and its UID is reset.
			"4-8 - volume bound by user to claim bound somewhere else",
			newVolumeArray("volume4-8", "10Gi", "uid4-8", "claim4-8", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume4-8", "10Gi", "", "claim4-8", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newClaimArray("claim4-8", "uid4-8", "10Gi", "volume4-8-x", v1.ClaimBound, nil),
			newClaimArray("claim4-8", "uid4-8", "10Gi", "volume4-8-x", v1.ClaimBound, nil),
			noevents, noerrors, testSyncVolume,
		},

		// PVC with class
		{
			// syncVolume binds a claim to requested class even if there is a
			// smaller PV available
			"13-1 - binding to class",
			[]*v1.PersistentVolume{
				newVolume("volume13-1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume13-1-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			},
			[]*v1.PersistentVolume{
				newVolume("volume13-1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume13-1-2", "10Gi", "uid13-1", "claim13-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classGold, pvutil.AnnBoundByController),
			},
			newClaimArray("claim13-1", "uid13-1", "1Gi", "", v1.ClaimPending, &classGold),
			withExpectedCapacity("10Gi", newClaimArray("claim13-1", "uid13-1", "1Gi", "volume13-1-2", v1.ClaimBound, &classGold, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds a claim without a class even if there is a
			// smaller PV with a class available
			"13-2 - binding without a class",
			[]*v1.PersistentVolume{
				newVolume("volume13-2-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
				newVolume("volume13-2-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			[]*v1.PersistentVolume{
				newVolume("volume13-2-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
				newVolume("volume13-2-2", "10Gi", "uid13-2", "claim13-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			},
			newClaimArray("claim13-2", "uid13-2", "1Gi", "", v1.ClaimPending, nil),
			withExpectedCapacity("10Gi", newClaimArray("claim13-2", "uid13-2", "1Gi", "volume13-2-2", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds a claim with given class even if there is a
			// smaller PV with different class available
			"13-3 - binding to specific a class",
			[]*v1.PersistentVolume{
				newVolume("volume13-3-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classSilver),
				newVolume("volume13-3-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			},
			[]*v1.PersistentVolume{
				newVolume("volume13-3-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classSilver),
				newVolume("volume13-3-2", "10Gi", "uid13-3", "claim13-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classGold, pvutil.AnnBoundByController),
			},
			newClaimArray("claim13-3", "uid13-3", "1Gi", "", v1.ClaimPending, &classGold),
			withExpectedCapacity("10Gi", newClaimArray("claim13-3", "uid13-3", "1Gi", "volume13-3-2", v1.ClaimBound, &classGold, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds claim requesting class "" to claim to PV with
			// class=""
			"13-4 - empty class",
			newVolumeArray("volume13-4", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume13-4", "1Gi", "uid13-4", "claim13-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim13-4", "uid13-4", "1Gi", "", v1.ClaimPending, &classEmpty),
			newClaimArray("claim13-4", "uid13-4", "1Gi", "volume13-4", v1.ClaimBound, &classEmpty, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds claim requesting class nil to claim to PV with
			// class = ""
			"13-5 - nil class",
			newVolumeArray("volume13-5", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume13-5", "1Gi", "uid13-5", "claim13-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim13-5", "uid13-5", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim13-5", "uid13-5", "1Gi", "volume13-5", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
	}

	runSyncTests(t, tests, []*storage.StorageClass{
		{
			ObjectMeta:        metav1.ObjectMeta{Name: classWait},
			VolumeBindingMode: &modeWait,
		},
	}, []*v1.Pod{})
}

func TestSyncBlockVolume(t *testing.T) {
	modeBlock := v1.PersistentVolumeBlock
	modeFile := v1.PersistentVolumeFilesystem

	// Tests assume defaulting, so feature enabled will never have nil volumeMode
	tests := []controllerTest{
		// PVC with VolumeMode
		{
			// syncVolume binds a requested block claim to a block volume
			"14-1 - binding to volumeMode block",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-1", "10Gi", "uid14-1", "claim14-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-1", "uid14-1", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-1", "uid14-1", "10Gi", "volume14-1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds a requested filesystem claim to a filesystem volume
			"14-2 - binding to volumeMode filesystem",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-2", "10Gi", "uid14-2", "claim14-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-2", "uid14-2", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-2", "uid14-2", "10Gi", "volume14-2", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind to an unspecified volumemode for claim to a specified filesystem volume
			"14-3 - do not bind pv volumeMode filesystem and pvc volumeMode block",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-3", "uid14-3", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-3", "uid14-3", "10Gi", "", v1.ClaimPending, nil)),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind a requested filesystem claim to an unspecified volumeMode for volume
			"14-4 - do not bind pv volumeMode block and pvc volumeMode filesystem",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-4", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-4", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-4", "uid14-4", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-4", "uid14-4", "10Gi", "", v1.ClaimPending, nil)),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind when matching class but not matching volumeModes
			"14-5 - do not bind when matching class but not volumeMode",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-5", "uid14-5", "10Gi", "", v1.ClaimPending, &classGold)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-5", "uid14-5", "10Gi", "", v1.ClaimPending, &classGold)),
			[]string{"Warning ProvisioningFailed"},
			noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind when matching volumeModes but class does not match
			"14-5-1 - do not bind when matching volumeModes but class does not match",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-1", "uid14-5-1", "10Gi", "", v1.ClaimPending, &classSilver)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-1", "uid14-5-1", "10Gi", "", v1.ClaimPending, &classSilver)),
			[]string{"Warning ProvisioningFailed"},
			noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind when pvc is prebound to pv with matching volumeModes but class does not match
			"14-5-2 - do not bind when pvc is prebound to pv with matching volumeModes but class does not match",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-2", "uid14-5-2", "10Gi", "volume14-5-2", v1.ClaimPending, &classSilver)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-2", "uid14-5-2", "10Gi", "volume14-5-2", v1.ClaimPending, &classSilver)),
			[]string{"Warning VolumeMismatch"},
			noerrors, testSyncClaim,
		},
		{
			// syncVolume bind when pv is prebound and volumeModes match
			"14-7 - bind when pv volume is prebound and volumeModes match",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-7", "10Gi", "", "claim14-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-7", "10Gi", "uid14-7", "claim14-7", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-7", "uid14-7", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-7", "uid14-7", "10Gi", "volume14-7", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind when pvc is prebound to pv with mismatching volumeModes
			"14-8 - do not bind when pvc is prebound to pv with mismatching volumeModes",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-8", "uid14-8", "10Gi", "volume14-8", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-8", "uid14-8", "10Gi", "volume14-8", v1.ClaimPending, nil)),
			[]string{"Warning VolumeMismatch"},
			noerrors, testSyncClaim,
		},
		{
			// failed syncVolume do not bind when pvc is prebound to pv with mismatching volumeModes
			"14-8-1 - do not bind when pvc is prebound to pv with mismatching volumeModes",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8-1", "10Gi", "", "claim14-8-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8-1", "10Gi", "", "claim14-8-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-8-1", "uid14-8-1", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-8-1", "uid14-8-1", "10Gi", "", v1.ClaimPending, nil)),
			[]string{"Normal FailedBinding"},
			noerrors, testSyncClaim,
		},
		{
			// syncVolume binds when pvc is prebound to pv with matching volumeModes block
			"14-9 - bind when pvc is prebound to pv with matching volumeModes block",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-9", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-9", "10Gi", "uid14-9", "claim14-9", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-9", "uid14-9", "10Gi", "volume14-9", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-9", "uid14-9", "10Gi", "volume14-9", v1.ClaimBound, nil, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds when pv is prebound to pvc with matching volumeModes block
			"14-10 - bind when pv is prebound to pvc with matching volumeModes block",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-10", "10Gi", "", "claim14-10", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-10", "10Gi", "uid14-10", "claim14-10", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-10", "uid14-10", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-10", "uid14-10", "10Gi", "volume14-10", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds when pvc is prebound to pv with matching volumeModes filesystem
			"14-11 - bind when pvc is prebound to pv with matching volumeModes filesystem",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-11", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-11", "10Gi", "uid14-11", "claim14-11", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-11", "uid14-11", "10Gi", "volume14-11", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-11", "uid14-11", "10Gi", "volume14-11", v1.ClaimBound, nil, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume binds when pv is prebound to pvc with matching volumeModes filesystem
			"14-12 - bind when pv is prebound to pvc with matching volumeModes filesystem",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-12", "10Gi", "", "claim14-12", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-12", "10Gi", "uid14-12", "claim14-12", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-12", "uid14-12", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-12", "uid14-12", "10Gi", "volume14-12", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			noevents, noerrors, testSyncClaim,
		},
		{
			// syncVolume output warning when pv is prebound to pvc with mismatching volumeMode
			"14-13 - output warning when pv is prebound to pvc with different volumeModes",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-13", "10Gi", "uid14-13", "claim14-13", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-13", "10Gi", "uid14-13", "claim14-13", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-13", "uid14-13", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-13", "uid14-13", "10Gi", "", v1.ClaimPending, nil)),
			[]string{"Warning VolumeMismatch"},
			noerrors, testSyncVolume,
		},
		{
			// syncVolume output warning when pv is prebound to pvc with mismatching volumeMode
			"14-13-1 - output warning when pv is prebound to pvc with different volumeModes",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-13-1", "10Gi", "uid14-13-1", "claim14-13-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-13-1", "10Gi", "uid14-13-1", "claim14-13-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-13-1", "uid14-13-1", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-13-1", "uid14-13-1", "10Gi", "", v1.ClaimPending, nil)),
			[]string{"Warning VolumeMismatch"},
			noerrors, testSyncVolume,
		},
		{
			// syncVolume waits for synClaim without warning when pv is prebound to pvc with matching volumeMode block
			"14-14 - wait for synClaim without warning when pv is prebound to pvc with matching volumeModes block",
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-14", "10Gi", "uid14-14", "claim14-14", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-14", "10Gi", "uid14-14", "claim14-14", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-14", "uid14-14", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeBlock, newClaimArray("claim14-14", "uid14-14", "10Gi", "", v1.ClaimPending, nil)),
			noevents, noerrors, testSyncVolume,
		},
		{
			// syncVolume waits for synClaim without warning when pv is prebound to pvc with matching volumeMode file
			"14-14-1 - wait for synClaim without warning when pv is prebound to pvc with matching volumeModes file",
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-14-1", "10Gi", "uid14-14-1", "claim14-14-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-14-1", "10Gi", "uid14-14-1", "claim14-14-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-14-1", "uid14-14-1", "10Gi", "", v1.ClaimPending, nil)),
			withClaimVolumeMode(&modeFile, newClaimArray("claim14-14-1", "uid14-14-1", "10Gi", "", v1.ClaimPending, nil)),
			noevents, noerrors, testSyncVolume,
		},
	}

	runSyncTests(t, tests, []*storage.StorageClass{}, []*v1.Pod{})
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
func TestMultiSync(t *testing.T) {
	tests := []controllerTest{
		// Test simple binding
		{
			// syncClaim binds to a matching unbound volume.
			"10-1 - successful bind",
			newVolumeArray("volume10-1", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
			newVolumeArray("volume10-1", "1Gi", "uid10-1", "claim10-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			newClaimArray("claim10-1", "uid10-1", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim10-1", "uid10-1", "1Gi", "volume10-1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
		{
			// Two controllers bound two PVs to single claim. Test one of them
			// wins and the second rolls back.
			"10-2 - bind PV race",
			[]*v1.PersistentVolume{
				newVolume("volume10-2-1", "1Gi", "uid10-2", "claim10-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
				newVolume("volume10-2-2", "1Gi", "uid10-2", "claim10-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
			},
			[]*v1.PersistentVolume{
				newVolume("volume10-2-1", "1Gi", "uid10-2", "claim10-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, pvutil.AnnBoundByController),
				newVolume("volume10-2-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			newClaimArray("claim10-2", "uid10-2", "1Gi", "volume10-2-1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			newClaimArray("claim10-2", "uid10-2", "1Gi", "volume10-2-1", v1.ClaimBound, nil, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors, testSyncClaim,
		},
	}

	runMultisyncTests(t, tests, []*storage.StorageClass{}, "")
}
