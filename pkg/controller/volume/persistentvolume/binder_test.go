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
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
)

// Test single call to syncClaim and syncVolume methods.
//  1. Fill in the controller with initial data
//  2. Call the tested function (syncClaim/syncVolume) via
//     controllerTest.testCall *once*.
//  3. Compare resulting volumes and claims with expected volumes and claims.
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
			// syncClaim reserves a matching unbound volume: it writes only the
			// PV's ClaimRef (+AnnBoundByController) and leaves the claim
			// untouched. syncVolume completes the bind (see 4-13).
			name:            "1-1 - reserve unbound volume",
			initialVolumes:  newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume1-1", "1Gi", "uid1-1", "claim1-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim does not do anything when there is no matching volume.
			name:            "1-2 - noop",
			initialVolumes:  newVolumeArray("volume1-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume1-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim1-2", "uid1-2", "10Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim1-2", "uid1-2", "10Gi", "", v1.ClaimPending, nil),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim resets claim.Status to Pending when there is no
			// matching volume.
			name:            "1-3 - reset to Pending",
			initialVolumes:  newVolumeArray("volume1-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume1-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim1-3", "uid1-3", "10Gi", "", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim1-3", "uid1-3", "10Gi", "", v1.ClaimPending, nil),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves the smallest matching volume (writes only the
			// PV). syncVolume completes the bind (see 4-13).
			name: "1-4 - reserve smallest volume",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume1-4_1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-4_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume1-4_1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-4_2", "1Gi", "uid1-4", "claim1-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			},
			initialClaims:  newClaimArray("claim1-4", "uid1-4", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims: newClaimArray("claim1-4", "uid1-4", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim selects only the volume that points to the claim (by
			// name), even though a smaller one is available. The PV already has a
			// ClaimRef, so it is past the reserve handoff: syncClaim writes nothing
			// and enqueues it for syncVolume, which stamps the UID and completes
			// the bind (see 4-14).
			name: "1-5 - defer prebound volume by name",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume1-5_1", "10Gi", "", "claim1-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-5_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume1-5_1", "10Gi", "", "claim1-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-5_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			initialClaims:  newClaimArray("claim1-5", "uid1-5", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims: newClaimArray("claim1-5", "uid1-5", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim finds the volume already reserved to it (by UID). The PV
			// already has a ClaimRef, so syncClaim writes nothing and enqueues it
			// for syncVolume, which completes the bind (see 4-5-1).
			name: "1-6 - already reserved volume by UID - defer",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume1-6_1", "10Gi", "uid1-6", "claim1-6", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-6_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume1-6_1", "10Gi", "uid1-6", "claim1-6", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-6_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			initialClaims:  newClaimArray("claim1-6", "uid1-6", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims: newClaimArray("claim1-6", "uid1-6", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim does not bind claim to a volume prebound to a claim with
			// same name and different UID
			name:            "1-7 - prebound volume to different claim",
			initialVolumes:  newVolumeArray("volume1-7", "10Gi", "uid1-777", "claim1-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume1-7", "10Gi", "uid1-777", "claim1-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim1-7", "uid1-7", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim1-7", "uid1-7", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncVolume completes binding - simulates controller crash after
			// PV.ClaimRef is saved.
			name:            "1-8 - complete bind after crash - PV bound",
			initialVolumes:  newVolumeArray("volume1-8", "1Gi", "uid1-8", "claim1-8", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume1-8", "1Gi", "uid1-8", "claim1-8", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim1-8", "uid1-8", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim1-8", "uid1-8", "1Gi", "volume1-8", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes binding - simulates controller crash after
			// PV.Status is saved.
			name:            "1-9 - complete bind after crash - PV status saved",
			initialVolumes:  newVolumeArray("volume1-9", "1Gi", "uid1-9", "claim1-9", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume1-9", "1Gi", "uid1-9", "claim1-9", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim1-9", "uid1-9", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim1-9", "uid1-9", "1Gi", "volume1-9", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes binding - simulates controller crash after
			// PVC.VolumeName is saved
			name:            "1-10 - complete bind after crash - PVC bound",
			initialVolumes:  newVolumeArray("volume1-10", "1Gi", "uid1-10", "claim1-10", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume1-10", "1Gi", "uid1-10", "claim1-10", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim1-10", "uid1-10", "1Gi", "volume1-10", v1.ClaimPending, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim1-10", "uid1-10", "1Gi", "volume1-10", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncClaim reserves a volume only when the label selector matches it.
			// syncVolume completes the bind (see 4-13).
			name:            "1-11 - reserve when selector matches",
			initialVolumes:  withLabels(labels, newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withLabels(labels, newVolumeArray("volume1-1", "1Gi", "uid1-1", "claim1-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim does not bind a claim when the label selector doesn't match
			name:            "1-12 - do not bind when selector does not match",
			initialVolumes:  newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withLabelSelector(labels, newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim does not do anything when binding is delayed
			name:            "1-13 - delayed binding",
			initialVolumes:  newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			expectedVolumes: newVolumeArray("volume1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			initialClaims:   newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			expectedClaims:  newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			expectedEvents:  []string{"Normal WaitForFirstConsumer"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim defers when binding is delayed but PV is prebound to PVC
			// (by name). The PV already has a ClaimRef, so it is past the reserve
			// handoff: syncClaim writes nothing and enqueues it for syncVolume,
			// which stamps the UID and completes the bind (see 4-14).
			name:            "1-14 - defer prebound PV",
			initialVolumes:  newVolumeArray("volume1-1", "1Gi", "", "claim1-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			expectedVolumes: newVolumeArray("volume1-1", "1Gi", "", "claim1-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			initialClaims:   newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			expectedClaims:  newClaimArray("claim1-1", "uid1-1", "1Gi", "", v1.ClaimPending, &classWait),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves the pre-bound PVC's target volume only (writes
			// only the PV), even if a smaller volume is available. The claim
			// already names this PV, so syncVolume completes it via the
			// volume.Name branch (see 4-17).
			name: "1-15 - reserve prebound PVC target",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume1-15_1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-15_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume1-15_1", "10Gi", "uid1-15", "claim1-15", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
				newVolume("volume1-15_2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			initialClaims:  newClaimArray("claim1-15", "uid1-15", "1Gi", "volume1-15_1", v1.ClaimPending, nil),
			expectedClaims: newClaimArray("claim1-15", "uid1-15", "1Gi", "volume1-15_1", v1.ClaimPending, nil),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim does not bind pre-bound PVC to PV with different AccessMode
			name: "1-16 - successful prebound PVC",
			// PV has ReadWriteOnce
			initialVolumes:  newVolumeArray("volume1-16", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume1-16", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   claimWithAccessMode([]v1.PersistentVolumeAccessMode{v1.ReadWriteMany}, newClaimArray("claim1-16", "uid1-16", "1Gi", "volume1-16", v1.ClaimPending, nil)),
			expectedClaims:  claimWithAccessMode([]v1.PersistentVolumeAccessMode{v1.ReadWriteMany}, newClaimArray("claim1-16", "uid1-16", "1Gi", "volume1-16", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim does not bind PVC to non-available PV if it's not pre-bind
			name: "1-17 - skip non-available PV if it's not pre-bind",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume1-17-pending", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-failed", "1Gi", "", "", v1.VolumeFailed, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-released", "1Gi", "", "", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-empty", "1Gi", "", "", "", v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume1-17-pending", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-failed", "1Gi", "", "", v1.VolumeFailed, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-released", "1Gi", "", "", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume1-17-empty", "1Gi", "", "", "", v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			initialClaims: []*v1.PersistentVolumeClaim{
				newClaim("claim1-17", "uid1-17", "1Gi", "", v1.ClaimPending, nil),
			},
			expectedClaims: []*v1.PersistentVolumeClaim{
				newClaim("claim1-17", "uid1-17", "1Gi", "", v1.ClaimPending, nil),
			},
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim on a claim scheduled to a selected node whose PV is
			// already reserved to it (by UID). The PV already has a ClaimRef, so
			// syncClaim writes nothing and enqueues it for syncVolume, which
			// completes the bind (see 4-5-1).
			name:            "1-18 - already reserved pre-bound PV - defer",
			initialVolumes:  newVolumeArray("volume1-18", "1Gi", "uid1-18", "claim1-18", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			expectedVolumes: newVolumeArray("volume1-18", "1Gi", "uid1-18", "claim1-18", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classWait),
			initialClaims: claimWithAnnotation(volume.AnnSelectedNode, "node1",
				newClaimArray("claim1-18", "uid1-18", "1Gi", "", v1.ClaimPending, &classWait)),
			expectedClaims: claimWithAnnotation(volume.AnnSelectedNode, "node1",
				newClaimArray("claim1-18", "uid1-18", "1Gi", "", v1.ClaimPending, &classWait)),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},

		// [Unit test set 2] User asked for a specific PV.
		// Test the binding when pv.ClaimRef is already set by controller or
		// by user.
		{
			// syncClaim with claim pre-bound to a PV that does not exist
			name:            "2-1 - claim prebound to non-existing volume - noop",
			initialVolumes:  novolumes,
			expectedVolumes: novolumes,
			initialClaims:   newClaimArray("claim2-1", "uid2-1", "10Gi", "volume2-1", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim2-1", "uid2-1", "10Gi", "volume2-1", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that does not exist.
			// Check that the claim status is reset to Pending
			name:            "2-2 - claim prebound to non-existing volume - reset status",
			initialVolumes:  novolumes,
			expectedVolumes: novolumes,
			initialClaims:   newClaimArray("claim2-2", "uid2-2", "10Gi", "volume2-2", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim2-2", "uid2-2", "10Gi", "volume2-2", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is unbound:
			// reserve writes the full ClaimRef + AnnBoundByController on the PV
			// (the controller is establishing the binding) and leaves the claim
			// untouched. The claim already names this PV, so syncVolume completes
			// it via the volume.Name branch (see 4-17).
			name:            "2-3 - reserve claim-prebound unbound volume",
			initialVolumes:  newVolumeArray("volume2-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-3", "1Gi", "uid2-3", "claim2-3", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim2-3", "uid2-3", "1Gi", "volume2-3", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim2-3", "uid2-3", "1Gi", "volume2-3", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// claim pre-bound to a PV that is pre-bound back to the claim (mutual
			// prebind, by name). syncClaim writes nothing and defers to syncVolume,
			// which completes the bind (see 4-18).
			name:            "2-4 - mutual prebind by name - defer to syncVolume",
			initialVolumes:  newVolumeArray("volume2-4", "1Gi", "", "claim2-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-4", "1Gi", "", "claim2-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim2-4", "uid2-4", "1Gi", "volume2-4", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim2-4", "uid2-4", "1Gi", "volume2-4", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// claim pre-bound to a PV that is pre-bound back to the claim (mutual
			// prebind, by UID). syncClaim writes nothing and defers to syncVolume,
			// which completes the bind (see 4-16).
			name:            "2-5 - mutual prebind by UID - defer to syncVolume",
			initialVolumes:  newVolumeArray("volume2-5", "1Gi", "uid2-5", "claim2-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-5", "1Gi", "uid2-5", "claim2-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim2-5", "uid2-5", "1Gi", "volume2-5", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim2-5", "uid2-5", "1Gi", "volume2-5", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that is bound to different
			// claim. Check it's reset to Pending.
			name:            "2-6 - claim prebound to already bound volume",
			initialVolumes:  newVolumeArray("volume2-6", "1Gi", "uid2-6_1", "claim2-6_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-6", "1Gi", "uid2-6_1", "claim2-6_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim2-6", "uid2-6", "1Gi", "volume2-6", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim2-6", "uid2-6", "1Gi", "volume2-6", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim bound by controller to a PV that is bound to
			// different claim. Check it throws an error.
			name:            "2-7 - claim bound by controller to already bound volume",
			initialVolumes:  newVolumeArray("volume2-7", "1Gi", "uid2-7_1", "claim2-7_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-7", "1Gi", "uid2-7_1", "claim2-7_1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim2-7", "uid2-7", "1Gi", "volume2-7", v1.ClaimBound, nil, volume.AnnBoundByController),
			expectedClaims:  newClaimArray("claim2-7", "uid2-7", "1Gi", "volume2-7", v1.ClaimBound, nil, volume.AnnBoundByController),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaimError,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is unbound,
			// but does not match the selector: reserve writes the full ClaimRef +
			// AnnBoundByController on the PV (the selector is ignored for a
			// user-named PV) and leaves the claim untouched. The claim already
			// names this PV, so syncVolume completes it via the volume.Name
			// branch (see 4-17).
			name:            "2-8 - reserve claim-prebound unbound volume ignoring selector",
			initialVolumes:  newVolumeArray("volume2-8", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-8", "1Gi", "uid2-8", "claim2-8", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   withLabelSelector(labels, newClaimArray("claim2-8", "uid2-8", "1Gi", "volume2-8", v1.ClaimPending, nil)),
			expectedClaims:  withLabelSelector(labels, newClaimArray("claim2-8", "uid2-8", "1Gi", "volume2-8", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, but its size is smaller than requested.
			//Check that the claim status is reset to Pending
			name:            "2-9 - claim prebound to unbound volume that size is smaller than requested",
			initialVolumes:  newVolumeArray("volume2-9", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume2-9", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim2-9", "uid2-9", "2Gi", "volume2-9", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim2-9", "uid2-9", "2Gi", "volume2-9", v1.ClaimPending, nil),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, but its class does not match. Check that the claim status is reset to Pending
			name:            "2-10 - claim prebound to unbound volume that class is different",
			initialVolumes:  newVolumeArray("volume2-10", "1Gi", "1", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			expectedVolumes: newVolumeArray("volume2-10", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			initialClaims:   newClaimArray("claim2-10", "uid2-10", "1Gi", "volume2-10", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim2-10", "uid2-10", "1Gi", "volume2-10", v1.ClaimPending, nil),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncClaim,
		},

		// [Unit test set 3] Syncing bound claim
		{
			// syncClaim with claim  bound and its claim.Spec.VolumeName is
			// removed. Check it's marked as Lost.
			name:            "3-1 - bound claim with missing VolumeName",
			initialVolumes:  novolumes,
			expectedVolumes: novolumes,
			initialClaims:   newClaimArray("claim3-1", "uid3-1", "10Gi", "", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim3-1", "uid3-1", "10Gi", "", v1.ClaimLost, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  []string{"Warning ClaimLost"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim bound to non-existing volume. Check it's
			// marked as Lost.
			name:            "3-2 - bound claim with missing volume",
			initialVolumes:  novolumes,
			expectedVolumes: novolumes,
			initialClaims:   newClaimArray("claim3-2", "uid3-2", "10Gi", "volume3-2", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim3-2", "uid3-2", "10Gi", "volume3-2", v1.ClaimLost, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  []string{"Warning ClaimLost"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves the volume for a bound claim whose volume
			// lost its ClaimRef. syncVolume completes the bind (see 4-5).
			name:            "3-3 - bound claim with unbound volume",
			initialVolumes:  newVolumeArray("volume3-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume3-3", "10Gi", "uid3-3", "claim3-3", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimPending, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimPending, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim bound to volume with missing (or different)
			// volume.Spec.ClaimRef.UID. Check that the claim is marked as lost.
			name:            "3-4 - bound claim with prebound volume",
			initialVolumes:  newVolumeArray("volume3-4", "10Gi", "claim3-4-x", "claim3-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume3-4", "10Gi", "claim3-4-x", "claim3-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim3-4", "uid3-4", "10Gi", "volume3-4", v1.ClaimPending, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim3-4", "uid3-4", "10Gi", "volume3-4", v1.ClaimLost, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  []string{"Warning ClaimMisbound"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncVolume completes the bind for a claim that has AnnBindCompleted
			// but is still Pending (e.g. after a crash).
			name:            "3-5 - bound claim with bound volume",
			initialVolumes:  newVolumeArray("volume3-5", "10Gi", "uid3-5", "claim3-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume3-5", "10Gi", "uid3-5", "claim3-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim3-5", "uid3-5", "10Gi", "volume3-5", v1.ClaimPending, nil, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim3-5", "uid3-5", "10Gi", "volume3-5", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncClaim with claim bound to a volume that is bound to different
			// claim. Check that the claim is marked as lost.
			// TODO: test that an event is emitted
			name:            "3-6 - bound claim with bound volume",
			initialVolumes:  newVolumeArray("volume3-6", "10Gi", "uid3-6-x", "claim3-6-x", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume3-6", "10Gi", "uid3-6-x", "claim3-6-x", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim3-6", "uid3-6", "10Gi", "volume3-6", v1.ClaimPending, nil, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim3-6", "uid3-6", "10Gi", "volume3-6", v1.ClaimLost, nil, volume.AnnBindCompleted),
			expectedEvents:  []string{"Warning ClaimMisbound"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves the volume even if the claim's selector
			// doesn't match. syncVolume completes the bind (see 4-5-2).
			name:            "3-7 - bound claim with unbound volume where selector doesn't match",
			initialVolumes:  newVolumeArray("volume3-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume3-3", "10Gi", "uid3-3", "claim3-3", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   withLabelSelector(labels, newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimPending, nil, volume.AnnBoundByController, volume.AnnBindCompleted)),
			expectedClaims:  withLabelSelector(labels, newClaimArray("claim3-3", "uid3-3", "10Gi", "volume3-3", v1.ClaimPending, nil, volume.AnnBoundByController, volume.AnnBindCompleted)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		// [Unit test set 4] All syncVolume tests.
		{
			// syncVolume with pending volume. Check it's marked as Available.
			name:            "4-1 - pending volume",
			initialVolumes:  newVolumeArray("volume4-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with prebound pending volume. Check it's marked as
			// Available.
			name:            "4-2 - pending prebound volume",
			initialVolumes:  newVolumeArray("volume4-2", "10Gi", "", "claim4-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-2", "10Gi", "", "claim4-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound to missing claim.
			// Check the volume gets Released
			name:            "4-3 - bound volume with missing claim",
			initialVolumes:  newVolumeArray("volume4-3", "10Gi", "uid4-3", "claim4-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-3", "10Gi", "uid4-3", "claim4-3", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   noclaims,
			expectedClaims:  noclaims,
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound to claim with different UID.
			// Check the volume gets Released.
			name:            "4-4 - volume bound to claim with different UID",
			initialVolumes:  newVolumeArray("volume4-4", "10Gi", "uid4-4", "claim4-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-4", "10Gi", "uid4-4", "claim4-4", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-4", "uid4-4-x", "10Gi", "volume4-4", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim4-4", "uid4-4-x", "10Gi", "volume4-4", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with a controller-reserved volume (ClaimRef+UID) whose
			// claim is not yet bound: syncVolume completes the bind.
			name:            "4-5 - volume reserved by controller, complete bind",
			initialVolumes:  newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim4-5", "uid4-5", "10Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-5", "uid4-5", "10Gi", "volume4-5", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with a user-prebound volume (ClaimRef+UID, no
			// AnnBoundByController) whose claim is not yet bound: syncVolume
			// completes the bind and stamps AnnBoundByController on the claim.
			name:            "4-5-1 - volume prebound by user, complete bind",
			initialVolumes:  newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-5", "10Gi", "uid4-5", "claim4-5", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-5", "uid4-5", "10Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-5", "uid4-5", "10Gi", "volume4-5", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes the bind even if the claim's selector
			// doesn't match the volume (see 3-7 for the reserve side).
			name:            "4-5-2 - complete bind for reserved volume, claim selector doesn't match",
			initialVolumes:  newVolumeArray("volume4-5-2", "10Gi", "uid4-5-2", "claim4-5-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume4-5-2", "10Gi", "uid4-5-2", "claim4-5-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   withLabelSelector(labels, newClaimArray("claim4-5-2", "uid4-5-2", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withLabelSelector(labels, newClaimArray("claim4-5-2", "uid4-5-2", "10Gi", "volume4-5-2", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound to bound claim.
			// Check that the volume is marked as Bound.
			name:            "4-6 - volume bound by to bound claim",
			initialVolumes:  newVolumeArray("volume4-6", "10Gi", "uid4-6", "claim4-6", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-6", "10Gi", "uid4-6", "claim4-6", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-6", "uid4-6", "10Gi", "volume4-6", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim4-6", "uid4-6", "10Gi", "volume4-6", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound by controller to claim bound to
			// another volume. Check that the volume is rolled back.
			name:            "4-7 - volume bound by controller to claim bound somewhere else",
			initialVolumes:  newVolumeArray("volume4-7", "10Gi", "uid4-7", "claim4-7", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume4-7", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-7", "uid4-7", "10Gi", "volume4-7-x", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim4-7", "uid4-7", "10Gi", "volume4-7-x", v1.ClaimBound, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound by user to claim bound to
			// another volume. Check that the volume is marked as Available
			// and its UID is reset.
			name:            "4-8 - volume bound by user to claim bound somewhere else",
			initialVolumes:  newVolumeArray("volume4-8", "10Gi", "uid4-8", "claim4-8", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-8", "10Gi", "", "claim4-8", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-8", "uid4-8", "10Gi", "volume4-8-x", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim4-8", "uid4-8", "10Gi", "volume4-8-x", v1.ClaimBound, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with a volume only pre-bound to a claim (ClaimRef.UID
			// unset) that is bound to another volume: the prebind is kept and the
			// volume stays Available.
			name:            "4-8-1 - volume prebound by user to claim bound somewhere else",
			initialVolumes:  newVolumeArray("volume4-8-1", "10Gi", "", "claim4-8-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-8-1", "10Gi", "", "claim4-8-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-8-1", "uid4-8-1", "10Gi", "volume4-8-1-x", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim4-8-1", "uid4-8-1", "10Gi", "volume4-8-1-x", v1.ClaimBound, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound to bound claim.
			// Check that the volume is not deleted.
			name:            "4-9 - volume bound to bound claim, with PersistentVolumeReclaimDelete",
			initialVolumes:  newVolumeArray("volume4-9", "10Gi", "uid4-9", "claim4-9", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty),
			expectedVolumes: newVolumeArray("volume4-9", "10Gi", "uid4-9", "claim4-9", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classEmpty),
			initialClaims:   newClaimArray("claim4-9", "uid4-9", "10Gi", "volume4-9", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedClaims:  newClaimArray("claim4-9", "uid4-9", "10Gi", "volume4-9", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume with volume bound to missing claim.
			// Check that a volume deletion is attempted. It fails because there is no deleter.
			name:           "4-10 - volume bound to missing claim",
			initialVolumes: newVolumeArray("volume4-10", "10Gi", "uid4-10", "claim4-10", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty),
			expectedVolumes: func() []*v1.PersistentVolume {
				volumes := newVolumeArray("volume4-10", "10Gi", "uid4-10", "claim4-10", v1.VolumeFailed, v1.PersistentVolumeReclaimDelete, classEmpty)
				volumes[0].Status.Message = `error getting deleter volume plugin for volume "volume4-10": no volume plugin matched`
				return volumes
			}(),
			initialClaims:  noclaims,
			expectedClaims: noclaims,
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncVolume,
		},
		{
			// syncVolume with volume bound to claim which exists in etcd but not in the local cache.
			// Check that nothing changes, in contrast to case 4-10 above.
			name:            "4-11 - volume bound to unknown claim",
			initialVolumes:  newVolumeArray("volume4-11", "10Gi", "uid4-11", "claim4-11", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty),
			expectedVolumes: newVolumeArray("volume4-11", "10Gi", "uid4-11", "claim4-11", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty),
			initialClaims:   newClaimArray("claim4-11", "uid4-11", "10Gi", "volume4-11", v1.ClaimBound, nil, annSkipLocalStore),
			expectedClaims:  newClaimArray("claim4-11", "uid4-11", "10Gi", "volume4-11", v1.ClaimBound, nil, annSkipLocalStore),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},

		{
			// syncVolume with volume bound to claim which exists in etcd but not updated to local cache.
			name:            "4-12 - volume bound to newest claim but not updated to local cache",
			initialVolumes:  newVolumeArray("volume4-12", "10Gi", "uid4-12-new", "claim4-12", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty),
			expectedVolumes: newVolumeArray("volume4-12", "10Gi", "uid4-12-new", "claim4-12", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classEmpty),
			initialClaims: func() []*v1.PersistentVolumeClaim {
				newClaim := newClaimArray("claim4-12", "uid4-12", "10Gi", "volume4-12", v1.ClaimBound, nil, "")
				// update uid to new-uid and not sync to cache.
				newClaim = append(newClaim, newClaimArray("claim4-12", "uid4-12-new", "10Gi", "volume4-12", v1.ClaimBound, nil, annSkipLocalStore)...)
				return newClaim
			}(),
			expectedClaims: newClaimArray("claim4-12", "uid4-12-new", "10Gi", "volume4-12", v1.ClaimBound, nil, annSkipLocalStore),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncVolume,
		},
		{
			// syncVolume completes a bind for a volume reserved by syncClaim
			// (ClaimRef+UID+AnnBoundByController, still Available). This is the
			// completion half of reserve case 1-1: PV phase->Bound, PVC bound.
			name:            "4-13 - complete bind for reserved volume",
			initialVolumes:  newVolumeArray("volume4-13", "1Gi", "uid4-13", "claim4-13", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume4-13", "1Gi", "uid4-13", "claim4-13", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim4-13", "uid4-13", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-13", "uid4-13", "1Gi", "volume4-13", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes a bind for a PV prebound by an admin with an
			// empty ClaimRef.UID and a claim that has not chosen a PV
			// (VolumeName==""). syncVolume stamps the UID and completes the bind.
			name:            "4-14 - complete bind for empty-UID prebound volume",
			initialVolumes:  newVolumeArray("volume4-14", "1Gi", "", "claim4-14", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-14", "1Gi", "uid4-14", "claim4-14", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-14", "uid4-14", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-14", "uid4-14", "1Gi", "volume4-14", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume refuses to complete a bind for an empty-UID prebound PV
			// that is too small for a claim that has not chosen a PV: it keeps the
			// PV Available and emits VolumeMismatch.
			name:            "4-15 - do not complete bind for too-small empty-UID prebound volume",
			initialVolumes:  newVolumeArray("volume4-15", "1Gi", "", "claim4-15", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-15", "1Gi", "", "claim4-15", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-15", "uid4-15", "10Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-15", "uid4-15", "10Gi", "", v1.ClaimPending, nil),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes a mutual prebind (PV.ClaimRef with UID + claim
			// pointing back by name) whose claim is still Pending: it must finish the
			// claim side, not just set the PV phase. Completion half of reserve case 2-5.
			name:            "4-16 - complete bind for mutual prebind, claim still pending",
			initialVolumes:  newVolumeArray("volume4-16", "1Gi", "uid4-16", "claim4-16", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-16", "1Gi", "uid4-16", "claim4-16", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-16", "uid4-16", "1Gi", "volume4-16", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-16", "uid4-16", "1Gi", "volume4-16", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes a bind for a PV reserved by the controller
			// (AnnBoundByController) whose claim already references it by name but is
			// still Pending. Exercises the VolumeName==volume.Name completion branch
			// for a controller-driven bind; the claim must gain AnnBindCompleted.
			name:            "4-17 - complete bind for controller-reserved volume, claim points back",
			initialVolumes:  newVolumeArray("volume4-17", "1Gi", "uid4-17", "claim4-17", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			expectedVolumes: newVolumeArray("volume4-17", "1Gi", "uid4-17", "claim4-17", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim4-17", "uid4-17", "1Gi", "volume4-17", v1.ClaimPending, nil, volume.AnnBoundByController),
			expectedClaims:  newClaimArray("claim4-17", "uid4-17", "1Gi", "volume4-17", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes a mutual prebind by name (PV.ClaimRef with an
			// empty UID + claim pointing back by name) whose claim is still
			// Pending. It stamps the UID and completes the bind via the
			// volume.Name branch, without adding AnnBoundByController to either
			// object (the admin established the prebind). Completion half of
			// reserve case 2-4.
			name:            "4-18 - complete bind for mutual prebind by name, empty UID",
			initialVolumes:  newVolumeArray("volume4-18", "1Gi", "", "claim4-18", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume4-18", "1Gi", "uid4-18", "claim4-18", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty),
			initialClaims:   newClaimArray("claim4-18", "uid4-18", "1Gi", "volume4-18", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim4-18", "uid4-18", "1Gi", "volume4-18", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes the bind for a PV pre-bound to the claim (by
			// name) whose storageClass does not match the claim's. findBestMatchForClaim
			// returns such a PV without checking the class, so checkVolumeSatisfyClaim
			// must not refuse it here.
			name:            "4-19 - complete bind for prebound volume with mismatching class",
			initialVolumes:  newVolumeArray("volume4-19", "1Gi", "", "claim4-19", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			expectedVolumes: newVolumeArray("volume4-19", "1Gi", "uid4-19", "claim4-19", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classGold),
			initialClaims:   newClaimArray("claim4-19", "uid4-19", "1Gi", "", v1.ClaimPending, &classSilver),
			expectedClaims:  newClaimArray("claim4-19", "uid4-19", "1Gi", "volume4-19", v1.ClaimBound, &classSilver, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},

		// PVC with class
		{
			// syncClaim reserves a volume of the requested class even if there is
			// a smaller PV available (writes only the PV). syncVolume completes
			// the bind (see 4-13).
			name: "13-1 - reserve by class",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume13-1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume13-1-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume13-1-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("volume13-1-2", "10Gi", "uid13-1", "claim13-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold, volume.AnnBoundByController),
			},
			initialClaims:  newClaimArray("claim13-1", "uid13-1", "1Gi", "", v1.ClaimPending, &classGold),
			expectedClaims: newClaimArray("claim13-1", "uid13-1", "1Gi", "", v1.ClaimPending, &classGold),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim reserves a volume without a class even if there is a
			// smaller PV with a class available. syncVolume completes the bind
			// (see 4-13).
			name: "13-2 - reserve without a class",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume13-2-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
				newVolume("volume13-2-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume13-2-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
				newVolume("volume13-2-2", "10Gi", "uid13-2", "claim13-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			},
			initialClaims:  newClaimArray("claim13-2", "uid13-2", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims: newClaimArray("claim13-2", "uid13-2", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim reserves a volume with the given class even if there is a
			// smaller PV with a different class available. syncVolume completes
			// the bind (see 4-13).
			name: "13-3 - reserve to specific a class",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume13-3-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classSilver),
				newVolume("volume13-3-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume13-3-1", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classSilver),
				newVolume("volume13-3-2", "10Gi", "uid13-3", "claim13-3", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold, volume.AnnBoundByController),
			},
			initialClaims:  newClaimArray("claim13-3", "uid13-3", "1Gi", "", v1.ClaimPending, &classGold),
			expectedClaims: newClaimArray("claim13-3", "uid13-3", "1Gi", "", v1.ClaimPending, &classGold),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// syncClaim reserves a class="" PV for a claim requesting class "".
			// syncVolume completes the bind (see 4-13).
			name:            "13-4 - empty class",
			initialVolumes:  newVolumeArray("volume13-4", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume13-4", "1Gi", "uid13-4", "claim13-4", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim13-4", "uid13-4", "1Gi", "", v1.ClaimPending, &classEmpty),
			expectedClaims:  newClaimArray("claim13-4", "uid13-4", "1Gi", "", v1.ClaimPending, &classEmpty),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves a class="" PV for a claim requesting class nil.
			// syncVolume completes the bind (see 4-13).
			name:            "13-5 - nil class",
			initialVolumes:  newVolumeArray("volume13-5", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume13-5", "1Gi", "uid13-5", "claim13-5", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim13-5", "uid13-5", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim13-5", "uid13-5", "1Gi", "", v1.ClaimPending, nil),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
	}

	// Once the feature-gate VolumeAttributesClass is promoted to GA, merge it with the above tests
	whenFeatureGateEnabled := []controllerTest{
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, but its volume attributes class does not match. Check that the claim status is reset to Pending
			name:            "2-11 - claim prebound to unbound volume that volume attributes class is different",
			initialVolumes:  volumesWithVAC(classGold, newVolumeArray("volume2-11", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: volumesWithVAC(classGold, newVolumeArray("volume2-11", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   newClaimArray("claim2-11", "uid2-11", "1Gi", "volume2-11", v1.ClaimBound, nil),
			expectedClaims:  newClaimArray("claim2-11", "uid2-11", "1Gi", "volume2-11", v1.ClaimPending, nil),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim with claim pre-bound to an unbound PV with the same volume
			// attributes class: reserve writes the full ClaimRef +
			// AnnBoundByController on the PV and leaves the claim untouched. The
			// claim already names this PV, so syncVolume completes it via the
			// volume.Name branch (see 4-17).
			name:            "2-12 - reserve claim-prebound unbound volume with matching VAC",
			initialVolumes:  volumesWithVAC(classGold, newVolumeArray("volume2-12", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: volumesWithVAC(classGold, newVolumeArray("volume2-12", "1Gi", "uid2-12", "claim2-12", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   claimWithVAC(&classGold, newClaimArray("claim2-12", "uid2-12", "1Gi", "volume2-12", v1.ClaimPending, nil)),
			expectedClaims:  claimWithVAC(&classGold, newClaimArray("claim2-12", "uid2-12", "1Gi", "volume2-12", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
	}

	// Once the feature-gate VolumeAttributesClass is promoted to GA, remove it
	whenFeatureGateDisabled := []controllerTest{
		{
			// syncClaim with claim pre-bound to a PV that exists and is
			// unbound, they have the same volume attributes class but the feature-gate is disabled. Check that the claim status is reset to Pending
			name:            "2-13 - claim prebound to unbound volume that volume attributes class is same",
			initialVolumes:  volumesWithVAC(classGold, newVolumeArray("volume2-13", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: volumesWithVAC(classGold, newVolumeArray("volume2-13", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withExpectedVAC(&classGold, claimWithVAC(&classGold, newClaimArray("claim2-13", "uid2-13", "1Gi", "volume2-13", v1.ClaimBound, nil))),
			expectedClaims:  claimWithVAC(&classGold, newClaimArray("claim2-13", "uid2-13", "1Gi", "volume2-13", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
	}

	for _, isEnabled := range []bool{true, false} {
		if !isEnabled {
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.35"))
		}
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeAttributesClass, isEnabled)

		allTests := tests
		if isEnabled {
			allTests = append(allTests, whenFeatureGateEnabled...)
		} else {
			allTests = append(allTests, whenFeatureGateDisabled...)
		}

		for i := range allTests {
			allTests[i].name = fmt.Sprintf("features.VolumeAttributesClass=%v %s", isEnabled, allTests[i].name)
		}

		_, ctx := ktesting.NewTestContext(t)
		runSyncTests(t, ctx, allTests, []*storage.StorageClass{
			{
				ObjectMeta:        metav1.ObjectMeta{Name: classWait},
				VolumeBindingMode: &modeWait,
			},
		}, []*v1.Pod{})
	}
}

func TestSyncBlockVolume(t *testing.T) {
	modeBlock := v1.PersistentVolumeBlock
	modeFile := v1.PersistentVolumeFilesystem

	// Tests assume defaulting, so feature enabled will never have nil volumeMode
	tests := []controllerTest{
		// PVC with VolumeMode
		{
			// syncClaim reserves a block volume for a requested block claim.
			// syncVolume completes the bind (see 14-14).
			name:            "14-1 - reserve volumeMode block",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-1", "10Gi", "uid14-1", "claim14-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-1", "uid14-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-1", "uid14-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves a filesystem volume for a requested filesystem
			// claim. syncVolume completes the bind (see 14-14-1).
			name:            "14-2 - reserve volumeMode filesystem",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-2", "10Gi", "uid14-2", "claim14-2", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-2", "uid14-2", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-2", "uid14-2", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind to an unspecified volumemode for claim to a specified filesystem volume
			name:            "14-3 - do not bind pv volumeMode filesystem and pvc volumeMode block",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-3", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-3", "uid14-3", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-3", "uid14-3", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind a requested filesystem claim to an unspecified volumeMode for volume
			name:            "14-4 - do not bind pv volumeMode block and pvc volumeMode filesystem",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-4", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-4", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-4", "uid14-4", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-4", "uid14-4", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind when matching class but not matching volumeModes
			name:            "14-5 - do not bind when matching class but not volumeMode",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-5", "uid14-5", "10Gi", "", v1.ClaimPending, &classGold)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-5", "uid14-5", "10Gi", "", v1.ClaimPending, &classGold)),
			expectedEvents:  []string{"Warning ProvisioningFailed"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind when matching volumeModes but class does not match
			name:            "14-5-1 - do not bind when matching volumeModes but class does not match",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-1", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-1", "uid14-5-1", "10Gi", "", v1.ClaimPending, &classSilver)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-1", "uid14-5-1", "10Gi", "", v1.ClaimPending, &classSilver)),
			expectedEvents:  []string{"Warning ProvisioningFailed"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind when pvc is prebound to pv with matching volumeModes but class does not match
			name:            "14-5-2 - do not bind when pvc is prebound to pv with matching volumeModes but class does not match",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-5-2", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-2", "uid14-5-2", "10Gi", "volume14-5-2", v1.ClaimPending, &classSilver)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-5-2", "uid14-5-2", "10Gi", "volume14-5-2", v1.ClaimPending, &classSilver)),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim defers when pv is prebound (by name) and volumeModes
			// match. The PV already has a ClaimRef (empty UID), so it is past the
			// reserve handoff: syncClaim writes nothing and enqueues it for
			// syncVolume. The claim did not name the PV, so syncVolume completes it
			// via the case-"" branch, stamping the UID (see 4-14; block-mode
			// completion is covered by 14-14).
			name:            "14-7 - defer when pv is prebound and volumeModes match",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-7", "10Gi", "", "claim14-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-7", "10Gi", "", "claim14-7", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-7", "uid14-7", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-7", "uid14-7", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind when pvc is prebound to pv with mismatching volumeModes
			name:            "14-8 - do not bind when pvc is prebound to pv with mismatching volumeModes",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-8", "uid14-8", "10Gi", "volume14-8", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-8", "uid14-8", "10Gi", "volume14-8", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// failed syncVolume do not bind when pvc is prebound to pv with mismatching volumeModes
			name:            "14-8-1 - do not bind when pvc is prebound to pv with mismatching volumeModes",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8-1", "10Gi", "", "claim14-8-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-8-1", "10Gi", "", "claim14-8-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-8-1", "uid14-8-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-8-1", "uid14-8-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Normal FailedBinding"},
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves when pvc is prebound to a pv with matching
			// volumeModes (block): reserve writes the full ClaimRef +
			// AnnBoundByController on the PV. The claim already names this PV, so
			// syncVolume completes it via the volume.Name branch, which does not
			// re-check the volumeMode (see 4-17).
			name:            "14-9 - reserve when pvc is prebound to pv with matching volumeModes block",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-9", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-9", "10Gi", "uid14-9", "claim14-9", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-9", "uid14-9", "10Gi", "volume14-9", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-9", "uid14-9", "10Gi", "volume14-9", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim defers when pv is prebound (by name) to the pvc with
			// matching volumeModes (block). The PV already has a ClaimRef (empty
			// UID), so it is past the reserve handoff: syncClaim writes nothing and
			// enqueues it for syncVolume. The claim did not name the PV, so
			// syncVolume completes it via the case-"" branch, stamping the UID
			// (see 4-14; block-mode completion is covered by 14-14).
			name:            "14-10 - defer when pv is prebound to pvc with matching volumeModes block",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-10", "10Gi", "", "claim14-10", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-10", "10Gi", "", "claim14-10", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-10", "uid14-10", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-10", "uid14-10", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim reserves when pvc is prebound to a pv with matching
			// volumeModes (filesystem). The claim already names this PV, so
			// syncVolume completes it via the volume.Name branch (see 4-17).
			name:            "14-11 - reserve when pvc is prebound to pv with matching volumeModes filesystem",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-11", "10Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-11", "10Gi", "uid14-11", "claim14-11", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-11", "uid14-11", "10Gi", "volume14-11", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-11", "uid14-11", "10Gi", "volume14-11", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncClaim defers when pv is prebound (by name) to the pvc with
			// matching volumeModes (filesystem). The PV already has a ClaimRef
			// (empty UID), so it is past the reserve handoff: syncClaim writes
			// nothing and enqueues it for syncVolume. The claim did not name the
			// PV, so syncVolume completes it via the case-"" branch, stamping the
			// UID (see 4-14; file-mode completion is covered by 14-14-1).
			name:            "14-12 - defer when pv is prebound to pvc with matching volumeModes filesystem",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-12", "10Gi", "", "claim14-12", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-12", "10Gi", "", "claim14-12", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-12", "uid14-12", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-12", "uid14-12", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// syncVolume output warning when pv is prebound to pvc with mismatching volumeMode
			name:            "14-13 - output warning when pv is prebound to pvc with different volumeModes",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-13", "10Gi", "uid14-13", "claim14-13", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-13", "10Gi", "uid14-13", "claim14-13", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-13", "uid14-13", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-13", "uid14-13", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume output warning when pv is prebound to pvc with mismatching volumeMode
			name:            "14-13-1 - output warning when pv is prebound to pvc with different volumeModes",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-13-1", "10Gi", "uid14-13-1", "claim14-13-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-13-1", "10Gi", "uid14-13-1", "claim14-13-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-13-1", "uid14-13-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-13-1", "uid14-13-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedEvents:  []string{"Warning VolumeMismatch"},
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes the bind (no warning) when a reserved pv is
			// prebound to a pvc with matching volumeMode block. syncVolume owns
			// completion now; the claim gets bound on the volume worker.
			name:            "14-14 - complete bind when pv is prebound to pvc with matching volumeModes block",
			initialVolumes:  withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-14", "10Gi", "uid14-14", "claim14-14", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			expectedVolumes: withVolumeVolumeMode(&modeBlock, newVolumeArray("volume14-14", "10Gi", "uid14-14", "claim14-14", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeBlock, newClaimArray("claim14-14", "uid14-14", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeBlock, newClaimArray("claim14-14", "uid14-14", "10Gi", "volume14-14", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
		{
			// syncVolume completes the bind (no warning) when a reserved pv is
			// prebound to a pvc with matching volumeMode file.
			name:            "14-14-1 - complete bind when pv is prebound to pvc with matching volumeModes file",
			initialVolumes:  withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-14-1", "10Gi", "uid14-14-1", "claim14-14-1", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			expectedVolumes: withVolumeVolumeMode(&modeFile, newVolumeArray("volume14-14-1", "10Gi", "uid14-14-1", "claim14-14-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController)),
			initialClaims:   withClaimVolumeMode(&modeFile, newClaimArray("claim14-14-1", "uid14-14-1", "10Gi", "", v1.ClaimPending, nil)),
			expectedClaims:  withClaimVolumeMode(&modeFile, newClaimArray("claim14-14-1", "uid14-14-1", "10Gi", "volume14-14-1", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted)),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncVolume,
		},
	}
	_, ctx := ktesting.NewTestContext(t)
	runSyncTests(t, ctx, tests, []*storage.StorageClass{}, []*v1.Pod{})
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
func TestMultiSync(t *testing.T) {
	tests := []controllerTest{
		// Test simple binding
		{
			// syncClaim binds to a matching unbound volume.
			name:            "10-1 - successful bind",
			initialVolumes:  newVolumeArray("volume10-1", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume10-1", "1Gi", "uid10-1", "claim10-1", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim10-1", "uid10-1", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim10-1", "uid10-1", "1Gi", "volume10-1", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// Two controllers bound two PVs to single claim. Test one of them
			// wins and the second rolls back.
			name: "10-2 - bind PV race",
			initialVolumes: []*v1.PersistentVolume{
				newVolume("volume10-2-1", "1Gi", "uid10-2", "claim10-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
				newVolume("volume10-2-2", "1Gi", "uid10-2", "claim10-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			},
			expectedVolumes: []*v1.PersistentVolume{
				newVolume("volume10-2-1", "1Gi", "uid10-2", "claim10-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
				newVolume("volume10-2-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			},
			initialClaims:  newClaimArray("claim10-2", "uid10-2", "1Gi", "volume10-2-1", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedClaims: newClaimArray("claim10-2", "uid10-2", "1Gi", "volume10-2-1", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents: noevents,
			errors:         noerrors,
			test:           testSyncClaim,
		},
		{
			// End-to-end reserve+complete: a claim that requests a specific
			// available PV by name must reach Bound. syncClaim reserves the PV
			// (claim stays Pending) and syncVolume completes the bind on the
			// volume worker.
			name:            "10-3 - reserve by name then complete",
			initialVolumes:  newVolumeArray("volume10-3", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume10-3", "1Gi", "uid10-3", "claim10-3", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim10-3", "uid10-3", "1Gi", "volume10-3", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim10-3", "uid10-3", "1Gi", "volume10-3", v1.ClaimBound, nil, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
		{
			// End-to-end: a claim with no VolumeName matched to an available PV.
			// syncClaim reserves it (AnnBoundByController on the PV) and syncVolume
			// completes, binding the claim. Guards the common dynamic/auto-bind path.
			name:            "10-4 - reserve unbound match then complete",
			initialVolumes:  newVolumeArray("volume10-4", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty),
			expectedVolumes: newVolumeArray("volume10-4", "1Gi", "uid10-4", "claim10-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty, volume.AnnBoundByController),
			initialClaims:   newClaimArray("claim10-4", "uid10-4", "1Gi", "", v1.ClaimPending, nil),
			expectedClaims:  newClaimArray("claim10-4", "uid10-4", "1Gi", "volume10-4", v1.ClaimBound, nil, volume.AnnBoundByController, volume.AnnBindCompleted),
			expectedEvents:  noevents,
			errors:          noerrors,
			test:            testSyncClaim,
		},
	}
	_, ctx := ktesting.NewTestContext(t)
	runMultisyncTests(t, ctx, tests, []*storage.StorageClass{}, "")
}
