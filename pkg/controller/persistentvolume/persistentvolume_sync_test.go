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
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

// Test single call to syncClaim and syncVolume methods.
// 1. Fill in the controller with initial data
// 2. Call the tested function (syncClaim/syncVolume) via
//    controllerTest.testCall *once*.
// 3. Compare resulting volumes and claims with expected volumes and claims.
func TestSync(t *testing.T) {
	tests := []controllerTest{
		// [Unit test set 1] User did not care which PV they get.
		// Test the matching with no claim.Spec.VolumeName and with various
		// volumes.
		{
			// syncClaim binds to a matching unbound volume.
			"1-1 - successful bind",
			newVolumeArray("volume1-1", "1Gi", "", "", api.VolumePending),
			newVolumeArray("volume1-1", "1Gi", "uid1-1", "claim1-1", api.VolumeBound, annBoundByController),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-1", "uid1-1", "1Gi", "volume1-1", api.ClaimBound, annBoundByController, annBindCompleted),
			testSyncClaim,
		},
		{
			// syncClaim does not do anything when there is no matching volume.
			"1-2 - noop",
			newVolumeArray("volume1-2", "1Gi", "", "", api.VolumePending),
			newVolumeArray("volume1-2", "1Gi", "", "", api.VolumePending),
			newClaimArray("claim1-2", "uid1-2", "10Gi", "", api.ClaimPending),
			newClaimArray("claim1-2", "uid1-2", "10Gi", "", api.ClaimPending),
			testSyncClaim,
		},
		{
			// syncClaim resets claim.Status to Pending when there is no
			// matching volume.
			"1-3 - reset to Pending",
			newVolumeArray("volume1-3", "1Gi", "", "", api.VolumePending),
			newVolumeArray("volume1-3", "1Gi", "", "", api.VolumePending),
			newClaimArray("claim1-3", "uid1-3", "10Gi", "", api.ClaimBound),
			newClaimArray("claim1-3", "uid1-3", "10Gi", "", api.ClaimPending),
			testSyncClaim,
		},
		{
			// syncClaim binds claims to the smallest matching volume
			"1-4 - smallest volume",
			[]*api.PersistentVolume{
				newVolume("volume1-4_1", "10Gi", "", "", api.VolumePending),
				newVolume("volume1-4_2", "1Gi", "", "", api.VolumePending),
			},
			[]*api.PersistentVolume{
				newVolume("volume1-4_1", "10Gi", "", "", api.VolumePending),
				newVolume("volume1-4_2", "1Gi", "uid1-4", "claim1-4", api.VolumeBound, annBoundByController),
			},
			newClaimArray("claim1-4", "uid1-4", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-4", "uid1-4", "1Gi", "volume1-4_2", api.ClaimBound, annBoundByController, annBindCompleted),
			testSyncClaim,
		},
		{
			// syncClaim binds a claim only to volume that points to it (by
			// name), even though a smaller one is available.
			"1-5 - prebound volume by name - success",
			[]*api.PersistentVolume{
				newVolume("volume1-5_1", "10Gi", "", "claim1-5", api.VolumePending),
				newVolume("volume1-5_2", "1Gi", "", "", api.VolumePending),
			},
			[]*api.PersistentVolume{
				newVolume("volume1-5_1", "10Gi", "uid1-5", "claim1-5", api.VolumeBound),
				newVolume("volume1-5_2", "1Gi", "", "", api.VolumePending),
			},
			newClaimArray("claim1-5", "uid1-5", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-5", "uid1-5", "1Gi", "volume1-5_1", api.ClaimBound, annBoundByController, annBindCompleted),
			testSyncClaim,
		},
		{
			// syncClaim binds a claim only to volume that points to it (by
			// UID), even though a smaller one is available.
			"1-6 - prebound volume by UID - success",
			[]*api.PersistentVolume{
				newVolume("volume1-6_1", "10Gi", "uid1-6", "claim1-6", api.VolumePending),
				newVolume("volume1-6_2", "1Gi", "", "", api.VolumePending),
			},
			[]*api.PersistentVolume{
				newVolume("volume1-6_1", "10Gi", "uid1-6", "claim1-6", api.VolumeBound),
				newVolume("volume1-6_2", "1Gi", "", "", api.VolumePending),
			},
			newClaimArray("claim1-6", "uid1-6", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-6", "uid1-6", "1Gi", "volume1-6_1", api.ClaimBound, annBoundByController, annBindCompleted),
			testSyncClaim,
		},
		{
			// syncClaim does not bind claim to a volume prebound to a claim with
			// same name and different UID
			"1-7 - prebound volume to different claim",
			newVolumeArray("volume1-7", "10Gi", "uid1-777", "claim1-7", api.VolumePending),
			newVolumeArray("volume1-7", "10Gi", "uid1-777", "claim1-7", api.VolumePending),
			newClaimArray("claim1-7", "uid1-7", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-7", "uid1-7", "1Gi", "", api.ClaimPending),
			testSyncClaim,
		},
		{
			// syncClaim completes binding - simulates controller crash after
			// PV.ClaimRef is saved
			"1-8 - complete bind after crash - PV bound",
			newVolumeArray("volume1-8", "1Gi", "uid1-8", "claim1-8", api.VolumePending, annBoundByController),
			newVolumeArray("volume1-8", "1Gi", "uid1-8", "claim1-8", api.VolumeBound, annBoundByController),
			newClaimArray("claim1-8", "uid1-8", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-8", "uid1-8", "1Gi", "volume1-8", api.ClaimBound, annBoundByController, annBindCompleted),
			testSyncClaim,
		},
		{
			// syncClaim completes binding - simulates controller crash after
			// PV.Status is saved
			"1-9 - complete bind after crash - PV status saved",
			newVolumeArray("volume1-9", "1Gi", "uid1-9", "claim1-9", api.VolumeBound, annBoundByController),
			newVolumeArray("volume1-9", "1Gi", "uid1-9", "claim1-9", api.VolumeBound, annBoundByController),
			newClaimArray("claim1-9", "uid1-9", "1Gi", "", api.ClaimPending),
			newClaimArray("claim1-9", "uid1-9", "1Gi", "volume1-9", api.ClaimBound, annBoundByController, annBindCompleted),
			testSyncClaim,
		},
		/*		TODO: enable when syncClaim with annBindCompleted is implemented
				controllerTest{
					// syncClaim completes binding - simulates controller crash after
					// PVC.VolumeName is saved
					"10 - complete bind after crash - PVC bound",
					[]*api.PersistentVolume{
						newVolume("volume1-10", "1Gi", "uid1-10", "claim1-10", api.VolumeBound, annBoundByController),
					},
					[]*api.PersistentVolume{
						newVolume("volume1-10", "1Gi", "uid1-10", "claim1-10", api.VolumeBound, annBoundByController),
					},
					[]*api.PersistentVolumeClaim{
						newClaim("claim1-10", "uid1-10", "1Gi", "volume1-10", api.ClaimPending, annBoundByController, annBindCompleted),
					},
					[]*api.PersistentVolumeClaim{
						newClaim("claim1-10", "uid1-10", "1Gi", "volume1-10", api.ClaimBound, annBoundByController, annBindCompleted),
					},
					testSyncClaim,
				},
		*/
	}
	runSyncTests(t, tests)
}
