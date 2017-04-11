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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
)

var class1Parameters = map[string]string{
	"param1": "value1",
}
var class2Parameters = map[string]string{
	"param2": "value2",
}
var storageClasses = []*storage.StorageClass{
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},

		ObjectMeta: metav1.ObjectMeta{
			Name: "gold",
		},

		Provisioner: mockPluginName,
		Parameters:  class1Parameters,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "silver",
		},
		Provisioner: mockPluginName,
		Parameters:  class2Parameters,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "external",
		},
		Provisioner: "vendor.com/my-volume",
		Parameters:  class1Parameters,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "unknown-internal",
		},
		Provisioner: "kubernetes.io/unknown",
		Parameters:  class1Parameters,
	},
}

// call to storageClass 1, returning an error
var provision1Error = provisionCall{
	ret:                errors.New("Mock provisioner error"),
	expectedParameters: class1Parameters,
}

// call to storageClass 1, returning a valid PV
var provision1Success = provisionCall{
	ret:                nil,
	expectedParameters: class1Parameters,
}

// call to storageClass 2, returning a valid PV
var provision2Success = provisionCall{
	ret:                nil,
	expectedParameters: class2Parameters,
}

var provisionAlphaSuccess = provisionCall{
	ret: nil,
}

// Test single call to syncVolume, expecting provisioning to happen.
// 1. Fill in the controller with initial data
// 2. Call the syncVolume *once*.
// 3. Compare resulting volumes with expected volumes.
func TestProvisionSync(t *testing.T) {
	tests := []controllerTest{
		{
			// Provision a volume (with a default class)
			"11-1 - successful provision with storage class 1",
			novolumes,
			newVolumeArray("pvc-uid11-1", "1Gi", "uid11-1", "claim11-1", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim11-1", "uid11-1", "1Gi", "", v1.ClaimPending, &classGold),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-1", "uid11-1", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Normal ProvisioningSucceeded"}, noerrors, wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// Provision failure - plugin not found
			"11-2 - plugin not found",
			novolumes,
			novolumes,
			newClaimArray("claim11-2", "uid11-2", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-2", "uid11-2", "1Gi", "", v1.ClaimPending, &classGold),
			[]string{"Warning ProvisioningFailed"}, noerrors,
			testSyncClaim,
		},
		{
			// Provision failure - newProvisioner returns error
			"11-3 - newProvisioner failure",
			novolumes,
			novolumes,
			newClaimArray("claim11-3", "uid11-3", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-3", "uid11-3", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Warning ProvisioningFailed"}, noerrors,
			wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// Provision failure - Provision returns error
			"11-4 - provision failure",
			novolumes,
			novolumes,
			newClaimArray("claim11-4", "uid11-4", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-4", "uid11-4", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Warning ProvisioningFailed"}, noerrors,
			wrapTestWithProvisionCalls([]provisionCall{provision1Error}, testSyncClaim),
		},
		{
			// No provisioning if there is a matching volume available
			"11-6 - provisioning when there is a volume available",
			newVolumeArray("volume11-6", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classGold),
			newVolumeArray("volume11-6", "1Gi", "uid11-6", "claim11-6", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classGold, annBoundByController),
			newClaimArray("claim11-6", "uid11-6", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-6", "uid11-6", "1Gi", "volume11-6", v1.ClaimBound, &classGold, annBoundByController, annBindCompleted),
			noevents, noerrors,
			// No provisioning plugin confingure - makes the test fail when
			// the controller errorneously tries to provision something
			wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// Provision success? - claim is bound before provisioner creates
			// a volume.
			"11-7 - claim is bound before provisioning",
			novolumes,
			newVolumeArray("pvc-uid11-7", "1Gi", "uid11-7", "claim11-7", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim11-7", "uid11-7", "1Gi", "", v1.ClaimPending, &classGold),
			// The claim would be bound in next syncClaim
			newClaimArray("claim11-7", "uid11-7", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim), func(ctrl *PersistentVolumeController, reactor *volumeReactor) {
				// Create a volume before provisionClaimOperation starts.
				// This similates a parallel controller provisioning the volume.
				reactor.lock.Lock()
				volume := newVolume("pvc-uid11-7", "1Gi", "uid11-7", "claim11-7", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, annBoundByController, annDynamicallyProvisioned)
				reactor.volumes[volume.Name] = volume
				reactor.lock.Unlock()
			}),
		},
		{
			// Provision success - cannot save provisioned PV once,
			// second retry succeeds
			"11-8 - cannot save provisioned volume",
			novolumes,
			newVolumeArray("pvc-uid11-8", "1Gi", "uid11-8", "claim11-8", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim11-8", "uid11-8", "1Gi", "", v1.ClaimPending, &classGold),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-8", "uid11-8", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Normal ProvisioningSucceeded"},
			[]reactorError{
				// Inject error to the first
				// kubeclient.PersistentVolumes.Create() call. All other calls
				// will succeed.
				{"create", "persistentvolumes", errors.New("Mock creation error")},
			},
			wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// Provision success? - cannot save provisioned PV five times,
			// volume is deleted and delete succeeds
			"11-9 - cannot save provisioned volume, delete succeeds",
			novolumes,
			novolumes,
			newClaimArray("claim11-9", "uid11-9", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-9", "uid11-9", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Warning ProvisioningFailed"},
			[]reactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{"create", "persistentvolumes", errors.New("Mock creation error1")},
				{"create", "persistentvolumes", errors.New("Mock creation error2")},
				{"create", "persistentvolumes", errors.New("Mock creation error3")},
				{"create", "persistentvolumes", errors.New("Mock creation error4")},
				{"create", "persistentvolumes", errors.New("Mock creation error5")},
			},
			wrapTestWithPluginCalls(
				nil,                                // recycle calls
				[]error{nil},                       // delete calls
				[]provisionCall{provision1Success}, // provision calls
				testSyncClaim,
			),
		},
		{
			// Provision failure - cannot save provisioned PV five times,
			// volume delete failed - no plugin found
			"11-10 - cannot save provisioned volume, no delete plugin found",
			novolumes,
			novolumes,
			newClaimArray("claim11-10", "uid11-10", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-10", "uid11-10", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Warning ProvisioningFailed", "Warning ProvisioningCleanupFailed"},
			[]reactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{"create", "persistentvolumes", errors.New("Mock creation error1")},
				{"create", "persistentvolumes", errors.New("Mock creation error2")},
				{"create", "persistentvolumes", errors.New("Mock creation error3")},
				{"create", "persistentvolumes", errors.New("Mock creation error4")},
				{"create", "persistentvolumes", errors.New("Mock creation error5")},
			},
			// No deleteCalls are configured, which results into no deleter plugin available for the volume
			wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// Provision failure - cannot save provisioned PV five times,
			// volume delete failed - deleter returns error five times
			"11-11 - cannot save provisioned volume, deleter fails",
			novolumes,
			novolumes,
			newClaimArray("claim11-11", "uid11-11", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-11", "uid11-11", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Warning ProvisioningFailed", "Warning ProvisioningCleanupFailed"},
			[]reactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{"create", "persistentvolumes", errors.New("Mock creation error1")},
				{"create", "persistentvolumes", errors.New("Mock creation error2")},
				{"create", "persistentvolumes", errors.New("Mock creation error3")},
				{"create", "persistentvolumes", errors.New("Mock creation error4")},
				{"create", "persistentvolumes", errors.New("Mock creation error5")},
			},
			wrapTestWithPluginCalls(
				nil, // recycle calls
				[]error{ // delete calls
					errors.New("Mock deletion error1"),
					errors.New("Mock deletion error2"),
					errors.New("Mock deletion error3"),
					errors.New("Mock deletion error4"),
					errors.New("Mock deletion error5"),
				},
				[]provisionCall{provision1Success}, // provision calls
				testSyncClaim),
		},
		{
			// Provision failure - cannot save provisioned PV five times,
			// volume delete succeeds 2nd time
			"11-12 - cannot save provisioned volume, delete succeeds 2nd time",
			novolumes,
			novolumes,
			newClaimArray("claim11-12", "uid11-12", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-12", "uid11-12", "1Gi", "", v1.ClaimPending, &classGold, annStorageProvisioner),
			[]string{"Warning ProvisioningFailed"},
			[]reactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{"create", "persistentvolumes", errors.New("Mock creation error1")},
				{"create", "persistentvolumes", errors.New("Mock creation error2")},
				{"create", "persistentvolumes", errors.New("Mock creation error3")},
				{"create", "persistentvolumes", errors.New("Mock creation error4")},
				{"create", "persistentvolumes", errors.New("Mock creation error5")},
			},
			wrapTestWithPluginCalls(
				nil, // recycle calls
				[]error{ // delete calls
					errors.New("Mock deletion error1"),
					nil,
				}, //  provison calls
				[]provisionCall{provision1Success},
				testSyncClaim,
			),
		},
		{
			// Provision a volume (with non-default class)
			"11-13 - successful provision with storage class 2",
			novolumes,
			newVolumeArray("pvc-uid11-13", "1Gi", "uid11-13", "claim11-13", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classSilver, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim11-13", "uid11-13", "1Gi", "", v1.ClaimPending, &classSilver),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-13", "uid11-13", "1Gi", "", v1.ClaimPending, &classSilver, annStorageProvisioner),
			[]string{"Normal ProvisioningSucceeded"}, noerrors, wrapTestWithProvisionCalls([]provisionCall{provision2Success}, testSyncClaim),
		},
		{
			// Provision error - non existing class
			"11-14 - fail due to non-existing class",
			novolumes,
			novolumes,
			newClaimArray("claim11-14", "uid11-14", "1Gi", "", v1.ClaimPending, &classNonExisting),
			newClaimArray("claim11-14", "uid11-14", "1Gi", "", v1.ClaimPending, &classNonExisting),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// No provisioning with class=""
			"11-15 - no provisioning with class=''",
			novolumes,
			novolumes,
			newClaimArray("claim11-15", "uid11-15", "1Gi", "", v1.ClaimPending, &classEmpty),
			newClaimArray("claim11-15", "uid11-15", "1Gi", "", v1.ClaimPending, &classEmpty),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// No provisioning with class=nil
			"11-16 - no provisioning with class=nil",
			novolumes,
			novolumes,
			newClaimArray("claim11-15", "uid11-15", "1Gi", "", v1.ClaimPending, nil),
			newClaimArray("claim11-15", "uid11-15", "1Gi", "", v1.ClaimPending, nil),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// No provisioning + normal event with external provisioner
			"11-17 - external provisioner",
			novolumes,
			novolumes,
			newClaimArray("claim11-17", "uid11-17", "1Gi", "", v1.ClaimPending, &classExternal),
			claimWithAnnotation(annStorageProvisioner, "vendor.com/my-volume",
				newClaimArray("claim11-17", "uid11-17", "1Gi", "", v1.ClaimPending, &classExternal)),
			[]string{"Normal ExternalProvisioning"},
			noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// No provisioning + warning event with unknown internal provisioner
			"11-18 - unknown internal provisioner",
			novolumes,
			novolumes,
			newClaimArray("claim11-18", "uid11-18", "1Gi", "", v1.ClaimPending, &classUnknownInternal),
			newClaimArray("claim11-18", "uid11-18", "1Gi", "", v1.ClaimPending, &classUnknownInternal),
			[]string{"Warning ProvisioningFailed"},
			noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
	}
	runSyncTests(t, tests, storageClasses)
}

func TestAlphaProvisionSync(t *testing.T) {
	tests := []controllerTest{
		{
			// Provision a volume with alpha annotation
			"14-1 - successful alpha provisioning",
			novolumes,
			newVolumeArray("pvc-uid14-1", "1Gi", "uid14-1", "claim14-1", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classEmpty, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim14-1", "uid14-1", "1Gi", "", v1.ClaimPending, nil, v1.AlphaStorageClassAnnotation),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim14-1", "uid14-1", "1Gi", "", v1.ClaimPending, nil, v1.AlphaStorageClassAnnotation, annStorageProvisioner),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{provisionAlphaSuccess}, testSyncClaim),
		},
		{
			// Provision success - there is already a volume available, still
			// we provision a new one when requested.
			"14-2 - no alpha provisioning when there is a volume available",
			newVolumeArray("volume14-2", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
			[]*v1.PersistentVolume{
				newVolume("volume14-2", "1Gi", "", "", v1.VolumePending, v1.PersistentVolumeReclaimRetain, classEmpty),
				newVolume("pvc-uid14-2", "1Gi", "uid14-2", "claim14-2", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classEmpty, annBoundByController, annDynamicallyProvisioned),
			},
			newClaimArray("claim14-2", "uid14-2", "1Gi", "", v1.ClaimPending, nil, v1.AlphaStorageClassAnnotation),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim14-2", "uid14-2", "1Gi", "", v1.ClaimPending, nil, v1.AlphaStorageClassAnnotation, annStorageProvisioner),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{provisionAlphaSuccess}, testSyncClaim),
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
func TestProvisionMultiSync(t *testing.T) {
	tests := []controllerTest{
		{
			// Provision a volume with binding
			"12-1 - successful provision",
			novolumes,
			newVolumeArray("pvc-uid12-1", "1Gi", "uid12-1", "claim12-1", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, annBoundByController, annDynamicallyProvisioned),
			newClaimArray("claim12-1", "uid12-1", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim12-1", "uid12-1", "1Gi", "pvc-uid12-1", v1.ClaimBound, &classGold, annBoundByController, annBindCompleted, annStorageProvisioner),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
	}

	runMultisyncTests(t, tests, storageClasses, storageClasses[0].Name)
}

// When provisioning is disabled, provisioning a claim should instantly return nil
func TestDisablingDynamicProvisioner(t *testing.T) {
	ctrl := newTestController(nil, nil, false)
	retVal := ctrl.provisionClaim(nil)
	if retVal != nil {
		t.Errorf("Expected nil return but got %v", retVal)
	}
}
