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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	api "k8s.io/kubernetes/pkg/apis/core"
	pvtesting "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/testing"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
)

var class1Parameters = map[string]string{
	"param1": "value1",
}
var class2Parameters = map[string]string{
	"param2": "value2",
}
var deleteReclaimPolicy = v1.PersistentVolumeReclaimDelete
var modeImmediate = storage.VolumeBindingImmediate
var storageClasses = []*storage.StorageClass{
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},

		ObjectMeta: metav1.ObjectMeta{
			Name: "gold",
		},

		Provisioner:       mockPluginName,
		Parameters:        class1Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: &modeImmediate,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "silver",
		},
		Provisioner:       mockPluginName,
		Parameters:        class2Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: &modeImmediate,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "copper",
		},
		Provisioner:       mockPluginName,
		Parameters:        class1Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: &modeWait,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "external",
		},
		Provisioner:       "vendor.com/my-volume",
		Parameters:        class1Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: &modeImmediate,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "external-wait",
		},
		Provisioner:       "vendor.com/my-volume-wait",
		Parameters:        class1Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: &modeWait,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "unknown-internal",
		},
		Provisioner:       "kubernetes.io/unknown",
		Parameters:        class1Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: &modeImmediate,
	},
	{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "unsupported-mountoptions",
		},
		Provisioner:       mockPluginName,
		Parameters:        class1Parameters,
		ReclaimPolicy:     &deleteReclaimPolicy,
		MountOptions:      []string{"foo"},
		VolumeBindingMode: &modeImmediate,
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
			newVolumeArray("pvc-uid11-1", "1Gi", "uid11-1", "claim11-1", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned),
			newClaimArray("claim11-1", "uid11-1", "1Gi", "", v1.ClaimPending, &classGold),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-1", "uid11-1", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
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
			newClaimArray("claim11-3", "uid11-3", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Warning ProvisioningFailed"}, noerrors,
			wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// Provision failure - Provision returns error
			"11-4 - provision failure",
			novolumes,
			novolumes,
			newClaimArray("claim11-4", "uid11-4", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-4", "uid11-4", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Warning ProvisioningFailed"}, noerrors,
			wrapTestWithProvisionCalls([]provisionCall{provision1Error}, testSyncClaim),
		},
		{
			// No provisioning if there is a matching volume available
			"11-6 - provisioning when there is a volume available",
			newVolumeArray("volume11-6", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classGold),
			newVolumeArray("volume11-6", "1Gi", "uid11-6", "claim11-6", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classGold, pvutil.AnnBoundByController),
			newClaimArray("claim11-6", "uid11-6", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-6", "uid11-6", "1Gi", "volume11-6", v1.ClaimBound, &classGold, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents, noerrors,
			// No provisioning plugin confingure - makes the test fail when
			// the controller erroneously tries to provision something
			wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// Provision success? - claim is bound before provisioner creates
			// a volume.
			"11-7 - claim is bound before provisioning",
			novolumes,
			newVolumeArray("pvc-uid11-7", "1Gi", "uid11-7", "claim11-7", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned),
			newClaimArray("claim11-7", "uid11-7", "1Gi", "", v1.ClaimPending, &classGold),
			// The claim would be bound in next syncClaim
			newClaimArray("claim11-7", "uid11-7", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			noevents, noerrors,
			wrapTestWithInjectedOperation(wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim), func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
				// Create a volume before provisionClaimOperation starts.
				// This similates a parallel controller provisioning the volume.
				volume := newVolume("pvc-uid11-7", "1Gi", "uid11-7", "claim11-7", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned)
				reactor.AddVolume(volume)
			}),
		},
		{
			// Provision success - cannot save provisioned PV once,
			// second retry succeeds
			"11-8 - cannot save provisioned volume",
			novolumes,
			newVolumeArray("pvc-uid11-8", "1Gi", "uid11-8", "claim11-8", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned),
			newClaimArray("claim11-8", "uid11-8", "1Gi", "", v1.ClaimPending, &classGold),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-8", "uid11-8", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Normal ProvisioningSucceeded"},
			[]pvtesting.ReactorError{
				// Inject error to the first
				// kubeclient.PersistentVolumes.Create() call. All other calls
				// will succeed.
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error")},
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
			newClaimArray("claim11-9", "uid11-9", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Warning ProvisioningFailed"},
			[]pvtesting.ReactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error1")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error2")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error3")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error4")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error5")},
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
			newClaimArray("claim11-10", "uid11-10", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Warning ProvisioningFailed", "Warning ProvisioningCleanupFailed"},
			[]pvtesting.ReactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error1")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error2")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error3")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error4")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error5")},
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
			newClaimArray("claim11-11", "uid11-11", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Warning ProvisioningFailed", "Warning ProvisioningCleanupFailed"},
			[]pvtesting.ReactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error1")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error2")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error3")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error4")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error5")},
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
			newClaimArray("claim11-12", "uid11-12", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			[]string{"Warning ProvisioningFailed"},
			[]pvtesting.ReactorError{
				// Inject error to five kubeclient.PersistentVolumes.Create()
				// calls
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error1")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error2")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error3")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error4")},
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error5")},
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
			newVolumeArray("pvc-uid11-13", "1Gi", "uid11-13", "claim11-13", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classSilver, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned),
			newClaimArray("claim11-13", "uid11-13", "1Gi", "", v1.ClaimPending, &classSilver),
			// Binding will be completed in the next syncClaim
			newClaimArray("claim11-13", "uid11-13", "1Gi", "", v1.ClaimPending, &classSilver, pvutil.AnnStorageProvisioner),
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
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "vendor.com/my-volume",
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
		{
			// Provision success - first save of a PV to API server fails (API
			// server has written the object to etcd, but crashed before sending
			// 200 OK response to the controller). Controller retries and the
			// second save of the PV returns "AlreadyExists" because the PV
			// object already is in the API server.
			//
			"11-19 - provisioned volume saved but API server crashed",
			novolumes,
			// We don't actually simulate API server saving the object and
			// crashing afterwards, Create() just returns error without saving
			// the volume in this test. So the set of expected volumes at the
			// end of the test is empty.
			novolumes,
			newClaimArray("claim11-19", "uid11-19", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim11-19", "uid11-19", "1Gi", "", v1.ClaimPending, &classGold, pvutil.AnnStorageProvisioner),
			noevents,
			[]pvtesting.ReactorError{
				// Inject errors to simulate crashed API server during
				// kubeclient.PersistentVolumes.Create()
				{Verb: "create", Resource: "persistentvolumes", Error: errors.New("Mock creation error1")},
				{Verb: "create", Resource: "persistentvolumes", Error: apierrors.NewAlreadyExists(api.Resource("persistentvolumes"), "")},
			},
			wrapTestWithPluginCalls(
				nil, // recycle calls
				nil, // delete calls - if Delete was called the test would fail
				[]provisionCall{provision1Success},
				testSyncClaim,
			),
		},
		{
			// No provisioning + warning event with unsupported storageClass.mountOptions
			"11-20 - unsupported storageClass.mountOptions",
			novolumes,
			novolumes,
			newClaimArray("claim11-20", "uid11-20", "1Gi", "", v1.ClaimPending, &classUnsupportedMountOptions),
			newClaimArray("claim11-20", "uid11-20", "1Gi", "", v1.ClaimPending, &classUnsupportedMountOptions, pvutil.AnnStorageProvisioner),
			// Expect event to be prefixed with "Mount options" because saving PV will fail anyway
			[]string{"Warning ProvisioningFailed Mount options"},
			noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// No provisioning due to CSI migration + normal event with external provisioner
			"11-21 - external provisioner for CSI migration",
			novolumes,
			novolumes,
			newClaimArray("claim11-21", "uid11-21", "1Gi", "", v1.ClaimPending, &classGold),
			[]*v1.PersistentVolumeClaim{
				annotateClaim(
					newClaim("claim11-21", "uid11-21", "1Gi", "", v1.ClaimPending, &classGold),
					map[string]string{
						pvutil.AnnStorageProvisioner: "vendor.com/MockCSIDriver",
						pvutil.AnnMigratedTo:         "vendor.com/MockCSIDriver",
					}),
			},
			[]string{"Normal ExternalProvisioning"},
			noerrors, wrapTestWithCSIMigrationProvisionCalls(testSyncClaim),
		},
		{
			// volume provisioned and available
			// in this case, NO normal event with external provisioner should be issued
			"11-22 - external provisioner with volume available",
			newVolumeArray("volume11-22", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classExternal),
			newVolumeArray("volume11-22", "1Gi", "uid11-22", "claim11-22", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classExternal, pvutil.AnnBoundByController),
			newClaimArray("claim11-22", "uid11-22", "1Gi", "", v1.ClaimPending, &classExternal),
			newClaimArray("claim11-22", "uid11-22", "1Gi", "volume11-22", v1.ClaimBound, &classExternal, pvutil.AnnBoundByController, pvutil.AnnBindCompleted),
			noevents,
			noerrors, wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim),
		},
		{
			// volume provision for PVC scheduled
			"11-23 - skip finding PV and provision for PVC annotated with AnnSelectedNode",
			newVolumeArray("volume11-23", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classCopper),
			[]*v1.PersistentVolume{
				newVolume("volume11-23", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classCopper),
				newVolume("pvc-uid11-23", "1Gi", "uid11-23", "claim11-23", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classCopper, pvutil.AnnDynamicallyProvisioned, pvutil.AnnBoundByController),
			},
			claimWithAnnotation(pvutil.AnnSelectedNode, "node1",
				newClaimArray("claim11-23", "uid11-23", "1Gi", "", v1.ClaimPending, &classCopper)),
			claimWithAnnotation(pvutil.AnnSelectedNode, "node1",
				newClaimArray("claim11-23", "uid11-23", "1Gi", "", v1.ClaimPending, &classCopper, pvutil.AnnStorageProvisioner)),
			[]string{"Normal ProvisioningSucceeded"},
			noerrors,
			wrapTestWithInjectedOperation(wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
				func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
					nodesIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
					node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}}
					nodesIndexer.Add(node)
					ctrl.NodeLister = corelisters.NewNodeLister(nodesIndexer)
				}),
		},
		{
			// volume provision for PVC that scheduled
			"11-24 - skip finding PV and wait external provisioner for PVC annotated with AnnSelectedNode",
			newVolumeArray("volume11-24", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classExternalWait),
			newVolumeArray("volume11-24", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimDelete, classExternalWait),
			claimWithAnnotation(pvutil.AnnSelectedNode, "node1",
				newClaimArray("claim11-24", "uid11-24", "1Gi", "", v1.ClaimPending, &classExternalWait)),
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "vendor.com/my-volume-wait",
				claimWithAnnotation(pvutil.AnnSelectedNode, "node1",
					newClaimArray("claim11-24", "uid11-24", "1Gi", "", v1.ClaimPending, &classExternalWait))),
			[]string{"Normal ExternalProvisioning"},
			noerrors, testSyncClaim,
		},
	}
	runSyncTests(t, tests, storageClasses, []*v1.Pod{})
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
			newVolumeArray("pvc-uid12-1", "1Gi", "uid12-1", "claim12-1", v1.VolumeBound, v1.PersistentVolumeReclaimDelete, classGold, pvutil.AnnBoundByController, pvutil.AnnDynamicallyProvisioned),
			newClaimArray("claim12-1", "uid12-1", "1Gi", "", v1.ClaimPending, &classGold),
			newClaimArray("claim12-1", "uid12-1", "1Gi", "pvc-uid12-1", v1.ClaimBound, &classGold, pvutil.AnnBoundByController, pvutil.AnnBindCompleted, pvutil.AnnStorageProvisioner),
			noevents, noerrors, wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// provision a volume (external provisioner) and binding + normal event with external provisioner
			"12-2 - external provisioner with volume provisioned success",
			novolumes,
			newVolumeArray("pvc-uid12-2", "1Gi", "uid12-2", "claim12-2", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classExternal, pvutil.AnnBoundByController),
			newClaimArray("claim12-2", "uid12-2", "1Gi", "", v1.ClaimPending, &classExternal),
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "vendor.com/my-volume",
				newClaimArray("claim12-2", "uid12-2", "1Gi", "pvc-uid12-2", v1.ClaimBound, &classExternal, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			[]string{"Normal ExternalProvisioning"},
			noerrors,
			wrapTestWithInjectedOperation(wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim), func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
				// Create a volume before syncClaim tries to bind a PV to PVC
				// This simulates external provisioner creating a volume while the controller
				// is waiting for a volume to bind to the existed claim
				// the external provisioner workflow implemented in "provisionClaimOperationCSI"
				// should issue an ExternalProvisioning event to signal that some external provisioner
				// is working on provisioning the PV, also add the operation start timestamp into local cache
				// operationTimestamps. Rely on the existences of the start time stamp to create a PV for binding
				if ctrl.operationTimestamps.Has("default/claim12-2") {
					volume := newVolume("pvc-uid12-2", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classExternal)
					ctrl.volumes.store.Add(volume) // add the volume to controller
					reactor.AddVolume(volume)
				}
			}),
		},
		{
			// provision a volume (external provisioner) but binding will not happen + normal event with external provisioner
			"12-3 - external provisioner with volume to be provisioned",
			novolumes,
			novolumes,
			newClaimArray("claim12-3", "uid12-3", "1Gi", "", v1.ClaimPending, &classExternal),
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "vendor.com/my-volume",
				newClaimArray("claim12-3", "uid12-3", "1Gi", "", v1.ClaimPending, &classExternal)),
			[]string{"Normal ExternalProvisioning"},
			noerrors,
			wrapTestWithProvisionCalls([]provisionCall{provision1Success}, testSyncClaim),
		},
		{
			// provision a volume (external provisioner) and binding + normal event with external provisioner
			"12-4 - external provisioner with volume provisioned/bound success",
			novolumes,
			newVolumeArray("pvc-uid12-4", "1Gi", "uid12-4", "claim12-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classExternal, pvutil.AnnBoundByController),
			newClaimArray("claim12-4", "uid12-4", "1Gi", "", v1.ClaimPending, &classExternal),
			claimWithAnnotation(pvutil.AnnStorageProvisioner, "vendor.com/my-volume",
				newClaimArray("claim12-4", "uid12-4", "1Gi", "pvc-uid12-4", v1.ClaimBound, &classExternal, pvutil.AnnBoundByController, pvutil.AnnBindCompleted)),
			[]string{"Normal ExternalProvisioning"},
			noerrors,
			wrapTestWithInjectedOperation(wrapTestWithProvisionCalls([]provisionCall{}, testSyncClaim), func(ctrl *PersistentVolumeController, reactor *pvtesting.VolumeReactor) {
				// Create a volume before syncClaim tries to bind a PV to PVC
				// This simulates external provisioner creating a volume while the controller
				// is waiting for a volume to bind to the existed claim
				// the external provisioner workflow implemented in "provisionClaimOperationCSI"
				// should issue an ExternalProvisioning event to signal that some external provisioner
				// is working on provisioning the PV, also add the operation start timestamp into local cache
				// operationTimestamps. Rely on the existences of the start time stamp to create a PV for binding
				if ctrl.operationTimestamps.Has("default/claim12-4") {
					volume := newVolume("pvc-uid12-4", "1Gi", "uid12-4", "claim12-4", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classExternal, pvutil.AnnBoundByController)
					ctrl.volumes.store.Add(volume) // add the volume to controller
					reactor.AddVolume(volume)
				}
			}),
		},
	}

	runMultisyncTests(t, tests, storageClasses, storageClasses[0].Name)
}

// When provisioning is disabled, provisioning a claim should instantly return nil
func TestDisablingDynamicProvisioner(t *testing.T) {
	ctrl, err := newTestController(nil, nil, false)
	if err != nil {
		t.Fatalf("Construct PersistentVolume controller failed: %v", err)
	}
	retVal := ctrl.provisionClaim(nil)
	if retVal != nil {
		t.Errorf("Expected nil return but got %v", retVal)
	}
}
