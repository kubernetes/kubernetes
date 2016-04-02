/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"os"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/types"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/host_path"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestRunStop(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	binder := NewPersistentVolumeClaimBinder(clientset, 1*time.Second)

	if len(binder.stopChannels) != 0 {
		t.Errorf("Non-running binder should not have any stopChannels.  Got %v", len(binder.stopChannels))
	}

	binder.Run()

	if len(binder.stopChannels) != 2 {
		t.Errorf("Running binder should have exactly 2 stopChannels.  Got %v", len(binder.stopChannels))
	}

	binder.Stop()

	if len(binder.stopChannels) != 0 {
		t.Errorf("Non-running binder should not have any stopChannels.  Got %v", len(binder.stopChannels))
	}
}

func TestClaimRace(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("claimbinder-test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	c1 := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: "c1",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimPending,
		},
	}
	c1.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")

	c2 := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: "c2",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimPending,
		},
	}
	c2.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")

	v := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fmt.Sprintf("%s/data01", tmpDir),
				},
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumePending,
		},
	}

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{}
	mockClient.volume = v

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, v)
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}
	if _, exists, _ := volumeIndex.Get(v); !exists {
		t.Errorf("Expected to find volume in index but it did not exist")
	}

	// add the claim to fake API server
	mockClient.UpdatePersistentVolumeClaim(c1)
	// an initial sync for a claim matches the volume
	err = syncClaim(volumeIndex, mockClient, c1)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if c1.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, c1.Status.Phase)
	}

	// before the volume gets updated w/ claimRef, a 2nd claim can attempt to bind and find the same volume
	// add the 2nd claim to fake API server
	mockClient.UpdatePersistentVolumeClaim(c2)
	err = syncClaim(volumeIndex, mockClient, c2)
	if err != nil {
		t.Errorf("unexpected error for unmatched claim: %v", err)
	}
	if c2.Status.Phase != api.ClaimPending {
		t.Errorf("Expected phase %s but got %s", api.ClaimPending, c2.Status.Phase)
	}
}

func TestNewClaimWithSameNameAsOldClaim(t *testing.T) {
	c1 := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "c1",
			Namespace: "foo",
			UID:       "12345",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}
	c1.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")

	v := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{
				Name:      c1.Name,
				Namespace: c1.Namespace,
				UID:       "45678",
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/tmp/data01",
				},
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumeBound,
		},
	}

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{
		claim:  c1,
		volume: v,
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil))

	syncVolume(volumeIndex, mockClient, v)
	if mockClient.volume.Status.Phase != api.VolumeReleased {
		t.Errorf("Expected phase %s but got %s", api.VolumeReleased, mockClient.volume.Status.Phase)
	}

}

func TestClaimSyncAfterVolumeProvisioning(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("claimbinder-test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Tests that binder.syncVolume will also syncClaim if the PV has completed
	// provisioning but the claim is still Pending.  We want to advance to Bound
	// without having to wait until the binder's next sync period.
	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "bar",
			Annotations: map[string]string{
				qosProvisioningKey: "foo",
			},
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimPending,
		},
	}
	claim.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")
	claimRef, _ := api.GetReference(claim)

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Annotations: map[string]string{
				pvProvisioningRequiredAnnotationKey: pvProvisioningCompletedAnnotationValue,
			},
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fmt.Sprintf("%s/data01", tmpDir),
				},
			},
			ClaimRef: claimRef,
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumePending,
		},
	}

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{
		claim:  claim,
		volume: pv,
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	// adds the volume to the index, making the volume available.
	// pv also completed provisioning, so syncClaim should cause claim's phase to advance to Bound
	syncVolume(volumeIndex, mockClient, pv)
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}
	if mockClient.claim.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
}

func TestExampleObjects(t *testing.T) {
	scenarios := map[string]struct {
		expected interface{}
	}{
		"claims/claim-01.yaml": {
			expected: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
						},
					},
				},
			},
		},
		"claims/claim-02.yaml": {
			expected: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("8Gi"),
						},
					},
				},
			},
		},
		"volumes/local-01.yaml": {
			expected: &api.PersistentVolume{
				Spec: api.PersistentVolumeSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("10Gi"),
					},
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/somepath/data01",
						},
					},
				},
			},
		},
		"volumes/local-02.yaml": {
			expected: &api.PersistentVolume{
				Spec: api.PersistentVolumeSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Capacity: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("8Gi"),
					},
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/somepath/data02",
						},
					},
					PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
				},
			},
		},
	}

	for name, scenario := range scenarios {
		codec := api.Codecs.UniversalDecoder()
		o := core.NewObjects(api.Scheme, codec)
		if err := core.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/"+name, o, codec); err != nil {
			t.Fatal(err)
		}

		clientset := &fake.Clientset{}
		clientset.AddReactor("*", "*", core.ObjectReaction(o, registered.RESTMapper()))

		if reflect.TypeOf(scenario.expected) == reflect.TypeOf(&api.PersistentVolumeClaim{}) {
			pvc, err := clientset.Core().PersistentVolumeClaims("ns").Get("doesntmatter")
			if err != nil {
				t.Fatalf("Error retrieving object: %v", err)
			}

			expected := scenario.expected.(*api.PersistentVolumeClaim)
			if pvc.Spec.AccessModes[0] != expected.Spec.AccessModes[0] {
				t.Errorf("Unexpected mismatch.  Got %v wanted %v", pvc.Spec.AccessModes[0], expected.Spec.AccessModes[0])
			}

			aQty := pvc.Spec.Resources.Requests[api.ResourceStorage]
			bQty := expected.Spec.Resources.Requests[api.ResourceStorage]
			aSize := aQty.Value()
			bSize := bQty.Value()

			if aSize != bSize {
				t.Errorf("Unexpected mismatch.  Got %v wanted %v", aSize, bSize)
			}
		}

		if reflect.TypeOf(scenario.expected) == reflect.TypeOf(&api.PersistentVolume{}) {
			pv, err := clientset.Core().PersistentVolumes().Get("doesntmatter")
			if err != nil {
				t.Fatalf("Error retrieving object: %v", err)
			}

			expected := scenario.expected.(*api.PersistentVolume)
			if pv.Spec.AccessModes[0] != expected.Spec.AccessModes[0] {
				t.Errorf("Unexpected mismatch.  Got %v wanted %v", pv.Spec.AccessModes[0], expected.Spec.AccessModes[0])
			}

			aQty := pv.Spec.Capacity[api.ResourceStorage]
			bQty := expected.Spec.Capacity[api.ResourceStorage]
			aSize := aQty.Value()
			bSize := bQty.Value()

			if aSize != bSize {
				t.Errorf("Unexpected mismatch.  Got %v wanted %v", aSize, bSize)
			}

			if pv.Spec.HostPath.Path != expected.Spec.HostPath.Path {
				t.Errorf("Unexpected mismatch.  Got %v wanted %v", pv.Spec.HostPath.Path, expected.Spec.HostPath.Path)
			}
		}
	}
}

func TestBindingWithExamples(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("claimbinder-test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	codec := api.Codecs.UniversalDecoder()
	o := core.NewObjects(api.Scheme, codec)
	if err := core.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/claims/claim-01.yaml", o, codec); err != nil {
		t.Fatal(err)
	}
	if err := core.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/volumes/local-01.yaml", o, codec); err != nil {
		t.Fatal(err)
	}

	clientset := &fake.Clientset{}
	clientset.AddReactor("*", "*", core.ObjectReaction(o, registered.RESTMapper()))

	pv, err := clientset.Core().PersistentVolumes().Get("any")
	if err != nil {
		t.Errorf("Unexpected error getting PV from client: %v", err)
	}
	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimRecycle
	if err != nil {
		t.Errorf("Unexpected error getting PV from client: %v", err)
	}
	pv.ObjectMeta.SelfLink = testapi.Default.SelfLink("pv", "")

	// the default value of the PV is Pending. if processed at least once, its status in etcd is Available.
	// There was a bug where only Pending volumes were being indexed and made ready for claims.
	// Test that !Pending gets correctly added
	pv.Status.Phase = api.VolumeAvailable

	claim, error := clientset.Core().PersistentVolumeClaims("ns").Get("any")
	if error != nil {
		t.Errorf("Unexpected error getting PVC from client: %v", err)
	}
	claim.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{
		volume: pv,
		claim:  claim,
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	recycler := &PersistentVolumeRecycler{
		kubeClient: clientset,
		client:     mockClient,
		pluginMgr:  plugMgr,
	}

	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, pv)
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}

	// add the claim to fake API server
	mockClient.UpdatePersistentVolumeClaim(claim)
	// an initial sync for a claim will bind it to an unbound volume
	syncClaim(volumeIndex, mockClient, claim)

	// bind expected on pv.Spec but status update hasn't happened yet
	if mockClient.volume.Spec.ClaimRef == nil {
		t.Errorf("Expected ClaimRef but got nil for pv.Status.ClaimRef\n")
	}
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}
	if mockClient.claim.Spec.VolumeName != pv.Name {
		t.Errorf("Expected claim.Spec.VolumeName %s but got %s", mockClient.claim.Spec.VolumeName, pv.Name)
	}
	if mockClient.claim.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}

	// state changes in pvc triggers sync that sets pv attributes to pvc.Status
	syncClaim(volumeIndex, mockClient, claim)
	if len(mockClient.claim.Status.AccessModes) == 0 {
		t.Errorf("Expected %d access modes but got 0", len(pv.Spec.AccessModes))
	}

	// persisting the bind to pv.Spec.ClaimRef triggers a sync
	syncVolume(volumeIndex, mockClient, mockClient.volume)
	if mockClient.volume.Status.Phase != api.VolumeBound {
		t.Errorf("Expected phase %s but got %s", api.VolumeBound, mockClient.volume.Status.Phase)
	}

	// pretend the user deleted their claim. periodic resync picks it up.
	mockClient.claim = nil
	syncVolume(volumeIndex, mockClient, mockClient.volume)

	if mockClient.volume.Status.Phase != api.VolumeReleased {
		t.Errorf("Expected phase %s but got %s", api.VolumeReleased, mockClient.volume.Status.Phase)
	}

	// released volumes with a PersistentVolumeReclaimPolicy (recycle/delete) can have further processing
	err = recycler.reclaimVolume(mockClient.volume)
	if err != nil {
		t.Errorf("Unexpected error reclaiming volume: %+v", err)
	}
	if mockClient.volume.Status.Phase != api.VolumePending {
		t.Errorf("Expected phase %s but got %s", api.VolumePending, mockClient.volume.Status.Phase)
	}

	// after the recycling changes the phase to Pending, the binder picks up again
	// to remove any vestiges of binding and make the volume Available again
	syncVolume(volumeIndex, mockClient, mockClient.volume)

	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}
	if mockClient.volume.Spec.ClaimRef != nil {
		t.Errorf("Expected nil ClaimRef: %+v", mockClient.volume.Spec.ClaimRef)
	}
}

func TestCasting(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	binder := NewPersistentVolumeClaimBinder(clientset, 1*time.Second)

	pv := &api.PersistentVolume{}
	unk := cache.DeletedFinalStateUnknown{}
	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.PersistentVolumeClaimStatus{Phase: api.ClaimBound},
	}

	// Inject mockClient into the binder. This prevents weird errors on stderr
	// as the binder wants to load PV/PVC from API server.
	mockClient := &mockBinderClient{
		volume: pv,
		claim:  pvc,
	}
	binder.client = mockClient

	// none of these should fail casting.
	// the real test is not failing when passed DeletedFinalStateUnknown in the deleteHandler
	binder.addVolume(pv)
	binder.updateVolume(pv, pv)
	binder.deleteVolume(pv)
	binder.deleteVolume(unk)
	binder.addClaim(pvc)
	binder.updateClaim(pvc, pvc)
}

func TestRecycledPersistentVolumeUID(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("claimbinder-test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	codec := api.Codecs.UniversalDecoder()
	o := core.NewObjects(api.Scheme, codec)
	if err := core.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/claims/claim-01.yaml", o, codec); err != nil {
		t.Fatal(err)
	}
	if err := core.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/volumes/local-01.yaml", o, codec); err != nil {
		t.Fatal(err)
	}

	clientset := &fake.Clientset{}
	clientset.AddReactor("*", "*", core.ObjectReaction(o, registered.RESTMapper()))

	pv, err := clientset.Core().PersistentVolumes().Get("any")
	if err != nil {
		t.Errorf("Unexpected error getting PV from client: %v", err)
	}
	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimRecycle
	if err != nil {
		t.Errorf("Unexpected error getting PV from client: %v", err)
	}
	pv.ObjectMeta.SelfLink = testapi.Default.SelfLink("pv", "")

	// the default value of the PV is Pending. if processed at least once, its status in etcd is Available.
	// There was a bug where only Pending volumes were being indexed and made ready for claims.
	// Test that !Pending gets correctly added
	pv.Status.Phase = api.VolumeAvailable

	claim, error := clientset.Core().PersistentVolumeClaims("ns").Get("any")
	if error != nil {
		t.Errorf("Unexpected error getting PVC from client: %v", err)
	}
	claim.ObjectMeta.SelfLink = testapi.Default.SelfLink("pvc", "")
	claim.ObjectMeta.UID = types.UID("uid1")

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{
		volume: pv,
		claim:  claim,
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	recycler := &PersistentVolumeRecycler{
		kubeClient: clientset,
		client:     mockClient,
		pluginMgr:  plugMgr,
	}

	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, pv)
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}

	// add the claim to fake API server
	mockClient.UpdatePersistentVolumeClaim(claim)
	// an initial sync for a claim will bind it to an unbound volume
	syncClaim(volumeIndex, mockClient, claim)

	// pretend the user deleted their claim. periodic resync picks it up.
	mockClient.claim = nil
	syncVolume(volumeIndex, mockClient, mockClient.volume)

	if mockClient.volume.Status.Phase != api.VolumeReleased {
		t.Errorf("Expected phase %s but got %s", api.VolumeReleased, mockClient.volume.Status.Phase)
	}

	// released volumes with a PersistentVolumeReclaimPolicy (recycle/delete) can have further processing
	err = recycler.reclaimVolume(mockClient.volume)
	if err != nil {
		t.Errorf("Unexpected error reclaiming volume: %+v", err)
	}
	if mockClient.volume.Status.Phase != api.VolumePending {
		t.Errorf("Expected phase %s but got %s", api.VolumePending, mockClient.volume.Status.Phase)
	}

	// after the recycling changes the phase to Pending, the binder picks up again
	// to remove any vestiges of binding and make the volume Available again
	//
	// explicitly set the claim's UID to a different value to ensure that a new claim with the same
	// name as what the PV was previously bound still yields an available volume
	claim.ObjectMeta.UID = types.UID("uid2")
	mockClient.claim = claim
	syncVolume(volumeIndex, mockClient, mockClient.volume)

	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}
	if mockClient.volume.Spec.ClaimRef != nil {
		t.Errorf("Expected nil ClaimRef: %+v", mockClient.volume.Spec.ClaimRef)
	}
}

type mockBinderClient struct {
	volume *api.PersistentVolume
	claim  *api.PersistentVolumeClaim
}

func (c *mockBinderClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.volume, nil
}

func (c *mockBinderClient) UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	c.volume = volume
	return c.volume, nil
}

func (c *mockBinderClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	c.volume = nil
	return nil
}

func (c *mockBinderClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	c.volume = volume
	return c.volume, nil
}

func (c *mockBinderClient) GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error) {
	if c.claim != nil {
		return c.claim, nil
	} else {
		return nil, errors.NewNotFound(api.Resource("persistentvolumes"), name)
	}
}

func (c *mockBinderClient) UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	c.claim = claim
	return c.claim, nil
}

func (c *mockBinderClient) UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	c.claim = claim
	return c.claim, nil
}

func newMockRecycler(spec *volume.Spec, host volume.VolumeHost, config volume.VolumeConfig) (volume.Recycler, error) {
	return &mockRecycler{
		path: spec.PersistentVolume.Spec.HostPath.Path,
	}, nil
}

type mockRecycler struct {
	path string
	host volume.VolumeHost
	volume.MetricsNil
}

func (r *mockRecycler) GetPath() string {
	return r.path
}

func (r *mockRecycler) Recycle() error {
	// return nil means recycle passed
	return nil
}
