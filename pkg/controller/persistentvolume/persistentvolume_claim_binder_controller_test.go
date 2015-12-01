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
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/host_path"
)

func TestRunStop(t *testing.T) {
	client := &testclient.Fake{}
	binder := NewPersistentVolumeClaimBinder(client, 1*time.Second)

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
					Path: "/tmp/data01",
				},
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumePending,
		},
	}

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, v)
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}
	if _, exists, _ := volumeIndex.Get(v); !exists {
		t.Errorf("Expected to find volume in index but it did not exist")
	}

	// an initial sync for a claim matches the volume
	err := syncClaim(volumeIndex, mockClient, c1)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if c1.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, c1.Status.Phase)
	}

	// before the volume gets updated w/ claimRef, a 2nd claim can attempt to bind and find the same volume
	err = syncClaim(volumeIndex, mockClient, c2)
	if err != nil {
		t.Errorf("unexpected error for unmatched claim: %v", err)
	}
	if c2.Status.Phase != api.ClaimPending {
		t.Errorf("Expected phase %s but got %s", api.ClaimPending, c2.Status.Phase)
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
							Path: "/tmp/data01",
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
							Path: "/tmp/data02",
						},
					},
					PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
				},
			},
		},
	}

	for name, scenario := range scenarios {
		o := testclient.NewObjects(api.Scheme, api.Scheme)
		if err := testclient.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/"+name, o, api.Scheme); err != nil {
			t.Fatal(err)
		}

		client := &testclient.Fake{}
		client.AddReactor("*", "*", testclient.ObjectReaction(o, api.RESTMapper))

		if reflect.TypeOf(scenario.expected) == reflect.TypeOf(&api.PersistentVolumeClaim{}) {
			pvc, err := client.PersistentVolumeClaims("ns").Get("doesntmatter")
			if err != nil {
				t.Errorf("Error retrieving object: %v", err)
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
			pv, err := client.PersistentVolumes().Get("doesntmatter")
			if err != nil {
				t.Errorf("Error retrieving object: %v", err)
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
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	if err := testclient.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/claims/claim-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}
	if err := testclient.AddObjectsFromPath("../../../docs/user-guide/persistent-volumes/volumes/local-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}

	client := &testclient.Fake{}
	client.AddReactor("*", "*", testclient.ObjectReaction(o, api.RESTMapper))

	pv, err := client.PersistentVolumes().Get("any")
	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimRecycle
	if err != nil {
		t.Errorf("Unexpected error getting PV from client: %v", err)
	}
	pv.ObjectMeta.SelfLink = testapi.Default.SelfLink("pv", "")

	// the default value of the PV is Pending. if processed at least once, its status in etcd is Available.
	// There was a bug where only Pending volumes were being indexed and made ready for claims.
	// Test that !Pending gets correctly added
	pv.Status.Phase = api.VolumeAvailable

	claim, error := client.PersistentVolumeClaims("ns").Get("any")
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
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler, volume.VolumeConfig{}), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	recycler := &PersistentVolumeRecycler{
		kubeClient: client,
		client:     mockClient,
		pluginMgr:  plugMgr,
	}

	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, pv)
	if mockClient.volume.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, mockClient.volume.Status.Phase)
	}

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
	client := &testclient.Fake{}
	binder := NewPersistentVolumeClaimBinder(client, 1*time.Second)

	pv := &api.PersistentVolume{}
	unk := cache.DeletedFinalStateUnknown{}
	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.PersistentVolumeClaimStatus{Phase: api.ClaimBound},
	}

	// none of these should fail casting.
	// the real test is not failing when passed DeletedFinalStateUnknown in the deleteHandler
	binder.addVolume(pv)
	binder.updateVolume(pv, pv)
	binder.deleteVolume(pv)
	binder.deleteVolume(unk)
	binder.addClaim(pvc)
	binder.updateClaim(pvc, pvc)
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
		return nil, errors.NewNotFound("persistentVolume", name)
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
