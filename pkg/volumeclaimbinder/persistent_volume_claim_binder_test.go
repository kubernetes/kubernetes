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

package volumeclaimbinder

import (
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/host_path"
)

func TestRunStop(t *testing.T) {
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, api.RESTMapper)}
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
		if err := testclient.AddObjectsFromPath("../../docs/user-guide/persistent-volumes/"+name, o, api.Scheme); err != nil {
			t.Fatal(err)
		}

		client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, api.RESTMapper)}

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
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	if err := testclient.AddObjectsFromPath("../../docs/user-guide/persistent-volumes/claims/claim-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}
	if err := testclient.AddObjectsFromPath("../../docs/user-guide/persistent-volumes/volumes/local-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}

	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, api.RESTMapper)}

	pv, err := client.PersistentVolumes().Get("any")
	pv.Spec.PersistentVolumeReclaimPolicy = api.PersistentVolumeReclaimRecycle
	if err != nil {
		t.Error("Unexpected error getting PV from client: %v", err)
	}

	claim, error := client.PersistentVolumeClaims("ns").Get("any")
	if error != nil {
		t.Errorf("Unexpected error getting PVC from client: %v", err)
	}

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{
		volume: pv,
		claim:  claim,
	}

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(host_path.ProbeRecyclableVolumePlugins(newMockRecycler), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	recycler := &PersistentVolumeRecycler{
		kubeClient: client,
		client:     mockClient,
		pluginMgr:  plugMgr,
	}

	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, pv)
	if pv.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeBound, pv.Status.Phase)
	}

	// an initial sync for a claim will bind it to an unbound volume, triggers state change
	syncClaim(volumeIndex, mockClient, claim)
	// state change causes another syncClaim to update statuses
	syncClaim(volumeIndex, mockClient, claim)
	// claim updated volume's status, causing an update and syncVolume call
	syncVolume(volumeIndex, mockClient, pv)

	if pv.Spec.ClaimRef == nil {
		t.Errorf("Expected ClaimRef but got nil for pv.Status.ClaimRef: %+v\n", pv)
	}

	if pv.Status.Phase != api.VolumeBound {
		t.Errorf("Expected phase %s but got %s", api.VolumeBound, pv.Status.Phase)
	}

	if claim.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if len(claim.Status.AccessModes) != len(pv.Spec.AccessModes) {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if claim.Status.AccessModes[0] != pv.Spec.AccessModes[0] {
		t.Errorf("Expected access mode %s but got %s", claim.Status.AccessModes[0], pv.Spec.AccessModes[0])
	}

	// pretend the user deleted their claim
	mockClient.claim = nil
	syncVolume(volumeIndex, mockClient, pv)

	if pv.Status.Phase != api.VolumeReleased {
		t.Errorf("Expected phase %s but got %s", api.VolumeReleased, pv.Status.Phase)
	}
	if pv.Spec.ClaimRef == nil {
		t.Errorf("Expected non-nil ClaimRef: %+v", pv.Spec)
	}

	mockClient.volume = pv

	// released volumes with a PersistentVolumeReclaimPolicy (recycle/delete) can have further processing
	err = recycler.reclaimVolume(pv)
	if err != nil {
		t.Errorf("Unexpected error reclaiming volume: %+v", err)
	}
	if pv.Status.Phase != api.VolumePending {
		t.Errorf("Expected phase %s but got %s", api.VolumePending, pv.Status.Phase)
	}

	// after the recycling changes the phase to Pending, the binder picks up again
	// to remove any vestiges of binding and make the volume Available again
	syncVolume(volumeIndex, mockClient, pv)

	if pv.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeAvailable, pv.Status.Phase)
	}
	if pv.Spec.ClaimRef != nil {
		t.Errorf("Expected nil ClaimRef: %+v", pv.Spec)
	}
}

func TestMissingFromIndex(t *testing.T) {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	if err := testclient.AddObjectsFromPath("../../docs/user-guide/persistent-volumes/claims/claim-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}
	if err := testclient.AddObjectsFromPath("../../docs/user-guide/persistent-volumes/volumes/local-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}

	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, api.RESTMapper)}

	pv, err := client.PersistentVolumes().Get("any")
	if err != nil {
		t.Error("Unexpected error getting PV from client: %v", err)
	}

	claim, error := client.PersistentVolumeClaims("ns").Get("any")
	if error != nil {
		t.Errorf("Unexpected error getting PVC from client: %v", err)
	}

	volumeIndex := NewPersistentVolumeOrderedIndex()
	mockClient := &mockBinderClient{
		volume: pv,
		claim:  claim,
	}

	// the default value of the PV is Pending.
	// if has previously been processed by the binder, it's status in etcd would be Available.
	// Only Pending volumes were being indexed and made ready for claims.
	pv.Status.Phase = api.VolumeAvailable

	// adds the volume to the index, making the volume available
	syncVolume(volumeIndex, mockClient, pv)
	if pv.Status.Phase != api.VolumeAvailable {
		t.Errorf("Expected phase %s but got %s", api.VolumeBound, pv.Status.Phase)
	}

	// an initial sync for a claim will bind it to an unbound volume, triggers state change
	err = syncClaim(volumeIndex, mockClient, claim)
	if err != nil {
		t.Fatalf("Expected Clam to be bound, instead got an error: %+v\n", err)
	}

	// state change causes another syncClaim to update statuses
	syncClaim(volumeIndex, mockClient, claim)
	// claim updated volume's status, causing an update and syncVolume call
	syncVolume(volumeIndex, mockClient, pv)

	if pv.Spec.ClaimRef == nil {
		t.Errorf("Expected ClaimRef but got nil for pv.Status.ClaimRef: %+v\n", pv)
	}

	if pv.Status.Phase != api.VolumeBound {
		t.Errorf("Expected phase %s but got %s", api.VolumeBound, pv.Status.Phase)
	}

	if claim.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if len(claim.Status.AccessModes) != len(pv.Spec.AccessModes) {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if claim.Status.AccessModes[0] != pv.Spec.AccessModes[0] {
		t.Errorf("Expected access mode %s but got %s", claim.Status.AccessModes[0], pv.Spec.AccessModes[0])
	}

	// pretend the user deleted their claim
	mockClient.claim = nil
	syncVolume(volumeIndex, mockClient, pv)

	if pv.Status.Phase != api.VolumeReleased {
		t.Errorf("Expected phase %s but got %s", api.VolumeReleased, pv.Status.Phase)
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
	return volume, nil
}

func (c *mockBinderClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	c.volume = nil
	return nil
}

func (c *mockBinderClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return volume, nil
}

func (c *mockBinderClient) GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error) {
	if c.claim != nil {
		return c.claim, nil
	} else {
		return nil, errors.NewNotFound("persistentVolume", name)
	}
}

func (c *mockBinderClient) UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return claim, nil
}

func (c *mockBinderClient) UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return claim, nil
}

func newMockRecycler(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error) {
	return &mockRecycler{
		path: spec.PersistentVolumeSource.HostPath.Path,
	}, nil
}

type mockRecycler struct {
	path string
	host volume.VolumeHost
}

func (r *mockRecycler) GetPath() string {
	return r.path
}

func (r *mockRecycler) Recycle() error {
	// return nil means recycle passed
	return nil
}
