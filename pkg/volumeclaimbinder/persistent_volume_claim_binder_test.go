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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

func TestRunStop(t *testing.T) {
	o := testclient.NewObjects(api.Scheme, api.Scheme)
	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, latest.RESTMapper)}
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
						api.ResourceName(api.ResourceStorage): resource.MustParse("5Gi"),
					},
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/tmp/data02",
						},
					},
				},
			},
		},
	}

	for name, scenario := range scenarios {
		o := testclient.NewObjects(api.Scheme, api.Scheme)
		if err := testclient.AddObjectsFromPath("../../examples/persistent-volumes/"+name, o, api.Scheme); err != nil {
			t.Fatal(err)
		}

		client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, latest.RESTMapper)}

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
	if err := testclient.AddObjectsFromPath("../../examples/persistent-volumes/claims/claim-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}
	if err := testclient.AddObjectsFromPath("../../examples/persistent-volumes/volumes/local-01.yaml", o, api.Scheme); err != nil {
		t.Fatal(err)
	}

	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, latest.RESTMapper)}

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
