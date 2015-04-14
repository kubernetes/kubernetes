/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

func TestExampleObjects(t *testing.T) {

	scenarios := map[string]struct {
		expected interface{}
	}{
		"claims/claim-01.yaml": {
			expected: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.AccessModeType{api.ReadWriteOnce},
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
					AccessModes: []api.AccessModeType{api.ReadWriteOnce},
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
					AccessModes: []api.AccessModeType{api.ReadWriteOnce},
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
					AccessModes: []api.AccessModeType{api.ReadWriteOnce},
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
		o := testclient.NewObjects(api.Scheme)
		if err := testclient.AddObjectsFromPath("../../examples/persistent-volumes/"+name, o); err != nil {
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
	o := testclient.NewObjects(api.Scheme)
	if err := testclient.AddObjectsFromPath("../../examples/persistent-volumes/claims/claim-01.yaml", o); err != nil {
		t.Fatal(err)
	}
	if err := testclient.AddObjectsFromPath("../../examples/persistent-volumes/volumes/local-01.yaml", o); err != nil {
		t.Fatal(err)
	}

	client := &testclient.Fake{ReactFn: testclient.ObjectReaction(o, latest.RESTMapper)}

	pv, err := client.PersistentVolumes().Get("any")
	if err != nil {
		t.Error("Unexpected error getting PV from client: %v", err)
	}

	claim, error := client.PersistentVolumeClaims("ns").Get("any")
	if error != nil {
		t.Error("Unexpected error getting PVC from client: %v", err)
	}

	controller := NewPersistentVolumeClaimBinder(client)
	err = controller.volumeStore.Add(pv)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	if _, exists, _ := controller.volumeStore.Get(pv); !exists {
		t.Error("Expected to find volume in the index")
	}

	err = controller.syncPersistentVolumeClaim(claim)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	if claim.Status.VolumeRef == nil {
		t.Error("Expected claim to be bound to volume")
	}

	if claim.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if len(claim.Status.AccessModes) != len(pv.Spec.AccessModes) {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if claim.Status.AccessModes[0] != pv.Spec.AccessModes[0] {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
	if claim.Status.Phase != api.ClaimBound {
		t.Errorf("Expected phase %s but got %s", api.ClaimBound, claim.Status.Phase)
	}
}
