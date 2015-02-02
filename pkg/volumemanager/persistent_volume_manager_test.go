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

package volumemanager

import (
	"io/ioutil"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakeWatcher struct {
	w *watch.FakeWatcher
	*client.Fake
}

func fakeClient() *client.Fake {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	fakeWatch := watch.NewFake()
	fake := &client.Fake{Watch: fakeWatch}
	return fake
}

func TestVolumeController(t *testing.T) {

	controller := NewPersistentVolumeManager(fakeClient())

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			UID: "gce-pd-5",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceSize): resource.MustParse("5G"),
			},
			Source: api.VolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
			},
		},
	}

	claimA := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.AccessModeType{ api.ReadOnlyMany, api.ReadWriteOnce },
			Resources: api.ResourceList{
				api.ResourceName(api.ResourceSize): resource.MustParse("5G"),
			},
		},
	}

	claimB := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim02",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.AccessModeType{ api.ReadOnlyMany, api.ReadWriteOnce },
			Resources: api.ResourceList{
				api.ResourceName(api.ResourceSize): resource.MustParse("5G"),
			},
		},
	}

	_, err := controller.syncPersistentVolume(pv)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	if !controller.volumeIndex.Exists(pv){
		t.Error("Expected to find volume in the index")
	}

	obj, err := controller.syncPersistentVolumeClaim(claimA)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	retClaimA := obj.(*api.PersistentVolumeClaim)
	if retClaimA.Status.VolumeRef == nil {
		t.Error("Expected claim to be bound to volume")
	}

	//claimA now owns the volume
	obj, err = controller.syncPersistentVolumeClaim(claimB)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	retClaimB := obj.(*api.PersistentVolumeClaim)
	if retClaimB.Status.VolumeRef != nil {
		t.Error("Unexpected claim found.")
	}
}


func TestVolumeExamples(t *testing.T){

	controller := NewPersistentVolumeManager(fakeClient())

	volumeA := readAndDecodeVolume("local-01.yaml", t)
	claimA := readAndDecodeClaim("claim-01.yaml", t)


	_, err := controller.syncPersistentVolume(volumeA)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	if !controller.volumeIndex.Exists(volumeA){
		t.Error("Expected to find volume in the index")
	}

	obj, err := controller.syncPersistentVolumeClaim(claimA)
	if err != nil {
		t.Error("Unexpected error: %v", err)
	}

	retClaimA := obj.(*api.PersistentVolumeClaim)
	if retClaimA.Status.VolumeRef == nil {
		t.Error("Expected claim to be bound to volume")
	}
}


func readAndDecodeVolume(name string, t *testing.T) *api.PersistentVolume {
	data, err := ioutil.ReadFile("../../examples/storage/volumes/" + name)
	if err != nil {
		t.Error("Unexpected error attempting to read example volume file: %s", name)
	}

	volume := &api.PersistentVolume{}
	if err := latest.Codec.DecodeInto([]byte(data), volume); err != nil {
		t.Errorf("Error decoding volume: %v", err)
	}

	return volume
}

func readAndDecodeClaim(name string, t *testing.T) *api.PersistentVolumeClaim {
	data, err := ioutil.ReadFile("../../examples/storage/claims/" + name)
	if err != nil {
		t.Error("Unexpected error attempting to read example volume file: %s", name)
	}

	claim := &api.PersistentVolumeClaim{}
	if err := latest.Codec.DecodeInto([]byte(data), claim); err != nil {
		t.Errorf("Error decoding volume: %v", err)
	}

	return claim
}
