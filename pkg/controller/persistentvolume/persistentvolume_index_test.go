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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func TestMatchVolume(t *testing.T) {
	volList := NewPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		volList.Add(pv)
	}

	scenarios := map[string]struct {
		expectedMatch string
		claim         *api.PersistentVolumeClaim
	}{
		"successful-match-gce-10": {
			expectedMatch: "gce-pd-10",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("8G"),
						},
					},
				},
			},
		},
		"successful-match-nfs-5": {
			expectedMatch: "nfs-5",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce, api.ReadWriteMany},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
						},
					},
				},
			},
		},
		"successful-skip-1g-bound-volume": {
			expectedMatch: "gce-pd-5",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
						},
					},
				},
			},
		},
		"successful-no-match": {
			expectedMatch: "",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("999G"),
						},
					},
				},
			},
		},
	}

	for name, scenario := range scenarios {
		volume, err := volList.FindBestMatchForClaim(scenario.claim)
		if err != nil {
			t.Errorf("Unexpected error matching volume by claim: %v", err)
		}
		if scenario.expectedMatch != "" && volume == nil {
			t.Errorf("Expected match but received nil volume for scenario: %s", name)
		}
		if scenario.expectedMatch != "" && volume != nil && string(volume.UID) != scenario.expectedMatch {
			t.Errorf("Expected %s but got volume %s in scenario %s", scenario.expectedMatch, volume.UID, name)
		}
		if scenario.expectedMatch == "" && volume != nil {
			t.Errorf("Unexpected match for scenario: %s", name)
		}
	}
}

func TestMatchingWithBoundVolumes(t *testing.T) {
	volumeIndex := NewPersistentVolumeOrderedIndex()
	// two similar volumes, one is bound
	pv1 := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			UID:  "gce-pd-1",
			Name: "gce001",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany},
			// this one we're pretending is already bound
			ClaimRef: &api.ObjectReference{UID: "abc123"},
		},
	}

	pv2 := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			UID:  "gce-pd-2",
			Name: "gce002",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany},
		},
	}

	volumeIndex.Add(pv1)
	volumeIndex.Add(pv2)

	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
				},
			},
		},
	}

	volume, err := volumeIndex.FindBestMatchForClaim(claim)
	if err != nil {
		t.Fatalf("Unexpected error matching volume by claim: %v", err)
	}
	if volume == nil {
		t.Fatalf("Unexpected nil volume.  Expected %s", pv2.Name)
	}
	if pv2.Name != volume.Name {
		t.Errorf("Expected %s but got volume %s instead", pv2.Name, volume.Name)
	}
}

func TestSort(t *testing.T) {
	volList := NewPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		volList.Add(pv)
	}

	volumes, err := volList.ListByAccessModes([]api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany})
	if err != nil {
		t.Error("Unexpected error retrieving volumes by access modes:", err)
	}

	for i, expected := range []string{"gce-pd-1", "gce-pd-5", "gce-pd-10"} {
		if string(volumes[i].UID) != expected {
			t.Error("Incorrect ordering of persistent volumes.  Expected %s but got %s", expected, volumes[i].UID)
		}
	}

	volumes, err = volList.ListByAccessModes([]api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany})
	if err != nil {
		t.Error("Unexpected error retrieving volumes by access modes:", err)
	}

	for i, expected := range []string{"nfs-1", "nfs-5", "nfs-10"} {
		if string(volumes[i].UID) != expected {
			t.Error("Incorrect ordering of persistent volumes.  Expected %s but got %s", expected, volumes[i].UID)
		}
	}
}

func createTestVolumes() []*api.PersistentVolume {
	// these volumes are deliberately out-of-order to test indexing and sorting
	return []*api.PersistentVolume{
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-10",
				Name: "gce003",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-20",
				Name: "gce004",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("20G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
				// this one we're pretending is already bound
				ClaimRef: &api.ObjectReference{UID: "def456"},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "nfs-5",
				Name: "nfs002",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					Glusterfs: &api.GlusterfsVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
					api.ReadWriteMany,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-1",
				Name: "gce001",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
				// this one we're pretending is already bound
				ClaimRef: &api.ObjectReference{UID: "abc123"},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "nfs-10",
				Name: "nfs003",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					Glusterfs: &api.GlusterfsVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
					api.ReadWriteMany,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-5",
				Name: "gce002",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "nfs-1",
				Name: "nfs001",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					Glusterfs: &api.GlusterfsVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
					api.ReadWriteMany,
				},
			},
		},
	}
}
