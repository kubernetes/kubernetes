/*
Copyright 2014 The Kubernetes Authors.

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
	"sort"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
)

func TestMatchVolume(t *testing.T) {
	volList := newPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		volList.store.Add(pv)
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
		"successful-no-match-due-to-label": {
			expectedMatch: "",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					Selector: &unversioned.LabelSelector{
						MatchLabels: map[string]string{
							"should-not-exist": "true",
						},
					},
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("999G"),
						},
					},
				},
			},
		},
		"successful-no-match-due-to-size-constraint-with-label-selector": {
			expectedMatch: "",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					Selector: &unversioned.LabelSelector{
						MatchLabels: map[string]string{
							"should-exist": "true",
						},
					},
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany, api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("20000G"),
						},
					},
				},
			},
		},
		"successful-match-due-with-constraint-and-label-selector": {
			expectedMatch: "gce-pd-2",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
				},
				Spec: api.PersistentVolumeClaimSpec{
					Selector: &unversioned.LabelSelector{
						MatchLabels: map[string]string{
							"should-exist": "true",
						},
					},
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("20000G"),
						},
					},
				},
			},
		},
		"successful-match-with-class": {
			expectedMatch: "gce-pd-silver1",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
					Annotations: map[string]string{
						storageutil.StorageClassAnnotation: "silver",
					},
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Selector: &unversioned.LabelSelector{
						MatchLabels: map[string]string{
							"should-exist": "true",
						},
					},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
						},
					},
				},
			},
		},
		"successful-match-with-class-and-labels": {
			expectedMatch: "gce-pd-silver2",
			claim: &api.PersistentVolumeClaim{
				ObjectMeta: api.ObjectMeta{
					Name:      "claim01",
					Namespace: "myns",
					Annotations: map[string]string{
						storageutil.StorageClassAnnotation: "silver",
					},
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceName(api.ResourceStorage): resource.MustParse("1G"),
						},
					},
				},
			},
		},
	}

	for name, scenario := range scenarios {
		volume, err := volList.findBestMatchForClaim(scenario.claim)
		if err != nil {
			t.Errorf("Unexpected error matching volume by claim: %v", err)
		}
		if len(scenario.expectedMatch) != 0 && volume == nil {
			t.Errorf("Expected match but received nil volume for scenario: %s", name)
		}
		if len(scenario.expectedMatch) != 0 && volume != nil && string(volume.UID) != scenario.expectedMatch {
			t.Errorf("Expected %s but got volume %s in scenario %s", scenario.expectedMatch, volume.UID, name)
		}
		if len(scenario.expectedMatch) == 0 && volume != nil {
			t.Errorf("Unexpected match for scenario: %s", name)
		}
	}
}

func TestMatchingWithBoundVolumes(t *testing.T) {
	volumeIndex := newPersistentVolumeOrderedIndex()
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

	volumeIndex.store.Add(pv1)
	volumeIndex.store.Add(pv2)

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

	volume, err := volumeIndex.findBestMatchForClaim(claim)
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

func TestListByAccessModes(t *testing.T) {
	volList := newPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		volList.store.Add(pv)
	}

	volumes, err := volList.listByAccessModes([]api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany})
	if err != nil {
		t.Error("Unexpected error retrieving volumes by access modes:", err)
	}
	sort.Sort(byCapacity{volumes})

	for i, expected := range []string{"gce-pd-1", "gce-pd-5", "gce-pd-10"} {
		if string(volumes[i].UID) != expected {
			t.Errorf("Incorrect ordering of persistent volumes.  Expected %s but got %s", expected, volumes[i].UID)
		}
	}

	volumes, err = volList.listByAccessModes([]api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany})
	if err != nil {
		t.Error("Unexpected error retrieving volumes by access modes:", err)
	}
	sort.Sort(byCapacity{volumes})

	for i, expected := range []string{"nfs-1", "nfs-5", "nfs-10"} {
		if string(volumes[i].UID) != expected {
			t.Errorf("Incorrect ordering of persistent volumes.  Expected %s but got %s", expected, volumes[i].UID)
		}
	}
}

func TestAllPossibleAccessModes(t *testing.T) {
	index := newPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		index.store.Add(pv)
	}

	// the mock PVs creates contain 2 types of accessmodes:   RWO+ROX and RWO+ROW+RWX
	possibleModes := index.allPossibleMatchingAccessModes([]api.PersistentVolumeAccessMode{api.ReadWriteOnce})
	if len(possibleModes) != 3 {
		t.Errorf("Expected 3 arrays of modes that match RWO, but got %v", len(possibleModes))
	}
	for _, m := range possibleModes {
		if !contains(m, api.ReadWriteOnce) {
			t.Errorf("AccessModes does not contain %s", api.ReadWriteOnce)
		}
	}

	possibleModes = index.allPossibleMatchingAccessModes([]api.PersistentVolumeAccessMode{api.ReadWriteMany})
	if len(possibleModes) != 1 {
		t.Errorf("Expected 1 array of modes that match RWX, but got %v", len(possibleModes))
	}
	if !contains(possibleModes[0], api.ReadWriteMany) {
		t.Errorf("AccessModes does not contain %s", api.ReadWriteOnce)
	}

}

func TestFindingVolumeWithDifferentAccessModes(t *testing.T) {
	gce := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{UID: "001", Name: "gce"},
		Spec: api.PersistentVolumeSpec{
			Capacity:               api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: api.PersistentVolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{}},
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
			},
		},
	}

	ebs := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{UID: "002", Name: "ebs"},
		Spec: api.PersistentVolumeSpec{
			Capacity:               api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: api.PersistentVolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{}},
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
		},
	}

	nfs := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{UID: "003", Name: "nfs"},
		Spec: api.PersistentVolumeSpec{
			Capacity:               api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: api.PersistentVolumeSource{NFS: &api.NFSVolumeSource{}},
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
		},
	}

	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("1G")}},
		},
	}

	index := newPersistentVolumeOrderedIndex()
	index.store.Add(gce)
	index.store.Add(ebs)
	index.store.Add(nfs)

	volume, _ := index.findBestMatchForClaim(claim)
	if volume.Name != ebs.Name {
		t.Errorf("Expected %s but got volume %s instead", ebs.Name, volume.Name)
	}

	claim.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != gce.Name {
		t.Errorf("Expected %s but got volume %s instead", gce.Name, volume.Name)
	}

	// order of the requested modes should not matter
	claim.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadWriteMany, api.ReadWriteOnce, api.ReadOnlyMany}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != nfs.Name {
		t.Errorf("Expected %s but got volume %s instead", nfs.Name, volume.Name)
	}

	// fewer modes requested should still match
	claim.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadWriteMany}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != nfs.Name {
		t.Errorf("Expected %s but got volume %s instead", nfs.Name, volume.Name)
	}

	// pretend the exact match is bound.  should get the next level up of modes.
	ebs.Spec.ClaimRef = &api.ObjectReference{}
	claim.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadWriteOnce}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != gce.Name {
		t.Errorf("Expected %s but got volume %s instead", gce.Name, volume.Name)
	}

	// continue up the levels of modes.
	gce.Spec.ClaimRef = &api.ObjectReference{}
	claim.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadWriteOnce}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != nfs.Name {
		t.Errorf("Expected %s but got volume %s instead", nfs.Name, volume.Name)
	}

	// partial mode request
	gce.Spec.ClaimRef = nil
	claim.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadOnlyMany}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != gce.Name {
		t.Errorf("Expected %s but got volume %s instead", gce.Name, volume.Name)
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
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-2",
				Name: "gce0022",
				Labels: map[string]string{
					"should-exist": "true",
				},
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("20000G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-silver1",
				Name: "gce0023",
				Labels: map[string]string{
					"should-exist": "true",
				},
				Annotations: map[string]string{
					storageutil.StorageClassAnnotation: "silver",
				},
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10000G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-silver2",
				Name: "gce0024",
				Annotations: map[string]string{
					storageutil.StorageClassAnnotation: "silver",
				},
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("100G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:  "gce-pd-gold",
				Name: "gce0025",
				Annotations: map[string]string{
					storageutil.StorageClassAnnotation: "gold",
				},
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("50G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			},
		},
	}
}

func testVolume(name, size string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
		Spec: api.PersistentVolumeSpec{
			Capacity:               api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse(size)},
			PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{}},
			AccessModes:            []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
		},
	}
}

func TestFindingPreboundVolumes(t *testing.T) {
	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
			SelfLink:  testapi.Default.SelfLink("pvc", ""),
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi")}},
		},
	}
	claimRef, err := api.GetReference(claim)
	if err != nil {
		t.Errorf("error getting claimRef: %v", err)
	}

	pv1 := testVolume("pv1", "1Gi")
	pv5 := testVolume("pv5", "5Gi")
	pv8 := testVolume("pv8", "8Gi")
	pvBadSize := testVolume("pvBadSize", "1Mi")
	pvBadMode := testVolume("pvBadMode", "1Gi")
	pvBadMode.Spec.AccessModes = []api.PersistentVolumeAccessMode{api.ReadOnlyMany}

	index := newPersistentVolumeOrderedIndex()
	index.store.Add(pv1)
	index.store.Add(pv5)
	index.store.Add(pv8)
	index.store.Add(pvBadSize)
	index.store.Add(pvBadMode)

	// expected exact match on size
	volume, _ := index.findBestMatchForClaim(claim)
	if volume.Name != pv1.Name {
		t.Errorf("Expected %s but got volume %s instead", pv1.Name, volume.Name)
	}

	// pretend the exact match is pre-bound.  should get the next size up.
	pv1.Spec.ClaimRef = &api.ObjectReference{Name: "foo", Namespace: "bar"}
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != pv5.Name {
		t.Errorf("Expected %s but got volume %s instead", pv5.Name, volume.Name)
	}

	// pretend the exact match is available but the largest volume is pre-bound to the claim.
	pv1.Spec.ClaimRef = nil
	pv8.Spec.ClaimRef = claimRef
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != pv8.Name {
		t.Errorf("Expected %s but got volume %s instead", pv8.Name, volume.Name)
	}

	// pretend the volume with too small a size is pre-bound to the claim. should get the exact match.
	pv8.Spec.ClaimRef = nil
	pvBadSize.Spec.ClaimRef = claimRef
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != pv1.Name {
		t.Errorf("Expected %s but got volume %s instead", pv1.Name, volume.Name)
	}

	// pretend the volume without the right access mode is pre-bound to the claim. should get the exact match.
	pvBadSize.Spec.ClaimRef = nil
	pvBadMode.Spec.ClaimRef = claimRef
	volume, _ = index.findBestMatchForClaim(claim)
	if volume.Name != pv1.Name {
		t.Errorf("Expected %s but got volume %s instead", pv1.Name, volume.Name)
	}
}

// byCapacity is used to order volumes by ascending storage size
type byCapacity struct {
	volumes []*api.PersistentVolume
}

func (c byCapacity) Less(i, j int) bool {
	return matchStorageCapacity(c.volumes[i], c.volumes[j])
}

func (c byCapacity) Swap(i, j int) {
	c.volumes[i], c.volumes[j] = c.volumes[j], c.volumes[i]
}

func (c byCapacity) Len() int {
	return len(c.volumes)
}
