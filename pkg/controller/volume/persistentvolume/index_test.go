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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/scheme"
	ref "k8s.io/client-go/tools/reference"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume/util"
)

func makePVC(size string, modfn func(*v1.PersistentVolumeClaim)) *v1.PersistentVolumeClaim {
	fs := v1.PersistentVolumeFilesystem
	pvc := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany, v1.ReadWriteOnce},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(size),
				},
			},
			VolumeMode: &fs,
		},
	}
	if modfn != nil {
		modfn(&pvc)
	}
	return &pvc
}

func makeVolumeModePVC(size string, mode *v1.PersistentVolumeMode, modfn func(*v1.PersistentVolumeClaim)) *v1.PersistentVolumeClaim {
	pvc := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeMode:  mode,
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(size),
				},
			},
		},
	}
	if modfn != nil {
		modfn(&pvc)
	}
	return &pvc
}

func TestMatchVolume(t *testing.T) {
	volList := newPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		volList.store.Add(pv)
	}

	scenarios := map[string]struct {
		expectedMatch string
		claim         *v1.PersistentVolumeClaim
	}{
		"successful-match-gce-10": {
			expectedMatch: "gce-pd-10",
			claim:         makePVC("8G", nil),
		},
		"successful-match-nfs-5": {
			expectedMatch: "nfs-5",
			claim: makePVC("5G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany, v1.ReadWriteOnce, v1.ReadWriteMany}
			}),
		},
		"successful-skip-1g-bound-volume": {
			expectedMatch: "gce-pd-5",
			claim:         makePVC("1G", nil),
		},
		"successful-no-match": {
			expectedMatch: "",
			claim:         makePVC("999G", nil),
		},
		"successful-no-match-due-to-label": {
			expectedMatch: "",
			claim: makePVC("999G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"should-not-exist": "true",
					},
				}
			}),
		},
		"successful-no-match-due-to-size-constraint-with-label-selector": {
			expectedMatch: "",
			claim: makePVC("20000G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"should-exist": "true",
					},
				}
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany, v1.ReadWriteOnce}
			}),
		},
		"successful-match-due-with-constraint-and-label-selector": {
			expectedMatch: "gce-pd-2",
			claim: makePVC("20000G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"should-exist": "true",
					},
				}
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
			}),
		},
		"successful-match-with-class": {
			expectedMatch: "gce-pd-silver1",
			claim: makePVC("1G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"should-exist": "true",
					},
				}
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classSilver
			}),
		},
		"successful-match-with-class-and-labels": {
			expectedMatch: "gce-pd-silver2",
			claim: makePVC("1G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classSilver
			}),
		},
		"successful-match-very-large": {
			expectedMatch: "local-pd-very-large",
			// we keep the pvc size less than int64 so that in case the pv overflows
			// the pvc does not overflow equally and give us false matching signals.
			claim: makePVC("1E", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classLarge
			}),
		},
		"successful-match-exact-extremely-large": {
			expectedMatch: "local-pd-extremely-large",
			claim: makePVC("800E", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classLarge
			}),
		},
		"successful-no-match-way-too-large": {
			expectedMatch: "",
			claim: makePVC("950E", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classLarge
			}),
		},
	}

	for name, scenario := range scenarios {
		volume, err := volList.findBestMatchForClaim(scenario.claim, false)
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
			t.Errorf("Unexpected match for scenario: %s, matched with %s instead", name, volume.UID)
		}
	}
}

func TestMatchingWithBoundVolumes(t *testing.T) {
	fs := v1.PersistentVolumeFilesystem
	volumeIndex := newPersistentVolumeOrderedIndex()
	// two similar volumes, one is bound
	pv1 := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "gce-pd-1",
			Name: "gce001",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany},
			// this one we're pretending is already bound
			ClaimRef:   &v1.ObjectReference{UID: "abc123"},
			VolumeMode: &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeBound,
		},
	}

	pv2 := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "gce-pd-2",
			Name: "gce002",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany},
			VolumeMode:  &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}

	volumeIndex.store.Add(pv1)
	volumeIndex.store.Add(pv2)

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany, v1.ReadWriteOnce},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G"),
				},
			},
			VolumeMode: &fs,
		},
	}

	volume, err := volumeIndex.findBestMatchForClaim(claim, false)
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

	volumes, err := volList.listByAccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany})
	if err != nil {
		t.Error("Unexpected error retrieving volumes by access modes:", err)
	}
	sort.Sort(byCapacity{volumes})

	for i, expected := range []string{"gce-pd-1", "gce-pd-5", "gce-pd-10"} {
		if string(volumes[i].UID) != expected {
			t.Errorf("Incorrect ordering of persistent volumes.  Expected %s but got %s", expected, volumes[i].UID)
		}
	}

	volumes, err = volList.listByAccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany, v1.ReadWriteMany})
	if err != nil {
		t.Error("Unexpected error retrieving volumes by access modes:", err)
	}
	sort.Sort(byCapacity{volumes})

	for i, expected := range []string{"nfs-1", "nfs-5", "nfs-10", "local-pd-very-large", "local-pd-extremely-large"} {
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
	possibleModes := index.allPossibleMatchingAccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteOnce})
	if len(possibleModes) != 3 {
		t.Errorf("Expected 3 arrays of modes that match RWO, but got %v", len(possibleModes))
	}
	for _, m := range possibleModes {
		if !util.AccessModesContains(m, v1.ReadWriteOnce) {
			t.Errorf("AccessModes does not contain %s", v1.ReadWriteOnce)
		}
	}

	possibleModes = index.allPossibleMatchingAccessModes([]v1.PersistentVolumeAccessMode{v1.ReadWriteMany})
	if len(possibleModes) != 1 {
		t.Errorf("Expected 1 array of modes that match RWX, but got %v", len(possibleModes))
	}
	if !util.AccessModesContains(possibleModes[0], v1.ReadWriteMany) {
		t.Errorf("AccessModes does not contain %s", v1.ReadWriteOnce)
	}

}

func TestFindingVolumeWithDifferentAccessModes(t *testing.T) {
	fs := v1.PersistentVolumeFilesystem
	gce := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "001", Name: "gce"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			VolumeMode: &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}

	ebs := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "002", Name: "ebs"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			VolumeMode: &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}

	nfs := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{UID: "003", Name: "nfs"},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{NFS: &v1.NFSVolumeSource{}},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
				v1.ReadWriteMany,
			},
			VolumeMode: &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources:   v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G")}},
			VolumeMode:  &fs,
		},
	}

	index := newPersistentVolumeOrderedIndex()
	index.store.Add(gce)
	index.store.Add(ebs)
	index.store.Add(nfs)

	volume, _ := index.findBestMatchForClaim(claim, false)
	if volume.Name != ebs.Name {
		t.Errorf("Expected %s but got volume %s instead", ebs.Name, volume.Name)
	}

	claim.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadOnlyMany}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != gce.Name {
		t.Errorf("Expected %s but got volume %s instead", gce.Name, volume.Name)
	}

	// order of the requested modes should not matter
	claim.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany, v1.ReadWriteOnce, v1.ReadOnlyMany}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != nfs.Name {
		t.Errorf("Expected %s but got volume %s instead", nfs.Name, volume.Name)
	}

	// fewer modes requested should still match
	claim.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != nfs.Name {
		t.Errorf("Expected %s but got volume %s instead", nfs.Name, volume.Name)
	}

	// pretend the exact match is bound.  should get the next level up of modes.
	ebs.Spec.ClaimRef = &v1.ObjectReference{}
	claim.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != gce.Name {
		t.Errorf("Expected %s but got volume %s instead", gce.Name, volume.Name)
	}

	// continue up the levels of modes.
	gce.Spec.ClaimRef = &v1.ObjectReference{}
	claim.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != nfs.Name {
		t.Errorf("Expected %s but got volume %s instead", nfs.Name, volume.Name)
	}

	// partial mode request
	gce.Spec.ClaimRef = nil
	claim.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != gce.Name {
		t.Errorf("Expected %s but got volume %s instead", gce.Name, volume.Name)
	}
}

func createTestVolumes() []*v1.PersistentVolume {
	fs := v1.PersistentVolumeFilesystem
	// these volumes are deliberately out-of-order to test indexing and sorting
	return []*v1.PersistentVolume{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-10",
				Name: "gce003",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-20",
				Name: "gce004",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("20G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				// this one we're pretending is already bound
				ClaimRef:   &v1.ObjectReference{UID: "def456"},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeBound,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "nfs-5",
				Name: "nfs002",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Glusterfs: &v1.GlusterfsPersistentVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
					v1.ReadWriteMany,
				},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-1",
				Name: "gce001",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				// this one we're pretending is already bound
				ClaimRef:   &v1.ObjectReference{UID: "abc123"},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeBound,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "nfs-10",
				Name: "nfs003",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Glusterfs: &v1.GlusterfsPersistentVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
					v1.ReadWriteMany,
				},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-5",
				Name: "gce002",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "nfs-1",
				Name: "nfs001",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Glusterfs: &v1.GlusterfsPersistentVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
					v1.ReadWriteMany,
				},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-2",
				Name: "gce0022",
				Labels: map[string]string{
					"should-exist": "true",
				},
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("20000G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				VolumeMode: &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-silver1",
				Name: "gce0023",
				Labels: map[string]string{
					"should-exist": "true",
				},
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("10000G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				StorageClassName: classSilver,
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-silver2",
				Name: "gce0024",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("100G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				StorageClassName: classSilver,
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "gce-pd-gold",
				Name: "gce0025",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("50G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
				},
				StorageClassName: classGold,
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "local-pd-very-large",
				Name: "local001",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("200E"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
					v1.ReadWriteMany,
				},
				StorageClassName: classLarge,
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "local-pd-extremely-large",
				Name: "local002",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("800E"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
					v1.ReadWriteMany,
				},
				StorageClassName: classLarge,
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv",
				Name: "affinity001",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("100G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value1"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv2",
				Name: "affinity002",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("150G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value1"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-prebound",
				Name: "affinity003",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("100G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				ClaimRef:         &v1.ObjectReference{Name: "claim02", Namespace: "myns"},
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value1"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv3",
				Name: "affinity003",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("200G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value3"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeAvailable,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv4-pending",
				Name: "affinity004-pending",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("200G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value4"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumePending,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv4-failed",
				Name: "affinity004-failed",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("200G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value4"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeFailed,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv4-released",
				Name: "affinity004-released",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("200G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value4"),
				VolumeMode:       &fs,
			},
			Status: v1.PersistentVolumeStatus{
				Phase: v1.VolumeReleased,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:  "affinity-pv4-empty",
				Name: "affinity004-empty",
			},
			Spec: v1.PersistentVolumeSpec{
				Capacity: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("200G"),
				},
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Local: &v1.LocalVolumeSource{},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{
					v1.ReadWriteOnce,
					v1.ReadOnlyMany,
				},
				StorageClassName: classWait,
				NodeAffinity:     pvutil.GetVolumeNodeAffinity("key1", "value4"),
				VolumeMode:       &fs,
			},
		},
	}
}

func testVolume(name, size string) *v1.PersistentVolume {
	fs := v1.PersistentVolumeFilesystem
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse(size)},
			PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{}},
			AccessModes:            []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			VolumeMode:             &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
}

func createVolumeModeBlockTestVolume() *v1.PersistentVolume {
	blockMode := v1.PersistentVolumeBlock

	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "local-1",
			Name: "block",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			VolumeMode: &blockMode,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
}

func createVolumeModeFilesystemTestVolume() *v1.PersistentVolume {
	filesystemMode := v1.PersistentVolumeFilesystem

	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "local-1",
			Name: "block",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			VolumeMode: &filesystemMode,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
}

func createVolumeModeNilTestVolume() *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  "local-1",
			Name: "nil-mode",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
}

func createTestVolOrderedIndex(pv *v1.PersistentVolume) persistentVolumeOrderedIndex {
	volFile := newPersistentVolumeOrderedIndex()
	volFile.store.Add(pv)
	return volFile
}

func TestVolumeModeCheck(t *testing.T) {

	blockMode := v1.PersistentVolumeBlock
	filesystemMode := v1.PersistentVolumeFilesystem

	// If feature gate is enabled, VolumeMode will always be defaulted
	// If feature gate is disabled, VolumeMode is dropped by API and ignored
	scenarios := map[string]struct {
		isExpectedMismatch bool
		vol                *v1.PersistentVolume
		pvc                *v1.PersistentVolumeClaim
		enableBlock        bool
	}{
		"feature enabled - pvc block and pv filesystem": {
			isExpectedMismatch: true,
			vol:                createVolumeModeFilesystemTestVolume(),
			pvc:                makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc filesystem and pv block": {
			isExpectedMismatch: true,
			vol:                createVolumeModeBlockTestVolume(),
			pvc:                makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc block and pv block": {
			isExpectedMismatch: false,
			vol:                createVolumeModeBlockTestVolume(),
			pvc:                makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc filesystem and pv filesystem": {
			isExpectedMismatch: false,
			vol:                createVolumeModeFilesystemTestVolume(),
			pvc:                makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc filesystem and pv nil": {
			isExpectedMismatch: false,
			vol:                createVolumeModeNilTestVolume(),
			pvc:                makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc nil and pv filesystem": {
			isExpectedMismatch: false,
			vol:                createVolumeModeFilesystemTestVolume(),
			pvc:                makeVolumeModePVC("8G", nil, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc nil and pv nil": {
			isExpectedMismatch: false,
			vol:                createVolumeModeNilTestVolume(),
			pvc:                makeVolumeModePVC("8G", nil, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc nil and pv block": {
			isExpectedMismatch: true,
			vol:                createVolumeModeBlockTestVolume(),
			pvc:                makeVolumeModePVC("8G", nil, nil),
			enableBlock:        true,
		},
		"feature enabled - pvc block and pv nil": {
			isExpectedMismatch: true,
			vol:                createVolumeModeNilTestVolume(),
			pvc:                makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:        true,
		},
		"feature disabled - pvc block and pv filesystem": {
			isExpectedMismatch: true,
			vol:                createVolumeModeFilesystemTestVolume(),
			pvc:                makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:        false,
		},
		"feature disabled - pvc filesystem and pv block": {
			isExpectedMismatch: true,
			vol:                createVolumeModeBlockTestVolume(),
			pvc:                makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:        false,
		},
		"feature disabled - pvc block and pv block": {
			isExpectedMismatch: true,
			vol:                createVolumeModeBlockTestVolume(),
			pvc:                makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:        false,
		},
		"feature disabled - pvc filesystem and pv filesystem": {
			isExpectedMismatch: false,
			vol:                createVolumeModeFilesystemTestVolume(),
			pvc:                makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:        false,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, scenario.enableBlock)()
			expectedMismatch := pvutil.CheckVolumeModeMismatches(&scenario.pvc.Spec, &scenario.vol.Spec)
			// expected to match but either got an error or no returned pvmatch
			if expectedMismatch && !scenario.isExpectedMismatch {
				t.Errorf("Unexpected failure for scenario, expected not to mismatch on modes but did: %s", name)
			}
			if !expectedMismatch && scenario.isExpectedMismatch {
				t.Errorf("Unexpected failure for scenario, did not mismatch on mode when expected to mismatch: %s", name)
			}
		})
	}
}

func TestFilteringVolumeModes(t *testing.T) {
	blockMode := v1.PersistentVolumeBlock
	filesystemMode := v1.PersistentVolumeFilesystem

	// If feature gate is enabled, VolumeMode will always be defaulted
	// If feature gate is disabled, VolumeMode is dropped by API and ignored
	scenarios := map[string]struct {
		isExpectedMatch bool
		vol             persistentVolumeOrderedIndex
		pvc             *v1.PersistentVolumeClaim
		enableBlock     bool
	}{
		"1-1 feature enabled - pvc block and pv filesystem": {
			isExpectedMatch: false,
			vol:             createTestVolOrderedIndex(createVolumeModeFilesystemTestVolume()),
			pvc:             makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:     true,
		},
		"1-2 feature enabled - pvc filesystem and pv block": {
			isExpectedMatch: false,
			vol:             createTestVolOrderedIndex(createVolumeModeBlockTestVolume()),
			pvc:             makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:     true,
		},
		"1-3 feature enabled - pvc block and pv no mode with default filesystem": {
			isExpectedMatch: false,
			vol:             createTestVolOrderedIndex(createVolumeModeFilesystemTestVolume()),
			pvc:             makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:     true,
		},
		"1-4 feature enabled - pvc no mode defaulted to filesystem and pv block": {
			isExpectedMatch: false,
			vol:             createTestVolOrderedIndex(createVolumeModeBlockTestVolume()),
			pvc:             makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:     true,
		},
		"1-5 feature enabled - pvc block and pv block": {
			isExpectedMatch: true,
			vol:             createTestVolOrderedIndex(createVolumeModeBlockTestVolume()),
			pvc:             makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:     true,
		},
		"1-6 feature enabled - pvc filesystem and pv filesystem": {
			isExpectedMatch: true,
			vol:             createTestVolOrderedIndex(createVolumeModeFilesystemTestVolume()),
			pvc:             makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:     true,
		},
		"1-7 feature enabled - pvc mode is nil and defaulted and pv mode is nil and defaulted": {
			isExpectedMatch: true,
			vol:             createTestVolOrderedIndex(createVolumeModeFilesystemTestVolume()),
			pvc:             makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:     true,
		},
		"2-1 feature disabled - pvc mode is nil and pv mode is nil": {
			isExpectedMatch: true,
			vol:             createTestVolOrderedIndex(testVolume("nomode-1", "8G")),
			pvc:             makeVolumeModePVC("8G", nil, nil),
			enableBlock:     false,
		},
		"2-2 feature disabled - pvc mode is block and pv mode is block - fields should be dropped by api and not analyzed with gate disabled": {
			isExpectedMatch: false,
			vol:             createTestVolOrderedIndex(createVolumeModeBlockTestVolume()),
			pvc:             makeVolumeModePVC("8G", &blockMode, nil),
			enableBlock:     false,
		},
		"2-3 feature disabled - pvc mode is filesystem and pv mode is filesystem - fields should be dropped by api and not analyzed with gate disabled": {
			isExpectedMatch: true,
			vol:             createTestVolOrderedIndex(createVolumeModeFilesystemTestVolume()),
			pvc:             makeVolumeModePVC("8G", &filesystemMode, nil),
			enableBlock:     false,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, scenario.enableBlock)()
			pvmatch, err := scenario.vol.findBestMatchForClaim(scenario.pvc, false)
			// expected to match but either got an error or no returned pvmatch
			if pvmatch == nil && scenario.isExpectedMatch {
				t.Errorf("Unexpected failure for scenario, no matching volume: %s", name)
			}
			if err != nil && scenario.isExpectedMatch {
				t.Errorf("Unexpected failure for scenario: %s - %+v", name, err)
			}
			// expected to not match but either got an error or a returned pvmatch
			if pvmatch != nil && !scenario.isExpectedMatch {
				t.Errorf("Unexpected failure for scenario, expected no matching volume: %s", name)
			}
			if err != nil && !scenario.isExpectedMatch {
				t.Errorf("Unexpected failure for scenario: %s - %+v", name, err)
			}
		})
	}
}

func TestStorageObjectInUseProtectionFiltering(t *testing.T) {
	fs := v1.PersistentVolumeFilesystem
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "pv1",
			Annotations: map[string]string{},
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity:               v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G")},
			PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{}},
			AccessModes:            []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			VolumeMode:             &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}

	pvToDelete := pv.DeepCopy()
	now := metav1.Now()
	pvToDelete.ObjectMeta.DeletionTimestamp = &now

	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pvc1",
			Namespace: "myns",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources:   v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G")}},
			VolumeMode:  &fs,
		},
	}

	satisfyingTestCases := map[string]struct {
		isExpectedMatch                    bool
		vol                                *v1.PersistentVolume
		pvc                                *v1.PersistentVolumeClaim
		enableStorageObjectInUseProtection bool
	}{
		"feature enabled - pv deletionTimeStamp not set": {
			isExpectedMatch:                    true,
			vol:                                pv,
			pvc:                                pvc,
			enableStorageObjectInUseProtection: true,
		},
		"feature enabled - pv deletionTimeStamp set": {
			isExpectedMatch:                    false,
			vol:                                pvToDelete,
			pvc:                                pvc,
			enableStorageObjectInUseProtection: true,
		},
		"feature disabled - pv deletionTimeStamp not set": {
			isExpectedMatch:                    true,
			vol:                                pv,
			pvc:                                pvc,
			enableStorageObjectInUseProtection: false,
		},
		"feature disabled - pv deletionTimeStamp set": {
			isExpectedMatch:                    true,
			vol:                                pvToDelete,
			pvc:                                pvc,
			enableStorageObjectInUseProtection: false,
		},
	}

	for name, testCase := range satisfyingTestCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageObjectInUseProtection, testCase.enableStorageObjectInUseProtection)()

			err := checkVolumeSatisfyClaim(testCase.vol, testCase.pvc)
			// expected to match but got an error
			if err != nil && testCase.isExpectedMatch {
				t.Errorf("%s: expected to match but got an error: %v", name, err)
			}
			// not expected to match but did
			if err == nil && !testCase.isExpectedMatch {
				t.Errorf("%s: not expected to match but did", name)
			}
		})
	}

	filteringTestCases := map[string]struct {
		isExpectedMatch                    bool
		vol                                persistentVolumeOrderedIndex
		pvc                                *v1.PersistentVolumeClaim
		enableStorageObjectInUseProtection bool
	}{
		"feature enabled - pv deletionTimeStamp not set": {
			isExpectedMatch:                    true,
			vol:                                createTestVolOrderedIndex(pv),
			pvc:                                pvc,
			enableStorageObjectInUseProtection: true,
		},
		"feature enabled - pv deletionTimeStamp set": {
			isExpectedMatch:                    false,
			vol:                                createTestVolOrderedIndex(pvToDelete),
			pvc:                                pvc,
			enableStorageObjectInUseProtection: true,
		},
		"feature disabled - pv deletionTimeStamp not set": {
			isExpectedMatch:                    true,
			vol:                                createTestVolOrderedIndex(pv),
			pvc:                                pvc,
			enableStorageObjectInUseProtection: false,
		},
		"feature disabled - pv deletionTimeStamp set": {
			isExpectedMatch:                    true,
			vol:                                createTestVolOrderedIndex(pvToDelete),
			pvc:                                pvc,
			enableStorageObjectInUseProtection: false,
		},
	}
	for name, testCase := range filteringTestCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageObjectInUseProtection, testCase.enableStorageObjectInUseProtection)()

			pvmatch, err := testCase.vol.findBestMatchForClaim(testCase.pvc, false)
			// expected to match but either got an error or no returned pvmatch
			if pvmatch == nil && testCase.isExpectedMatch {
				t.Errorf("Unexpected failure for testcase, no matching volume: %s", name)
			}
			if err != nil && testCase.isExpectedMatch {
				t.Errorf("Unexpected failure for testcase: %s - %+v", name, err)
			}
			// expected to not match but either got an error or a returned pvmatch
			if pvmatch != nil && !testCase.isExpectedMatch {
				t.Errorf("Unexpected failure for testcase, expected no matching volume: %s", name)
			}
			if err != nil && !testCase.isExpectedMatch {
				t.Errorf("Unexpected failure for testcase: %s - %+v", name, err)
			}
		})
	}
}

func TestFindingPreboundVolumes(t *testing.T) {
	fs := v1.PersistentVolumeFilesystem
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim01",
			Namespace: "myns",
			SelfLink:  "/api/v1/namespaces/myns/persistentvolumeclaims/claim01",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources:   v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi")}},
			VolumeMode:  &fs,
		},
	}
	claimRef, err := ref.GetReference(scheme.Scheme, claim)
	if err != nil {
		t.Errorf("error getting claimRef: %v", err)
	}

	pv1 := testVolume("pv1", "1Gi")
	pv5 := testVolume("pv5", "5Gi")
	pv8 := testVolume("pv8", "8Gi")
	pvBadSize := testVolume("pvBadSize", "1Mi")
	pvBadMode := testVolume("pvBadMode", "1Gi")
	pvBadMode.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany}

	index := newPersistentVolumeOrderedIndex()
	index.store.Add(pv1)
	index.store.Add(pv5)
	index.store.Add(pv8)
	index.store.Add(pvBadSize)
	index.store.Add(pvBadMode)

	// expected exact match on size
	volume, _ := index.findBestMatchForClaim(claim, false)
	if volume.Name != pv1.Name {
		t.Errorf("Expected %s but got volume %s instead", pv1.Name, volume.Name)
	}

	// pretend the exact match is pre-bound.  should get the next size up.
	pv1.Spec.ClaimRef = &v1.ObjectReference{Name: "foo", Namespace: "bar"}
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != pv5.Name {
		t.Errorf("Expected %s but got volume %s instead", pv5.Name, volume.Name)
	}

	// pretend the exact match is available but the largest volume is pre-bound to the claim.
	pv1.Spec.ClaimRef = nil
	pv8.Spec.ClaimRef = claimRef
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != pv8.Name {
		t.Errorf("Expected %s but got volume %s instead", pv8.Name, volume.Name)
	}

	// pretend the volume with too small a size is pre-bound to the claim. should get the exact match.
	pv8.Spec.ClaimRef = nil
	pvBadSize.Spec.ClaimRef = claimRef
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != pv1.Name {
		t.Errorf("Expected %s but got volume %s instead", pv1.Name, volume.Name)
	}

	// pretend the volume without the right access mode is pre-bound to the claim. should get the exact match.
	pvBadSize.Spec.ClaimRef = nil
	pvBadMode.Spec.ClaimRef = claimRef
	volume, _ = index.findBestMatchForClaim(claim, false)
	if volume.Name != pv1.Name {
		t.Errorf("Expected %s but got volume %s instead", pv1.Name, volume.Name)
	}
}

func TestBestMatchDelayed(t *testing.T) {
	volList := newPersistentVolumeOrderedIndex()
	for _, pv := range createTestVolumes() {
		volList.store.Add(pv)
	}

	// binding through PV controller should be delayed
	claim := makePVC("8G", nil)
	volume, err := volList.findBestMatchForClaim(claim, true)
	if err != nil {
		t.Errorf("Unexpected error matching volume by claim: %v", err)
	}
	if volume != nil {
		t.Errorf("Unexpected match with %q", volume.UID)
	}
}

func TestFindMatchVolumeWithNode(t *testing.T) {
	volumes := createTestVolumes()
	node1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"key1": "value1"},
		},
	}
	node2 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"key1": "value2"},
		},
	}
	node3 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"key1": "value3"},
		},
	}
	node4 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"key1": "value4"},
		},
	}

	scenarios := map[string]struct {
		expectedMatch   string
		claim           *v1.PersistentVolumeClaim
		node            *v1.Node
		excludedVolumes map[string]*v1.PersistentVolume
	}{
		"success-match": {
			expectedMatch: "affinity-pv",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
			}),
			node: node1,
		},
		"success-prebound": {
			expectedMatch: "affinity-prebound",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
				pvc.Name = "claim02"
			}),
			node: node1,
		},
		"success-exclusion": {
			expectedMatch: "affinity-pv2",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
			}),
			node:            node1,
			excludedVolumes: map[string]*v1.PersistentVolume{"affinity001": nil},
		},
		"fail-exclusion": {
			expectedMatch: "",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
			}),
			node:            node1,
			excludedVolumes: map[string]*v1.PersistentVolume{"affinity001": nil, "affinity002": nil},
		},
		"fail-accessmode": {
			expectedMatch: "",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
				pvc.Spec.StorageClassName = &classWait
			}),
			node: node1,
		},
		"fail-nodeaffinity": {
			expectedMatch: "",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
			}),
			node: node2,
		},
		"fail-prebound-node-affinity": {
			expectedMatch: "",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
				pvc.Name = "claim02"
			}),
			node: node3,
		},
		"fail-nonavaiable": {
			expectedMatch: "",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
				pvc.Name = "claim04"
			}),
			node: node4,
		},
		"success-bad-and-good-node-affinity": {
			expectedMatch: "affinity-pv3",
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
				pvc.Spec.StorageClassName = &classWait
				pvc.Name = "claim03"
			}),
			node: node3,
		},
	}

	for name, scenario := range scenarios {
		volume, err := pvutil.FindMatchingVolume(scenario.claim, volumes, scenario.node, scenario.excludedVolumes, true)
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
			t.Errorf("Unexpected match for scenario: %s, matched with %s instead", name, volume.UID)
		}
	}
}

func TestCheckAccessModes(t *testing.T) {
	volume := &v1.PersistentVolume{
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce, v1.ReadWriteMany},
		},
	}

	scenarios := map[string]struct {
		shouldSucceed bool
		claim         *v1.PersistentVolumeClaim
	}{
		"success-single-mode": {
			shouldSucceed: true,
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
			}),
		},
		"success-many-modes": {
			shouldSucceed: true,
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany, v1.ReadWriteOnce}
			}),
		},
		"fail-single-mode": {
			shouldSucceed: false,
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany}
			}),
		},
		"fail-many-modes": {
			shouldSucceed: false,
			claim: makePVC("100G", func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany, v1.ReadOnlyMany}
			}),
		},
	}

	for name, scenario := range scenarios {
		result := pvutil.CheckAccessModes(scenario.claim, volume)
		if result != scenario.shouldSucceed {
			t.Errorf("Test %q failed: Expected %v, got %v", name, scenario.shouldSucceed, result)
		}
	}
}

// byCapacity is used to order volumes by ascending storage size
type byCapacity struct {
	volumes []*v1.PersistentVolume
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

// matchStorageCapacity is a matchPredicate used to sort and find volumes
func matchStorageCapacity(pvA, pvB *v1.PersistentVolume) bool {
	aQty := pvA.Spec.Capacity[v1.ResourceStorage]
	bQty := pvB.Spec.Capacity[v1.ResourceStorage]
	return aQty.Cmp(bQty) <= 0
}
