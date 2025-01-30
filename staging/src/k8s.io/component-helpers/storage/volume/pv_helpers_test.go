/*
Copyright 2021 The Kubernetes Authors.

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

package volume

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
)

var (
	classNotHere       = "not-here"
	classNoMode        = "no-mode"
	classImmediateMode = "immediate-mode"
	classWaitMode      = "wait-mode"
	classGold          = "gold"
	classSilver        = "silver"

	modeImmediate = storagev1.VolumeBindingImmediate
	modeWait      = storagev1.VolumeBindingWaitForFirstConsumer
)

func makePVCClass(scName *string) *v1.PersistentVolumeClaim {
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: scName,
		},
	}

	return claim
}

func makeStorageClass(scName string, mode *storagev1.VolumeBindingMode) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		VolumeBindingMode: mode,
	}
}

func TestDelayBindingMode(t *testing.T) {
	tests := map[string]struct {
		pvc         *v1.PersistentVolumeClaim
		shouldDelay bool
		shouldFail  bool
	}{
		"nil-class": {
			pvc:         makePVCClass(nil),
			shouldDelay: false,
		},
		"class-not-found": {
			pvc:         makePVCClass(&classNotHere),
			shouldDelay: false,
		},
		"no-mode-class": {
			pvc:         makePVCClass(&classNoMode),
			shouldDelay: false,
			shouldFail:  true,
		},
		"immediate-mode-class": {
			pvc:         makePVCClass(&classImmediateMode),
			shouldDelay: false,
		},
		"wait-mode-class": {
			pvc:         makePVCClass(&classWaitMode),
			shouldDelay: true,
		},
	}

	classes := []*storagev1.StorageClass{
		makeStorageClass(classNoMode, nil),
		makeStorageClass(classImmediateMode, &modeImmediate),
		makeStorageClass(classWaitMode, &modeWait),
	}

	client := &fake.Clientset{}
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	classInformer := informerFactory.Storage().V1().StorageClasses()

	for _, class := range classes {
		if err := classInformer.Informer().GetIndexer().Add(class); err != nil {
			t.Fatalf("Failed to add storage class %q: %v", class.Name, err)
		}
	}

	for name, test := range tests {
		shouldDelay, err := IsDelayBindingMode(test.pvc, classInformer.Lister())
		if err != nil && !test.shouldFail {
			t.Errorf("Test %q returned error: %v", name, err)
		}
		if err == nil && test.shouldFail {
			t.Errorf("Test %q returned success, expected error", name)
		}
		if shouldDelay != test.shouldDelay {
			t.Errorf("Test %q returned unexpected %v", name, test.shouldDelay)
		}
	}
}

// makeVolumeNodeAffinity returns a VolumeNodeAffinity for given key and value.
func makeNodeAffinity(key string, value string) *v1.VolumeNodeAffinity {
	return &v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      key,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{value},
						},
					},
				},
			},
		},
	}
}

func TestFindMatchVolumeWithNode(t *testing.T) {
	volumes := []*v1.PersistentVolume{
		makeTestVolume("local-small", "local001", "5G", true, nil),
		makeTestVolume("local-pd-very-large", "local002", "200E", true, func(pv *v1.PersistentVolume) {
			pv.Spec.StorageClassName = "large"
		}),
		makeTestVolume("affinity-pv", "affinity001", "100G", true, func(pv *v1.PersistentVolume) {
			pv.Spec.StorageClassName = "wait"
			pv.Spec.NodeAffinity = makeNodeAffinity("key1", "value1")
		}),
		makeTestVolume("affinity-pv2", "affinity002", "150G", true, func(pv *v1.PersistentVolume) {
			pv.Spec.StorageClassName = "wait"
			pv.Spec.NodeAffinity = makeNodeAffinity("key1", "value1")
		}),
		makeTestVolume("affinity-prebound", "affinity003", "100G", true, func(pv *v1.PersistentVolume) {
			pv.Spec.StorageClassName = "wait"
			pv.Spec.ClaimRef = &v1.ObjectReference{Name: "claim02", Namespace: "myns"}
			pv.Spec.NodeAffinity = makeNodeAffinity("key1", "value1")
		}),
		makeTestVolume("affinity-pv3", "affinity003", "200G", true, func(pv *v1.PersistentVolume) {
			pv.Spec.StorageClassName = "wait"
			pv.Spec.NodeAffinity = makeNodeAffinity("key1", "value3")
		}),
		makeTestVolume("affinity-pv4", "affinity004", "200G", false, func(pv *v1.PersistentVolume) {
			pv.Spec.StorageClassName = "wait"
			pv.Spec.NodeAffinity = makeNodeAffinity("key1", "value4")
		}),
	}

	var volumesWithVAC = func(name string, input []*v1.PersistentVolume) []*v1.PersistentVolume {
		output := make([]*v1.PersistentVolume, len(input))
		for i, volume := range input {
			output[i] = volume.DeepCopy()
			output[i].Spec.VolumeAttributesClassName = &name
		}
		return output
	}

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
		expectErr       bool
		claim           *v1.PersistentVolumeClaim
		node            *v1.Node
		volumes         []*v1.PersistentVolume
		excludedVolumes map[string]*v1.PersistentVolume
		vacEnabled      []bool
	}{
		"success-match": {
			expectedMatch: "affinity-pv",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node1,
			vacEnabled:    []bool{true, false},
		},
		"success-prebound": {
			expectedMatch: "affinity-prebound",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim02", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node1,
			vacEnabled:    []bool{true, false},
		},
		"success-exclusion": {
			expectedMatch:   "affinity-pv2",
			volumes:         volumes,
			claim:           makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:            node1,
			excludedVolumes: map[string]*v1.PersistentVolume{"affinity001": nil},
			vacEnabled:      []bool{true, false},
		},
		"fail-exclusion": {
			expectedMatch:   "",
			volumes:         volumes,
			claim:           makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:            node1,
			excludedVolumes: map[string]*v1.PersistentVolume{"affinity001": nil, "affinity002": nil, "affinity002-vac": nil},
			vacEnabled:      []bool{true, false},
		},
		"fail-accessmode": {
			expectedMatch: "",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}, nil),
			node:          node1,
			vacEnabled:    []bool{true, false},
		},
		"fail-nodeaffinity": {
			expectedMatch: "",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node2,
			vacEnabled:    []bool{true, false},
		},
		"fail-prebound-node-affinity": {
			expectedMatch: "",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim02", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node3,
			vacEnabled:    []bool{true, false},
		},
		"fail-nonavaliable": {
			expectedMatch: "",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim04", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node4,
			vacEnabled:    []bool{true, false},
		},
		"success-bad-and-good-node-affinity": {
			expectedMatch: "affinity-pv3",
			volumes:       volumes,
			claim:         makeTestPersistentVolumeClaim("claim03", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node3,
			vacEnabled:    []bool{true, false},
		},
		"success-match-with-vac": {
			expectedMatch: "affinity-pv",
			volumes:       volumesWithVAC(classGold, volumes),
			claim: makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.VolumeAttributesClassName = &classGold
			}),
			node:       node1,
			vacEnabled: []bool{true},
		},
		"fail-vac": { // claim has a given vac and volumes don't have the same vac.
			expectedMatch: "",
			volumes:       volumes,
			claim: makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.VolumeAttributesClassName = &classSilver
			}),
			node:       node1,
			vacEnabled: []bool{true},
		},
		"fail-prebound-vac": { // claim has a given vac and volume name but the given volume has a different vac.
			expectedMatch: "",
			volumes:       volumesWithVAC(classGold, volumes),
			claim: makeTestPersistentVolumeClaim("claim02", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.VolumeAttributesClassName = &classSilver
			}),
			node:       node1,
			vacEnabled: []bool{true},
		},
		"fail-on-error": { // claim has a given vac when feature-gate is disabled.
			expectedMatch: "",
			expectErr:     true,
			volumes:       volumes,
			claim: makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, func(pvc *v1.PersistentVolumeClaim) {
				pvc.Spec.VolumeAttributesClassName = &classGold
			}),
			node:       node1,
			vacEnabled: []bool{false},
		},
		"fail-volumes-vac": { // claim has no vac and all volumes have vac when feature-gate is disabled.
			expectedMatch: "",
			volumes:       volumesWithVAC(classGold, volumes),
			claim:         makeTestPersistentVolumeClaim("claim01", "100G", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}, nil),
			node:          node1,
			vacEnabled:    []bool{false},
		},
	}

	for name, scenario := range scenarios {
		for _, enabled := range scenario.vacEnabled {
			name := fmt.Sprintf("[VolumeAttributiesClass: %v] %s", enabled, name)
			volume, err := FindMatchingVolume(scenario.claim, scenario.volumes, scenario.node, scenario.excludedVolumes, true, enabled)
			if scenario.expectErr && err == nil {
				t.Errorf("Expected error for scenario: %s", name)
			}
			if !scenario.expectErr && err != nil {
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
}

func makeTestPersistentVolumeClaim(name string, size string, accessMode []v1.PersistentVolumeAccessMode, modfn func(*v1.PersistentVolumeClaim)) *v1.PersistentVolumeClaim {
	fs := v1.PersistentVolumeFilesystem
	sc := "wait"
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "myns",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: accessMode,
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(size),
				},
			},
			StorageClassName: &sc,
			VolumeMode:       &fs,
		},
	}

	if modfn != nil {
		modfn(pvc)
	}

	return pvc
}

func makeTestVolume(uid types.UID, name string, capacity string, available bool, modfn func(*v1.PersistentVolume)) *v1.PersistentVolume {
	var status v1.PersistentVolumeStatus
	if available {
		status = v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		}
	}

	fs := v1.PersistentVolumeFilesystem

	pv := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			UID:  uid,
			Name: name,
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			VolumeMode: &fs,
		},
		Status: status,
	}

	if modfn != nil {
		modfn(&pv)
	}
	return &pv
}
