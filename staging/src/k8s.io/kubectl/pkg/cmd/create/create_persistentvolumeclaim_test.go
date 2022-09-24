/*
Copyright 2020 The Kubernetes Authors.

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

package create

import (
	"encoding/json"
	"testing"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	res "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreatePersistentVolumeClaimValidation(t *testing.T) {
	tests := map[string]struct {
		storage     string
		accessModes []string
		expected    string
	}{
		"invalid storage resource": {
			storage:  "10Mb",
			expected: `10Mb is not a valid storage resource value`,
		},
		"invalid accessMode": {
			storage: "10Mi",
			accessModes: []string{
				"ReadWriteOnce",
				"RWO",
			},
			expected: `invalid accessMode value (RWO)`,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreatePersistentVolumeClaimOptions{
				AccessModes: tc.accessModes,
				Storage:     tc.storage,
			}

			err := o.Validate()
			if err != nil && err.Error() != tc.expected {
				t.Errorf("unexpected error: %v", err)
			}
			if tc.expected != "" && err == nil {
				t.Errorf("expected error, got no error")
			}
		})
	}
}

func TestCreatePersistentVolumeClaim(t *testing.T) {
	objectMeta := metav1.ObjectMeta{
		Name: "my-pv",
	}
	typeMeta := metav1.TypeMeta{
		APIVersion: v1.SchemeGroupVersion.String(),
		Kind:       "PersistentVolumeClaim",
	}
	tests := map[string]struct {
		accessModes  []string
		storage      string
		storageClass string
		expected     *v1.PersistentVolumeClaim
	}{
		"Create a persistent volume claim.": {
			storage: "1Gi",
			expected: &v1.PersistentVolumeClaim{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeClaimSpec{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceStorage: res.MustParse("1Gi"),
						},
					},
				},
			},
		},
		"Create a persistent volume claim and define a storageClassName.": {
			storage:      "1Gi",
			storageClass: "manual",
			expected: &v1.PersistentVolumeClaim{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeClaimSpec{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceStorage: res.MustParse("1Gi"),
						},
					},
					StorageClassName: func() *string { s := "manual"; return &s }(),
				},
			},
		},
		"Create a persistent volume claim includes ReadOnlyMany and ReadWriteMany access modes": {
			storage:     "1Gi",
			accessModes: []string{"ReadWriteOnce", "ReadWriteOncePod"},
			expected: &v1.PersistentVolumeClaim{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeClaimSpec{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceStorage: res.MustParse("1Gi"),
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
						v1.ReadWriteOncePod,
					},
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreatePersistentVolumeClaimOptions{
				Name:         objectMeta.Name,
				AccessModes:  tc.accessModes,
				Storage:      tc.storage,
				StorageClass: tc.storageClass,
			}
			persistentVolume, _ := o.createPersistentVolumeClaim()
			if !apiequality.Semantic.DeepEqual(persistentVolume, tc.expected) {
				j1, e1 := json.Marshal(tc.expected)
				j2, e2 := json.Marshal(persistentVolume)
				if e1 == nil && e2 == nil {
					t.Errorf("expected:\n%#v\ngot:\n%#v", string(j1), string(j2))
				} else {
					t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, persistentVolume)
				}
			}
		})
	}
}
