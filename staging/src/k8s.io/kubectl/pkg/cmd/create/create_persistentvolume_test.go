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

func TestCreatePersistentVolumeValidation(t *testing.T) {
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
			o := &CreatePersistentVolumeOptions{
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

func TestCreatePersistentVolume(t *testing.T) {
	objectMeta := metav1.ObjectMeta{
		Name: "my-pv",
	}
	typeMeta := metav1.TypeMeta{
		APIVersion: v1.SchemeGroupVersion.String(),
		Kind:       "PersistentVolume",
	}
	tests := map[string]struct {
		accessModes  []string
		hostPath     string
		storage      string
		storageClass string
		expected     *v1.PersistentVolume
	}{
		"create a persistent volume and set the default hostPath to the / directory.": {
			storage:  "1Gi",
			hostPath: "/",
			expected: &v1.PersistentVolume{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/"},
					},
					Capacity: v1.ResourceList{
						v1.ResourceStorage: res.MustParse("1Gi"),
					},
				},
			},
		},
		"Create a persistent volume and define /tmp hostPath location.": {
			storage:  "1Gi",
			hostPath: "/tmp",
			expected: &v1.PersistentVolume{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/tmp"},
					},
					Capacity: v1.ResourceList{
						v1.ResourceStorage: res.MustParse("1Gi"),
					},
				},
			},
		},
		"Create a persistent volume and define a storageClassName": {
			storage:      "1Gi",
			hostPath:     "/tmp",
			storageClass: "manual",
			expected: &v1.PersistentVolume{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/tmp"},
					},
					Capacity: v1.ResourceList{
						v1.ResourceStorage: res.MustParse("1Gi"),
					},
					StorageClassName: "manual",
				},
			},
		},
		"Create a persistent volume includes ReadWriteOnce and ReadWriteOncePod access modes": {
			storage:     "1Gi",
			hostPath:    "/tmp",
			accessModes: []string{"ReadWriteOnce", "ReadWriteOncePod"},
			expected: &v1.PersistentVolume{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/tmp"},
					},
					Capacity: v1.ResourceList{
						v1.ResourceStorage: res.MustParse("1Gi"),
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
			o := &CreatePersistentVolumeOptions{
				Name:         objectMeta.Name,
				AccessModes:  tc.accessModes,
				HostPath:     tc.hostPath,
				Storage:      tc.storage,
				StorageClass: tc.storageClass,
			}
			persistentVolume, _ := o.createPersistentVolume()
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

func TestCreatePersistentVolumeClaim(t *testing.T) {
	objectMeta := metav1.ObjectMeta{
		Name:      "my-pvc",
		Namespace: "default",
	}
	typeMeta := metav1.TypeMeta{
		APIVersion: v1.SchemeGroupVersion.String(),
		Kind:       "PersistentVolumeClaim",
	}
	tests := map[string]struct {
		namespace    string
		storage      string
		storageClass string
		accessModes  []string
		claim        bool
		expected     *v1.PersistentVolumeClaim
	}{
		"Create a persistent volume and persistent volume claim": {
			namespace:    "default",
			storage:      "1Gi",
			storageClass: "manual",
			accessModes:  []string{"ReadWriteOnce", "ReadWriteOncePod"},
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
					StorageClassName: func() *string { s := "manual"; return &s }(),
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreatePersistentVolumeOptions{
				Name:             objectMeta.Name,
				Namespace:        tc.namespace,
				Storage:          tc.storage,
				AccessModes:      tc.accessModes,
				StorageClass:     tc.storageClass,
				Claim:            tc.claim,
				EnforceNamespace: true,
			}
			persistentVolumeClaim, _ := o.createPersistentVolumeClaim()
			if !apiequality.Semantic.DeepEqual(persistentVolumeClaim, tc.expected) {
				j1, e1 := json.Marshal(tc.expected)
				j2, e2 := json.Marshal(persistentVolumeClaim)
				if e1 == nil && e2 == nil {
					t.Errorf("expected:\n%#v\ngot:\n%#v", string(j1), string(j2))
				} else {
					t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, persistentVolumeClaim)
				}
			}
		})
	}
}
