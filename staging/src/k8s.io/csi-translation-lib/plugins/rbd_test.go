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

package plugins

import (
	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"reflect"
	"testing"
)

func TestTranslateRBDInTreeStorageClassToCSI(t *testing.T) {
	translator := NewRBDCSITranslator()
	testCases := []struct {
		name     string
		inTreeSC *storage.StorageClass
		csiSC    *storage.StorageClass
		errorExp bool
	}{
		{
			name: "correct",
			inTreeSC: &storage.StorageClass{
				Provisioner: RBDVolumePluginName,
				Parameters: map[string]string{
					"adminId":              "kubeadmin",
					"monitors":             "10.70.53.126:6789,10.70.53.156:6789",
					"pool":                 "replicapool",
					"adminSecretName":      "ceph-admin-secret",
					"adminSecretNamespace": "default",
				},
			},
			csiSC: &storage.StorageClass{
				Provisioner: RBDDriverName,
				Parameters: map[string]string{
					"adminId":   "kubeadmin",
					"pool":      "replicapool",
					"migration": "true",
					"clusterID": "7982de6a23b77bce50b1ba9f2e879cce",
					"monitors":  "10.70.53.126:6789,10.70.53.156:6789",
					"csi.storage.k8s.io/controller-expand-secret-name":      "ceph-admin-secret",
					"csi.storage.k8s.io/controller-expand-secret-namespace": "default",
					"csi.storage.k8s.io/node-stage-secret-name":             "ceph-admin-secret",
					"csi.storage.k8s.io/node-stage-secret-namespace":        "default",
					"csi.storage.k8s.io/provisioner-secret-name":            "ceph-admin-secret",
					"csi.storage.k8s.io/provisioner-secret-namespace":       "default",
				},
			},
			errorExp: false,
		},
		{
			name: "missing monitor",
			inTreeSC: &storage.StorageClass{
				Provisioner: RBDVolumePluginName,
				Parameters: map[string]string{
					"adminId":              "kubeadmin",
					"monitors":             "",
					"pool":                 "replicapool",
					"adminSecretName":      "ceph-admin-secret",
					"adminSecretNamespace": "default",
				},
			},
			csiSC:    nil,
			errorExp: true,
		},
		{
			name: "monitor unavailable",
			inTreeSC: &storage.StorageClass{
				Provisioner: RBDVolumePluginName,
				Parameters: map[string]string{
					"adminId":              "kubeadmin",
					"pool":                 "replicapool",
					"adminSecretName":      "ceph-admin-secret",
					"adminSecretNamespace": "default",
				},
			},
			csiSC:    nil,
			errorExp: true,
		},
		{
			name: "admin secret unavailable",
			inTreeSC: &storage.StorageClass{
				Provisioner: RBDVolumePluginName,
				Parameters: map[string]string{
					"adminId":              "kubeadmin",
					"pool":                 "replicapool",
					"monitors":             "10.70.53.126:6789,10.70.53.156:6789",
					"adminSecretNamespace": "default",
				},
			},
			csiSC:    nil,
			errorExp: true,
		},

		{
			name:     "nil, err expected",
			inTreeSC: nil,
			csiSC:    nil,
			errorExp: true,
		},
	}
	for _, tc := range testCases {
		t.Logf("Testing %v", tc.name)
		result, err := translator.TranslateInTreeStorageClassToCSI(tc.inTreeSC)
		if err != nil && !tc.errorExp {
			t.Errorf("Did not expect error but got: %v", err)
		}
		if err == nil && tc.errorExp {
			t.Errorf("Expected error, but did not get one.")
		}
		if !reflect.DeepEqual(result, tc.csiSC) {
			t.Errorf("Got parameters: %v\n, expected :%v", result, tc.csiSC)
		}
	}
}

func TestTranslateRBDInTreeInlineVolumeToCSI(t *testing.T) {
	translator := NewRBDCSITranslator()
	testCases := []struct {
		name        string
		inLine      *v1.Volume
		csiVol      *v1.PersistentVolume
		errExpected bool
	}{
		{
			name: "normal",
			inLine: &v1.Volume{
				Name: "rbdVol",
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						CephMonitors: []string{"10.70.53.126:6789,10.70.53.156:6789"},
						RBDPool:      "replicapool",
						RBDImage:     "kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
						RadosUser:    "admin",
						SecretRef:    &v1.LocalObjectReference{Name: "ceph-secret"},
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},

			csiVol: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					// Must be unique per disk as it is used as the unique part of the
					// staging path
					Name: "rbd.csi.ceph.com-kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       RBDDriverName,
							VolumeHandle: "kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
							FSType:       "ext4",
							VolumeAttributes: map[string]string{
								"clusterID":     "7982de6a23b77bce50b1ba9f2e879cce",
								"imageFeatures": "layering",
								"pool":          "replicapool",
								"staticVolume":  "true",
							},
							NodeStageSecretRef:        &v1.SecretReference{Name: "ceph-secret", Namespace: "ns"},
							ControllerExpandSecretRef: &v1.SecretReference{Name: "ceph-secret", Namespace: "ns"},
						},
					},
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
				},
			},
			errExpected: false,
		},
		{
			name:        "nil",
			inLine:      nil,
			csiVol:      nil,
			errExpected: true,
		},
	}

	for _, tc := range testCases {
		t.Logf("Testing %v", tc.name)
		result, err := translator.TranslateInTreeInlineVolumeToCSI(tc.inLine, "ns")
		if err != nil && !tc.errExpected {
			t.Errorf("Did not expect error but got: %v", err)
		}
		if err == nil && tc.errExpected {
			t.Errorf("Expected error, but did not get one.")
		}
		if !reflect.DeepEqual(result, tc.csiVol) {
			t.Errorf("Got parameters: %v\n, expected :%v", result, tc.csiVol)
		}
	}
}

func TestTranslateRBDInTreePVToCSI(t *testing.T) {
	translator := NewRBDCSITranslator()
	testCases := []struct {
		name        string
		inTree      *v1.PersistentVolume
		csi         *v1.PersistentVolume
		errExpected bool
	}{
		{
			name: "no RBD volume",
			inTree: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "rbd.csi.ceph.com",
				},
				Spec: v1.PersistentVolumeSpec{
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
					ClaimRef: &v1.ObjectReference{
						Name:      "test-pvc",
						Namespace: "default",
					},
				},
			},
			csi:         nil,
			errExpected: true,
		},
		{
			name: "normal",
			inTree: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: RBDDriverName,
				},
				Spec: v1.PersistentVolumeSpec{
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
					ClaimRef: &v1.ObjectReference{
						Name:      "test-pvc",
						Namespace: "default",
					},
					PersistentVolumeSource: v1.PersistentVolumeSource{
						RBD: &v1.RBDPersistentVolumeSource{
							CephMonitors: []string{"10.70.53.126:6789"},
							RBDPool:      "replicapool",
							RBDImage:     "kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
							RadosUser:    "admin",
							FSType:       "ext4",
							ReadOnly:     false,
							SecretRef: &v1.SecretReference{
								Name:      "ceph-secret",
								Namespace: "default",
							},
						},
					},
				},
			},
			csi: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: RBDDriverName,
				},
				Spec: v1.PersistentVolumeSpec{
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
					ClaimRef: &v1.ObjectReference{
						Name:      "test-pvc",
						Namespace: "default",
					},
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       RBDDriverName,
							VolumeHandle: "mig_mons-b7f67366bb43f32e07d8a261a7840da9_image-e4111eb6-4088-11ec-b823-0242ac110003_7265706c696361706f6f6c",
							ReadOnly:     false,
							FSType:       "ext4",
							VolumeAttributes: map[string]string{
								"clusterID":        "b7f67366bb43f32e07d8a261a7840da9",
								"imageFeatures":    "layering",
								"imageFormat":      "",
								"imageName":        "kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
								"journalPool":      "",
								"migration":        "true",
								"pool":             "replicapool",
								"staticVolume":     "true",
								"tryOtherMounters": "true",
							},
							NodeStageSecretRef: &v1.SecretReference{
								Name:      "ceph-secret",
								Namespace: "default",
							},
							ControllerExpandSecretRef: &v1.SecretReference{
								Name:      "ceph-secret",
								Namespace: "default",
							},
						},
					},
				},
			},
			errExpected: false,
		},
		{
			name:        "nil PV",
			inTree:      nil,
			csi:         nil,
			errExpected: true,
		},
	}

	for _, tc := range testCases {
		t.Logf("Testing %v", tc.name)
		result, err := translator.TranslateInTreePVToCSI(tc.inTree)
		if err != nil && !tc.errExpected {
			t.Errorf("Did not expect error but got: %v", err)
		}
		if err == nil && tc.errExpected {
			t.Errorf("Expected error, but did not get one.")
		}
		if !reflect.DeepEqual(result, tc.csi) {
			t.Errorf("Got parameters: %v\n, expected :%v", result, tc.csi)
		}
	}
}
func TestTranslateCSIPvToInTree(t *testing.T) {
	translator := NewRBDCSITranslator()

	testCases := []struct {
		name        string
		csi         *v1.PersistentVolume
		inTree      *v1.PersistentVolume
		errExpected bool
	}{
		{
			name: "no CSI section",
			csi: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: RBDDriverName,
				},
				Spec: v1.PersistentVolumeSpec{
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
					ClaimRef: &v1.ObjectReference{
						Name:      "test-pvc",
						Namespace: "default",
					},
				},
			},
			inTree:      nil,
			errExpected: true,
		},
		{
			name: "normal",
			csi: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: RBDDriverName,
				},
				Spec: v1.PersistentVolumeSpec{
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
					ClaimRef: &v1.ObjectReference{
						Name:      "test-pvc",
						Namespace: "default",
					},
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       RBDDriverName,
							VolumeHandle: "dummy",
							ReadOnly:     false,
							FSType:       "ext4",
							VolumeAttributes: map[string]string{
								"clusterID":        "b7f67366bb43f32e07d8a261a7840da9",
								"imageFeatures":    "layering",
								"imageFormat":      "1",
								"imageName":        "kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
								"journalPool":      "some",
								"migration":        "true",
								"pool":             "replicapool",
								"staticVolume":     "true",
								"tryOtherMounters": "true",
							},
							NodeStageSecretRef: &v1.SecretReference{
								Name:      "ceph-secret",
								Namespace: "default",
							},
						},
					},
				},
			},
			inTree: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: RBDDriverName,
					Annotations: map[string]string{
						"clusterID":                      "b7f67366bb43f32e07d8a261a7840da9",
						"imageFeatures":                  "layering",
						"imageFormat":                    "1",
						"journalPool":                    "some",
						"rbd.csi.ceph.com/volume-handle": "dummy",
					},
				},
				Spec: v1.PersistentVolumeSpec{
					AccessModes: []v1.PersistentVolumeAccessMode{
						v1.ReadWriteOnce,
					},
					ClaimRef: &v1.ObjectReference{
						Name:      "test-pvc",
						Namespace: "default",
					},
					PersistentVolumeSource: v1.PersistentVolumeSource{
						RBD: &v1.RBDPersistentVolumeSource{
							CephMonitors: nil,
							RBDPool:      "replicapool",
							RBDImage:     "kubernetes-dynamic-pvc-e4111eb6-4088-11ec-b823-0242ac110003",
							RadosUser:    "admin",
							FSType:       "ext4",
							ReadOnly:     false,
							SecretRef: &v1.SecretReference{
								Name:      "ceph-secret",
								Namespace: "default",
							},
						},
					},
				},
			},
			errExpected: false,
		},
		{
			name:        "nil PV",
			inTree:      nil,
			csi:         nil,
			errExpected: true,
		},
	}
	for _, tc := range testCases {
		t.Logf("Testing %v", tc.name)
		result, err := translator.TranslateCSIPVToInTree(tc.csi)
		if err != nil && !tc.errExpected {
			t.Errorf("Did not expect error but got: %v", err)
		}
		if err == nil && tc.errExpected {
			t.Errorf("Expected error, but did not get one.")
		}
		if !reflect.DeepEqual(result, tc.inTree) {
			t.Errorf("Got parameters: %v\n, expected :%v", result, tc.inTree)
		}
	}
}
