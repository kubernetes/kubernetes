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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"
)

func TestTranslatePortworxInTreeStorageClassToCSI(t *testing.T) {
	translator := NewPortworxCSITranslator()
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		name     string
		inTreeSC *storage.StorageClass
		csiSC    *storage.StorageClass
		errorExp bool
	}{
		{
			name: "correct",
			inTreeSC: &storage.StorageClass{
				Provisioner: PortworxVolumePluginName,
				Parameters: map[string]string{
					"repl":        "1",
					"fs":          "ext4",
					"shared":      "true",
					"priority_io": "high",
				},
			},
			csiSC: &storage.StorageClass{
				Provisioner: PortworxDriverName,
				Parameters: map[string]string{
					"repl":        "1",
					"fs":          "ext4",
					"shared":      "true",
					"priority_io": "high",
				},
			},
			errorExp: false,
		},
		{
			name:     "nil, err expected",
			inTreeSC: nil,
			csiSC:    nil,
			errorExp: true,
		},
		{
			name:     "empty",
			inTreeSC: &storage.StorageClass{},
			csiSC: &storage.StorageClass{
				Provisioner: PortworxDriverName,
			},
			errorExp: false,
		},
	}
	for _, tc := range testCases {
		t.Logf("Testing %v", tc.name)
		result, err := translator.TranslateInTreeStorageClassToCSI(logger, tc.inTreeSC)
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

func TestTranslatePortworxInTreeInlineVolumeToCSI(t *testing.T) {
	translator := NewPortworxCSITranslator()
	logger, _ := ktesting.NewTestContext(t)

	testCases := []struct {
		name        string
		inLine      *v1.Volume
		csiVol      *v1.PersistentVolume
		errExpected bool
	}{
		{
			name: "normal",
			inLine: &v1.Volume{
				Name: "PortworxVol",
				VolumeSource: v1.VolumeSource{
					PortworxVolume: &v1.PortworxVolumeSource{
						VolumeID: "ID",
						FSType:   "type",
						ReadOnly: false,
					},
				},
			},
			csiVol: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					// Must be unique per disk as it is used as the unique part of the
					// staging path
					Name: "pxd.portworx.com-ID",
				},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:           PortworxDriverName,
							VolumeHandle:     "ID",
							FSType:           "type",
							VolumeAttributes: make(map[string]string),
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
		result, err := translator.TranslateInTreeInlineVolumeToCSI(logger, tc.inLine, "ns")
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

func TestTranslatePortworxInTreePVToCSI(t *testing.T) {
	translator := NewPortworxCSITranslator()
	logger, _ := ktesting.NewTestContext(t)

	testCases := []struct {
		name        string
		inTree      *v1.PersistentVolume
		csi         *v1.PersistentVolume
		errExpected bool
	}{
		{
			name: "no Portworx volume",
			inTree: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pxd.portworx.com",
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
					Name: "pxd.portworx.com",
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
						PortworxVolume: &v1.PortworxVolumeSource{
							VolumeID: "ID1111",
							FSType:   "type",
							ReadOnly: false,
						},
					},
				},
			},
			csi: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pxd.portworx.com",
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
							Driver:           PortworxDriverName,
							VolumeHandle:     "ID1111",
							FSType:           "type",
							VolumeAttributes: make(map[string]string),
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
		result, err := translator.TranslateInTreePVToCSI(logger, tc.inTree)
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

func TestTranslatePortworxCSIPvToInTree(t *testing.T) {
	translator := NewPortworxCSITranslator()

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
					// Must be unique per disk as it is used as the unique part of the
					// staging path
					Name: "pxd.portworx.com",
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
					Name: "pxd.portworx.com",
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
							Driver:           PortworxDriverName,
							VolumeHandle:     "ID1111",
							FSType:           "type",
							VolumeAttributes: make(map[string]string),
						},
					},
				},
			},
			inTree: &v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pxd.portworx.com",
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
						PortworxVolume: &v1.PortworxVolumeSource{
							VolumeID: "ID1111",
							FSType:   "type",
							ReadOnly: false,
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
