/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetFileShareInfo(t *testing.T) {
	tests := []struct {
		options   string
		expected1 string
		expected2 string
		expected3 string
		expected4 error
	}{
		{
			options:   "rg#f5713de20cde511e8ba4900#pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41",
			expected1: "rg",
			expected2: "f5713de20cde511e8ba4900",
			expected3: "pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41",
			expected4: nil,
		},
		{
			options:   "rg#f5713de20cde511e8ba4900",
			expected1: "",
			expected2: "",
			expected3: "",
			expected4: fmt.Errorf("error parsing volume id: \"rg#f5713de20cde511e8ba4900\", should at least contain two #"),
		},
		{
			options:   "rg",
			expected1: "",
			expected2: "",
			expected3: "",
			expected4: fmt.Errorf("error parsing volume id: \"rg\", should at least contain two #"),
		},
		{
			options:   "",
			expected1: "",
			expected2: "",
			expected3: "",
			expected4: fmt.Errorf("error parsing volume id: \"\", should at least contain two #"),
		},
	}

	for _, test := range tests {
		result1, result2, result3, result4 := getFileShareInfo(test.options)
		if !reflect.DeepEqual(result1, test.expected1) || !reflect.DeepEqual(result2, test.expected2) ||
			!reflect.DeepEqual(result3, test.expected3) || !reflect.DeepEqual(result4, test.expected4) {
			t.Errorf("input: %q, getFileShareInfo result1: %q, expected1: %q, result2: %q, expected2: %q, result3: %q, expected3: %q, result4: %q, expected4: %q", test.options, result1, test.expected1, result2, test.expected2,
				result3, test.expected3, result4, test.expected4)
		}
	}
}

func TestTranslateAzureFileInTreeStorageClassToCSI(t *testing.T) {
	translator := NewAzureFileCSITranslator()

	cases := []struct {
		name   string
		volume *corev1.Volume
		expVol *corev1.PersistentVolume
		expErr bool
	}{
		{
			name:   "empty volume",
			expErr: true,
		},
		{
			name:   "no azure file volume",
			volume: &corev1.Volume{},
			expErr: true,
		},
		{
			name: "azure file volume",
			volume: &corev1.Volume{
				VolumeSource: corev1.VolumeSource{
					AzureFile: &corev1.AzureFileVolumeSource{
						ReadOnly:   true,
						SecretName: "secretname",
						ShareName:  "sharename",
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "file.csi.azure.com-sharename",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver: "file.csi.azure.com",
							NodePublishSecretRef: &corev1.SecretReference{
								Name:      "sharename",
								Namespace: "default",
							},
							ReadOnly:         true,
							VolumeAttributes: map[string]string{azureFileShareName: "sharename"},
							VolumeHandle:     "#secretname#sharename",
						},
					},
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreeInlineVolumeToCSI(tc.volume)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}

		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}

		if !reflect.DeepEqual(got, tc.expVol) {
			t.Errorf("Got parameters: %v, expected :%v", got, tc.expVol)
		}
	}
}

func TestTranslateAzureFileInTreePVToCSI(t *testing.T) {
	translator := NewAzureFileCSITranslator()

	secretNamespace := "secretnamespace"

	cases := []struct {
		name   string
		volume *corev1.PersistentVolume
		expVol *corev1.PersistentVolume
		expErr bool
	}{
		{
			name:   "empty volume",
			expErr: true,
		},
		{
			name:   "no azure file volume",
			volume: &corev1.PersistentVolume{},
			expErr: true,
		},
		{
			name: "azure file volume",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "file.csi.azure.com-sharename",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							ShareName:       "sharename",
							SecretName:      "secretname",
							SecretNamespace: &secretNamespace,
							ReadOnly:        true,
						},
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "file.csi.azure.com-sharename",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver:   "file.csi.azure.com",
							ReadOnly: true,
							NodePublishSecretRef: &corev1.SecretReference{
								Name:      "sharename",
								Namespace: secretNamespace,
							},
							VolumeAttributes: map[string]string{azureFileShareName: "sharename"},
							VolumeHandle:     "#secretname#sharename",
						},
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreePVToCSI(tc.volume)
		if err != nil && !tc.expErr {
			t.Errorf("Did not expect error but got: %v", err)
		}

		if err == nil && tc.expErr {
			t.Errorf("Expected error, but did not get one.")
		}

		if !reflect.DeepEqual(got, tc.expVol) {
			t.Errorf("Got parameters: %v, expected :%v", got, tc.expVol)
		}
	}
}
