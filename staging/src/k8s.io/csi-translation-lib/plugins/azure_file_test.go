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
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"

	"github.com/stretchr/testify/assert"
)

func TestGetFileShareInfo(t *testing.T) {
	tests := []struct {
		id                string
		resourceGroupName string
		accountName       string
		fileShareName     string
		diskName          string
		expectedError     error
	}{
		{
			id:                "rg#f5713de20cde511e8ba4900#pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41#diskname.vhd",
			resourceGroupName: "rg",
			accountName:       "f5713de20cde511e8ba4900",
			fileShareName:     "pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41",
			diskName:          "diskname.vhd",
			expectedError:     nil,
		},
		{
			id:                "rg#f5713de20cde511e8ba4900#pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41",
			resourceGroupName: "rg",
			accountName:       "f5713de20cde511e8ba4900",
			fileShareName:     "pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41",
			diskName:          "",
			expectedError:     nil,
		},
		{
			id:                "rg#f5713de20cde511e8ba4900",
			resourceGroupName: "",
			accountName:       "",
			fileShareName:     "",
			diskName:          "",
			expectedError:     fmt.Errorf("error parsing volume id: \"rg#f5713de20cde511e8ba4900\", should at least contain two #"),
		},
		{
			id:                "rg",
			resourceGroupName: "",
			accountName:       "",
			fileShareName:     "",
			diskName:          "",
			expectedError:     fmt.Errorf("error parsing volume id: \"rg\", should at least contain two #"),
		},
		{
			id:                "",
			resourceGroupName: "",
			accountName:       "",
			fileShareName:     "",
			diskName:          "",
			expectedError:     fmt.Errorf("error parsing volume id: \"\", should at least contain two #"),
		},
	}

	for _, test := range tests {
		resourceGroupName, accountName, fileShareName, diskName, expectedError := getFileShareInfo(test.id)
		if resourceGroupName != test.resourceGroupName {
			t.Errorf("getFileShareInfo(%q) returned with: %q, expected: %q", test.id, resourceGroupName, test.resourceGroupName)
		}
		if accountName != test.accountName {
			t.Errorf("getFileShareInfo(%q) returned with: %q, expected: %q", test.id, accountName, test.accountName)
		}
		if fileShareName != test.fileShareName {
			t.Errorf("getFileShareInfo(%q) returned with: %q, expected: %q", test.id, fileShareName, test.fileShareName)
		}
		if diskName != test.diskName {
			t.Errorf("getFileShareInfo(%q) returned with: %q, expected: %q", test.id, diskName, test.diskName)
		}
		if !reflect.DeepEqual(expectedError, test.expectedError) {
			t.Errorf("getFileShareInfo(%q) returned with: %v, expected: %v", test.id, expectedError, test.expectedError)
		}
	}
}

func TestTranslateAzureFileInTreeStorageClassToCSI(t *testing.T) {
	translator := NewAzureFileCSITranslator()
	logger, _ := ktesting.NewTestContext(t)

	cases := []struct {
		name         string
		volume       *corev1.Volume
		podNamespace string
		expVol       *corev1.PersistentVolume
		expErr       bool
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
				Name: "name",
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
					Name: "#secretname#sharename#name#default",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver: "file.csi.azure.com",
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      "secretname",
								Namespace: "default",
							},
							ReadOnly:         true,
							VolumeAttributes: map[string]string{shareNameField: "sharename"},
							VolumeHandle:     "#secretname#sharename#name#default",
						},
					},
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
				},
			},
		},
		{
			name: "azure file volume with a pod namespace",
			volume: &corev1.Volume{
				Name: "name",
				VolumeSource: corev1.VolumeSource{
					AzureFile: &corev1.AzureFileVolumeSource{
						ReadOnly:   true,
						SecretName: "secretname",
						ShareName:  "sharename",
					},
				},
			},
			podNamespace: "test",
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "#secretname#sharename#name#test",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver: "file.csi.azure.com",
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      "secretname",
								Namespace: "test",
							},
							ReadOnly:         true,
							VolumeAttributes: map[string]string{shareNameField: "sharename"},
							VolumeHandle:     "#secretname#sharename#name#test",
						},
					},
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreeInlineVolumeToCSI(logger, tc.volume, tc.podNamespace)
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
	logger, _ := ktesting.NewTestContext(t)

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
			name: "return error if secret namespace could not be found",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "uuid",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							ShareName:  "sharename",
							SecretName: "secretname",
							ReadOnly:   true,
						},
					},
				},
			},
			expErr: true,
		},
		{
			name: "azure file volume",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "uuid",
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
					Name: "uuid",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver:   "file.csi.azure.com",
							ReadOnly: true,
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      "secretname",
								Namespace: secretNamespace,
							},
							VolumeAttributes: map[string]string{shareNameField: "sharename"},
							VolumeHandle:     "#secretname#sharename#uuid#secretnamespace",
						},
					},
				},
			},
		},
		{
			name: "azure file volume with rg annotation",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "uuid",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
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
					Name:        "uuid",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver:   "file.csi.azure.com",
							ReadOnly: true,
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      "secretname",
								Namespace: secretNamespace,
							},
							VolumeAttributes: map[string]string{shareNameField: "sharename"},
							VolumeHandle:     "rg#secretname#sharename#uuid#secretnamespace",
						},
					},
				},
			},
		},
		{
			name: "get secret namespace from ClaimRef when it's missing in pv spec source",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "uuid",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							ShareName:  "sharename",
							SecretName: "secretname",
							//SecretNamespace: &secretNamespace,
							ReadOnly: true,
						},
					},
					ClaimRef: &corev1.ObjectReference{
						Namespace: secretNamespace,
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "uuid",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							Driver:   "file.csi.azure.com",
							ReadOnly: true,
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      "secretname",
								Namespace: secretNamespace,
							},
							VolumeAttributes: map[string]string{shareNameField: "sharename"},
							VolumeHandle:     "rg#secretname#sharename#uuid#secretnamespace",
						},
					},
					ClaimRef: &corev1.ObjectReference{
						Namespace: secretNamespace,
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateInTreePVToCSI(logger, tc.volume)
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

func TestTranslateCSIPVToInTree(t *testing.T) {
	translator := NewAzureFileCSITranslator()

	secretName := "secretname"
	secretNamespace := "secretnamespace"
	shareName := "sharename"
	defaultNS := "default"
	mp := make(map[string]string)
	mp["shareName"] = shareName

	secretMap := make(map[string]string)
	secretMap["shareName"] = shareName
	secretMap["secretName"] = secretName
	secretMap["secretNamespace"] = secretNamespace

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
			name: "resource group empty",
			volume: &corev1.PersistentVolume{
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      "ut",
								Namespace: secretNamespace,
							},
							ReadOnly:         true,
							VolumeAttributes: mp,
						},
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							SecretName:      "ut",
							SecretNamespace: &secretNamespace,
							ReadOnly:        true,
							ShareName:       shareName,
						},
					},
				},
			},
			expErr: false,
		},
		{
			name: "translate from volume handle error",
			volume: &corev1.PersistentVolume{
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							VolumeHandle:     shareName,
							ReadOnly:         true,
							VolumeAttributes: mp,
						},
					},
				},
			},
			expErr: true,
		},
		{
			name: "translate from VolumeAttributes",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "file.csi.azure.com-sharename",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							VolumeHandle:     "rg#st#pvc-file-dynamic#diskname.vhd",
							ReadOnly:         true,
							VolumeAttributes: mp,
						},
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "file.csi.azure.com-sharename",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							SecretName:      "azure-storage-account-st-secret",
							ShareName:       shareName,
							SecretNamespace: &defaultNS,
							ReadOnly:        true,
						},
					},
				},
			},
			expErr: false,
		},
		{
			name: "translate from SecretMap VolumeAttributes",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "file.csi.azure.com-sharename",
					Annotations: map[string]string{},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							VolumeHandle:     "rg#st#pvc-file-dynamic#diskname.vhd",
							ReadOnly:         true,
							VolumeAttributes: secretMap,
						},
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "file.csi.azure.com-sharename",
					Annotations: map[string]string{},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							SecretName:      secretName,
							SecretNamespace: &secretNamespace,
							ShareName:       shareName,
							ReadOnly:        true,
						},
					},
				},
			},
			expErr: false,
		},
		{
			name: "translate from NodeStageSecretRef",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "file.csi.azure.com-sharename",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							VolumeHandle:     "rg#st#pvc-file-dynamic#diskname.vhd",
							ReadOnly:         true,
							VolumeAttributes: mp,
							NodeStageSecretRef: &corev1.SecretReference{
								Name:      secretName,
								Namespace: secretNamespace,
							},
						},
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "file.csi.azure.com-sharename",
					Annotations: map[string]string{},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							SecretName:      secretName,
							ShareName:       shareName,
							SecretNamespace: &secretNamespace,
							ReadOnly:        true,
						},
					},
				},
			},
			expErr: false,
		},
		{
			name: "translate from VolumeHandle",
			volume: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "file.csi.azure.com-sharename",
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						CSI: &corev1.CSIPersistentVolumeSource{
							VolumeHandle: "rg#st#pvc-file-dynamic#diskname.vhd",
							ReadOnly:     true,
						},
					},
				},
			},
			expVol: &corev1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "file.csi.azure.com-sharename",
					Annotations: map[string]string{resourceGroupAnnotation: "rg"},
				},
				Spec: corev1.PersistentVolumeSpec{
					PersistentVolumeSource: corev1.PersistentVolumeSource{
						AzureFile: &corev1.AzureFilePersistentVolumeSource{
							SecretName:      "azure-storage-account-st-secret",
							ShareName:       "pvc-file-dynamic",
							SecretNamespace: &defaultNS,
							ReadOnly:        true,
						},
					},
				},
			},
			expErr: false,
		},
	}

	for _, tc := range cases {
		t.Logf("Testing %v", tc.name)
		got, err := translator.TranslateCSIPVToInTree(tc.volume)
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

func TestGetStorageAccount(t *testing.T) {
	tests := []struct {
		secretName     string
		expectedError  bool
		expectedResult string
	}{
		{
			secretName:     "azure-storage-account-accountname-secret",
			expectedError:  false,
			expectedResult: "accountname",
		},
		{
			secretName:     "azure-storage-account-accountname-dup-secret",
			expectedError:  false,
			expectedResult: "accountname-dup",
		},
		{
			secretName:     "invalid",
			expectedError:  true,
			expectedResult: "",
		},
	}

	for i, test := range tests {
		accountName, err := getStorageAccountName(test.secretName)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]", i)
		assert.Equal(t, test.expectedResult, accountName, "TestCase[%d]", i)
	}
}
