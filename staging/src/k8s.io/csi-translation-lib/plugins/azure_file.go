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
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// AzureFileDriverName is the name of the CSI driver for Azure File
	AzureFileDriverName = "file.csi.azure.com"
	// AzureFileInTreePluginName is the name of the intree plugin for Azure file
	AzureFileInTreePluginName = "kubernetes.io/azure-file"

	separator        = "#"
	volumeIDTemplate = "%s#%s#%s"
	// Parameter names defined in azure file CSI driver, refer to
	// https://github.com/kubernetes-sigs/azurefile-csi-driver/blob/master/docs/driver-parameters.md
	azureFileShareName = "shareName"
)

var _ InTreePlugin = &azureFileCSITranslator{}

// azureFileCSITranslator handles translation of PV spec from In-tree
// Azure File to CSI Azure File and vice versa
type azureFileCSITranslator struct{}

// NewAzureFileCSITranslator returns a new instance of azureFileTranslator
func NewAzureFileCSITranslator() InTreePlugin {
	return &azureFileCSITranslator{}
}

// TranslateInTreeStorageClassParametersToCSI translates InTree Azure File storage class parameters to CSI storage class
func (t *azureFileCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with AzureFile set from in-tree
// and converts the AzureFile source to a CSIPersistentVolumeSource
func (t *azureFileCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume) (*v1.PersistentVolume, error) {
	if volume == nil || volume.AzureFile == nil {
		return nil, fmt.Errorf("volume is nil or AWS EBS not defined on volume")
	}

	azureSource := volume.AzureFile

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			// Must be unique per disk as it is used as the unique part of the
			// staging path
			Name: fmt.Sprintf("%s-%s", AzureFileDriverName, azureSource.ShareName),
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					VolumeHandle:     fmt.Sprintf(volumeIDTemplate, "", azureSource.SecretName, azureSource.ShareName),
					ReadOnly:         azureSource.ReadOnly,
					VolumeAttributes: map[string]string{azureFileShareName: azureSource.ShareName},
					NodePublishSecretRef: &v1.SecretReference{
						Name:      azureSource.ShareName,
						Namespace: "default",
					},
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
		},
	}
	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with AzureFile set from in-tree
// and converts the AzureFile source to a CSIPersistentVolumeSource
func (t *azureFileCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.AzureFile == nil {
		return nil, fmt.Errorf("pv is nil or Azure File source not defined on pv")
	}

	azureSource := pv.Spec.PersistentVolumeSource.AzureFile

	volumeID := fmt.Sprintf(volumeIDTemplate, "", azureSource.SecretName, azureSource.ShareName)
	// refer to https://github.com/kubernetes-sigs/azurefile-csi-driver/blob/master/docs/driver-parameters.md
	csiSource := &v1.CSIPersistentVolumeSource{
		VolumeHandle:     volumeID,
		ReadOnly:         azureSource.ReadOnly,
		VolumeAttributes: map[string]string{azureFileShareName: azureSource.ShareName},
	}

	csiSource.NodePublishSecretRef = &v1.SecretReference{
		Name:      azureSource.ShareName,
		Namespace: *azureSource.SecretNamespace,
	}

	pv.Spec.PersistentVolumeSource.AzureFile = nil
	pv.Spec.PersistentVolumeSource.CSI = csiSource
	pv.Spec.AccessModes = backwardCompatibleAccessModes(pv.Spec.AccessModes)

	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with CSIPersistentVolumeSource set and
// translates the Azure File CSI source to a AzureFile source.
func (t *azureFileCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	csiSource := pv.Spec.CSI

	// refer to https://github.com/kubernetes-sigs/azurefile-csi-driver/blob/master/docs/driver-parameters.md
	azureSource := &v1.AzureFilePersistentVolumeSource{
		ReadOnly: csiSource.ReadOnly,
	}

	if csiSource.NodePublishSecretRef != nil && csiSource.NodePublishSecretRef.Name != "" {
		azureSource.SecretName = csiSource.NodePublishSecretRef.Name
		azureSource.SecretNamespace = &csiSource.NodePublishSecretRef.Namespace
		if csiSource.VolumeAttributes != nil {
			if shareName, ok := csiSource.VolumeAttributes[azureFileShareName]; ok {
				azureSource.ShareName = shareName
			}
		}
	} else {
		_, _, fileShareName, err := getFileShareInfo(csiSource.VolumeHandle)
		if err != nil {
			return nil, err
		}
		azureSource.ShareName = fileShareName
		// to-do: for dynamic provision scenario in CSI, it uses cluster's identity to get storage account key
		// secret for the file share is not created, we may create a serect here
	}

	pv.Spec.CSI = nil
	pv.Spec.AzureFile = azureSource

	return pv, nil
}

// CanSupport tests whether the plugin supports a given volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *azureFileCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.AzureFile != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *azureFileCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.AzureFile != nil
}

// GetInTreePluginName returns the name of the intree plugin driver
func (t *azureFileCSITranslator) GetInTreePluginName() string {
	return AzureFileInTreePluginName
}

// GetCSIPluginName returns the name of the CSI plugin
func (t *azureFileCSITranslator) GetCSIPluginName() string {
	return AzureFileDriverName
}

func (t *azureFileCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
}

// get file share info according to volume id, e.g.
// input: "rg#f5713de20cde511e8ba4900#pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41"
// output: rg, f5713de20cde511e8ba4900, pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41
func getFileShareInfo(id string) (string, string, string, error) {
	segments := strings.Split(id, separator)
	if len(segments) < 3 {
		return "", "", "", fmt.Errorf("error parsing volume id: %q, should at least contain two #", id)
	}
	return segments[0], segments[1], segments[2], nil
}
