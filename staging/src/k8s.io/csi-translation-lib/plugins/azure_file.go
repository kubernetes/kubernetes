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
	"regexp"
	"strings"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

const (
	// AzureFileDriverName is the name of the CSI driver for Azure File
	AzureFileDriverName = "file.csi.azure.com"
	// AzureFileInTreePluginName is the name of the intree plugin for Azure file
	AzureFileInTreePluginName = "kubernetes.io/azure-file"

	separator        = "#"
	volumeIDTemplate = "%s#%s#%s#%s#%s"
	// Parameter names defined in azure file CSI driver, refer to
	// https://github.com/kubernetes-sigs/azurefile-csi-driver/blob/master/docs/driver-parameters.md
	shareNameField          = "sharename"
	secretNameField         = "secretname"
	secretNamespaceField    = "secretnamespace"
	secretNameTemplate      = "azure-storage-account-%s-secret"
	defaultSecretNamespace  = "default"
	resourceGroupAnnotation = "kubernetes.io/azure-file-resource-group"
)

var _ InTreePlugin = &azureFileCSITranslator{}

var secretNameFormatRE = regexp.MustCompile(`azure-storage-account-(.+)-secret`)

// azureFileCSITranslator handles translation of PV spec from In-tree
// Azure File to CSI Azure File and vice versa
type azureFileCSITranslator struct{}

// NewAzureFileCSITranslator returns a new instance of azureFileTranslator
func NewAzureFileCSITranslator() InTreePlugin {
	return &azureFileCSITranslator{}
}

// TranslateInTreeStorageClassToCSI translates InTree Azure File storage class parameters to CSI storage class
func (t *azureFileCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with AzureFile set from in-tree
// and converts the AzureFile source to a CSIPersistentVolumeSource
func (t *azureFileCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.AzureFile == nil {
		return nil, fmt.Errorf("volume is nil or Azure File not defined on volume")
	}

	azureSource := volume.AzureFile
	accountName, err := getStorageAccountName(azureSource.SecretName)
	if err != nil {
		klog.Warningf("getStorageAccountName(%s) returned with error: %v", azureSource.SecretName, err)
		accountName = azureSource.SecretName
	}

	secretNamespace := defaultSecretNamespace
	if podNamespace != "" {
		secretNamespace = podNamespace
	}
	volumeID := fmt.Sprintf(volumeIDTemplate, "", accountName, azureSource.ShareName, volume.Name, secretNamespace)

	var (
		pv = &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				// Must be unique as it is used as the unique part of the staging path
				Name: volumeID,
			},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					CSI: &v1.CSIPersistentVolumeSource{
						Driver:           AzureFileDriverName,
						VolumeHandle:     volumeID,
						ReadOnly:         azureSource.ReadOnly,
						VolumeAttributes: map[string]string{shareNameField: azureSource.ShareName},
						NodeStageSecretRef: &v1.SecretReference{
							Name:      azureSource.SecretName,
							Namespace: secretNamespace,
						},
					},
				},
				AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
			},
		}
	)

	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with AzureFile set from in-tree
// and converts the AzureFile source to a CSIPersistentVolumeSource
func (t *azureFileCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.AzureFile == nil {
		return nil, fmt.Errorf("pv is nil or Azure File source not defined on pv")
	}

	azureSource := pv.Spec.PersistentVolumeSource.AzureFile
	accountName, err := getStorageAccountName(azureSource.SecretName)
	if err != nil {
		klog.Warningf("getStorageAccountName(%s) returned with error: %v", azureSource.SecretName, err)
		accountName = azureSource.SecretName
	}
	resourceGroup := ""
	if pv.ObjectMeta.Annotations != nil {
		if v, ok := pv.ObjectMeta.Annotations[resourceGroupAnnotation]; ok {
			resourceGroup = v
		}
	}

	// Secret is required when mounting a volume but pod presence cannot be assumed - we should not try to read pod now.
	namespace := ""
	// Try to read SecretNamespace from source pv.
	if azureSource.SecretNamespace != nil {
		namespace = *azureSource.SecretNamespace
	} else {
		// Try to read namespace from ClaimRef which should be always present.
		if pv.Spec.ClaimRef != nil {
			namespace = pv.Spec.ClaimRef.Namespace
		}
	}

	if len(namespace) == 0 {
		return nil, fmt.Errorf("could not find a secret namespace in PersistentVolumeSource or ClaimRef")
	}

	volumeID := fmt.Sprintf(volumeIDTemplate, resourceGroup, accountName, azureSource.ShareName, pv.ObjectMeta.Name, namespace)

	var (
		// refer to https://github.com/kubernetes-sigs/azurefile-csi-driver/blob/master/docs/driver-parameters.md
		csiSource = &v1.CSIPersistentVolumeSource{
			Driver: AzureFileDriverName,
			NodeStageSecretRef: &v1.SecretReference{
				Name:      azureSource.SecretName,
				Namespace: namespace,
			},
			ReadOnly:         azureSource.ReadOnly,
			VolumeAttributes: map[string]string{shareNameField: azureSource.ShareName},
			VolumeHandle:     volumeID,
		}
	)

	pv.Spec.PersistentVolumeSource.AzureFile = nil
	pv.Spec.PersistentVolumeSource.CSI = csiSource

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

	for k, v := range csiSource.VolumeAttributes {
		switch strings.ToLower(k) {
		case shareNameField:
			azureSource.ShareName = v
		case secretNameField:
			azureSource.SecretName = v
		case secretNamespaceField:
			ns := v
			azureSource.SecretNamespace = &ns
		}
	}

	resourceGroup := ""
	if csiSource.NodeStageSecretRef != nil && csiSource.NodeStageSecretRef.Name != "" {
		azureSource.SecretName = csiSource.NodeStageSecretRef.Name
		azureSource.SecretNamespace = &csiSource.NodeStageSecretRef.Namespace
	}
	if azureSource.ShareName == "" || azureSource.SecretName == "" {
		rg, storageAccount, fileShareName, _, err := getFileShareInfo(csiSource.VolumeHandle)
		if err != nil {
			return nil, err
		}
		if azureSource.ShareName == "" {
			azureSource.ShareName = fileShareName
		}
		if azureSource.SecretName == "" {
			azureSource.SecretName = fmt.Sprintf(secretNameTemplate, storageAccount)
		}
		resourceGroup = rg
	}

	if azureSource.SecretNamespace == nil {
		ns := defaultSecretNamespace
		azureSource.SecretNamespace = &ns
	}

	pv.Spec.CSI = nil
	pv.Spec.AzureFile = azureSource
	if pv.ObjectMeta.Annotations == nil {
		pv.ObjectMeta.Annotations = map[string]string{}
	}
	if resourceGroup != "" {
		pv.ObjectMeta.Annotations[resourceGroupAnnotation] = resourceGroup
	}

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
// input: "rg#f5713de20cde511e8ba4900#pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41#diskname.vhd"
// output: rg, f5713de20cde511e8ba4900, pvc-file-dynamic-17e43f84-f474-11e8-acd0-000d3a00df41, diskname.vhd
func getFileShareInfo(id string) (string, string, string, string, error) {
	segments := strings.Split(id, separator)
	if len(segments) < 3 {
		return "", "", "", "", fmt.Errorf("error parsing volume id: %q, should at least contain two #", id)
	}
	var diskName string
	if len(segments) > 3 {
		diskName = segments[3]
	}
	return segments[0], segments[1], segments[2], diskName, nil
}

// get storage account name from secret name
func getStorageAccountName(secretName string) (string, error) {
	matches := secretNameFormatRE.FindStringSubmatch(secretName)
	if len(matches) != 2 {
		return "", fmt.Errorf("could not get account name from %s, correct format: %s", secretName, secretNameFormatRE)
	}
	return matches[1], nil
}
