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

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	// AzureDiskDriverName is the name of the CSI driver for Azure Disk
	AzureDiskDriverName = "disk.csi.azure.com"
	// AzureDiskInTreePluginName is the name of the intree plugin for Azure Disk
	AzureDiskInTreePluginName = "kubernetes.io/azure-disk"

	// Parameter names defined in azure disk CSI driver, refer to
	// https://github.com/kubernetes-sigs/azuredisk-csi-driver/blob/master/docs/driver-parameters.md
	azureDiskCachingMode = "cachingMode"
	azureDiskFSType      = "fsType"
)

var (
	managedDiskPathRE   = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(?:.*)/providers/Microsoft.Compute/disks/(.+)`)
	unmanagedDiskPathRE = regexp.MustCompile(`http(?:.*)://(?:.*)/vhds/(.+)`)
)

var _ InTreePlugin = &azureDiskCSITranslator{}

// azureDiskCSITranslator handles translation of PV spec from In-tree
// Azure Disk to CSI Azure Disk and vice versa
type azureDiskCSITranslator struct{}

// NewAzureDiskCSITranslator returns a new instance of azureDiskTranslator
func NewAzureDiskCSITranslator() InTreePlugin {
	return &azureDiskCSITranslator{}
}

// TranslateInTreeStorageClassParametersToCSI translates InTree Azure Disk storage class parameters to CSI storage class
func (t *azureDiskCSITranslator) TranslateInTreeStorageClassToCSI(sc *storage.StorageClass) (*storage.StorageClass, error) {
	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with AzureDisk set from in-tree
// and converts the AzureDisk source to a CSIPersistentVolumeSource
func (t *azureDiskCSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume) (*v1.PersistentVolume, error) {
	if volume == nil || volume.AzureDisk == nil {
		return nil, fmt.Errorf("volume is nil or Azure Disk not defined on volume")
	}

	azureSource := volume.AzureDisk
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			// A.K.A InnerVolumeSpecName required to match for Unmount
			Name: volume.Name,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:           AzureDiskDriverName,
					VolumeHandle:     azureSource.DataDiskURI,
					ReadOnly:         *azureSource.ReadOnly,
					FSType:           *azureSource.FSType,
					VolumeAttributes: map[string]string{},
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}

	if *azureSource.CachingMode != "" {
		pv.Spec.PersistentVolumeSource.CSI.VolumeAttributes[azureDiskCachingMode] = string(*azureSource.CachingMode)
	}
	if *azureSource.FSType != "" {
		pv.Spec.PersistentVolumeSource.CSI.VolumeAttributes[azureDiskFSType] = *azureSource.FSType
	}

	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with AzureDisk set from in-tree
// and converts the AzureDisk source to a CSIPersistentVolumeSource
func (t *azureDiskCSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.AzureDisk == nil {
		return nil, fmt.Errorf("pv is nil or Azure Disk source not defined on pv")
	}

	azureSource := pv.Spec.PersistentVolumeSource.AzureDisk

	// refer to https://github.com/kubernetes-sigs/azuredisk-csi-driver/blob/master/docs/driver-parameters.md
	csiSource := &v1.CSIPersistentVolumeSource{
		Driver:           AzureDiskDriverName,
		VolumeHandle:     azureSource.DataDiskURI,
		ReadOnly:         *azureSource.ReadOnly,
		FSType:           *azureSource.FSType,
		VolumeAttributes: map[string]string{},
	}

	if *azureSource.CachingMode != "" {
		csiSource.VolumeAttributes[azureDiskCachingMode] = string(*azureSource.CachingMode)
	}

	if *azureSource.FSType != "" {
		csiSource.VolumeAttributes[azureDiskFSType] = *azureSource.FSType
	}

	pv.Spec.PersistentVolumeSource.AzureDisk = nil
	pv.Spec.PersistentVolumeSource.CSI = csiSource
	pv.Spec.AccessModes = backwardCompatibleAccessModes(pv.Spec.AccessModes)

	return pv, nil
}

// TranslateCSIPVToInTree takes a PV with CSIPersistentVolumeSource set and
// translates the Azure Disk CSI source to a AzureDisk source.
func (t *azureDiskCSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, fmt.Errorf("pv is nil or CSI source not defined on pv")
	}
	csiSource := pv.Spec.CSI

	diskURI := csiSource.VolumeHandle
	diskName, err := getDiskName(diskURI)
	if err != nil {
		return nil, err
	}

	// refer to https://github.com/kubernetes-sigs/azuredisk-csi-driver/blob/master/docs/driver-parameters.md
	azureSource := &v1.AzureDiskVolumeSource{
		DiskName:    diskName,
		DataDiskURI: diskURI,
		FSType:      &csiSource.FSType,
		ReadOnly:    &csiSource.ReadOnly,
	}

	if csiSource.VolumeAttributes != nil {
		if cachingMode, ok := csiSource.VolumeAttributes[azureDiskCachingMode]; ok {
			mode := v1.AzureDataDiskCachingMode(cachingMode)
			azureSource.CachingMode = &mode
		}

		if fsType, ok := csiSource.VolumeAttributes[azureDiskFSType]; ok && fsType != "" {
			azureSource.FSType = &fsType
		}
	}

	pv.Spec.CSI = nil
	pv.Spec.AzureDisk = azureSource

	return pv, nil
}

// CanSupport tests whether the plugin supports a given volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *azureDiskCSITranslator) CanSupport(pv *v1.PersistentVolume) bool {
	return pv != nil && pv.Spec.AzureDisk != nil
}

// CanSupportInline tests whether the plugin supports a given inline volume
// specification from the API.  The spec pointer should be considered
// const.
func (t *azureDiskCSITranslator) CanSupportInline(volume *v1.Volume) bool {
	return volume != nil && volume.AzureDisk != nil
}

// GetInTreePluginName returns the name of the intree plugin driver
func (t *azureDiskCSITranslator) GetInTreePluginName() string {
	return AzureDiskInTreePluginName
}

// GetCSIPluginName returns the name of the CSI plugin
func (t *azureDiskCSITranslator) GetCSIPluginName() string {
	return AzureDiskDriverName
}

func isManagedDisk(diskURI string) bool {
	if len(diskURI) > 4 && strings.ToLower(diskURI[:4]) == "http" {
		return false
	}
	return true
}

func getDiskName(diskURI string) (string, error) {
	diskPathRE := managedDiskPathRE
	if !isManagedDisk(diskURI) {
		diskPathRE = unmanagedDiskPathRE
	}

	matches := diskPathRE.FindStringSubmatch(diskURI)
	if len(matches) != 2 {
		return "", fmt.Errorf("could not get disk name from %s, correct format: %s", diskURI, diskPathRE)
	}
	return matches[1], nil
}
