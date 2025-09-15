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
	// AzureDiskDriverName is the name of the CSI driver for Azure Disk
	AzureDiskDriverName = "disk.csi.azure.com"
	// AzureDiskTopologyKey is the topology key of Azure Disk CSI driver
	AzureDiskTopologyKey = "topology.disk.csi.azure.com/zone"
	// AzureDiskInTreePluginName is the name of the intree plugin for Azure Disk
	AzureDiskInTreePluginName = "kubernetes.io/azure-disk"

	// Parameter names defined in azure disk CSI driver, refer to
	// https://github.com/kubernetes-sigs/azuredisk-csi-driver/blob/master/docs/driver-parameters.md
	azureDiskKind        = "kind"
	azureDiskCachingMode = "cachingmode"
	azureDiskFSType      = "fstype"
)

var (
	managedDiskPathRE   = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(?:.*)/providers/Microsoft.Compute/disks/(.+)`)
	unmanagedDiskPathRE = regexp.MustCompile(`http(?:.*)://(?:.*)/vhds/(.+)`)
	managed             = string(v1.AzureManagedDisk)
	unzonedCSIRegionRE  = regexp.MustCompile(`^[0-9]+$`)
)

var _ InTreePlugin = &azureDiskCSITranslator{}

// azureDiskCSITranslator handles translation of PV spec from In-tree
// Azure Disk to CSI Azure Disk and vice versa
type azureDiskCSITranslator struct{}

// NewAzureDiskCSITranslator returns a new instance of azureDiskTranslator
func NewAzureDiskCSITranslator() InTreePlugin {
	return &azureDiskCSITranslator{}
}

// TranslateInTreeStorageClassToCSI translates InTree Azure Disk storage class parameters to CSI storage class
func (t *azureDiskCSITranslator) TranslateInTreeStorageClassToCSI(logger klog.Logger, sc *storage.StorageClass) (*storage.StorageClass, error) {
	var (
		generatedTopologies []v1.TopologySelectorTerm
		params              = map[string]string{}
	)
	for k, v := range sc.Parameters {
		switch strings.ToLower(k) {
		case zoneKey:
			generatedTopologies = generateToplogySelectors(AzureDiskTopologyKey, []string{v})
		case zonesKey:
			generatedTopologies = generateToplogySelectors(AzureDiskTopologyKey, strings.Split(v, ","))
		default:
			params[k] = v
		}
	}

	if len(generatedTopologies) > 0 && len(sc.AllowedTopologies) > 0 {
		return nil, fmt.Errorf("cannot simultaneously set allowed topologies and zone/zones parameters")
	} else if len(generatedTopologies) > 0 {
		sc.AllowedTopologies = generatedTopologies
	} else if len(sc.AllowedTopologies) > 0 {
		newTopologies, err := translateAllowedTopologies(sc.AllowedTopologies, AzureDiskTopologyKey)
		if err != nil {
			return nil, fmt.Errorf("failed translating allowed topologies: %v", err)
		}
		sc.AllowedTopologies = newTopologies
	}
	sc.AllowedTopologies = t.replaceFailureDomainsToCSI(sc.AllowedTopologies)

	sc.Parameters = params

	return sc, nil
}

// TranslateInTreeInlineVolumeToCSI takes a Volume with AzureDisk set from in-tree
// and converts the AzureDisk source to a CSIPersistentVolumeSource
func (t *azureDiskCSITranslator) TranslateInTreeInlineVolumeToCSI(logger klog.Logger, volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil || volume.AzureDisk == nil {
		return nil, fmt.Errorf("volume is nil or Azure Disk not defined on volume")
	}

	azureSource := volume.AzureDisk
	if azureSource.Kind != nil && !strings.EqualFold(string(*azureSource.Kind), managed) {
		return nil, fmt.Errorf("kind(%v) is not supported in csi migration", *azureSource.Kind)
	}
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			// Must be unique per disk as it is used as the unique part of the
			// staging path
			Name: azureSource.DataDiskURI,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:           AzureDiskDriverName,
					VolumeHandle:     azureSource.DataDiskURI,
					VolumeAttributes: map[string]string{azureDiskKind: managed},
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
	if azureSource.ReadOnly != nil {
		pv.Spec.PersistentVolumeSource.CSI.ReadOnly = *azureSource.ReadOnly
	}

	if azureSource.CachingMode != nil && *azureSource.CachingMode != "" {
		pv.Spec.PersistentVolumeSource.CSI.VolumeAttributes[azureDiskCachingMode] = string(*azureSource.CachingMode)
	}
	if azureSource.FSType != nil {
		pv.Spec.PersistentVolumeSource.CSI.FSType = *azureSource.FSType
		pv.Spec.PersistentVolumeSource.CSI.VolumeAttributes[azureDiskFSType] = *azureSource.FSType
	}
	pv.Spec.PersistentVolumeSource.CSI.VolumeAttributes[azureDiskKind] = managed

	return pv, nil
}

// TranslateInTreePVToCSI takes a PV with AzureDisk set from in-tree
// and converts the AzureDisk source to a CSIPersistentVolumeSource
func (t *azureDiskCSITranslator) TranslateInTreePVToCSI(logger klog.Logger, pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.AzureDisk == nil {
		return nil, fmt.Errorf("pv is nil or Azure Disk source not defined on pv")
	}

	var (
		azureSource = pv.Spec.PersistentVolumeSource.AzureDisk

		// refer to https://github.com/kubernetes-sigs/azuredisk-csi-driver/blob/master/docs/driver-parameters.md
		csiSource = &v1.CSIPersistentVolumeSource{
			Driver:           AzureDiskDriverName,
			VolumeAttributes: map[string]string{azureDiskKind: managed},
			VolumeHandle:     azureSource.DataDiskURI,
		}
	)

	if azureSource.Kind != nil && !strings.EqualFold(string(*azureSource.Kind), managed) {
		return nil, fmt.Errorf("kind(%v) is not supported in csi migration", *azureSource.Kind)
	}

	if azureSource.CachingMode != nil {
		csiSource.VolumeAttributes[azureDiskCachingMode] = string(*azureSource.CachingMode)
	}

	if azureSource.FSType != nil {
		csiSource.FSType = *azureSource.FSType
		csiSource.VolumeAttributes[azureDiskFSType] = *azureSource.FSType
	}
	csiSource.VolumeAttributes[azureDiskKind] = managed

	if azureSource.ReadOnly != nil {
		csiSource.ReadOnly = *azureSource.ReadOnly
	}

	pv.Spec.PersistentVolumeSource.AzureDisk = nil
	pv.Spec.PersistentVolumeSource.CSI = csiSource

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
	managed := v1.AzureManagedDisk
	azureSource := &v1.AzureDiskVolumeSource{
		DiskName:    diskName,
		DataDiskURI: diskURI,
		FSType:      &csiSource.FSType,
		ReadOnly:    &csiSource.ReadOnly,
		Kind:        &managed,
	}

	if csiSource.VolumeAttributes != nil {
		for k, v := range csiSource.VolumeAttributes {
			switch strings.ToLower(k) {
			case azureDiskCachingMode:
				if v != "" {
					mode := v1.AzureDataDiskCachingMode(v)
					azureSource.CachingMode = &mode
				}
			case azureDiskFSType:
				if v != "" {
					fsType := v
					azureSource.FSType = &fsType
				}
			}
		}
		azureSource.Kind = &managed
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

func (t *azureDiskCSITranslator) RepairVolumeHandle(volumeHandle, nodeID string) (string, error) {
	return volumeHandle, nil
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

// Replace topology values for failure domains ("<number>") to "",
// as it's the value that the CSI driver expects.
func (t *azureDiskCSITranslator) replaceFailureDomainsToCSI(terms []v1.TopologySelectorTerm) []v1.TopologySelectorTerm {
	if terms == nil {
		return nil
	}

	newTopologies := []v1.TopologySelectorTerm{}
	for _, term := range terms {
		newTerm := term.DeepCopy()
		for i := range newTerm.MatchLabelExpressions {
			exp := &newTerm.MatchLabelExpressions[i]
			if exp.Key == AzureDiskTopologyKey {
				for j := range exp.Values {
					if unzonedCSIRegionRE.Match([]byte(exp.Values[j])) {
						// Topologies "0", "1" etc are used when in-tree topology is translated to CSI in Azure
						// regions that don't have availability zones. E.g.:
						//    topology.kubernetes.io/region: westus
						//    topology.kubernetes.io/zone: "0"
						// The CSI driver uses zone "" instead of "0" in this case.
						//    topology.disk.csi.azure.com/zone": ""
						exp.Values[j] = ""
					}
				}
			}
		}
		newTopologies = append(newTopologies, *newTerm)
	}
	return newTopologies
}
