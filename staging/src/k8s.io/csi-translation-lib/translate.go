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

package csitranslation

import (
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/csi-translation-lib/plugins"
)

var (
	inTreePlugins = map[string]plugins.InTreePlugin{
		plugins.GCEPDDriverName:     plugins.NewGCEPersistentDiskCSITranslator(),
		plugins.AWSEBSDriverName:    plugins.NewAWSElasticBlockStoreCSITranslator(),
		plugins.CinderDriverName:    plugins.NewOpenStackCinderCSITranslator(),
		plugins.AzureDiskDriverName: plugins.NewAzureDiskCSITranslator(),
		plugins.AzureFileDriverName: plugins.NewAzureFileCSITranslator(),
		plugins.VSphereDriverName:   plugins.NewvSphereCSITranslator(),
	}
)

// CSITranslator translates in-tree storage API objects to their equivalent CSI
// API objects. It also provides many helper functions to determine whether
// translation logic exists and the mappings between "in-tree plugin <-> csi driver"
type CSITranslator struct{}

// New creates a new CSITranslator which does real translation
// for "in-tree plugins <-> csi drivers"
func New() CSITranslator {
	return CSITranslator{}
}

// TranslateInTreeStorageClassToCSI takes in-tree Storage Class
// and translates it to a set of parameters consumable by CSI plugin
func (CSITranslator) TranslateInTreeStorageClassToCSI(inTreePluginName string, sc *storage.StorageClass) (*storage.StorageClass, error) {
	newSC := sc.DeepCopy()
	for _, curPlugin := range inTreePlugins {
		if inTreePluginName == curPlugin.GetInTreePluginName() {
			return curPlugin.TranslateInTreeStorageClassToCSI(newSC)
		}
	}
	return nil, fmt.Errorf("could not find in-tree storage class parameter translation logic for %#v", inTreePluginName)
}

// TranslateInTreeInlineVolumeToCSI takes a inline volume and will translate
// the in-tree volume source to a CSIPersistentVolumeSource (wrapped in a PV)
// if the translation logic has been implemented.
func (CSITranslator) TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error) {
	if volume == nil {
		return nil, fmt.Errorf("persistent volume was nil")
	}
	for _, curPlugin := range inTreePlugins {
		if curPlugin.CanSupportInline(volume) {
			pv, err := curPlugin.TranslateInTreeInlineVolumeToCSI(volume, podNamespace)
			if err != nil {
				return nil, err
			}
			// Inline volumes only support PersistentVolumeFilesystem (and not block).
			// If VolumeMode has not been set explicitly by plugin-specific
			// translator, set it to Filesystem here.
			// This is only necessary for inline volumes as the default PV
			// initialization that populates VolumeMode does not apply to inline volumes.
			if pv.Spec.VolumeMode == nil {
				volumeMode := v1.PersistentVolumeFilesystem
				pv.Spec.VolumeMode = &volumeMode
			}
			return pv, nil
		}
	}
	return nil, fmt.Errorf("could not find in-tree plugin translation logic for %#v", volume.Name)
}

// TranslateInTreePVToCSI takes a persistent volume and will translate
// the in-tree source to a CSI Source if the translation logic
// has been implemented. The input persistent volume will not
// be modified
func (CSITranslator) TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil {
		return nil, errors.New("persistent volume was nil")
	}
	copiedPV := pv.DeepCopy()
	for _, curPlugin := range inTreePlugins {
		if curPlugin.CanSupport(copiedPV) {
			return curPlugin.TranslateInTreePVToCSI(copiedPV)
		}
	}
	return nil, fmt.Errorf("could not find in-tree plugin translation logic for %#v", copiedPV.Name)
}

// TranslateCSIPVToInTree takes a PV with a CSI PersistentVolume Source and will translate
// it to a in-tree Persistent Volume Source for the specific in-tree volume specified
// by the `Driver` field in the CSI Source. The input PV object will not be modified.
func (CSITranslator) TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	if pv == nil || pv.Spec.CSI == nil {
		return nil, errors.New("CSI persistent volume was nil")
	}
	copiedPV := pv.DeepCopy()
	for driverName, curPlugin := range inTreePlugins {
		if copiedPV.Spec.CSI.Driver == driverName {
			return curPlugin.TranslateCSIPVToInTree(copiedPV)
		}
	}
	return nil, fmt.Errorf("could not find in-tree plugin translation logic for %s", copiedPV.Spec.CSI.Driver)
}

// IsMigratableIntreePluginByName tests whether there is migration logic for the in-tree plugin
// whose name matches the given name
func (CSITranslator) IsMigratableIntreePluginByName(inTreePluginName string) bool {
	for _, curPlugin := range inTreePlugins {
		if curPlugin.GetInTreePluginName() == inTreePluginName {
			return true
		}
	}
	return false
}

// IsMigratedCSIDriverByName tests whether there exists an in-tree plugin with logic
// to migrate to the CSI driver with given name
func (CSITranslator) IsMigratedCSIDriverByName(csiPluginName string) bool {
	if _, ok := inTreePlugins[csiPluginName]; ok {
		return true
	}
	return false
}

// GetInTreePluginNameFromSpec returns the plugin name
func (CSITranslator) GetInTreePluginNameFromSpec(pv *v1.PersistentVolume, vol *v1.Volume) (string, error) {
	if pv != nil {
		for _, curPlugin := range inTreePlugins {
			if curPlugin.CanSupport(pv) {
				return curPlugin.GetInTreePluginName(), nil
			}
		}
		return "", fmt.Errorf("could not find in-tree plugin name from persistent volume %v", pv)
	} else if vol != nil {
		for _, curPlugin := range inTreePlugins {
			if curPlugin.CanSupportInline(vol) {
				return curPlugin.GetInTreePluginName(), nil
			}
		}
		return "", fmt.Errorf("could not find in-tree plugin name from volume %v", vol)
	} else {
		return "", errors.New("both persistent volume and volume are nil")
	}
}

// GetCSINameFromInTreeName returns the name of a CSI driver that supersedes the
// in-tree plugin with the given name
func (CSITranslator) GetCSINameFromInTreeName(pluginName string) (string, error) {
	for csiDriverName, curPlugin := range inTreePlugins {
		if curPlugin.GetInTreePluginName() == pluginName {
			return csiDriverName, nil
		}
	}
	return "", fmt.Errorf("could not find CSI Driver name for plugin %v", pluginName)
}

// GetInTreeNameFromCSIName returns the name of the in-tree plugin superseded by
// a CSI driver with the given name
func (CSITranslator) GetInTreeNameFromCSIName(pluginName string) (string, error) {
	if plugin, ok := inTreePlugins[pluginName]; ok {
		return plugin.GetInTreePluginName(), nil
	}
	return "", fmt.Errorf("could not find In-Tree driver name for CSI plugin %v", pluginName)
}

// IsPVMigratable tests whether there is migration logic for the given Persistent Volume
func (CSITranslator) IsPVMigratable(pv *v1.PersistentVolume) bool {
	for _, curPlugin := range inTreePlugins {
		if curPlugin.CanSupport(pv) {
			return true
		}
	}
	return false
}

// IsInlineMigratable tests whether there is Migration logic for the given Inline Volume
func (CSITranslator) IsInlineMigratable(vol *v1.Volume) bool {
	for _, curPlugin := range inTreePlugins {
		if curPlugin.CanSupportInline(vol) {
			return true
		}
	}
	return false
}

// RepairVolumeHandle generates a correct volume handle based on node ID information.
func (CSITranslator) RepairVolumeHandle(driverName, volumeHandle, nodeID string) (string, error) {
	if plugin, ok := inTreePlugins[driverName]; ok {
		return plugin.RepairVolumeHandle(volumeHandle, nodeID)
	}
	return "", fmt.Errorf("could not find In-Tree driver name for CSI plugin %v", driverName)
}
