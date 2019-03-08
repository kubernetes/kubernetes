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

	"k8s.io/api/core/v1"
	"k8s.io/csi-translation-lib/plugins"
)

var (
	inTreePlugins = map[string]plugins.InTreePlugin{
		plugins.GCEPDDriverName:  plugins.NewGCEPersistentDiskCSITranslator(),
		plugins.AWSEBSDriverName: plugins.NewAWSElasticBlockStoreCSITranslator(),
		plugins.CinderDriverName: plugins.NewOpenStackCinderCSITranslator(),
	}
)

// TranslateInTreeStorageClassParametersToCSI takes in-tree storage class
// parameters and translates them to a set of parameters consumable by CSI plugin
func TranslateInTreeStorageClassParametersToCSI(inTreePluginName string, scParameters map[string]string) (map[string]string, error) {
	for _, curPlugin := range inTreePlugins {
		if inTreePluginName == curPlugin.GetInTreePluginName() {
			return curPlugin.TranslateInTreeStorageClassParametersToCSI(scParameters)
		}
	}
	return nil, fmt.Errorf("could not find in-tree storage class parameter translation logic for %#v", inTreePluginName)
}

// TranslateInTreePVToCSI takes a persistent volume and will translate
// the in-tree source to a CSI Source if the translation logic
// has been implemented. The input persistent volume will not
// be modified
func TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
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
func TranslateCSIPVToInTree(pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
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
func IsMigratableIntreePluginByName(inTreePluginName string) bool {
	for _, curPlugin := range inTreePlugins {
		if curPlugin.GetInTreePluginName() == inTreePluginName {
			return true
		}
	}
	return false
}

// IsMigratedCSIDriverByName tests whether there exists an in-tree plugin with logic
// to migrate to the CSI driver with given name
func IsMigratedCSIDriverByName(csiPluginName string) bool {
	if _, ok := inTreePlugins[csiPluginName]; ok {
		return true
	}
	return false
}

// GetInTreePluginNameFromSpec returns the plugin name
func GetInTreePluginNameFromSpec(pv *v1.PersistentVolume, vol *v1.Volume) (string, error) {
	if pv != nil {
		for _, curPlugin := range inTreePlugins {
			if curPlugin.CanSupport(pv) {
				return curPlugin.GetInTreePluginName(), nil
			}
		}
		return "", fmt.Errorf("could not find in-tree plugin name from persistent volume %v", pv)
	} else if vol != nil {
		// TODO(dyzz): Implement inline volume migration support
		return "", errors.New("inline volume migration not yet supported")
	} else {
		return "", errors.New("both persistent volume and volume are nil")
	}
}

// GetCSINameFromInTreeName returns the name of a CSI driver that supersedes the
// in-tree plugin with the given name
func GetCSINameFromInTreeName(pluginName string) (string, error) {
	for csiDriverName, curPlugin := range inTreePlugins {
		if curPlugin.GetInTreePluginName() == pluginName {
			return csiDriverName, nil
		}
	}
	return "", fmt.Errorf("could not find CSI Driver name for plugin %v", pluginName)
}

// GetInTreeNameFromCSIName returns the name of the in-tree plugin superseded by
// a CSI driver with the given name
func GetInTreeNameFromCSIName(pluginName string) (string, error) {
	if plugin, ok := inTreePlugins[pluginName]; ok {
		return plugin.GetInTreePluginName(), nil
	}
	return "", fmt.Errorf("Could not find In-Tree driver name for CSI plugin %v", pluginName)
}

// IsPVMigratable tests whether there is migration logic for the given Persistent Volume
func IsPVMigratable(pv *v1.PersistentVolume) bool {
	for _, curPlugin := range inTreePlugins {
		if curPlugin.CanSupport(pv) {
			return true
		}
	}
	return false
}

// IsInlineMigratable tests whether there is Migration logic for the given Inline Volume
func IsInlineMigratable(vol *v1.Volume) bool {
	return false
}
