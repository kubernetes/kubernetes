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
		return nil, fmt.Errorf("persistent volume was nil")
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
		return nil, fmt.Errorf("CSI persistent volume was nil")
	}
	copiedPV := pv.DeepCopy()
	for driverName, curPlugin := range inTreePlugins {
		if copiedPV.Spec.CSI.Driver == driverName {
			return curPlugin.TranslateCSIPVToInTree(copiedPV)
		}
	}
	return nil, fmt.Errorf("could not find in-tree plugin translation logic for %s", copiedPV.Spec.CSI.Driver)
}

// IsMigratableByName tests whether there is Migration logic for the in-tree plugin
// for the given `pluginName`
func IsMigratableByName(pluginName string) bool {
	for _, curPlugin := range inTreePlugins {
		if curPlugin.GetInTreePluginName() == pluginName {
			return true
		}
	}
	return false
}

// GetCSINameFromIntreeName maps the name of a CSI driver to its in-tree version
func GetCSINameFromIntreeName(pluginName string) (string, error) {
	for csiDriverName, curPlugin := range inTreePlugins {
		if curPlugin.GetInTreePluginName() == pluginName {
			return csiDriverName, nil
		}
	}
	return "", fmt.Errorf("Could not find CSI Driver name for plugin %v", pluginName)
}

// IsPVMigratable tests whether there is Migration logic for the given Persistent Volume
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
