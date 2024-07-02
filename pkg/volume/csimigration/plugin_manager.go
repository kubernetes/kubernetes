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

package csimigration

import (
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/featuregate"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
)

// PluginNameMapper contains utility methods to retrieve names of plugins
// that support a spec, map intree <=> migrated CSI plugin names, etc
type PluginNameMapper interface {
	GetInTreePluginNameFromSpec(pv *v1.PersistentVolume, vol *v1.Volume) (string, error)
	GetCSINameFromInTreeName(pluginName string) (string, error)
}

// PluginManager keeps track of migrated state of in-tree plugins
type PluginManager struct {
	PluginNameMapper
	featureGate featuregate.FeatureGate
}

// NewPluginManager returns a new PluginManager instance
func NewPluginManager(m PluginNameMapper, featureGate featuregate.FeatureGate) PluginManager {
	return PluginManager{
		PluginNameMapper: m,
		featureGate:      featureGate,
	}
}

// IsMigrationCompleteForPlugin indicates whether CSI migration has been completed
// for a particular storage plugin. A complete migration will need to:
// 1. Enable CSIMigrationXX for the plugin
// 2. Unregister the in-tree plugin by setting the InTreePluginXXUnregister feature gate
func (pm PluginManager) IsMigrationCompleteForPlugin(pluginName string) bool {
	// CSIMigration feature and plugin specific InTreePluginUnregister feature flags should
	// be enabled for plugin specific migration completion to be take effect
	if !pm.IsMigrationEnabledForPlugin(pluginName) {
		return false
	}

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		return true
	case csilibplugins.GCEPDInTreePluginName:
		return true
	case csilibplugins.AzureFileInTreePluginName:
		return true
	case csilibplugins.AzureDiskInTreePluginName:
		return true
	case csilibplugins.CinderInTreePluginName:
		return true
	case csilibplugins.VSphereInTreePluginName:
		return true
	case csilibplugins.PortworxVolumePluginName:
		return pm.featureGate.Enabled(features.InTreePluginPortworxUnregister)
	default:
		return false
	}
}

// IsMigrationEnabledForPlugin indicates whether CSI migration has been enabled
// for a particular storage plugin
func (pm PluginManager) IsMigrationEnabledForPlugin(pluginName string) bool {
	// CSIMigration feature should be enabled along with the plugin-specific one
	// CSIMigration has been GA. It will be enabled by default.

	switch pluginName {
	case csilibplugins.AWSEBSInTreePluginName:
		return true
	case csilibplugins.GCEPDInTreePluginName:
		return true
	case csilibplugins.AzureFileInTreePluginName:
		return true
	case csilibplugins.AzureDiskInTreePluginName:
		return true
	case csilibplugins.CinderInTreePluginName:
		return true
	case csilibplugins.VSphereInTreePluginName:
		return true
	case csilibplugins.PortworxVolumePluginName:
		return pm.featureGate.Enabled(features.CSIMigrationPortworx)
	default:
		return false
	}
}

// IsMigratable indicates whether CSI migration has been enabled for a volume
// plugin that the spec refers to
func (pm PluginManager) IsMigratable(spec *volume.Spec) (bool, error) {
	if spec == nil {
		return false, fmt.Errorf("could not find if plugin is migratable because volume spec is nil")
	}

	pluginName, _ := pm.GetInTreePluginNameFromSpec(spec.PersistentVolume, spec.Volume)
	if pluginName == "" {
		return false, nil
	}
	// found an in-tree plugin that supports the spec
	return pm.IsMigrationEnabledForPlugin(pluginName), nil
}

// InTreeToCSITranslator performs translation of Volume sources for PV and Volume objects
// from references to in-tree plugins to migrated CSI plugins
type InTreeToCSITranslator interface {
	TranslateInTreePVToCSI(pv *v1.PersistentVolume) (*v1.PersistentVolume, error)
	TranslateInTreeInlineVolumeToCSI(volume *v1.Volume, podNamespace string) (*v1.PersistentVolume, error)
}

// TranslateInTreeSpecToCSI translates a volume spec (either PV or inline volume)
// supported by an in-tree plugin to CSI
func TranslateInTreeSpecToCSI(spec *volume.Spec, podNamespace string, translator InTreeToCSITranslator) (*volume.Spec, error) {
	var csiPV *v1.PersistentVolume
	var err error
	inlineVolume := false
	if spec.PersistentVolume != nil {
		csiPV, err = translator.TranslateInTreePVToCSI(spec.PersistentVolume)
	} else if spec.Volume != nil {
		csiPV, err = translator.TranslateInTreeInlineVolumeToCSI(spec.Volume, podNamespace)
		inlineVolume = true
	} else {
		err = errors.New("not a valid volume spec")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to translate in-tree pv to CSI: %v", err)
	}
	return &volume.Spec{
		Migrated:                        true,
		PersistentVolume:                csiPV,
		ReadOnly:                        spec.ReadOnly,
		InlineVolumeSpecForCSIMigration: inlineVolume,
	}, nil
}

// CheckMigrationFeatureFlags checks the configuration of feature flags related
// to CSI Migration is valid. It will return whether the migration is complete
// by looking up the pluginUnregister flag
func CheckMigrationFeatureFlags(f featuregate.FeatureGate, pluginMigration,
	pluginUnregister featuregate.Feature) (migrationComplete bool, err error) {
	// This is for in-tree plugin that get migration finished
	if f.Enabled(pluginMigration) && f.Enabled(pluginUnregister) {
		return true, nil
	}
	return false, nil
}
