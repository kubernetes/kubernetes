// +build !providerless

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

package app

import (
	"k8s.io/component-base/featuregate"
	"k8s.io/csi-translation-lib/plugins"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
	"k8s.io/kubernetes/pkg/volume/azure_dd"
	"k8s.io/kubernetes/pkg/volume/azure_file"
	"k8s.io/kubernetes/pkg/volume/cinder"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/gcepd"
	"k8s.io/kubernetes/pkg/volume/vsphere_volume"
)

type probeFn func() []volume.VolumePlugin

func appendPluginBasedOnMigrationFeatureFlags(plugins []volume.VolumePlugin, inTreePluginName string, featureGate featuregate.FeatureGate, pluginMigration, pluginMigrationComplete featuregate.Feature, fn probeFn) ([]volume.VolumePlugin, error) {
	// Skip appending the in-tree plugin to the list of plugins to be probed/initialized
	// if the CSIMigration feature flag and plugin specific feature flag indicating
	// CSI migration is complete
	err := csimigration.CheckMigrationFeatureFlags(featureGate, pluginMigration, pluginMigrationComplete)
	if err != nil {
		klog.Warningf("Unexpected CSI Migration Feature Flags combination detected: %v. CSI Migration may not take effect", err)
		// TODO: fail and return here once alpha only tests can set the feature flags for a plugin correctly
	}
	if featureGate.Enabled(features.CSIMigration) && featureGate.Enabled(pluginMigration) && featureGate.Enabled(pluginMigrationComplete) {
		klog.Infof("Skip registration of plugin %s since feature flag %v is enabled", inTreePluginName, pluginMigrationComplete)
		return plugins, nil
	}
	plugins = append(plugins, fn()...)
	return plugins, nil
}

type pluginInfo struct {
	pluginMigrationFeature         featuregate.Feature
	pluginMigrationCompleteFeature featuregate.Feature
	pluginProbeFunction            probeFn
}

func appendLegacyProviderVolumes(allPlugins []volume.VolumePlugin, featureGate featuregate.FeatureGate) ([]volume.VolumePlugin, error) {
	pluginMigrationStatus := make(map[string]pluginInfo)
	pluginMigrationStatus[plugins.AWSEBSInTreePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationAWS, pluginMigrationCompleteFeature: features.CSIMigrationAWSComplete, pluginProbeFunction: awsebs.ProbeVolumePlugins}
	pluginMigrationStatus[plugins.GCEPDInTreePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationGCE, pluginMigrationCompleteFeature: features.CSIMigrationGCEComplete, pluginProbeFunction: gcepd.ProbeVolumePlugins}
	pluginMigrationStatus[plugins.CinderInTreePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationOpenStack, pluginMigrationCompleteFeature: features.CSIMigrationOpenStackComplete, pluginProbeFunction: cinder.ProbeVolumePlugins}
	pluginMigrationStatus[plugins.AzureDiskInTreePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationAzureDisk, pluginMigrationCompleteFeature: features.CSIMigrationAzureDiskComplete, pluginProbeFunction: azure_dd.ProbeVolumePlugins}
	pluginMigrationStatus[plugins.AzureFileInTreePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationAzureFile, pluginMigrationCompleteFeature: features.CSIMigrationAzureFileComplete, pluginProbeFunction: azure_file.ProbeVolumePlugins}
	pluginMigrationStatus[plugins.VSphereInTreePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationvSphere, pluginMigrationCompleteFeature: features.CSIMigrationvSphereComplete, pluginProbeFunction: vsphere_volume.ProbeVolumePlugins}

	var err error
	for pluginName, pluginInfo := range pluginMigrationStatus {
		allPlugins, err = appendPluginBasedOnMigrationFeatureFlags(allPlugins, pluginName, featureGate, pluginInfo.pluginMigrationFeature, pluginInfo.pluginMigrationCompleteFeature, pluginInfo.pluginProbeFunction)
		if err != nil {
			return allPlugins, err
		}
	}
	return allPlugins, nil
}
