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
	"k8s.io/kubernetes/pkg/volume/portworx"
)

type probeFn func() []volume.VolumePlugin

func appendPluginBasedOnFeatureFlags(logger klog.Logger, plugins []volume.VolumePlugin, inTreePluginName string, featureGate featuregate.FeatureGate, pluginInfo pluginInfo) []volume.VolumePlugin {
	// Skip appending the in-tree plugin to the list of plugins to be probed/initialized
	// if the plugin unregister feature flag is set
	if featureGate.Enabled(pluginInfo.pluginUnregisterFeature) {
		logger.Info("Skip registration of plugin since feature flag is enabled", "plugin", inTreePluginName, "feature", pluginInfo.pluginUnregisterFeature)
		return plugins
	}
	plugins = append(plugins, pluginInfo.pluginProbeFunction()...)
	return plugins
}

type pluginInfo struct {
	pluginMigrationFeature  featuregate.Feature
	pluginUnregisterFeature featuregate.Feature
	pluginProbeFunction     probeFn
}

func appendLegacyControllerProviders(logger klog.Logger, allPlugins []volume.VolumePlugin, featureGate featuregate.FeatureGate) []volume.VolumePlugin {
	pluginMigrationStatus := make(map[string]pluginInfo)
	pluginMigrationStatus[plugins.PortworxVolumePluginName] = pluginInfo{pluginMigrationFeature: features.CSIMigrationPortworx, pluginUnregisterFeature: features.InTreePluginPortworxUnregister, pluginProbeFunction: portworx.ProbeVolumePlugins}
	for pluginName, pluginInfo := range pluginMigrationStatus {
		allPlugins = appendPluginBasedOnFeatureFlags(logger, allPlugins, pluginName, featureGate, pluginInfo)
	}
	return allPlugins
}
