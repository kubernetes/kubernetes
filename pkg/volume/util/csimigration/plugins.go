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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/features"

	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
	"k8s.io/kubernetes/pkg/volume/azure_dd"
	"k8s.io/kubernetes/pkg/volume/azure_file"
	"k8s.io/kubernetes/pkg/volume/cinder"
	"k8s.io/kubernetes/pkg/volume/gcepd"
)

type probeFn func() []volume.VolumePlugin

func probePluginWithCSIMigrationDisabled(plugins []volume.VolumePlugin, key featuregate.Feature, fn probeFn) []volume.VolumePlugin {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIMigration) || !utilfeature.DefaultFeatureGate.Enabled(key) {
		plugins = append(plugins, fn()...)
	}
	return plugins
}

// ProbePluginsWithCSIMigrationDisabled probes/registers in-tree plugins based on feature flags
func ProbePluginsWithCSIMigrationDisabled(plugins []volume.VolumePlugin) []volume.VolumePlugin {
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationAWS, awsebs.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationGCE, gcepd.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationOpenStack, cinder.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationAzureDisk, azure_dd.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationAzureFile, azure_file.ProbeVolumePlugins)
	return plugins
}

// ProbeAttachablePluginsWithCSIMigrationDisabled probes/registers in-tree attachable plugins based on feature flags
func ProbeAttachablePluginsWithCSIMigrationDisabled(plugins []volume.VolumePlugin) []volume.VolumePlugin {
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationAWS, awsebs.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationGCE, gcepd.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationAzureDisk, azure_dd.ProbeVolumePlugins)
	plugins = probePluginWithCSIMigrationDisabled(plugins, features.CSIMigrationAzureDisk, cinder.ProbeVolumePlugins)
	return plugins
}
