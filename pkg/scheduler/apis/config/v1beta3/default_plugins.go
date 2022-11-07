/*
Copyright 2021 The Kubernetes Authors.

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

package v1beta3

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kube-scheduler/config/v1beta3"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

// getDefaultPlugins returns the default set of plugins.
func getDefaultPlugins() *v1beta3.Plugins {
	plugins := &v1beta3.Plugins{
		MultiPoint: v1beta3.PluginSet{
			Enabled: []v1beta3.Plugin{
				{Name: names.PrioritySort},
				{Name: names.NodeUnschedulable},
				{Name: names.NodeName},
				{Name: names.TaintToleration, Weight: pointer.Int32(3)},
				{Name: names.NodeAffinity, Weight: pointer.Int32(2)},
				{Name: names.NodePorts},
				{Name: names.NodeResourcesFit, Weight: pointer.Int32(1)},
				{Name: names.VolumeRestrictions},
				{Name: names.EBSLimits},
				{Name: names.GCEPDLimits},
				{Name: names.NodeVolumeLimits},
				{Name: names.AzureDiskLimits},
				{Name: names.VolumeBinding},
				{Name: names.VolumeZone},
				{Name: names.PodTopologySpread, Weight: pointer.Int32(2)},
				{Name: names.InterPodAffinity, Weight: pointer.Int32(2)},
				{Name: names.DefaultPreemption},
				{Name: names.NodeResourcesBalancedAllocation, Weight: pointer.Int32(1)},
				{Name: names.ImageLocality, Weight: pointer.Int32(1)},
				{Name: names.DefaultBinder},
			},
		},
	}
	applyFeatureGates(plugins)

	return plugins
}

func applyFeatureGates(config *v1beta3.Plugins) {
	if utilfeature.DefaultFeatureGate.Enabled(features.PodSchedulingReadiness) {
		config.MultiPoint.Enabled = append(config.MultiPoint.Enabled, v1beta3.Plugin{Name: names.SchedulingGates})
	}
}

// mergePlugins merges the custom set into the given default one, handling disabled sets.
func mergePlugins(defaultPlugins, customPlugins *v1beta3.Plugins) *v1beta3.Plugins {
	if customPlugins == nil {
		return defaultPlugins
	}

	defaultPlugins.MultiPoint = mergePluginSet(defaultPlugins.MultiPoint, customPlugins.MultiPoint)
	defaultPlugins.QueueSort = mergePluginSet(defaultPlugins.QueueSort, customPlugins.QueueSort)
	defaultPlugins.PreFilter = mergePluginSet(defaultPlugins.PreFilter, customPlugins.PreFilter)
	defaultPlugins.Filter = mergePluginSet(defaultPlugins.Filter, customPlugins.Filter)
	defaultPlugins.PostFilter = mergePluginSet(defaultPlugins.PostFilter, customPlugins.PostFilter)
	defaultPlugins.PreScore = mergePluginSet(defaultPlugins.PreScore, customPlugins.PreScore)
	defaultPlugins.Score = mergePluginSet(defaultPlugins.Score, customPlugins.Score)
	defaultPlugins.Reserve = mergePluginSet(defaultPlugins.Reserve, customPlugins.Reserve)
	defaultPlugins.Permit = mergePluginSet(defaultPlugins.Permit, customPlugins.Permit)
	defaultPlugins.PreBind = mergePluginSet(defaultPlugins.PreBind, customPlugins.PreBind)
	defaultPlugins.Bind = mergePluginSet(defaultPlugins.Bind, customPlugins.Bind)
	defaultPlugins.PostBind = mergePluginSet(defaultPlugins.PostBind, customPlugins.PostBind)
	return defaultPlugins
}

type pluginIndex struct {
	index  int
	plugin v1beta3.Plugin
}

func mergePluginSet(defaultPluginSet, customPluginSet v1beta3.PluginSet) v1beta3.PluginSet {
	disabledPlugins := sets.NewString()
	enabledCustomPlugins := make(map[string]pluginIndex)
	// replacedPluginIndex is a set of index of plugins, which have replaced the default plugins.
	replacedPluginIndex := sets.NewInt()
	var disabled []v1beta3.Plugin
	for _, disabledPlugin := range customPluginSet.Disabled {
		// if the user is manually disabling any (or all, with "*") default plugins for an extension point,
		// we need to track that so that the MultiPoint extension logic in the framework can know to skip
		// inserting unspecified default plugins to this point.
		disabled = append(disabled, v1beta3.Plugin{Name: disabledPlugin.Name})
		disabledPlugins.Insert(disabledPlugin.Name)
	}

	// With MultiPoint, we may now have some disabledPlugins in the default registry
	// For example, we enable PluginX with Filter+Score through MultiPoint but disable its Score plugin by default.
	for _, disabledPlugin := range defaultPluginSet.Disabled {
		disabled = append(disabled, v1beta3.Plugin{Name: disabledPlugin.Name})
		disabledPlugins.Insert(disabledPlugin.Name)
	}

	for index, enabledPlugin := range customPluginSet.Enabled {
		enabledCustomPlugins[enabledPlugin.Name] = pluginIndex{index, enabledPlugin}
	}
	var enabledPlugins []v1beta3.Plugin
	if !disabledPlugins.Has("*") {
		for _, defaultEnabledPlugin := range defaultPluginSet.Enabled {
			if disabledPlugins.Has(defaultEnabledPlugin.Name) {
				continue
			}
			// The default plugin is explicitly re-configured, update the default plugin accordingly.
			if customPlugin, ok := enabledCustomPlugins[defaultEnabledPlugin.Name]; ok {
				klog.InfoS("Default plugin is explicitly re-configured; overriding", "plugin", defaultEnabledPlugin.Name)
				// Update the default plugin in place to preserve order.
				defaultEnabledPlugin = customPlugin.plugin
				replacedPluginIndex.Insert(customPlugin.index)
			}
			enabledPlugins = append(enabledPlugins, defaultEnabledPlugin)
		}
	}

	// Append all the custom plugins which haven't replaced any default plugins.
	// Note: duplicated custom plugins will still be appended here.
	// If so, the instantiation of scheduler framework will detect it and abort.
	for index, plugin := range customPluginSet.Enabled {
		if !replacedPluginIndex.Has(index) {
			enabledPlugins = append(enabledPlugins, plugin)
		}
	}
	return v1beta3.PluginSet{Enabled: enabledPlugins, Disabled: disabled}
}
