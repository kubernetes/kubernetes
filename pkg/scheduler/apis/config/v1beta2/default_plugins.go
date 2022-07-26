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

package v1beta2

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kube-scheduler/config/v1beta2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

// getDefaultPlugins returns the default set of plugins.
func getDefaultPlugins() *v1beta2.Plugins {
	plugins := &v1beta2.Plugins{
		QueueSort: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.PrioritySort},
			},
		},
		PreFilter: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.NodeResourcesFit},
				{Name: names.NodePorts},
				{Name: names.VolumeRestrictions},
				{Name: names.PodTopologySpread},
				{Name: names.InterPodAffinity},
				{Name: names.VolumeBinding},
				{Name: names.NodeAffinity},
			},
		},
		Filter: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.NodeUnschedulable},
				{Name: names.NodeName},
				{Name: names.TaintToleration},
				{Name: names.NodeAffinity},
				{Name: names.NodePorts},
				{Name: names.NodeResourcesFit},
				{Name: names.VolumeRestrictions},
				{Name: names.EBSLimits},
				{Name: names.GCEPDLimits},
				{Name: names.NodeVolumeLimits},
				{Name: names.AzureDiskLimits},
				{Name: names.VolumeBinding},
				{Name: names.VolumeZone},
				{Name: names.PodTopologySpread},
				{Name: names.InterPodAffinity},
			},
		},
		PostFilter: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.DefaultPreemption},
			},
		},
		PreScore: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.InterPodAffinity},
				{Name: names.PodTopologySpread},
				{Name: names.TaintToleration},
				{Name: names.NodeAffinity},
			},
		},
		Score: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.NodeResourcesBalancedAllocation, Weight: pointer.Int32(1)},
				{Name: names.ImageLocality, Weight: pointer.Int32(1)},
				{Name: names.InterPodAffinity, Weight: pointer.Int32(1)},
				{Name: names.NodeResourcesFit, Weight: pointer.Int32(1)},
				{Name: names.NodeAffinity, Weight: pointer.Int32(1)},
				// Weight is doubled because:
				// - This is a score coming from user preference.
				// - It makes its signal comparable to NodeResourcesFit.LeastAllocated.
				{Name: names.PodTopologySpread, Weight: pointer.Int32(2)},
				{Name: names.TaintToleration, Weight: pointer.Int32(1)},
			},
		},
		Reserve: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.VolumeBinding},
			},
		},
		PreBind: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.VolumeBinding},
			},
		},
		Bind: v1beta2.PluginSet{
			Enabled: []v1beta2.Plugin{
				{Name: names.DefaultBinder},
			},
		},
	}
	applyFeatureGates(plugins)

	return plugins
}

func applyFeatureGates(config *v1beta2.Plugins) {
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeCapacityPriority) {
		config.Score.Enabled = append(config.Score.Enabled, v1beta2.Plugin{Name: names.VolumeBinding, Weight: pointer.Int32(1)})
	}
}

// mergePlugins merges the custom set into the given default one, handling disabled sets.
func mergePlugins(defaultPlugins, customPlugins *v1beta2.Plugins) *v1beta2.Plugins {
	if customPlugins == nil {
		return defaultPlugins
	}

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
	plugin v1beta2.Plugin
}

func mergePluginSet(defaultPluginSet, customPluginSet v1beta2.PluginSet) v1beta2.PluginSet {
	disabledPlugins := sets.NewString()
	enabledCustomPlugins := make(map[string]pluginIndex)
	// replacedPluginIndex is a set of index of plugins, which have replaced the default plugins.
	replacedPluginIndex := sets.NewInt()
	for _, disabledPlugin := range customPluginSet.Disabled {
		disabledPlugins.Insert(disabledPlugin.Name)
	}
	for index, enabledPlugin := range customPluginSet.Enabled {
		enabledCustomPlugins[enabledPlugin.Name] = pluginIndex{index, enabledPlugin}
	}
	var enabledPlugins []v1beta2.Plugin
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
	return v1beta2.PluginSet{Enabled: enabledPlugins}
}
