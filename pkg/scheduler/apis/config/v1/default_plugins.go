/*
Copyright 2022 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	v1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

// getDefaultPlugins returns the default set of plugins.
func getDefaultPlugins() *v1.Plugins {
	plugins := &v1.Plugins{
		MultiPoint: v1.PluginSet{
			Enabled: []v1.Plugin{
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

func applyFeatureGates(config *v1.Plugins) {
	if utilfeature.DefaultFeatureGate.Enabled(features.PodSchedulingReadiness) {
		config.MultiPoint.Enabled = append(config.MultiPoint.Enabled, v1.Plugin{Name: names.SchedulingGates})
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		// This plugin should come before DefaultPreemption because if
		// there is a problem with a Pod and PostFilter gets called to
		// resolve the problem, it is better to first deallocate an
		// idle ResourceClaim than it is to evict some Pod that might
		// be doing useful work.
		for i := range config.MultiPoint.Enabled {
			if config.MultiPoint.Enabled[i].Name == names.DefaultPreemption {
				extended := make([]v1.Plugin, 0, len(config.MultiPoint.Enabled)+1)
				extended = append(extended, config.MultiPoint.Enabled[:i]...)
				extended = append(extended, v1.Plugin{Name: names.DynamicResources})
				extended = append(extended, config.MultiPoint.Enabled[i:]...)
				config.MultiPoint.Enabled = extended
				break
			}
		}
	}
}

// mergePlugins merges the custom set into the given default one, handling disabled sets.
func mergePlugins(logger klog.Logger, defaultPlugins, customPlugins *v1.Plugins) *v1.Plugins {
	if customPlugins == nil {
		return defaultPlugins
	}

	defaultPlugins.MultiPoint = mergePluginSet(logger, defaultPlugins.MultiPoint, customPlugins.MultiPoint)
	defaultPlugins.PreEnqueue = mergePluginSet(logger, defaultPlugins.PreEnqueue, customPlugins.PreEnqueue)
	defaultPlugins.QueueSort = mergePluginSet(logger, defaultPlugins.QueueSort, customPlugins.QueueSort)
	defaultPlugins.PreFilter = mergePluginSet(logger, defaultPlugins.PreFilter, customPlugins.PreFilter)
	defaultPlugins.Filter = mergePluginSet(logger, defaultPlugins.Filter, customPlugins.Filter)
	defaultPlugins.PostFilter = mergePluginSet(logger, defaultPlugins.PostFilter, customPlugins.PostFilter)
	defaultPlugins.PreScore = mergePluginSet(logger, defaultPlugins.PreScore, customPlugins.PreScore)
	defaultPlugins.Score = mergePluginSet(logger, defaultPlugins.Score, customPlugins.Score)
	defaultPlugins.Reserve = mergePluginSet(logger, defaultPlugins.Reserve, customPlugins.Reserve)
	defaultPlugins.Permit = mergePluginSet(logger, defaultPlugins.Permit, customPlugins.Permit)
	defaultPlugins.PreBind = mergePluginSet(logger, defaultPlugins.PreBind, customPlugins.PreBind)
	defaultPlugins.Bind = mergePluginSet(logger, defaultPlugins.Bind, customPlugins.Bind)
	defaultPlugins.PostBind = mergePluginSet(logger, defaultPlugins.PostBind, customPlugins.PostBind)
	return defaultPlugins
}

type pluginIndex struct {
	index  int
	plugin v1.Plugin
}

func mergePluginSet(logger klog.Logger, defaultPluginSet, customPluginSet v1.PluginSet) v1.PluginSet {
	disabledPlugins := sets.NewString()
	enabledCustomPlugins := make(map[string]pluginIndex)
	// replacedPluginIndex is a set of index of plugins, which have replaced the default plugins.
	replacedPluginIndex := sets.NewInt()
	var disabled []v1.Plugin
	for _, disabledPlugin := range customPluginSet.Disabled {
		// if the user is manually disabling any (or all, with "*") default plugins for an extension point,
		// we need to track that so that the MultiPoint extension logic in the framework can know to skip
		// inserting unspecified default plugins to this point.
		disabled = append(disabled, v1.Plugin{Name: disabledPlugin.Name})
		disabledPlugins.Insert(disabledPlugin.Name)
	}

	// With MultiPoint, we may now have some disabledPlugins in the default registry
	// For example, we enable PluginX with Filter+Score through MultiPoint but disable its Score plugin by default.
	for _, disabledPlugin := range defaultPluginSet.Disabled {
		disabled = append(disabled, v1.Plugin{Name: disabledPlugin.Name})
		disabledPlugins.Insert(disabledPlugin.Name)
	}

	for index, enabledPlugin := range customPluginSet.Enabled {
		enabledCustomPlugins[enabledPlugin.Name] = pluginIndex{index, enabledPlugin}
	}
	var enabledPlugins []v1.Plugin
	if !disabledPlugins.Has("*") {
		for _, defaultEnabledPlugin := range defaultPluginSet.Enabled {
			if disabledPlugins.Has(defaultEnabledPlugin.Name) {
				continue
			}
			// The default plugin is explicitly re-configured, update the default plugin accordingly.
			if customPlugin, ok := enabledCustomPlugins[defaultEnabledPlugin.Name]; ok {
				logger.Info("Default plugin is explicitly re-configured; overriding", "plugin", defaultEnabledPlugin.Name)
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
	return v1.PluginSet{Enabled: enabledPlugins, Disabled: disabled}
}
