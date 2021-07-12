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

package v1beta1

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/utils/pointer"
)

// getDefaultPlugins returns the default set of plugins.
func getDefaultPlugins() *v1beta1.Plugins {
	plugins := &v1beta1.Plugins{
		QueueSort: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.PrioritySort},
			},
		},
		PreFilter: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.NodeResourcesFit},
				{Name: names.NodePorts},
				{Name: names.VolumeRestrictions},
				{Name: names.PodTopologySpread},
				{Name: names.InterPodAffinity},
				{Name: names.VolumeBinding},
				{Name: names.NodeAffinity},
			},
		},
		Filter: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
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
		PostFilter: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.DefaultPreemption},
			},
		},
		PreScore: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.InterPodAffinity},
				{Name: names.PodTopologySpread},
				{Name: names.TaintToleration},
				{Name: names.NodeAffinity},
			},
		},
		Score: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.NodeResourcesBalancedAllocation, Weight: pointer.Int32Ptr(1)},
				{Name: names.ImageLocality, Weight: pointer.Int32Ptr(1)},
				{Name: names.InterPodAffinity, Weight: pointer.Int32Ptr(1)},
				{Name: names.NodeResourcesLeastAllocated, Weight: pointer.Int32Ptr(1)},
				{Name: names.NodeAffinity, Weight: pointer.Int32Ptr(1)},
				{Name: names.NodePreferAvoidPods, Weight: pointer.Int32Ptr(10000)},
				// Weight is doubled because:
				// - This is a score coming from user preference.
				// - It makes its signal comparable to NodeResourcesLeastAllocated.
				{Name: names.PodTopologySpread, Weight: pointer.Int32Ptr(2)},
				{Name: names.TaintToleration, Weight: pointer.Int32Ptr(1)},
			},
		},
		Reserve: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.VolumeBinding},
			},
		},
		PreBind: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.VolumeBinding},
			},
		},
		Bind: &v1beta1.PluginSet{
			Enabled: []v1beta1.Plugin{
				{Name: names.DefaultBinder},
			},
		},
	}
	applyFeatureGates(plugins)

	return plugins
}

func applyFeatureGates(config *v1beta1.Plugins) {
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeCapacityPriority) {
		config.Score.Enabled = append(config.Score.Enabled, v1beta1.Plugin{Name: names.VolumeBinding, Weight: pointer.Int32Ptr(1)})
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.DefaultPodTopologySpread) {
		// When feature is enabled, the default spreading is done by
		// PodTopologySpread plugin, which is enabled by default.
		klog.InfoS("Registering SelectorSpread plugin")
		s := v1beta1.Plugin{Name: names.SelectorSpread}
		config.PreScore.Enabled = append(config.PreScore.Enabled, s)
		s.Weight = pointer.Int32Ptr(1)
		config.Score.Enabled = append(config.Score.Enabled, s)
	}
}

// mergePlugins merges the custom set into the given default one, handling disabled sets.
func mergePlugins(defaultPlugins, customPlugins *v1beta1.Plugins) *v1beta1.Plugins {
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

func mergePluginSet(defaultPluginSet, customPluginSet *v1beta1.PluginSet) *v1beta1.PluginSet {
	disabledPlugins := sets.NewString()
	if customPluginSet != nil {
		for _, disabledPlugin := range customPluginSet.Disabled {
			disabledPlugins.Insert(disabledPlugin.Name)
		}
	}

	var enabledPlugins []v1beta1.Plugin
	if defaultPluginSet != nil && !disabledPlugins.Has("*") {
		for _, defaultEnabledPlugin := range defaultPluginSet.Enabled {
			if disabledPlugins.Has(defaultEnabledPlugin.Name) {
				continue
			}

			enabledPlugins = append(enabledPlugins, defaultEnabledPlugin)
		}
	}

	if customPluginSet != nil {
		enabledPlugins = append(enabledPlugins, customPluginSet.Enabled...)
	}

	if len(enabledPlugins) == 0 {
		return nil
	}

	return &v1beta1.PluginSet{Enabled: enabledPlugins}
}
