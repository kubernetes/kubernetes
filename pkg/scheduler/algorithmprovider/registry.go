/*
Copyright 2014 The Kubernetes Authors.

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

package algorithmprovider

import (
	"fmt"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/imagelocality"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodepreferavoidpods"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodevolumelimits"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/selectorspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
)

// ClusterAutoscalerProvider defines the default autoscaler provider
const ClusterAutoscalerProvider = "ClusterAutoscalerProvider"

// Registry is a collection of all available algorithm providers.
type Registry map[string]*schedulerapi.Plugins

// NewRegistry returns an algorithm provider registry instance.
func NewRegistry() Registry {
	defaultConfig := getDefaultConfig()
	applyFeatureGates(defaultConfig)

	caConfig := getClusterAutoscalerConfig()
	applyFeatureGates(caConfig)

	return Registry{
		schedulerapi.SchedulerDefaultProviderName: defaultConfig,
		ClusterAutoscalerProvider:                 caConfig,
	}
}

// ListAlgorithmProviders lists registered algorithm providers.
func ListAlgorithmProviders() string {
	return fmt.Sprintf("%s | %s", ClusterAutoscalerProvider, schedulerapi.SchedulerDefaultProviderName)
}

func getDefaultConfig() *schedulerapi.Plugins {
	plugins := &schedulerapi.Plugins{
		QueueSort: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: queuesort.Name},
			},
		},
		PreFilter: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: noderesources.FitName},
				{Name: nodeports.Name},
				{Name: podtopologyspread.Name},
				{Name: interpodaffinity.Name},
				{Name: volumebinding.Name},
				{Name: nodeaffinity.Name},
			},
		},
		Filter: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: nodeunschedulable.Name},
				{Name: nodename.Name},
				{Name: tainttoleration.Name},
				{Name: nodeaffinity.Name},
				{Name: nodeports.Name},
				{Name: noderesources.FitName},
				{Name: volumerestrictions.Name},
				{Name: nodevolumelimits.EBSName},
				{Name: nodevolumelimits.GCEPDName},
				{Name: nodevolumelimits.CSIName},
				{Name: nodevolumelimits.AzureDiskName},
				{Name: volumebinding.Name},
				{Name: volumezone.Name},
				{Name: podtopologyspread.Name},
				{Name: interpodaffinity.Name},
			},
		},
		PostFilter: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: defaultpreemption.Name},
			},
		},
		PreScore: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: interpodaffinity.Name},
				{Name: podtopologyspread.Name},
				{Name: tainttoleration.Name},
				{Name: nodeaffinity.Name},
			},
		},
		Score: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: noderesources.BalancedAllocationName, Weight: 1},
				{Name: imagelocality.Name, Weight: 1},
				{Name: interpodaffinity.Name, Weight: 1},
				{Name: noderesources.LeastAllocatedName, Weight: 1},
				{Name: nodeaffinity.Name, Weight: 1},
				{Name: nodepreferavoidpods.Name, Weight: 10000},
				// Weight is doubled because:
				// - This is a score coming from user preference.
				// - It makes its signal comparable to NodeResourcesLeastAllocated.
				{Name: podtopologyspread.Name, Weight: 2},
				{Name: tainttoleration.Name, Weight: 1},
			},
		},
		Reserve: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: volumebinding.Name},
			},
		},
		PreBind: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: volumebinding.Name},
			},
		},
		Bind: schedulerapi.PluginSet{
			Enabled: []schedulerapi.Plugin{
				{Name: defaultbinder.Name},
			},
		},
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeCapacityPriority) {
		plugins.Score.Enabled = append(plugins.Score.Enabled, schedulerapi.Plugin{Name: volumebinding.Name, Weight: 1})
	}
	return plugins
}

func getClusterAutoscalerConfig() *schedulerapi.Plugins {
	caConfig := getDefaultConfig()
	// Replace least with most requested.
	for i := range caConfig.Score.Enabled {
		if caConfig.Score.Enabled[i].Name == noderesources.LeastAllocatedName {
			caConfig.Score.Enabled[i].Name = noderesources.MostAllocatedName
		}
	}
	return caConfig
}

func applyFeatureGates(config *schedulerapi.Plugins) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DefaultPodTopologySpread) {
		// When feature is enabled, the default spreading is done by
		// PodTopologySpread plugin, which is enabled by default.
		klog.Infof("Registering SelectorSpread plugin")
		s := schedulerapi.Plugin{Name: selectorspread.Name}
		config.PreScore.Enabled = append(config.PreScore.Enabled, s)
		s.Weight = 1
		config.Score.Enabled = append(config.Score.Enabled, s)
	}
}
