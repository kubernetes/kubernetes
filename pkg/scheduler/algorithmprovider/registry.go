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
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpodtopologyspread"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
)

// ClusterAutoscalerProvider defines the default autoscaler provider
const ClusterAutoscalerProvider = "ClusterAutoscalerProvider"

// Config the configuration of an algorithm provider.
type Config struct {
	FrameworkPlugins      *schedulerapi.Plugins
	FrameworkPluginConfig []schedulerapi.PluginConfig
}

// Registry is a collection of all available algorithm providers.
type Registry map[string]*Config

// NewRegistry returns an algorithm provider registry instance.
func NewRegistry(hardPodAffinityWeight int64) Registry {
	defaultConfig := getDefaultConfig(hardPodAffinityWeight)
	applyFeatureGates(defaultConfig)

	caConfig := getClusterAutoscalerConfig(hardPodAffinityWeight)
	applyFeatureGates(caConfig)

	return Registry{
		schedulerapi.SchedulerDefaultProviderName: defaultConfig,
		ClusterAutoscalerProvider:                 caConfig,
	}
}

// ListAlgorithmProviders lists registered algorithm providers.
func ListAlgorithmProviders() string {
	r := NewRegistry(1)
	var providers []string
	for k := range r {
		providers = append(providers, k)
	}
	sort.Strings(providers)
	return strings.Join(providers, " | ")
}

func getDefaultConfig(hardPodAffinityWeight int64) *Config {
	return &Config{
		FrameworkPlugins: &schedulerapi.Plugins{
			QueueSort: &schedulerapi.PluginSet{
				Enabled: []schedulerapi.Plugin{
					{Name: queuesort.Name},
				},
			},
			PreFilter: &schedulerapi.PluginSet{
				Enabled: []schedulerapi.Plugin{
					{Name: noderesources.FitName},
					{Name: nodeports.Name},
					{Name: interpodaffinity.Name},
				},
			},
			Filter: &schedulerapi.PluginSet{
				Enabled: []schedulerapi.Plugin{
					{Name: nodeunschedulable.Name},
					{Name: noderesources.FitName},
					{Name: nodename.Name},
					{Name: nodeports.Name},
					{Name: nodeaffinity.Name},
					{Name: volumerestrictions.Name},
					{Name: tainttoleration.Name},
					{Name: nodevolumelimits.EBSName},
					{Name: nodevolumelimits.GCEPDName},
					{Name: nodevolumelimits.CSIName},
					{Name: nodevolumelimits.AzureDiskName},
					{Name: volumebinding.Name},
					{Name: volumezone.Name},
					{Name: interpodaffinity.Name},
				},
			},
			PostFilter: &schedulerapi.PluginSet{
				Enabled: []schedulerapi.Plugin{
					{Name: interpodaffinity.Name},
					{Name: defaultpodtopologyspread.Name},
					{Name: tainttoleration.Name},
				},
			},
			Score: &schedulerapi.PluginSet{
				Enabled: []schedulerapi.Plugin{
					{Name: noderesources.BalancedAllocationName, Weight: 1},
					{Name: imagelocality.Name, Weight: 1},
					{Name: interpodaffinity.Name, Weight: 1},
					{Name: noderesources.LeastAllocatedName, Weight: 1},
					{Name: nodeaffinity.Name, Weight: 1},
					{Name: nodepreferavoidpods.Name, Weight: 10000},
					{Name: defaultpodtopologyspread.Name, Weight: 1},
					{Name: tainttoleration.Name, Weight: 1},
				},
			},
			Bind: &schedulerapi.PluginSet{
				Enabled: []schedulerapi.Plugin{
					{Name: defaultbinder.Name},
				},
			},
		},
		FrameworkPluginConfig: []schedulerapi.PluginConfig{
			{
				Name: interpodaffinity.Name,
				Args: runtime.Unknown{Raw: []byte(fmt.Sprintf(`{"hardPodAffinityWeight":%d}`, hardPodAffinityWeight))},
			},
		},
	}
}

func getClusterAutoscalerConfig(hardPodAffinityWeight int64) *Config {
	defaultConfig := getDefaultConfig(hardPodAffinityWeight)
	caConfig := Config{
		FrameworkPlugins: &schedulerapi.Plugins{},
	}
	defaultConfig.FrameworkPlugins.DeepCopyInto(caConfig.FrameworkPlugins)
	caConfig.FrameworkPluginConfig = append([]schedulerapi.PluginConfig(nil), defaultConfig.FrameworkPluginConfig...)

	// Replace least with most requested.
	for i := range caConfig.FrameworkPlugins.Score.Enabled {
		if caConfig.FrameworkPlugins.Score.Enabled[i].Name == noderesources.LeastAllocatedName {
			caConfig.FrameworkPlugins.Score.Enabled[i].Name = noderesources.MostAllocatedName
		}
	}

	return &caConfig
}

func applyFeatureGates(config *Config) {
	// Only add EvenPodsSpread if the feature is enabled.
	if utilfeature.DefaultFeatureGate.Enabled(features.EvenPodsSpread) {
		klog.Infof("Registering EvenPodsSpread predicate and priority function")
		f := schedulerapi.Plugin{Name: podtopologyspread.Name}
		config.FrameworkPlugins.PreFilter.Enabled = append(config.FrameworkPlugins.PreFilter.Enabled, f)
		config.FrameworkPlugins.Filter.Enabled = append(config.FrameworkPlugins.Filter.Enabled, f)
		config.FrameworkPlugins.PostFilter.Enabled = append(config.FrameworkPlugins.PostFilter.Enabled, f)
		s := schedulerapi.Plugin{Name: podtopologyspread.Name, Weight: 1}
		config.FrameworkPlugins.Score.Enabled = append(config.FrameworkPlugins.Score.Enabled, s)
	}

	// Prioritizes nodes that satisfy pod's resource limits
	if utilfeature.DefaultFeatureGate.Enabled(features.ResourceLimitsPriorityFunction) {
		klog.Infof("Registering resourcelimits priority function")
		s := schedulerapi.Plugin{Name: noderesources.ResourceLimitsName}
		config.FrameworkPlugins.PostFilter.Enabled = append(config.FrameworkPlugins.PostFilter.Enabled, s)
		s = schedulerapi.Plugin{Name: noderesources.ResourceLimitsName, Weight: 1}
		config.FrameworkPlugins.Score.Enabled = append(config.FrameworkPlugins.Score.Enabled, s)
	}
}
