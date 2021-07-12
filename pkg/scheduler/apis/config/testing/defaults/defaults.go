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

package defaults

import (
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

// PluginsV1beta1 default set of v1beta1 plugins.
var PluginsV1beta1 = &config.Plugins{
	QueueSort: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.PrioritySort},
		},
	},
	PreFilter: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.NodeResourcesFit},
			{Name: names.NodePorts},
			{Name: names.VolumeRestrictions},
			{Name: names.PodTopologySpread},
			{Name: names.InterPodAffinity},
			{Name: names.VolumeBinding},
			{Name: names.NodeAffinity},
		},
	},
	Filter: config.PluginSet{
		Enabled: []config.Plugin{
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
	PostFilter: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.DefaultPreemption},
		},
	},
	PreScore: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.InterPodAffinity},
			{Name: names.PodTopologySpread},
			{Name: names.TaintToleration},
			{Name: names.NodeAffinity},
		},
	},
	Score: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.NodeResourcesBalancedAllocation, Weight: 1},
			{Name: names.ImageLocality, Weight: 1},
			{Name: names.InterPodAffinity, Weight: 1},
			{Name: names.NodeResourcesLeastAllocated, Weight: 1},
			{Name: names.NodeAffinity, Weight: 1},
			{Name: names.NodePreferAvoidPods, Weight: 10000},
			// Weight is doubled because:
			// - This is a score coming from user preference.
			// - It makes its signal comparable to NodeResourcesLeastAllocated.
			{Name: names.PodTopologySpread, Weight: 2},
			{Name: names.TaintToleration, Weight: 1},
		},
	},
	Reserve: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.VolumeBinding},
		},
	},
	PreBind: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.VolumeBinding},
		},
	},
	Bind: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.DefaultBinder},
		},
	},
}

// PluginConfigsV1beta1 default plugin configurations. This could get versioned, but since
// all available versions produce the same defaults, we just have one for now.
var PluginConfigsV1beta1 = []config.PluginConfig{
	{
		Name: "DefaultPreemption",
		Args: &config.DefaultPreemptionArgs{
			MinCandidateNodesPercentage: 10,
			MinCandidateNodesAbsolute:   100,
		},
	},
	{
		Name: "InterPodAffinity",
		Args: &config.InterPodAffinityArgs{
			HardPodAffinityWeight: 1,
		},
	},
	{
		Name: "NodeAffinity",
		Args: &config.NodeAffinityArgs{},
	},
	{
		Name: "NodeResourcesBalancedAllocation",
		Args: &config.NodeResourcesBalancedAllocationArgs{
			Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
		},
	},
	{
		Name: "NodeResourcesFit",
		Args: &config.NodeResourcesFitArgs{
			ScoringStrategy: &config.ScoringStrategy{
				Type:      config.LeastAllocated,
				Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
			},
		},
	},
	{
		Name: "NodeResourcesLeastAllocated",
		Args: &config.NodeResourcesLeastAllocatedArgs{
			Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
		},
	},
	{
		Name: "PodTopologySpread",
		Args: &config.PodTopologySpreadArgs{
			DefaultingType: config.SystemDefaulting,
		},
	},
	{
		Name: "VolumeBinding",
		Args: &config.VolumeBindingArgs{
			BindTimeoutSeconds: 600,
		},
	},
}

// PluginsV1beta2 default set of v1beta2 plugins.
var PluginsV1beta2 = &config.Plugins{
	QueueSort: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.PrioritySort},
		},
	},
	PreFilter: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.NodeResourcesFit},
			{Name: names.NodePorts},
			{Name: names.VolumeRestrictions},
			{Name: names.PodTopologySpread},
			{Name: names.InterPodAffinity},
			{Name: names.VolumeBinding},
			{Name: names.NodeAffinity},
		},
	},
	Filter: config.PluginSet{
		Enabled: []config.Plugin{
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
	PostFilter: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.DefaultPreemption},
		},
	},
	PreScore: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.InterPodAffinity},
			{Name: names.PodTopologySpread},
			{Name: names.TaintToleration},
			{Name: names.NodeAffinity},
		},
	},
	Score: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.NodeResourcesBalancedAllocation, Weight: 1},
			{Name: names.ImageLocality, Weight: 1},
			{Name: names.InterPodAffinity, Weight: 1},
			{Name: names.NodeResourcesFit, Weight: 1},
			{Name: names.NodeAffinity, Weight: 1},
			// Weight is doubled because:
			// - This is a score coming from user preference.
			// - It makes its signal comparable to NodeResourcesLeastAllocated.
			{Name: names.PodTopologySpread, Weight: 2},
			{Name: names.TaintToleration, Weight: 1},
		},
	},
	Reserve: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.VolumeBinding},
		},
	},
	PreBind: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.VolumeBinding},
		},
	},
	Bind: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.DefaultBinder},
		},
	},
}

// PluginConfigsV1beta2 default plugin configurations. This could get versioned, but since
// all available versions produce the same defaults, we just have one for now.
var PluginConfigsV1beta2 = []config.PluginConfig{
	{
		Name: "DefaultPreemption",
		Args: &config.DefaultPreemptionArgs{
			MinCandidateNodesPercentage: 10,
			MinCandidateNodesAbsolute:   100,
		},
	},
	{
		Name: "InterPodAffinity",
		Args: &config.InterPodAffinityArgs{
			HardPodAffinityWeight: 1,
		},
	},
	{
		Name: "NodeAffinity",
		Args: &config.NodeAffinityArgs{},
	},
	{
		Name: "NodeResourcesBalancedAllocation",
		Args: &config.NodeResourcesBalancedAllocationArgs{
			Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
		},
	},
	{
		Name: "NodeResourcesFit",
		Args: &config.NodeResourcesFitArgs{
			ScoringStrategy: &config.ScoringStrategy{
				Type:      config.LeastAllocated,
				Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
			},
		},
	},
	{
		Name: "PodTopologySpread",
		Args: &config.PodTopologySpreadArgs{
			DefaultingType: config.SystemDefaulting,
		},
	},
	{
		Name: "VolumeBinding",
		Args: &config.VolumeBindingArgs{
			BindTimeoutSeconds: 600,
		},
	},
}
