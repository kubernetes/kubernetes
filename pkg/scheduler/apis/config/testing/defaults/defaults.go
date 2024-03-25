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

// PluginsV1 is the set of default v1 plugins (before MultiPoint expansion)
var PluginsV1 = &config.Plugins{
	MultiPoint: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.SchedulingGates},
			{Name: names.PrioritySort},
			{Name: names.NodeUnschedulable},
			{Name: names.NodeName},
			{Name: names.TaintToleration, Weight: 3},
			{Name: names.NodeAffinity, Weight: 2},
			{Name: names.NodePorts},
			{Name: names.NodeResourcesFit, Weight: 1},
			{Name: names.VolumeRestrictions},
			{Name: names.EBSLimits},
			{Name: names.GCEPDLimits},
			{Name: names.NodeVolumeLimits},
			{Name: names.AzureDiskLimits},
			{Name: names.VolumeBinding},
			{Name: names.VolumeZone},
			{Name: names.PodTopologySpread, Weight: 2},
			{Name: names.InterPodAffinity, Weight: 2},
			{Name: names.DefaultPreemption},
			{Name: names.NodeResourcesBalancedAllocation, Weight: 1},
			{Name: names.ImageLocality, Weight: 1},
			{Name: names.DefaultBinder},
		},
	},
}

// ExpandedPluginsV1 default set of v1 plugins after MultiPoint expansion
var ExpandedPluginsV1 = &config.Plugins{
	PreEnqueue: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.SchedulingGates},
		},
	},
	QueueSort: config.PluginSet{
		Enabled: []config.Plugin{
			{Name: names.PrioritySort},
		},
	},
	PreFilter: config.PluginSet{
		Enabled: []config.Plugin{
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
			{Name: names.TaintToleration},
			{Name: names.NodeAffinity},
			{Name: names.NodeResourcesFit},
			{Name: names.VolumeBinding},
			{Name: names.PodTopologySpread},
			{Name: names.InterPodAffinity},
			{Name: names.NodeResourcesBalancedAllocation},
		},
	},
	Score: config.PluginSet{
		Enabled: []config.Plugin{
			// Weight is tripled because:
			// - This is a score coming from user preference.
			// - Usage of node tainting to group nodes in the cluster is increasing becoming a use-case
			// for many user workloads
			{Name: names.TaintToleration, Weight: 3},
			// Weight is doubled because:
			// - This is a score coming from user preference.
			{Name: names.NodeAffinity, Weight: 2},
			{Name: names.NodeResourcesFit, Weight: 1},
			// Weight is tripled because:
			// - This is a score coming from user preference.
			// - Usage of node tainting to group nodes in the cluster is increasing becoming a use-case
			//	 for many user workloads
			{Name: names.VolumeBinding, Weight: 1},
			// Weight is doubled because:
			// - This is a score coming from user preference.
			// - It makes its signal comparable to NodeResourcesLeastAllocated.
			{Name: names.PodTopologySpread, Weight: 2},
			// Weight is doubled because:
			// - This is a score coming from user preference.
			{Name: names.InterPodAffinity, Weight: 2},
			{Name: names.NodeResourcesBalancedAllocation, Weight: 1},
			{Name: names.ImageLocality, Weight: 1},
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

// PluginConfigsV1 default plugin configurations.
var PluginConfigsV1 = []config.PluginConfig{
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
